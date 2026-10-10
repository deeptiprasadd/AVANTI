"""Tests for the HTTP layer.

The analysis engine is exercised by the other test modules, so the pipeline
is stubbed here. What these tests check is the contract the frontend
depends on: that an upload returns immediately with a job id, that polling
reports progress, that results come back in the documented shape, and that
bad input is rejected with a useful message rather than a 500.
"""

import time

import pytest
from fastapi.testclient import TestClient

from server import jobs, main, pipeline

client = TestClient(main.app)

MINIMAL_PDF = b"%PDF-1.4\n% a real parse is not attempted in these tests\n"


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    """Give every test its own job store, so they cannot interfere."""
    store = jobs.JobStore(max_workers=1)
    monkeypatch.setattr(main, "store", store)
    yield store
    store.shutdown()


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace the slow engine with something instant and predictable."""

    class FakeIndex:
        def __len__(self):
            return 3

        def ask(self, question, k=5):
            from avanti.qa import Answer

            return [Answer(text=f"Answer to {question}", page=4,
                           section="Results", score=0.42)][:k]

    class FakeAnalysis:
        class doc:
            title = "A Test Paper"
            text = "Extracted body text."

        class paper:
            body = "Body."

        index = FakeIndex()

    def fake_run(data, *, max_pages, target_words, ocr_mode, progress=None):
        if progress:
            progress("extracting", 0.3, "Reading the PDF")
            progress("indexing", 0.9, "Indexing")
        return FakeAnalysis(), {
            "document": {"title": "A Test Paper", "wordCount": 1234,
                         "usedOcr": False, "pagesProcessed": 5},
            "brief": {"objective": [], "metrics": [], "datasets": []},
            "summary": {"keyPoints": [], "sections": []},
            "sections": [],
            "suggestedQuestions": ["What problem does this paper address?"],
            "passageCount": 3,
        }

    monkeypatch.setattr(pipeline, "run", fake_run)


def wait_for(store, job_id, timeout=10.0):
    """Block until a job leaves the running state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job and job.status in (jobs.DONE, jobs.ERROR):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# ----------------------------------------------------------------- health


def test_health_reports_capabilities():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    # The frontend hides the OCR and AI controls based on these.
    assert isinstance(body["ocr"]["available"], bool)
    assert isinstance(body["aiReport"], bool)
    assert body["limits"]["maxUploadMb"] > 0


# ------------------------------------------------------------- validation


def test_non_pdf_is_rejected_by_content_not_filename():
    """A .pdf name on non-PDF bytes must still be refused."""
    r = client.post(
        "/api/analyze",
        files={"file": ("sneaky.pdf", b"GIF89a totally an image", "application/pdf")},
    )
    assert r.status_code == 422
    assert "not a PDF" in r.json()["detail"]


def test_empty_upload_is_rejected():
    r = client.post("/api/analyze", files={"file": ("x.pdf", b"", "application/pdf")})
    assert r.status_code == 422


def test_bad_ocr_mode_is_rejected():
    r = client.post(
        "/api/analyze",
        files={"file": ("x.pdf", MINIMAL_PDF, "application/pdf")},
        data={"ocr_mode": "turbo"},
    )
    assert r.status_code == 422
    assert "auto" in r.json()["detail"]


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    r = client.post(
        "/api/analyze",
        files={"file": ("big.pdf", b"%PDF" + b"x" * 2000, "application/pdf")},
    )
    assert r.status_code == 413


def test_non_http_url_is_rejected():
    """A file:// URL must not reach the downloader."""
    r = client.post("/api/analyze-url", json={"url": "file:///etc/passwd"})
    assert r.status_code == 422


def test_unknown_job_is_404():
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/result").status_code == 404


def test_empty_search_is_rejected():
    assert client.get("/api/search", params={"q": "   "}).status_code == 422


# --------------------------------------------------------- job lifecycle


def test_upload_returns_immediately_with_a_job_id(fresh_store, fake_pipeline):
    """The response must not wait for the analysis to finish."""
    r = client.post(
        "/api/analyze",
        files={"file": ("paper.pdf", MINIMAL_PDF, "application/pdf")},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["id"] and body["name"] == "paper.pdf"
    assert body["status"] in ("queued", "running", "done")


def test_job_runs_to_completion_and_returns_the_result(fresh_store, fake_pipeline):
    job_id = client.post(
        "/api/analyze",
        files={"file": ("paper.pdf", MINIMAL_PDF, "application/pdf")},
    ).json()["id"]

    wait_for(fresh_store, job_id)

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "done"
    assert status["progress"] == 1.0

    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["document"]["title"] == "A Test Paper"
    assert result["passageCount"] == 3
    assert "job" in result


def test_result_before_completion_is_409(fresh_store, monkeypatch):
    """Asking for a result too early must not 500 or return half a payload."""

    def slow_run(data, *, max_pages, target_words, ocr_mode, progress=None):
        time.sleep(0.6)
        return None, {}

    monkeypatch.setattr(pipeline, "run", slow_run)
    job_id = client.post(
        "/api/analyze",
        files={"file": ("paper.pdf", MINIMAL_PDF, "application/pdf")},
    ).json()["id"]

    r = client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 409
    assert "not finished" in r.json()["detail"]


def test_engine_failure_becomes_an_error_job_not_a_crash(fresh_store, monkeypatch):
    def boom(data, **kwargs):
        raise ValueError("corrupt page tree")

    monkeypatch.setattr(pipeline, "run", boom)
    job_id = client.post(
        "/api/analyze",
        files={"file": ("bad.pdf", MINIMAL_PDF, "application/pdf")},
    ).json()["id"]

    job = wait_for(fresh_store, job_id)
    assert job.status == jobs.ERROR

    status = client.get(f"/api/jobs/{job_id}").json()
    assert "corrupt page tree" in status["error"]
    # The result endpoint surfaces the failure rather than pretending.
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409


def test_ask_returns_cited_answers(fresh_store, fake_pipeline):
    job_id = client.post(
        "/api/analyze",
        files={"file": ("paper.pdf", MINIMAL_PDF, "application/pdf")},
    ).json()["id"]
    wait_for(fresh_store, job_id)

    r = client.post(
        f"/api/jobs/{job_id}/ask", json={"question": "what is the method"}
    )
    assert r.status_code == 200
    answer = r.json()["answers"][0]
    # A citation is the whole point: an answer without a page is useless.
    assert answer["page"] == 4
    assert answer["section"] == "Results"


def test_ask_rejects_an_empty_question(fresh_store, fake_pipeline):
    job_id = client.post(
        "/api/analyze",
        files={"file": ("paper.pdf", MINIMAL_PDF, "application/pdf")},
    ).json()["id"]
    wait_for(fresh_store, job_id)
    assert client.post(f"/api/jobs/{job_id}/ask", json={"question": "a"}).status_code == 422


def test_ai_report_is_503_without_a_key(fresh_store, fake_pipeline, monkeypatch):
    monkeypatch.setattr(main.llm, "has_key", lambda: False)
    job_id = client.post(
        "/api/analyze",
        files={"file": ("paper.pdf", MINIMAL_PDF, "application/pdf")},
    ).json()["id"]
    wait_for(fresh_store, job_id)

    r = client.post(f"/api/jobs/{job_id}/ai-report", json={"tone": "Professional"})
    assert r.status_code == 503


# ------------------------------------------------------------- job store


def test_store_evicts_finished_jobs_past_the_ceiling(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    store = jobs.JobStore(max_workers=1)
    try:
        for _ in range(8):
            job = store.submit(lambda p: (None, {}))
            wait_for(store, job.id, timeout=5)
        store._evict()
        assert len(store.listing()) <= 3
    finally:
        store.shutdown()


def test_progress_from_the_worker_is_visible_to_a_poller():
    """A long OCR run is only tolerable if progress is observable mid-flight."""
    store = jobs.JobStore(max_workers=1)
    release = __import__("threading").Event()
    try:
        def worker(progress):
            progress("extracting", 0.25, "page 1 of 4")
            release.wait(timeout=5)   # hold the job open mid-run
            return None, {}

        job = store.submit(worker)

        # Poll from this thread while the worker is still inside the job.
        deadline = time.time() + 5
        snapshot = None
        while time.time() < deadline:
            snapshot = store.get(job.id).snapshot()
            if snapshot["detail"]:
                break
            time.sleep(0.02)

        assert snapshot["stage"] == "extracting"
        assert snapshot["progress"] == pytest.approx(0.25)
        assert snapshot["detail"] == "page 1 of 4"
        assert snapshot["status"] == jobs.RUNNING

        release.set()
        wait_for(store, job.id, timeout=5)
        assert store.get(job.id).status == jobs.DONE
    finally:
        release.set()
        store.shutdown()
