"""AVANTI HTTP API.

Analysis is slow and CPU-bound, so uploads return a job id and the client
polls for progress. Everything the engine produces is available as JSON, so
the same API serves the web UI and any other caller.

Run locally:

    .venv/Scripts/python -m uvicorn server.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from avanti import __version__, compare, extract, llm, ocr, sources
from server import pipeline
from server.jobs import DONE, ERROR, JobStore

# A paper is a few MB; anything much larger is a mistake or an attack.
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

# Bounds on what a caller may ask for, so one request cannot monopolize
# the worker pool on a small free-tier host.
MAX_PAGES_LIMIT = 60
MAX_SUMMARY_WORDS = 1500

STATIC_DIR = Path(__file__).parent / "static"

store = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    store.shutdown()


app = FastAPI(
    title="AVANTI",
    description="Research paper analyzer: layout-aware extraction, "
                "structured findings, and cited question answering.",
    version=__version__,
    lifespan=lifespan,
)

# The UI is served from this same origin, so CORS is only needed if someone
# calls the API from their own page. Allowed origins are opt-in via env.
_origins = [o for o in os.environ.get("AVANTI_CORS", "").split(",") if o]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# ------------------------------------------------------------- schemas


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


class UrlRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    title: str = ""
    max_pages: int = Field(default=30, ge=1, le=MAX_PAGES_LIMIT)
    target_words: int = Field(default=450, ge=150, le=MAX_SUMMARY_WORDS)
    ocr_mode: str = "auto"


class ReportRequest(BaseModel):
    tone: str = "Professional"


# ------------------------------------------------------------- helpers


def _require_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job. It may have expired.")
    return job


def _require_result(job_id: str):
    job = _require_job(job_id)
    if job.status == ERROR:
        raise HTTPException(409, job.error or "Analysis failed.")
    if job.status != DONE or job.analysis is None:
        raise HTTPException(409, f"Analysis is not finished ({job.status}).")
    return job


def _validate_ocr_mode(mode: str) -> str:
    if mode not in ("auto", "always", "never"):
        raise HTTPException(422, "ocr_mode must be auto, always or never.")
    return mode


def _submit(data: bytes, name: str, max_pages: int,
            target_words: int, ocr_mode: str):
    def worker(progress):
        return pipeline.run(
            data,
            max_pages=max_pages,
            target_words=target_words,
            ocr_mode=ocr_mode,
            progress=progress,
        )

    return store.submit(worker, name=name)


# -------------------------------------------------------------- routes


@app.get("/api/health")
def health() -> dict:
    """Capabilities, so the UI can hide what this deployment cannot do."""
    return {
        "status": "ok",
        "version": __version__,
        "pdfBackend": extract.backend_name(),
        "ocr": {
            "available": ocr.available(),
            "backend": ocr.backend_label(),
            "secondsPerPage": 10,
        },
        "aiReport": llm.has_key(),
        "search": {
            "sources": [{"id": k, "name": v} for k, v in sources.SOURCES.items()],
            "maxResults": sources.MAX_RESULTS,
        },
        "limits": {
            "maxUploadMb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "maxPages": MAX_PAGES_LIMIT,
        },
    }


@app.post("/api/analyze", status_code=202)
async def analyze_upload(
    file: UploadFile = File(...),
    max_pages: int = Form(30),
    target_words: int = Form(450),
    ocr_mode: str = Form("auto"),
) -> dict:
    """Upload a PDF for analysis. Returns a job id to poll."""
    _validate_ocr_mode(ocr_mode)

    data = await file.read()
    if not data:
        raise HTTPException(422, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File is {len(data) / 1e6:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    # Check the magic bytes rather than trusting the filename.
    if data[:4] != b"%PDF":
        raise HTTPException(422, "That file is not a PDF.")

    job = _submit(
        data,
        file.filename or "upload.pdf",
        max(1, min(max_pages, MAX_PAGES_LIMIT)),
        max(150, min(target_words, MAX_SUMMARY_WORDS)),
        ocr_mode,
    )
    return job.snapshot()


@app.post("/api/analyze-url", status_code=202)
def analyze_url(req: UrlRequest) -> dict:
    """Analyze a PDF by URL, e.g. straight from an arXiv search result."""
    _validate_ocr_mode(req.ocr_mode)
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(422, "URL must be http or https.")

    try:
        data = sources.fetch_pdf(req.url, max_bytes=MAX_UPLOAD_BYTES)
    except sources.UnsafeURL as exc:
        raise HTTPException(422, str(exc))
    except requests.RequestException as exc:
        raise HTTPException(502, f"Could not download that PDF: {exc}")
    if data is None:
        raise HTTPException(
            422, "That link did not return a usable PDF within the size limit."
        )

    job = _submit(
        data, req.title or req.url, req.max_pages, req.target_words, req.ocr_mode
    )
    return job.snapshot()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    """Poll a job's stage and progress."""
    return _require_job(job_id).snapshot()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> dict:
    """The full analysis, once the job is done."""
    job = _require_result(job_id)
    return {"job": job.snapshot(), **(job.result or {})}


@app.post("/api/jobs/{job_id}/ask")
def ask(job_id: str, req: AskRequest) -> dict:
    """Answer a question from the paper's own sentences, cited to a page."""
    job = _require_result(job_id)
    answers = job.analysis.index.ask(req.question, k=req.k)
    return {
        "question": req.question,
        "answers": pipeline.answers_json(answers),
    }


@app.post("/api/jobs/{job_id}/ai-report")
def ai_report(job_id: str, req: ReportRequest) -> dict:
    """Optional Gemini write-up. Only useful when a key is configured."""
    if not llm.has_key():
        raise HTTPException(
            503, "No Gemini API key is configured on this deployment."
        )
    job = _require_result(job_id)
    text = llm.deep_analysis(
        job.analysis.paper.body, job.analysis.doc.title, req.tone
    )
    return {"report": text}


@app.get("/api/jobs/{job_id}/text")
def job_text(job_id: str) -> dict:
    """The extracted text the analysis actually read."""
    job = _require_result(job_id)
    return {"text": job.analysis.doc.text}


@app.get("/api/search")
def search(q: str, limit: int = 10, source: str = "all") -> dict:
    """Search arXiv, OpenAlex and Europe PMC -- one of them, or all at once."""
    if not q.strip():
        raise HTTPException(422, "Empty query.")
    if source != "all" and source not in sources.SOURCES:
        raise HTTPException(
            422, f"source must be all or one of: {', '.join(sources.SOURCES)}."
        )
    try:
        found = sources.search(q, max(1, min(limit, sources.MAX_RESULTS)), source)
    except requests.RequestException as exc:
        raise HTTPException(502, f"Search failed: {exc}")
    return {
        "query": q,
        "source": source,
        "failed": found.failed,
        "results": [
            {
                "title": r.title,
                "authors": r.authors,
                "authorLine": r.author_line,
                "published": r.published,
                "summary": r.summary,
                "abstractUrl": r.abstract_url,
                "pdfUrl": r.pdf_url,
                "source": r.source,
            }
            for r in found.results
        ],
    }


@app.get("/api/compare")
def compare_jobs(a: str, b: str) -> dict:
    """Compare two finished analyses side by side.

    There is deliberately no endpoint listing jobs: a job id is what grants
    access to an analysis, so the client remembers its own ids and nobody
    can browse anyone else's papers.
    """
    if a == b:
        raise HTTPException(422, "Pick two different papers to compare.")
    first, second = _require_result(a), _require_result(b)
    return pipeline.comparison_json(
        compare.compare(first.analysis, second.analysis)
    )


# ----------------------------------------------------------------- UI

# Mounted last so the API routes above always win. The directory may not
# exist yet while the frontend is being built.
if STATIC_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static"
    )
else:

    @app.get("/")
    def placeholder() -> JSONResponse:
        return JSONResponse(
            {
                "name": "AVANTI",
                "version": __version__,
                "note": "API is running. The web UI is not built yet.",
                "docs": "/docs",
            }
        )
