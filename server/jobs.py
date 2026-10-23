"""Background analysis jobs with progress reporting.

Analyzing a paper is slow and CPU-bound -- a scanned PDF runs about ten
seconds a page through OCR, so a twenty-page scan is several minutes. That
cannot happen inside a request: the connection would time out and the
caller would see nothing until the end.

So an upload returns a job id immediately, the work runs on a small thread
pool, and the caller polls for stage and progress. The pool is deliberately
small because the work is CPU-bound; more workers would only make every job
slower.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# CPU-bound work: more workers means every job finishes later, not sooner.
MAX_WORKERS = 2

# Finished jobs are dropped after this long, so a long-running server does
# not accumulate parsed documents in memory for ever.
JOB_TTL_SECONDS = 60 * 60

# Hard ceiling on retained jobs, whatever the TTL says.
MAX_JOBS = 60

QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"


@dataclass
class Job:
    """One analysis, from upload to result."""

    id: str
    name: str = ""
    status: str = QUEUED
    stage: str = "queued"
    progress: float = 0.0
    detail: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    # The live Analysis object, kept server-side: the question index holds a
    # sparse matrix and cannot be serialized, so follow-up questions are
    # answered here rather than in the client.
    analysis: Any = None
    result: Optional[dict] = None

    def snapshot(self) -> dict:
        """The JSON-safe status view a client polls for."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "detail": self.detail,
            "error": self.error,
            "elapsed": round((self.updated or time.time()) - self.created, 1),
        }


class Progress:
    """Handed to the worker so it can report where it has got to."""

    def __init__(self, job: Job, store: "JobStore"):
        self._job = job
        self._store = store

    def __call__(
        self, stage: str, progress: float, detail: str = ""
    ) -> None:
        self._store.update(
            self._job.id, stage=stage, progress=progress, detail=detail
        )


class JobStore:
    """Thread-safe registry of jobs plus the pool that runs them."""

    def __init__(self, max_workers: int = MAX_WORKERS):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="avanti"
        )

    # ---------------------------------------------------------- lifecycle

    def submit(
        self,
        worker: Callable[[Progress], Any],
        name: str = "",
    ) -> Job:
        """Queue a job. `worker` receives a Progress callable."""
        self._evict()

        job = Job(id=uuid.uuid4().hex[:12], name=name)
        with self._lock:
            self._jobs[job.id] = job

        self._pool.submit(self._run, job, worker)
        return job

    def _run(self, job: Job, worker: Callable[[Progress], Any]) -> None:
        self.update(job.id, status=RUNNING, stage="starting", progress=0.02)
        try:
            analysis, result = worker(Progress(job, self))
        except Exception as exc:  # a worker failure must not kill the pool
            self.update(
                job.id,
                status=ERROR,
                stage="failed",
                progress=1.0,
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        with self._lock:
            current = self._jobs.get(job.id)
            if current is not None:
                current.analysis = analysis
                current.result = result
        self.update(
            job.id, status=DONE, stage="done", progress=1.0, detail=""
        )

    # ------------------------------------------------------------ access

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def listing(self) -> List[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: -j.created)
        return [j.snapshot() for j in jobs]

    # ---------------------------------------------------------- eviction

    def _evict(self) -> None:
        """Drop finished jobs that are old, and trim to the hard ceiling."""
        now = time.time()
        with self._lock:
            stale = [
                jid
                for jid, job in self._jobs.items()
                if job.status in (DONE, ERROR)
                and now - job.updated > JOB_TTL_SECONDS
            ]
            for jid in stale:
                self._jobs.pop(jid, None)

            if len(self._jobs) > MAX_JOBS:
                # Oldest first, but never evict work still in flight.
                finished = sorted(
                    (j for j in self._jobs.values()
                     if j.status in (DONE, ERROR)),
                    key=lambda j: j.updated,
                )
                for job in finished[: len(self._jobs) - MAX_JOBS]:
                    self._jobs.pop(job.id, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
