"""Thread-safe in-memory jobs for long-running auto-balloon scans.

The OCR service is deployed as one local process, so Milestone 1 deliberately
keeps job state in memory.  The public API is independent of OCR internals: a
caller submits a function that receives a progress reporter and returns the
complete scan result.  Nothing exposes partial candidates.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from time import time
from typing import Any, Callable, Literal
from uuid import uuid4

ScanJobStatus = Literal["queued", "running", "succeeded", "failed"]
ProgressReporter = Callable[..., None]
ScanWork = Callable[[ProgressReporter], dict[str, Any]]


@dataclass
class _ScanJob:
    job_id: str
    status: ScanJobStatus = "queued"
    stage: str = "queued"
    message: str = "Scan queued"
    percent: int = 0
    completed: int = 0
    total: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


class ScanJobManager:
    """Run scan work off the request thread and expose immutable snapshots."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        retention_seconds: float = 60 * 60,
    ) -> None:
        self._jobs: dict[str, _ScanJob] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="auto-balloon-scan",
        )
        self._retention_seconds = retention_seconds

    def submit(
        self,
        work: ScanWork,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Queue one scan and return its initial public snapshot."""
        job = _ScanJob(job_id=str(uuid4()), metadata=dict(metadata or {}))
        with self._lock:
            self._discard_expired_locked()
            self._jobs[job.job_id] = job
        self._executor.submit(self._run, job.job_id, work)
        return self.get(job.job_id) or {}

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return a detached public snapshot, or ``None`` for an unknown job."""
        with self._lock:
            job = self._jobs.get(job_id)
            return self._snapshot(job) if job else None

    def shutdown(self, *, wait: bool = True) -> None:
        """Release worker threads; primarily used by fast unit tests."""
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run(self, job_id: str, work: ScanWork) -> None:
        self._update(
            job_id,
            status="running",
            stage="preparing",
            message="Preparing selected area",
            percent=1,
        )

        def report_progress(
            *,
            stage: str,
            message: str,
            percent: int,
            completed: int = 0,
            total: int = 0,
        ) -> None:
            self._update(
                job_id,
                status="running",
                stage=stage,
                message=message,
                percent=percent,
                completed=completed,
                total=total,
            )

        try:
            result = work(report_progress)
        except Exception as exc:  # noqa: BLE001 - the job must expose all failures
            self._update(
                job_id,
                status="failed",
                stage="failed",
                message="Scan failed",
                error=str(exc) or exc.__class__.__name__,
            )
            return

        self._update(
            job_id,
            status="succeeded",
            stage="complete",
            message="Scan complete",
            percent=100,
            result=result,
            error=None,
        )

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for name, value in changes.items():
                if name == "percent":
                    value = max(job.percent, min(100, int(value)))
                setattr(job, name, value)
            job.updated_at = time()

    def _discard_expired_locked(self) -> None:
        cutoff = time() - self._retention_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"succeeded", "failed"} and job.updated_at < cutoff
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)

    @staticmethod
    def _snapshot(job: _ScanJob) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage,
            "message": job.message,
            "percent": job.percent,
            "completed": job.completed,
            "total": job.total,
            "metadata": deepcopy(job.metadata),
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        # Results are published only after the complete scan succeeds.
        snapshot["result"] = (
            deepcopy(job.result) if job.status == "succeeded" else None
        )
        return snapshot
