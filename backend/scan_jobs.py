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
from math import sqrt
from threading import Event, Lock, Thread
from time import time
from typing import Any, Callable, Literal
from uuid import uuid4

ScanJobStatus = Literal["queued", "running", "succeeded", "failed"]
ScanLiveness = Literal[
    "queued",
    "working",
    "long_running",
    "slow_progress",
    "possibly_stalled",
    "complete",
    "failed",
]
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
    pass_current: int = 0
    pass_total: int = 0
    tile_current: int = 0
    tile_total: int = 0
    object_current: int = 0
    object_total: int = 0
    candidate_count: int = 0
    operation_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    started_at: float | None = None
    finished_at: float | None = None
    heartbeat_at: float = field(default_factory=time)
    last_progress_at: float = field(default_factory=time)
    step_started_at: float = field(default_factory=time)
    stage_started_at: float = field(default_factory=time)
    stage_start_completed: int = 0


class ScanJobManager:
    """Run scan work off the request thread and expose immutable snapshots."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        retention_seconds: float = 60 * 60,
        heartbeat_interval_seconds: float = 1.0,
        heartbeat_stale_seconds: float = 6.0,
        long_step_base_seconds: float = 12.0,
        stalled_step_multiplier: float = 4.0,
    ) -> None:
        self._jobs: dict[str, _ScanJob] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="auto-balloon-scan",
        )
        self._retention_seconds = retention_seconds
        self._heartbeat_interval_seconds = max(0.01, heartbeat_interval_seconds)
        self._heartbeat_stale_seconds = max(
            self._heartbeat_interval_seconds * 2,
            heartbeat_stale_seconds,
        )
        self._long_step_base_seconds = max(0.01, long_step_base_seconds)
        self._stalled_step_multiplier = max(2.0, stalled_step_multiplier)

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
        started_at = time()
        self._update(
            job_id,
            status="running",
            stage="preparing",
            message="Preparing selected area",
            percent=1,
            started_at=started_at,
            heartbeat_at=started_at,
        )
        heartbeat_stop = Event()
        heartbeat_thread = Thread(
            target=self._heartbeat_loop,
            args=(job_id, heartbeat_stop),
            name=f"scan-heartbeat-{job_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()

        def report_progress(
            *,
            stage: str,
            message: str,
            percent: int,
            completed: int = 0,
            total: int = 0,
            pass_current: int = 0,
            pass_total: int = 0,
            tile_current: int = 0,
            tile_total: int = 0,
            object_current: int = 0,
            object_total: int = 0,
            candidate_count: int = 0,
            operation_label: str = "",
        ) -> None:
            self._update(
                job_id,
                status="running",
                stage=stage,
                message=message,
                percent=percent,
                completed=completed,
                total=total,
                pass_current=pass_current,
                pass_total=pass_total,
                tile_current=tile_current,
                tile_total=tile_total,
                object_current=object_current,
                object_total=object_total,
                candidate_count=candidate_count,
                operation_label=operation_label,
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
                finished_at=time(),
            )
        else:
            self._update(
                job_id,
                status="succeeded",
                stage="complete",
                message="Scan complete",
                percent=100,
                result=result,
                error=None,
                finished_at=time(),
            )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(
                timeout=self._heartbeat_interval_seconds * 2,
            )

    def _heartbeat_loop(self, job_id: str, stop: Event) -> None:
        """Keep liveness current even while one OCR call blocks its worker."""

        while not stop.wait(self._heartbeat_interval_seconds):
            with self._lock:
                job = self._jobs.get(job_id)
                if not job or job.status != "running":
                    return
                job.heartbeat_at = time()

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            now = time()
            incoming_stage = changes.get("stage", job.stage)
            stage_changed = incoming_stage != job.stage
            progress_fields = {
                "stage",
                "message",
                "percent",
                "completed",
                "total",
                "pass_current",
                "pass_total",
                "tile_current",
                "tile_total",
                "object_current",
                "object_total",
                "candidate_count",
                "operation_label",
            }
            step_fields = progress_fields - {"percent", "completed", "total"}
            made_progress = False
            started_step = False
            for name, value in changes.items():
                if name == "percent":
                    value = max(job.percent, min(100, int(value)))
                if name in progress_fields and getattr(job, name) != value:
                    made_progress = True
                if name in step_fields and getattr(job, name) != value:
                    started_step = True
                setattr(job, name, value)
            if stage_changed:
                job.stage_started_at = now
                job.stage_start_completed = int(changes.get("completed", 0))
            elif (
                "completed" in changes
                and int(changes["completed"]) < job.stage_start_completed
            ):
                # A phase can discover a new work plan while retaining its
                # public stage name. Restart its rate sample instead of mixing
                # unrelated work-unit counts.
                job.stage_started_at = now
                job.stage_start_completed = int(changes["completed"])
            if made_progress:
                job.last_progress_at = now
            if started_step:
                job.step_started_at = now
            if changes.get("status") in {"succeeded", "failed"}:
                job.heartbeat_at = now
            job.updated_at = now

    def _discard_expired_locked(self) -> None:
        cutoff = time() - self._retention_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"succeeded", "failed"} and job.updated_at < cutoff
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)

    def _liveness_thresholds(self, job: _ScanJob) -> tuple[float, float]:
        scope_bbox = job.metadata.get("scope_bbox", {})
        area = float(
            job.metadata.get("scope_pixel_area")
            or (
                float(scope_bbox.get("width", 0))
                * float(scope_bbox.get("height", 0))
            )
        )
        area_megapixels = max(0.0, area / 1_000_000.0)
        long_running_after = self._long_step_base_seconds + min(
            18.0,
            4.0 * sqrt(area_megapixels),
        )
        stalled_after = max(
            long_running_after * self._stalled_step_multiplier,
            self._heartbeat_stale_seconds * 2,
        )
        return long_running_after, stalled_after

    def _snapshot(self, job: _ScanJob) -> dict[str, Any]:
        now = time()
        elapsed = (
            max(0.0, (job.finished_at or now) - job.started_at)
            if job.started_at is not None
            else 0.0
        )
        heartbeat_age = max(0.0, now - job.heartbeat_at)
        progress_age = max(0.0, now - job.last_progress_at)
        step_elapsed = (
            max(0.0, (job.finished_at or now) - job.step_started_at)
            if job.started_at is not None
            else 0.0
        )
        long_running_after, stalled_after = self._liveness_thresholds(job)

        liveness: ScanLiveness
        if job.status == "queued":
            liveness = "queued"
        elif job.status == "succeeded":
            liveness = "complete"
        elif job.status == "failed":
            liveness = "failed"
        elif heartbeat_age > self._heartbeat_stale_seconds:
            liveness = "possibly_stalled"
        elif progress_age > stalled_after:
            # The independent service heartbeat is current, so distinguish a
            # slow/hung work unit from an unavailable scan service.
            liveness = "slow_progress"
        elif step_elapsed > long_running_after:
            liveness = "long_running"
        else:
            liveness = "working"

        # ETA is intentionally stage-local. Global percent is weighted for UI
        # presentation and made the former estimate explode into hours when one
        # detector pass was slower than later work. A stage estimate appears
        # only after at least one comparable unit has actually completed.
        eta_seconds: int | None = None
        stage_elapsed = max(0.0, now - job.stage_started_at)
        stage_completed = max(0, job.completed - job.stage_start_completed)
        stage_remaining = max(0, job.total - job.completed)
        if (
            job.status == "running"
            and stage_remaining > 0
            and stage_completed >= 1
            and stage_elapsed >= 1.0
        ):
            seconds_per_unit = stage_elapsed / stage_completed
            eta_seconds = max(
                0,
                int(round(seconds_per_unit * stage_remaining)),
            )

        snapshot: dict[str, Any] = {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage,
            "message": job.message,
            "percent": job.percent,
            "completed": job.completed,
            "total": job.total,
            "pass_current": job.pass_current,
            "pass_total": job.pass_total,
            "tile_current": job.tile_current,
            "tile_total": job.tile_total,
            "object_current": job.object_current,
            "object_total": job.object_total,
            "candidate_count": job.candidate_count,
            "operation_label": job.operation_label,
            "metadata": deepcopy(job.metadata),
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "heartbeat_at": job.heartbeat_at,
            "last_progress_at": job.last_progress_at,
            "elapsed_seconds": int(elapsed),
            "step_elapsed_seconds": int(step_elapsed),
            "heartbeat_age_seconds": int(heartbeat_age),
            "progress_age_seconds": int(progress_age),
            "estimated_remaining_seconds": eta_seconds,
            "liveness": liveness,
        }
        # Results are published only after the complete scan succeeds.
        snapshot["result"] = (
            deepcopy(job.result) if job.status == "succeeded" else None
        )
        return snapshot
