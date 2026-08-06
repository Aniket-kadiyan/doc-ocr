"""Fast lifecycle tests for atomic, in-memory auto-balloon scan jobs."""

from __future__ import annotations

from threading import Event
from time import monotonic, sleep

from scan_jobs import ScanJobManager


def _wait_for_terminal(
    manager: ScanJobManager,
    job_id: str,
    *,
    timeout: float = 2.0,
) -> dict:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        snapshot = manager.get(job_id)
        if snapshot and snapshot["status"] in {"succeeded", "failed"}:
            return snapshot
        sleep(0.01)
    raise AssertionError("scan job did not finish")


def test_result_is_hidden_until_the_complete_job_succeeds() -> None:
    manager = ScanJobManager(max_workers=1)
    started = Event()
    release = Event()

    def work(report):
        report(
            stage="detecting",
            message="Detected candidates",
            percent=25,
            completed=3,
            total=3,
        )
        started.set()
        assert release.wait(1.0)
        report(
            stage="recognizing",
            message="Recognized candidates",
            percent=90,
            completed=3,
            total=3,
        )
        return {"count": 1, "regions": [{"text": "25"}]}

    try:
        initial = manager.submit(work, metadata={"scope_kind": "section"})
        assert initial["result"] is None
        assert started.wait(1.0)
        running = manager.get(initial["job_id"])
        assert running is not None
        assert running["status"] == "running"
        assert running["percent"] == 25
        assert running["result"] is None

        release.set()
        complete = _wait_for_terminal(manager, initial["job_id"])
        assert complete["status"] == "succeeded"
        assert complete["percent"] == 100
        assert complete["result"] == {
            "count": 1,
            "regions": [{"text": "25"}],
        }
        assert complete["metadata"] == {"scope_kind": "section"}
    finally:
        release.set()
        manager.shutdown()


def test_failure_publishes_an_error_but_no_partial_result() -> None:
    manager = ScanJobManager(max_workers=1)

    def work(report):
        report(
            stage="recognizing",
            message="Recognizing object 1 of 2",
            percent=60,
            completed=0,
            total=2,
        )
        raise RuntimeError("synthetic OCR failure")

    try:
        initial = manager.submit(work)
        failed = _wait_for_terminal(manager, initial["job_id"])
        assert failed["status"] == "failed"
        assert failed["error"] == "synthetic OCR failure"
        assert failed["result"] is None
        assert failed["percent"] == 60
    finally:
        manager.shutdown()


def test_reported_percentage_never_moves_backwards() -> None:
    manager = ScanJobManager(max_workers=1)
    reported = Event()
    release = Event()

    def work(report):
        report(stage="detecting", message="First", percent=70)
        report(stage="grouping", message="Second", percent=30)
        reported.set()
        assert release.wait(1.0)
        return {"count": 0, "regions": []}

    try:
        initial = manager.submit(work)
        assert reported.wait(1.0)
        running = manager.get(initial["job_id"])
        assert running is not None
        assert running["percent"] == 70
        release.set()
        complete = _wait_for_terminal(manager, initial["job_id"])
        assert complete["status"] == "succeeded"
        assert complete["percent"] == 100
    finally:
        release.set()
        manager.shutdown()


def test_heartbeat_and_area_aware_liveness_continue_during_blocking_work() -> None:
    manager = ScanJobManager(
        max_workers=1,
        heartbeat_interval_seconds=0.01,
        heartbeat_stale_seconds=1.0,
        long_step_base_seconds=0.02,
        stalled_step_multiplier=2.0,
    )
    started = Event()
    release = Event()

    def work(report):
        report(
            stage="detecting",
            message="Running detection pass 1 of 33",
            percent=7,
            completed=0,
            total=33,
            pass_current=1,
            pass_total=33,
            tile_current=1,
            tile_total=1,
            operation_label="source contrast",
        )
        started.set()
        assert release.wait(1.0)
        return {"count": 0, "regions": []}

    try:
        initial = manager.submit(
            work,
            metadata={"scope_pixel_area": 1},
        )
        assert started.wait(1.0)
        first = manager.get(initial["job_id"])
        assert first is not None
        first_heartbeat = first["heartbeat_at"]
        with manager._lock:
            manager._jobs[initial["job_id"]].started_at -= 10.0
        timing = manager.get(initial["job_id"])
        assert timing is not None
        assert timing["elapsed_seconds"] >= 10
        # No unit has completed yet, so a multi-hour percent extrapolation is
        # intentionally withheld while the UI says "Calculating estimate".
        assert timing["estimated_remaining_seconds"] is None

        sleep(0.07)
        long_running = manager.get(initial["job_id"])
        assert long_running is not None
        assert long_running["heartbeat_at"] > first_heartbeat
        assert long_running["heartbeat_age_seconds"] == 0
        assert long_running["liveness"] == "long_running"
        assert long_running["pass_current"] == 1
        assert long_running["pass_total"] == 33
        assert long_running["result"] is None

        # Move only the no-forward-progress timestamp beyond the configured
        # threshold. The independently ticking heartbeat remains current.
        with manager._lock:
            manager._jobs[initial["job_id"]].last_progress_at -= 3.0
        possibly_stalled = manager.get(initial["job_id"])
        assert possibly_stalled is not None
        assert possibly_stalled["liveness"] == "possibly_stalled"
        assert possibly_stalled["progress_age_seconds"] >= 3

        release.set()
        complete = _wait_for_terminal(manager, initial["job_id"])
        assert complete["liveness"] == "complete"
        assert complete["finished_at"] is not None
    finally:
        release.set()
        manager.shutdown()


def test_eta_uses_completed_units_from_the_current_stage() -> None:
    manager = ScanJobManager(max_workers=1)
    started = Event()
    finish = Event()

    def work(report):
        report(
            stage="detecting",
            message="Running primary detector 1 of 2",
            percent=10,
            completed=0,
            total=2,
            pass_current=1,
            pass_total=2,
        )
        report(
            stage="detecting",
            message="Completed primary detector 1 of 2",
            percent=16,
            completed=1,
            total=2,
            pass_current=1,
            pass_total=2,
        )
        started.set()
        assert finish.wait(1.0)
        return {"count": 0, "regions": []}

    try:
        initial = manager.submit(work)
        assert started.wait(1.0)
        with manager._lock:
            job = manager._jobs[initial["job_id"]]
            job.stage_started_at -= 10.0
            job.started_at -= 100.0

        snapshot = manager.get(initial["job_id"])
        assert snapshot is not None
        # One comparable unit took about ten seconds and one remains. The old
        # global-percent formula would incorrectly project several minutes.
        assert 9 <= snapshot["estimated_remaining_seconds"] <= 11

        finish.set()
        complete = _wait_for_terminal(manager, initial["job_id"])
        assert complete["status"] == "succeeded"
    finally:
        finish.set()
        manager.shutdown()
