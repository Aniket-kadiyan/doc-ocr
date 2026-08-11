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
        if snapshot and snapshot["status"] in {
            "cancelled",
            "succeeded",
            "failed",
        }:
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


def test_running_job_cancels_after_the_current_work_unit_without_result() -> None:
    manager = ScanJobManager(
        max_workers=1,
        heartbeat_interval_seconds=0.01,
    )
    model_call_started = Event()
    model_call_finished = Event()

    def work(report):
        report(
            stage="recognizing",
            message="Running one blocking OCR operation",
            percent=60,
            completed=0,
            total=2,
        )
        model_call_started.set()
        assert model_call_finished.wait(1.0)
        # This progress checkpoint observes the cancellation request and aborts
        # before the synthetic partial result can be returned.
        report(
            stage="recognizing",
            message="OCR operation complete",
            percent=80,
            completed=1,
            total=2,
        )
        return {"count": 1, "regions": [{"text": "25"}]}

    try:
        initial = manager.submit(work)
        assert model_call_started.wait(1.0)

        cancelling = manager.cancel(initial["job_id"])
        assert cancelling is not None
        assert cancelling["status"] == "cancelling"
        assert cancelling["liveness"] == "cancelling"
        assert cancelling["result"] is None

        model_call_finished.set()
        cancelled = _wait_for_terminal(manager, initial["job_id"])
        assert cancelled["status"] == "cancelled"
        assert cancelled["liveness"] == "cancelled"
        assert cancelled["result"] is None
        assert cancelled["error"] is None
        assert cancelled["percent"] == 60
    finally:
        model_call_finished.set()
        manager.shutdown()


def test_queued_job_can_be_cancelled_before_its_worker_starts() -> None:
    manager = ScanJobManager(max_workers=1)
    first_started = Event()
    release_first = Event()
    queued_work_ran = Event()

    def blocking_work(_report):
        first_started.set()
        assert release_first.wait(1.0)
        return {"count": 0, "regions": []}

    def queued_work(_report):
        queued_work_ran.set()
        return {"count": 1, "regions": [{"text": "50"}]}

    try:
        first = manager.submit(blocking_work)
        assert first_started.wait(1.0)
        queued = manager.submit(queued_work)
        cancelled = manager.cancel(queued["job_id"])
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        assert cancelled["result"] is None

        release_first.set()
        assert _wait_for_terminal(manager, first["job_id"])["status"] == "succeeded"
        sleep(0.05)
        assert not queued_work_ran.is_set()
        assert manager.get(queued["job_id"])["status"] == "cancelled"
    finally:
        release_first.set()
        manager.shutdown()


def test_cancel_request_wins_a_race_with_terminal_result_publication() -> None:
    manager = ScanJobManager(max_workers=1)
    work_started = Event()
    release_work = Event()

    def work(_report):
        work_started.set()
        assert release_work.wait(1.0)
        return {"count": 1, "regions": [{"text": "25"}]}

    try:
        initial = manager.submit(work)
        assert work_started.wait(1.0)
        assert manager.cancel(initial["job_id"])["status"] == "cancelling"

        # Simulate the worker reaching terminal publication in the same instant
        # as the cancellation request. The manager must discard the result.
        manager._update(
            initial["job_id"],
            status="succeeded",
            stage="complete",
            message="Scan complete",
            percent=100,
            result={"count": 1, "regions": [{"text": "25"}]},
        )
        raced = manager.get(initial["job_id"])
        assert raced is not None
        assert raced["status"] == "cancelled"
        assert raced["result"] is None

        release_work.set()
        assert _wait_for_terminal(manager, initial["job_id"])["status"] == "cancelled"
    finally:
        release_work.set()
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
            candidate_count=27,
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
        assert long_running["candidate_count"] == 27
        assert long_running["result"] is None

        # Move only the no-forward-progress timestamp beyond the configured
        # threshold. The independently ticking heartbeat remains current.
        with manager._lock:
            manager._jobs[initial["job_id"]].last_progress_at -= 3.0
        slow_progress = manager.get(initial["job_id"])
        assert slow_progress is not None
        assert slow_progress["liveness"] == "slow_progress"
        assert slow_progress["heartbeat_age_seconds"] == 0
        assert slow_progress["progress_age_seconds"] >= 3

        # Only a stale service heartbeat is labelled possibly stalled.
        with manager._lock:
            job = manager._jobs[initial["job_id"]]
            job.heartbeat_at -= 3.0
            possibly_stalled = manager._snapshot(job)
        assert possibly_stalled["liveness"] == "possibly_stalled"

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


def test_debug_overlay_is_visible_during_work_without_partial_results() -> None:
    manager = ScanJobManager(max_workers=1)
    started = Event()
    finish = Event()
    overlay = {
        "enabled": True,
        "page_width": 100,
        "page_height": 60,
        "scope_kind": "page",
        "table_masks": [{"x": 70, "y": 0, "width": 30, "height": 60}],
        "panels": [],
        "overlaps": [],
        "candidates": [],
    }

    def work(report):
        report(
            stage="layout",
            message="Layout ready",
            percent=8,
            overlay=overlay,
        )
        started.set()
        assert finish.wait(1.0)
        return {"count": 0, "regions": []}

    try:
        initial = manager.submit(work)
        assert started.wait(1.0)
        running = manager.get(initial["job_id"])
        assert running is not None
        assert running["overlay"] == overlay
        assert running["result"] is None

        finish.set()
        complete = _wait_for_terminal(manager, initial["job_id"])
        assert complete["overlay"] == overlay
        assert complete["result"] == {"count": 0, "regions": []}
    finally:
        finish.set()
        manager.shutdown()
