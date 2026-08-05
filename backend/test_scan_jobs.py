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
