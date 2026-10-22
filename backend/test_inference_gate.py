from __future__ import annotations

from threading import Event, Lock, Thread
from time import monotonic, sleep

from inference_gate import (
    InferenceCapacityGate,
    current_inference_priority,
    inference_priority,
)
from ocr_capacity import (
    AUTO_BALLOON_SCAN_CAPACITY,
    OCR_CONCURRENT_SCANS_PER_BUNDLE,
    OCR_MODEL_BUNDLE_COUNT,
)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.005)
    raise AssertionError("Timed out waiting for inference-gate state")


def test_capacity_matches_the_created_model_bundle_count() -> None:
    assert AUTO_BALLOON_SCAN_CAPACITY == (
        OCR_MODEL_BUNDLE_COUNT * OCR_CONCURRENT_SCANS_PER_BUNDLE
    )


def test_gate_never_exceeds_its_configured_capacity() -> None:
    gate = InferenceCapacityGate(1)
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    active = 0
    peak = 0
    counter_lock = Lock()

    def worker(entered: Event, release: Event | None = None) -> None:
        nonlocal active, peak
        with gate.slot("test"):
            with counter_lock:
                active += 1
                peak = max(peak, active)
            entered.set()
            if release is not None:
                assert release.wait(timeout=2)
            with counter_lock:
                active -= 1

    first = Thread(target=worker, args=(first_entered, release_first))
    second = Thread(target=worker, args=(second_entered,))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()

    _wait_until(lambda: gate.status()["waiting_interactive"] == 1)
    assert not second_entered.is_set()
    assert gate.status()["active_calls"] == 1

    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert peak == 1
    assert gate.status()["peak_active_calls"] == 1


def test_waiting_interactive_call_precedes_background_call() -> None:
    gate = InferenceCapacityGate(1)
    release_holder = Event()
    holder_entered = Event()
    order: list[str] = []

    def holder() -> None:
        with gate.slot("holder"):
            holder_entered.set()
            assert release_holder.wait(timeout=2)

    def contender(name: str, priority: str) -> None:
        with inference_priority(priority):  # type: ignore[arg-type]
            with gate.slot(name):
                order.append(name)

    holding_thread = Thread(target=holder)
    background_thread = Thread(
        target=contender,
        args=("background", "background"),
    )
    interactive_thread = Thread(
        target=contender,
        args=("interactive", "interactive"),
    )

    holding_thread.start()
    assert holder_entered.wait(timeout=2)
    background_thread.start()
    _wait_until(lambda: gate.status()["waiting_background"] == 1)
    interactive_thread.start()
    _wait_until(lambda: gate.status()["waiting_interactive"] == 1)

    release_holder.set()
    holding_thread.join(timeout=2)
    interactive_thread.join(timeout=2)
    background_thread.join(timeout=2)

    assert order == ["interactive", "background"]


def test_inference_priority_is_scoped_and_restored() -> None:
    assert current_inference_priority() == "interactive"
    with inference_priority("background"):
        assert current_inference_priority() == "background"
    assert current_inference_priority() == "interactive"


def test_zero_capacity_reports_unavailable_instead_of_waiting_forever() -> None:
    gate = InferenceCapacityGate(0)
    assert gate.status()["configured_slots"] == 0
    try:
        with gate.slot("test"):
            raise AssertionError("unreachable")
    except RuntimeError as exc:
        assert "No OCR model bundle" in str(exc)
    else:
        raise AssertionError("A zero-capacity gate accepted work")
