from __future__ import annotations

from threading import Event, Lock, Thread
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np

from ocr_model_pool import OcrModelBundlePool
from ocr_pipeline import OcrPipeline


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.005)
    raise AssertionError("Timed out waiting for OCR model-pool state")


class _FakePaddle:
    def __init__(self, bundle_id: int) -> None:
        self.bundle_id = bundle_id

    def predict(self, _arr):
        return [{"bundle_id": self.bundle_id}]


def _successful_loader():
    next_id = 0

    def load(pipeline: OcrPipeline) -> None:
        nonlocal next_id
        pipeline._paddle = _FakePaddle(next_id)
        pipeline._paddle_available = True
        pipeline._paddle_api = 3
        pipeline._paddle_version = "test"
        next_id += 1

    return load


def test_pipeline_loads_and_rotates_independent_model_bundles() -> None:
    pipeline = OcrPipeline()
    with patch.object(OcrPipeline, "_load_paddle", _successful_loader()):
        pipeline.load(model_bundle_count=3)

    assert pipeline.inference_capacity == 3
    assert pipeline.inference_gate.status()["configured_slots"] == 3
    assert pipeline.status["model_pool"] == {
        "requested": 3,
        "loaded": 3,
        "failed": 0,
        "load_errors": [],
        "loaded_bundles": 3,
        "leased_bundles": 0,
        "available_bundles": 3,
    }

    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    assert [
        pipeline._predict_3x(arr)[0]["bundle_id"]
        for _ in range(4)
    ] == [0, 1, 2, 0]


def test_failed_bundle_reduces_real_capacity_and_stops_further_loads() -> None:
    load_calls = 0

    def load(pipeline: OcrPipeline) -> None:
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            pipeline._paddle = _FakePaddle(0)
            pipeline._paddle_available = True
            pipeline._paddle_api = 3
            pipeline._paddle_version = "test"
        else:
            pipeline._init_errors.append("simulated load failure")

    pipeline = OcrPipeline()
    with patch.object(OcrPipeline, "_load_paddle", load):
        pipeline.load(model_bundle_count=3)

    assert load_calls == 2
    assert pipeline.inference_capacity == 1
    assert pipeline.inference_gate.status()["configured_slots"] == 1
    assert pipeline.status["model_pool"]["requested"] == 3
    assert pipeline.status["model_pool"]["loaded"] == 1
    assert pipeline.status["model_pool"]["failed"] == 2
    assert "simulated load failure" in (
        pipeline.status["model_pool"]["load_errors"][0]
    )


def test_pool_never_leases_one_bundle_to_two_callers() -> None:
    pool = OcrModelBundlePool(["bundle-a", "bundle-b"])
    release = Event()
    first_two_entered = Event()
    acquired: list[str] = []
    active: set[str] = set()
    errors: list[BaseException] = []
    state_lock = Lock()

    def worker() -> None:
        try:
            with pool.lease("test") as bundle:
                with state_lock:
                    assert bundle not in active
                    active.add(bundle)
                    acquired.append(bundle)
                    if len(active) == 2:
                        first_two_entered.set()
                assert release.wait(timeout=2)
                with state_lock:
                    active.remove(bundle)
        except BaseException as exc:
            errors.append(exc)

    threads = [Thread(target=worker) for _ in range(3)]
    threads[0].start()
    threads[1].start()
    assert first_two_entered.wait(timeout=2)
    threads[2].start()
    _wait_until(
        lambda: pool.inference_gate.status()["waiting_interactive"] == 1
    )

    assert pool.status() == {
        "loaded_bundles": 2,
        "leased_bundles": 2,
        "available_bundles": 0,
    }
    release.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert set(acquired[:2]) == {"bundle-a", "bundle-b"}
    assert len(acquired) == 3
    assert errors == []
