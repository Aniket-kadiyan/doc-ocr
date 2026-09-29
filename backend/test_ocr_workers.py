"""Unit tests for the OCR worker pool — no processes are spawned here."""

from __future__ import annotations

import numpy as np
from PIL import Image

import ocr_workers
from ocr_workers import OcrWorkerPool, image_to_png, plain, png_to_image, run_task


def test_plain_converts_numpy_scalars_and_arrays() -> None:
    value = {"a": np.float32(1.5), "b": [np.int64(2), np.array([1, 2])], "c": ("x", 3)}
    assert plain(value) == {"a": 1.5, "b": [2, [1, 2]], "c": ["x", 3]}


def test_png_round_trip_preserves_size() -> None:
    image = Image.new("RGB", (37, 11), "white")
    assert png_to_image(image_to_png(image)).size == (37, 11)


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def recognize(self, image, **kwargs):
        self.calls.append(("recognize", image.size))
        return {"text": "R1", "confidence": np.float32(0.9), "kwargs": sorted(kwargs)}

    def _recognize_page_batch(self, images, *, batch_size, profile):
        self.calls.append(("batch", (len(images), batch_size, profile)))
        return [{"text": f"v{i}"} for i in range(len(images))]

    def _angled_in_roi(self, image, **kwargs):
        self.calls.append(("angled", sorted(kwargs)))
        return [{"text": "0.5×45°", "bbox": {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}}]

    def _authoritative_attempts(self, tight, padded, *, debug_dump, debug_dump_force, **_kwargs):
        self.calls.append(("authoritative", (tight.profile, padded.profile, tight.image.size)))
        return [{"text": "23-0.1", "authoritative_valid": True}]


def test_run_task_dispatches_each_task_kind_and_returns_plain_data() -> None:
    pipeline = _FakePipeline()
    png = image_to_png(Image.new("RGB", (20, 10), "white"))

    result = run_task(pipeline, "recognize", {"png": png, "kwargs": {"compute_text_bbox": True}})
    assert result["text"] == "R1" and result["kwargs"] == ["compute_text_bbox"]
    assert isinstance(result["confidence"], float) and abs(result["confidence"] - 0.9) < 1e-6

    batch = run_task(pipeline, "batch", {"pngs": [png, png], "batch_size": 4, "profile": "p"})
    assert [item["text"] for item in batch] == ["v0", "v1"]
    assert ("batch", (2, 4, "p")) in pipeline.calls

    angled = run_task(pipeline, "angled_roi", {"png": png, "kwargs": {"sign": -1.0}})
    assert angled[0]["text"] == "0.5×45°"

    attempts = run_task(
        pipeline,
        "authoritative",
        {
            "crops": [
                {
                    "profile": "tight",
                    "png": png,
                    "target_bbox": {"x": 0, "y": 0, "width": 20, "height": 10},
                    "target_polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
                    "polygon_usable": True,
                },
                {
                    "profile": "padded",
                    "png": png,
                    "target_bbox": {"x": 0, "y": 0, "width": 20, "height": 10},
                    "target_polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
                },
            ]
        },
    )
    assert attempts == [{"text": "23-0.1", "authoritative_valid": True}]
    assert ("authoritative", ("tight", "padded", (20, 10))) in pipeline.calls


def test_map_without_workers_runs_everything_locally_in_order() -> None:
    pool = OcrWorkerPool(size=0)
    seen: list[int] = []
    results = pool.map(
        "recognize",
        [{"n": 3}, {"n": 1}, {"n": 2}],
        local=lambda payload: payload["n"] * 10,
        on_item_done=lambda index, value: seen.append(index),
    )
    assert results == [30, 10, 20]
    assert seen == [0, 1, 2]
    assert not pool.available


def test_pool_is_not_started_unless_asked(monkeypatch) -> None:
    monkeypatch.setattr(ocr_workers, "_pool", None)
    pool = ocr_workers.get_worker_pool(start=False)
    assert pool.status()["started"] is False
    assert pool.ready_workers() == 0
    monkeypatch.setattr(ocr_workers, "_pool", None)


def test_configured_worker_count_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("OCR_WORKERS", "99")
    assert 0 <= ocr_workers.configured_worker_count() <= ocr_workers.MAX_WORKERS
    monkeypatch.setenv("OCR_WORKERS", "0")
    assert ocr_workers.configured_worker_count() == 0
