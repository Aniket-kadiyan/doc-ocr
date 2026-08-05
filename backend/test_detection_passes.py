"""Fast Milestone 2 tests that do not load OCR models or require OpenCV."""

from __future__ import annotations

from typing import Any

from PIL import Image

from detection_passes import (
    DETECTION_ROTATIONS_CW,
    DETECTION_VARIANTS,
    build_detection_pass_plan,
    build_detection_variants,
    map_deskewed_box_to_original,
    map_quarter_turn_box_to_source,
)
from ocr_pipeline import OcrPipeline


def _assert_box(
    actual: dict[str, Any] | None,
    expected: tuple[float, float, float, float],
) -> None:
    assert actual is not None
    assert (
        actual["x"],
        actual["y"],
        actual["w"],
        actual["h"],
    ) == expected


def test_pass_plan_covers_variants_rotations_and_two_scales() -> None:
    image = Image.new("RGB", (320, 180), "white")
    plan = build_detection_pass_plan(image)

    assert len(plan) == 32
    assert {spec.variant for spec in plan} == set(DETECTION_VARIANTS)
    assert {spec.rotation_cw for spec in plan} == set(DETECTION_ROTATIONS_CW)
    assert {spec.target_long_edge for spec in plan} == {1100, 1800}
    assert len({spec.label for spec in plan}) == len(plan)


def test_detection_variants_keep_size_and_include_inverse_threshold() -> None:
    image = Image.new("RGB", (80, 40), "white")
    for x in range(20, 60):
        for y in range(14, 26):
            image.putpixel((x, y), (20, 70, 180))

    variants = build_detection_variants(image)

    assert set(variants) == set(DETECTION_VARIANTS)
    assert all(variant.size == image.size for variant in variants.values())
    threshold = variants["threshold"].convert("L")
    inverted = variants["inverted"].convert("L")
    assert all(
        threshold.getpixel((x, y)) + inverted.getpixel((x, y)) == 255
        for x, y in ((0, 0), (25, 20), (79, 39))
    )


def test_every_quarter_turn_maps_back_to_the_original_box() -> None:
    source_size = (100, 60)
    _assert_box(
        map_quarter_turn_box_to_source(
            {"x": 10, "y": 20, "w": 30, "h": 15},
            0,
            source_size,
        ),
        (10.0, 20.0, 30.0, 15.0),
    )
    _assert_box(
        map_quarter_turn_box_to_source(
            {"x": 25, "y": 10, "w": 15, "h": 30},
            90,
            source_size,
        ),
        (10.0, 20.0, 30.0, 15.0),
    )
    _assert_box(
        map_quarter_turn_box_to_source(
            {"x": 60, "y": 25, "w": 30, "h": 15},
            180,
            source_size,
        ),
        (10.0, 20.0, 30.0, 15.0),
    )
    _assert_box(
        map_quarter_turn_box_to_source(
            {"x": 20, "y": 60, "w": 15, "h": 30},
            270,
            source_size,
        ),
        (10.0, 20.0, 30.0, 15.0),
    )


def test_deskew_mapping_is_identity_at_zero_and_clips_to_source() -> None:
    identity = map_deskewed_box_to_original(
        {"x": 12, "y": 8, "w": 30, "h": 10},
        (100, 60),
        0.0,
    )
    _assert_box(identity, (12.0, 8.0, 30.0, 10.0))

    clipped = map_deskewed_box_to_original(
        {"x": -4, "y": -3, "w": 14, "h": 13},
        (100, 60),
        0.0,
    )
    _assert_box(clipped, (0.0, 0.0, 10.0, 10.0))


def test_paddle_scale_and_padding_are_mapped_back_to_input_pixels() -> None:
    pipeline = object.__new__(OcrPipeline)

    def fake_run(image: Image.Image, det: bool = True):
        assert det is True
        # (100 x 50) + 24px padding on each side, then exactly 5x upscale.
        assert image.size == (740, 490)
        return (
            "25",
            0.9,
            [
                {
                    "x": 170.0,
                    "y": 145.0,
                    "width": 100.0,
                    "height": 50.0,
                    "text": "25",
                    "confidence": 0.9,
                }
            ],
        )

    pipeline._run_paddle = fake_run
    boxes = pipeline._paddle_det_boxes(
        Image.new("RGB", (100, 50), "white"),
        target_long_edge=740,
        preprocess=False,
    )

    assert len(boxes) == 1
    _assert_box(boxes[0], (10.0, 5.0, 20.0, 10.0))


def test_accuracy_detector_runs_every_pass_and_keeps_low_confidence(
    monkeypatch,
) -> None:
    pipeline = object.__new__(OcrPipeline)
    calls: list[tuple[tuple[int, int], int, bool]] = []

    def fake_paddle(
        image: Image.Image,
        *,
        target_long_edge: int,
        preprocess: bool,
    ) -> list[dict[str, Any]]:
        calls.append((image.size, target_long_edge, preprocess))
        # A centred square maps to the same position through every quarter-turn.
        return [
            {
                "x": 40.0,
                "y": 40.0,
                "w": 20.0,
                "h": 20.0,
                "text": "?",
                "conf": 0.01,
            }
        ]

    pipeline._paddle_det_boxes = fake_paddle
    monkeypatch.setattr("region_detect.propose_text_regions", lambda _image: [])
    events: list[dict[str, Any]] = []

    boxes = pipeline.detect_regions(
        Image.new("RGB", (100, 100), "white"),
        progress_callback=lambda **event: events.append(event),
    )

    assert len(calls) == 32
    assert all(preprocess is False for _, _, preprocess in calls)
    assert len(boxes) == 1
    assert boxes[0]["conf"] == 0.01
    assert len(events) == 33  # 32 Paddle passes plus morphology.
    assert events[-1]["completed"] == events[-1]["total"] == 33
    assert events[-1]["label"] == "morphology"


def test_morphology_is_used_even_when_the_pass_plan_runs(monkeypatch) -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._paddle_det_boxes = lambda _image, **_kwargs: []
    monkeypatch.setattr(
        "region_detect.propose_text_regions",
        lambda _image: [{"x": 8.0, "y": 9.0, "w": 22.0, "h": 11.0}],
    )

    boxes = pipeline.detect_regions(Image.new("RGB", (80, 50), "white"))

    assert len(boxes) == 1
    assert boxes[0]["_detection_pass"] == "morphology"
    assert boxes[0]["conf"] == 0.0
