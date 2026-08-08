"""Fast Milestone 2 tests that do not load OCR models or require OpenCV."""

from __future__ import annotations

from typing import Any

from PIL import Image

from detection_passes import (
    DETECTION_PRIMARY_MAX_EDGE,
    DETECTION_ROTATIONS_CW,
    DETECTION_VARIANTS,
    build_detection_pass_plan,
    build_refinement_regions,
    build_detection_tiles,
    build_detection_variants,
    detection_primary_target_edge,
    detection_tile_target_edge,
    map_deskewed_box_to_original,
    map_quarter_turn_box_to_source,
    offset_tile_box,
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


def test_primary_plan_uses_two_orientations_and_bounded_resolution() -> None:
    image = Image.new("RGB", (320, 180), "white")
    plan = build_detection_pass_plan(image)

    assert len(plan) == 2
    assert {spec.variant for spec in plan} == {"source_contrast"}
    assert {spec.rotation_cw for spec in plan} == set(DETECTION_ROTATIONS_CW)
    assert {spec.target_long_edge for spec in plan} == {1100}
    assert len({spec.label for spec in plan}) == len(plan)
    assert detection_primary_target_edge((4200, 2400)) == DETECTION_PRIMARY_MAX_EDGE


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


def test_large_images_use_overlapping_tiles_with_complete_coverage() -> None:
    tiles = build_detection_tiles((4200, 2400), max_edge=2000, overlap=160)

    assert len(tiles) == 6
    assert tiles[0].box == (0, 0, 2000, 2000)
    assert tiles[-1].box == (3680, 1840, 4200, 2400)
    assert max(tile.x + tile.width for tile in tiles) == 4200
    assert max(tile.y + tile.height for tile in tiles) == 2400
    assert {tile.x for tile in tiles} == {0, 1840, 3680}
    assert {tile.y for tile in tiles} == {0, 1840}


def test_tile_boxes_and_pass_scale_restore_to_rotated_coordinates() -> None:
    tile = build_detection_tiles((2500, 1000))[1]

    assert tile.box == (1840, 0, 2500, 1000)
    assert offset_tile_box(
        {"x": 10, "y": 20, "w": 30, "h": 10},
        tile,
    ) == {"x": 1850.0, "y": 20.0, "w": 30, "h": 10}
    assert detection_tile_target_edge(
        tile,
        full_size=(2500, 1000),
        pass_target_edge=3750,
    ) == 1500


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


def test_quarter_turn_mapping_preserves_detector_polygon() -> None:
    mapped = map_quarter_turn_box_to_source(
        {
            "x": 25,
            "y": 10,
            "w": 15,
            "h": 30,
            "polygon": [[25, 10], [40, 10], [40, 40], [25, 40]],
        },
        90,
        (100, 60),
    )

    assert mapped is not None
    assert mapped["polygon"] == [
        [10.0, 35.0],
        [10.0, 20.0],
        [40.0, 20.0],
        [40.0, 35.0],
    ]


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


def test_detector_only_scale_and_padding_keep_low_confidence() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True

    def fake_detector(image: Image.Image):
        # (100 x 50) + 24px padding on each side, then exactly 5x upscale.
        assert image.size == (740, 490)
        return [
            {
                "x": 170.0,
                "y": 145.0,
                "width": 100.0,
                "height": 50.0,
                "confidence": 0.03,
                "polygon": [
                    [170.0, 145.0],
                    [270.0, 145.0],
                    [270.0, 195.0],
                    [170.0, 195.0],
                ],
            }
        ]

    pipeline._run_text_detector_regions = fake_detector
    boxes = pipeline._detector_only_boxes(
        Image.new("RGB", (100, 50), "white"),
        target_long_edge=740,
    )

    assert len(boxes) == 1
    _assert_box(boxes[0], (10.0, 5.0, 20.0, 10.0))
    assert boxes[0]["conf"] == 0.03
    assert boxes[0]["polygon"] == [
        [10.0, 5.0],
        [30.0, 5.0],
        [30.0, 15.0],
        [10.0, 15.0],
    ]


def test_detector_only_input_downscales_a_large_page_to_the_budget() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    received_sizes: list[tuple[int, int]] = []

    def fake_detector(image: Image.Image):
        received_sizes.append(image.size)
        return []

    pipeline._run_text_detector_regions = fake_detector
    pipeline._detector_only_boxes(
        Image.new("RGB", (4200, 2400), "white"),
        target_long_edge=DETECTION_PRIMARY_MAX_EDGE,
    )

    assert len(received_sizes) == 1
    assert max(received_sizes[0]) == DETECTION_PRIMARY_MAX_EDGE


def test_adaptive_detector_runs_two_primary_passes_and_keeps_low_confidence(
    monkeypatch,
) -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    calls: list[tuple[tuple[int, int], int]] = []

    def fake_detector(
        image: Image.Image,
        *,
        target_long_edge: int,
    ) -> list[dict[str, Any]]:
        calls.append((image.size, target_long_edge))
        # A centred square maps to the same position through both orientations.
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

    pipeline._detector_only_boxes = fake_detector
    monkeypatch.setattr("region_detect.propose_text_regions", lambda _image: [])
    events: list[dict[str, Any]] = []

    boxes = pipeline.detect_regions(
        Image.new("RGB", (100, 100), "white"),
        progress_callback=lambda **event: events.append(event),
    )

    assert len(calls) == 2
    assert {size for size, _target in calls} == {(100, 100)}
    assert len(boxes) == 1
    assert boxes[0]["conf"] == 0.01
    # Every blocking unit announces both its start and completion so the UI can
    # distinguish active work from the previously completed pass.
    assert len(events) == 6
    assert events[0]["state"] == "running"
    assert events[1]["state"] == "completed"
    assert events[0]["phase"] == "proposing"
    assert events[-1]["phase"] == "detecting"
    assert events[-1]["completed"] == events[-1]["total"] == 2
    assert events[-1]["pass_current"] == events[-1]["pass_total"] == 2


def test_large_area_uses_two_bounded_primary_calls_without_page_tiles(
    monkeypatch,
) -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    calls: list[tuple[tuple[int, int], int]] = []

    def fake_detector(
        image: Image.Image,
        *,
        target_long_edge: int,
    ) -> list[dict[str, Any]]:
        calls.append((image.size, target_long_edge))
        return []

    pipeline._detector_only_boxes = fake_detector
    monkeypatch.setattr("region_detect.propose_text_regions", lambda _image: [])
    events: list[dict[str, Any]] = []

    pipeline.detect_regions(
        Image.new("RGB", (4200, 2400), "white"),
        progress_callback=lambda **event: events.append(event),
    )

    assert len(calls) == 2
    assert all(target == DETECTION_PRIMARY_MAX_EDGE for _size, target in calls)
    assert {size for size, _target in calls} == {(4200, 2400), (2400, 4200)}
    assert all(event["tile_total"] == 1 for event in events)
    assert events[-1]["completed"] == events[-1]["total"] == 2


def test_uncovered_morphology_gap_gets_one_local_refinement(monkeypatch) -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    calls: list[tuple[int, int]] = []

    def fake_detector(image: Image.Image, **_kwargs) -> list[dict[str, Any]]:
        calls.append(image.size)
        if len(calls) == 1:
            return [
                {
                    "x": 20.0,
                    "y": 20.0,
                    "w": 40.0,
                    "h": 10.0,
                    "text": "",
                    "conf": 0.8,
                }
            ]
        return []

    pipeline._detector_only_boxes = fake_detector
    monkeypatch.setattr(
        "region_detect.propose_text_regions",
        lambda _image: [
            {"x": 20.0, "y": 20.0, "w": 40.0, "h": 10.0},
            {"x": 130.0, "y": 55.0, "w": 30.0, "h": 12.0},
        ],
    )
    events: list[dict[str, Any]] = []

    boxes = pipeline.detect_regions(
        Image.new("RGB", (200, 100), "white"),
        progress_callback=lambda **event: events.append(event),
    )

    assert len(calls) == 3  # two primary calls + one uncovered local gap
    assert len(boxes) == 2  # both morphology proposals remain available
    refinement_events = [event for event in events if event["phase"] == "refining"]
    assert len(refinement_events) == 2
    assert refinement_events[-1]["completed"] == refinement_events[-1]["total"] == 1


def test_refinement_plan_is_capped_without_dropping_morphology_candidates() -> None:
    morphology = [
        {"x": float(index * 100), "y": 10.0, "w": 8.0, "h": 8.0}
        for index in range(30)
    ]

    refinements = build_refinement_regions(
        morphology,
        [],
        source_size=(3200, 100),
        max_regions=5,
        margin_ratio=0.0,
    )

    assert len(refinements) == 5
    assert len(morphology) == 30


def test_detector_work_is_bounded_to_two_primary_plus_24_local_calls(
    monkeypatch,
) -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    calls = 0

    def fake_detector(_image: Image.Image, **_kwargs) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return []

    morphology = [
        {"x": float(index * 100), "y": 10.0, "w": 8.0, "h": 8.0}
        for index in range(30)
    ]
    pipeline._detector_only_boxes = fake_detector
    monkeypatch.setattr(
        "region_detect.propose_text_regions",
        lambda _image: morphology,
    )

    boxes = pipeline.detect_regions(Image.new("RGB", (3200, 100), "white"))

    assert calls == 26
    assert len(boxes) == len(morphology)


def test_morphology_is_used_even_when_the_pass_plan_runs(monkeypatch) -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    pipeline._detector_only_boxes = lambda _image, **_kwargs: []
    monkeypatch.setattr(
        "region_detect.propose_text_regions",
        lambda _image: [{"x": 8.0, "y": 9.0, "w": 22.0, "h": 11.0}],
    )

    boxes = pipeline.detect_regions(Image.new("RGB", (80, 50), "white"))

    assert len(boxes) == 1
    assert boxes[0]["_detection_pass"] == "morphology"
    assert boxes[0]["conf"] == 0.0
