"""Fast post-detection recovery checks that do not load PaddleOCR."""

from __future__ import annotations

from PIL import Image, ImageDraw

from page_candidate_recovery import (
    build_recovery_crops,
    reconstruct_line_context,
    resolve_recovery_consensus,
    result_needs_recovery,
    select_recovery_record_indexes,
)


def _result(
    text: str,
    *,
    raw: str | None = None,
    confidence: float = 0.96,
    corrected: bool = False,
) -> dict:
    return {
        "text": text,
        "raw_ocr": text if raw is None else raw,
        "confidence": confidence,
        "confusable_corrected": corrected,
        "orientation_confidence": 0.99,
        "ocr_profile": "test",
    }


def test_recovery_selection_is_bounded_and_prioritizes_missing_values() -> None:
    records = [
        {"result": _result("LONG LABEL")},
        {"result": _result("", confidence=0.0)},
        {"result": _result("25", confidence=0.55)},
        {"result": _result("NOTES")},
    ]

    assert select_recovery_record_indexes(records, maximum=2) == [1, 2]
    assert result_needs_recovery(records[0]["result"])


def test_consensus_recovers_a_dimension_without_full_pipeline_fallback() -> None:
    resolved = resolve_recovery_consensus(
        _result("", confidence=0.0),
        [
            _result("R5.00", confidence=0.81),
            _result("R5.00", confidence=0.84),
        ],
        attempted=True,
    )

    assert resolved["text"] == "R5.00"
    assert not resolved["needs_review"]
    assert resolved["agreement"] == 1.0


def test_stable_alphabetic_confusable_is_not_converted_to_a_balloon_value() -> None:
    resolved = resolve_recovery_consensus(
        _result("8", raw="B", corrected=True),
        [
            _result("8", raw="B", corrected=True),
            _result("8", raw="B", corrected=True),
        ],
        attempted=True,
    )

    assert resolved["text"] == "B"
    assert resolved["stable_alpha"]
    assert not resolved["needs_review"]


def test_numeric_alphabetic_conflict_remains_reviewable() -> None:
    resolved = resolve_recovery_consensus(
        _result("8", raw="8"),
        [
            _result("B", raw="B"),
            _result("B", raw="B"),
        ],
        attempted=True,
    )

    assert resolved["needs_review"]
    assert resolved["confusable_conflict"]
    assert "conflict" in resolved["review_reason"].lower()


def test_recovery_builds_two_complementary_source_crops() -> None:
    image = Image.new("RGB", (220, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((60, 56, 160, 48), fill="black", width=3)
    bbox = {"x": 58, "y": 42, "width": 105, "height": 24}
    polygon = [[60, 48], [160, 40], [163, 56], [62, 64]]

    crops = build_recovery_crops(image, bbox, polygon)

    assert [crop.profile for crop in crops] == [
        "recovery_rectified",
        "recovery_expanded_sharp",
    ]
    assert all(crop.image.width > 1 and crop.image.height > 1 for crop in crops)


def test_degenerate_polygon_uses_safe_axis_aligned_recovery() -> None:
    image = Image.new("RGB", (80, 40), "white")
    bbox = {"x": 10, "y": 10, "width": 30, "height": 12}

    crops = build_recovery_crops(
        image,
        bbox,
        [[10, 10], [10, 10], [10, 10], [10, 10]],
    )

    assert len(crops) == 2
    assert all(crop.image.width > 1 and crop.image.height > 1 for crop in crops)


def test_same_line_context_rejoins_split_scale_label() -> None:
    records = [
        {"bbox": {"x": 20, "y": 10, "width": 70, "height": 12}, "text": "SCALE"},
        {"bbox": {"x": 96, "y": 10, "width": 24, "height": 12}, "text": "2:1"},
        {"bbox": {"x": 20, "y": 70, "width": 30, "height": 12}, "text": "50"},
    ]

    assert reconstruct_line_context(records[1], records) == "SCALE 2:1"
