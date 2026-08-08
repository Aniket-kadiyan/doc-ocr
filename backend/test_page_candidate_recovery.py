"""Fast post-detection recovery checks that do not load PaddleOCR."""

from __future__ import annotations

from PIL import Image, ImageDraw

from page_candidate_recovery import (
    build_recovery_crops,
    engineering_values_agree,
    is_usable_engineering_value,
    reconstruct_line_context,
    resolve_recovery_consensus,
    result_needs_recovery,
    result_needs_second_recovery,
    select_recovery_record_indexes,
)


def _result(
    text: str,
    *,
    raw: str | None = None,
    confidence: float = 0.96,
    corrected: bool = False,
    orientation_confidence: float = 0.99,
) -> dict:
    return {
        "text": text,
        "raw_ocr": text if raw is None else raw,
        "confidence": confidence,
        "confusable_corrected": corrected,
        "orientation_confidence": orientation_confidence,
        "ocr_profile": "test",
    }


def test_recovery_selection_is_bounded_and_prioritizes_missing_values() -> None:
    records = [
        {"result": _result("LONG LABEL")},
        {"result": _result("", confidence=0.0)},
        {"result": _result("25", confidence=0.55)},
        {"result": _result("NOTES")},
        {"result": _result("15°±")},
        {"result": _result("B")},
    ]

    assert select_recovery_record_indexes(records, maximum=2) == [1, 4]
    assert not result_needs_recovery(records[0]["result"])
    assert not result_needs_recovery(records[2]["result"])
    assert not result_needs_recovery(records[3]["result"])
    assert result_needs_recovery(records[5]["result"])


def test_confidence_orientation_and_spacing_are_not_approval_gates() -> None:
    low_quality = _result(
        "30 ° +/- 3 °",
        confidence=0.31,
        orientation_confidence=0.12,
    )

    assert is_usable_engineering_value(low_quality["text"])
    assert not result_needs_recovery(low_quality)
    assert engineering_values_agree("30°±3°", "30 +/- 3")
    assert engineering_values_agree("R3", "r3.0")
    assert not engineering_values_agree("15.0", "75.0")


def test_second_recovery_runs_only_for_unresolved_or_conflicting_reads() -> None:
    assert not result_needs_second_recovery(
        _result("", confidence=0.0),
        _result("R5.00", confidence=0.72),
    )
    assert not result_needs_second_recovery(
        _result("30°±3°"),
        _result("30 +/- 3"),
    )
    assert result_needs_second_recovery(
        _result("15.0"),
        _result("75.0"),
    )
    assert result_needs_second_recovery(
        _result("8", raw="8"),
        _result("B", raw="B"),
    )


def test_semantic_consensus_selects_the_richest_equivalent_value() -> None:
    resolved = resolve_recovery_consensus(
        _result("30 ± 3", confidence=0.62),
        [
            _result("30 +/- 3°", confidence=0.81),
            _result("30°±3°", confidence=0.74),
        ],
        attempted=True,
    )

    assert resolved["text"] == "30°±3°"
    assert resolved["agreement"] == 1.0
    assert not resolved["needs_review"]


def test_numeric_conflict_requires_review_until_a_recovery_majority_exists() -> None:
    unresolved = resolve_recovery_consensus(
        _result("15.0"),
        [_result("75.0")],
        attempted=True,
    )
    resolved = resolve_recovery_consensus(
        _result("15.0"),
        [_result("75.0"), _result("75")],
        attempted=True,
    )

    assert unresolved["needs_review"]
    assert unresolved["numeric_conflict"]
    assert not resolved["needs_review"]
    assert resolved["text"] == "75.0"


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
