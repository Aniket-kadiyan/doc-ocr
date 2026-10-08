"""Fast tests for the native-PDF-first scan path; no OCR models are loaded."""

from __future__ import annotations

from typing import Any

import pytest

from pdf_text_first import (
    crop_text_layer_result,
    merge_native_and_ocr_results,
    native_text_segment_result,
    native_vector_result_is_primary,
)
from pdf_text_layer import (
    TextRun,
    _decode_symbols,
    has_usable_text_layer,
    text_layer_regions,
)


def _bbox(x: float = 10, y: float = 20) -> dict[str, float]:
    return {"x": x, "y": y, "width": 60, "height": 18}


def _native_region(
    text: str,
    *,
    x: float = 10,
    y: float = 20,
    needs_review: bool = False,
) -> dict[str, Any]:
    return {
        "text": text,
        "bbox": _bbox(x, y),
        "confidence": 0.995,
        "needs_review": needs_review,
        "recognized": True,
        "recognition_source": "native_pdf",
    }


def _native_child(
    candidate_id: str,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "text": text,
        "raw_text": text,
        "bbox": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "confidence": 0.995,
        "orientation": "horizontal",
        "rotation": 0.0,
        "recognition_source": "native_pdf",
    }


def _outcome(
    candidate_id: str,
    text: str,
    state: str,
    *,
    x: float = 10,
    y: float = 20,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "text": text,
        "raw_text": text,
        "bbox": _bbox(x, y),
        "state": state,
        "recognized": True,
        "reason": "test",
        "rule": "test",
        "recognition_source": "ocr",
    }


def _scan_result(*outcomes: dict[str, Any]) -> dict[str, Any]:
    regions = [item for item in outcomes if item["state"] == "eligible"]
    reviews = [item for item in outcomes if item["state"] == "review"]
    return {
        "count": len(regions),
        "detected_count": len(outcomes),
        "recognized_count": len(outcomes),
        "eligible_count": len(regions),
        "excluded_count": sum(item["state"] == "excluded" for item in outcomes),
        "review_count": len(reviews),
        "unread_count": 0,
        "skipped_existing_count": 0,
        "filter_rule_counts": {},
        "regions": regions,
        "review_candidates": reviews,
        "candidate_outcomes": list(outcomes),
        "source_profile": {"kind": "hybrid"},
        "recognition_source_counts": {"ocr": len(regions) + len(reviews)},
    }


def _run(text: str, font: str = "ArialMT") -> TextRun:
    return TextRun(
        text=text,
        x=10,
        y=10,
        width=30,
        height=12,
        angle=0,
        font=font,
        lx=10,
        ly=10,
        lw=30,
        lh=12,
        size=12,
    )


def test_subset_prefixed_cad_and_searchable_ocr_fonts_are_recognized() -> None:
    assert _decode_symbols("P", "ABCDEF+AIGDT") == "Ø"
    assert not has_usable_text_layer(
        [_run("25", "ABCDEF+GlyphLessFont")],
        {"ABCDEF+GlyphLessFont"},
    )


def test_vector_pdf_text_is_extracted_at_the_uploaded_raster_size() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 40), "R1 +0.25", fontsize=12)
    payload = document.tobytes()
    document.close()

    result = text_layer_regions(
        payload,
        target_size=(400, 200),
        recover_symbols=False,
    )

    assert result["usable"] is True
    assert result["page_size"] == (400, 200)
    assert result["regions"][0]["text"] == "R1+0.25"
    assert result["regions"][0]["bbox"]["x"] == pytest.approx(40, abs=8)
    assert result["regions"][0]["recognition_source"] == "native_pdf"


def test_native_outcomes_keep_review_and_other_candidates_visible() -> None:
    text_layer = {
        "usable": True,
        "page_size": (500, 300),
        "fonts": ["ArialMT"],
        "fallback_reason": None,
        "regions": [
            _native_region("R1+0.25"),
            _native_region("+0.5", x=100, needs_review=True),
        ],
        "excluded": [_native_region("DRAWN BY", x=200)],
    }

    result = native_text_segment_result(
        text_layer,
        page_number=1,
        scope_kind="page",
        source_profile={"kind": "vector"},
    )

    assert [item["text"] for item in result["regions"]] == ["R1+0.25"]
    assert [item["text"] for item in result["review_candidates"]] == ["+0.5"]
    assert [item["state"] for item in result["candidate_outcomes"]] == [
        "eligible",
        "review",
        "excluded",
    ]
    assert result["excluded_count"] == 1
    assert result["source_profile"]["text_layer_strategy"] == "native_text_first"


def test_existing_balloon_is_removed_before_native_results_are_published() -> None:
    text_layer = {
        "usable": True,
        "page_size": (500, 300),
        "regions": [_native_region("R1+0.25")],
        "excluded": [],
    }

    result = native_text_segment_result(
        text_layer,
        page_number=1,
        scope_kind="page",
        existing_value_boxes=[_bbox()],
    )

    assert result["skipped_existing_count"] == 1
    assert result["candidate_outcomes"] == []
    assert result["regions"] == []


def test_table_exclusion_wins_over_bare_number_review() -> None:
    text_layer = {
        "usable": True,
        "page_size": (500, 300),
        "regions": [_native_region("84", needs_review=True)],
        "excluded": [],
    }

    result = native_text_segment_result(
        text_layer,
        page_number=1,
        scope_kind="page",
        table_masks=[{"x": 0, "y": 0, "width": 100, "height": 100}],
    )

    assert result["review_count"] == 0
    assert result["excluded_count"] == 1
    assert result["candidate_outcomes"][0]["rule"] == "table_region"


def test_native_nominal_and_tolerance_are_assembled_before_filtering() -> None:
    nominal = _native_region("22.9", x=10, y=20)
    nominal["bbox"]["width"] = 30
    tolerance = _native_region("±1", x=43, y=21)
    tolerance["bbox"]["width"] = 16
    text_layer = {
        "usable": True,
        "page_size": (500, 300),
        "regions": [nominal],
        # Simulate the previous failure: the tolerance fragment was placed in
        # the non-callout list before it could be associated with its nominal.
        "excluded": [tolerance],
    }

    result = native_text_segment_result(
        text_layer,
        page_number=1,
        scope_kind="page",
    )

    assert result["eligible_count"] == 1
    assert result["excluded_count"] == 0
    assert result["regions"][0]["text"] == "22.9 ±1"
    assert len(result["regions"][0]["assembly_children"]) == 2
    assert result["regions"][0]["recognition_source"] == "native_pdf"
    assert result["assembly_stats"]["absorbed_fragment_count"] == 1


def test_native_geometry_repairs_lower_zero_appended_to_angle_nominal() -> None:
    malformed = _native_region("210", x=100, y=100, needs_review=True)
    malformed["bbox"].update({"width": 48, "height": 25})
    malformed["assembly_children"] = [
        _native_child("R1", "2", 104, 108, 6, 10),
        _native_child("R2", "1", 114, 108, 4, 10),
        _native_child("R3", "0", 132, 117, 10, 7),
    ]
    upper = _native_region("+1", x=128, y=99, needs_review=True)
    upper["bbox"].update({"width": 18, "height": 10})
    upper["assembly_children"] = [
        _native_child("R4", "+", 129, 101, 5, 6),
        _native_child("R5", "1", 138, 100, 4, 8),
    ]

    result = native_text_segment_result(
        {
            "usable": True,
            "page_size": (500, 300),
            "regions": [malformed],
            "excluded": [upper],
        },
        page_number=1,
        scope_kind="page",
    )

    candidate = result["candidate_outcomes"][0]
    assert candidate["text"] == "21 +1/0"
    assert candidate["state"] == "review"
    assert candidate["native_structure_ambiguous"] is False
    assert candidate["assembly_rule"] == (
        "native_geometry_stacked_tolerance_repair"
    )
    assert [child["text"] for child in candidate["assembly_children"]] == [
        "2",
        "1",
        "0",
        "+",
        "1",
    ]
    assert result["excluded_count"] == 0


def test_ordinary_native_210_on_one_baseline_is_not_reinterpreted() -> None:
    ordinary = _native_region("210", x=100, y=100, needs_review=True)
    ordinary["assembly_children"] = [
        _native_child("R1", "2", 104, 108, 6, 10),
        _native_child("R2", "1", 114, 108, 4, 10),
        _native_child("R3", "0", 122, 108, 6, 10),
    ]

    result = native_text_segment_result(
        {
            "usable": True,
            "page_size": (500, 300),
            "regions": [ordinary],
            "excluded": [],
        },
        page_number=1,
        scope_kind="page",
    )

    candidate = result["candidate_outcomes"][0]
    assert candidate["text"] == "210"
    assert candidate["state"] == "review"
    assert not candidate.get("native_structure_ambiguous", False)


def test_unpaired_native_stacked_zero_stays_visible_in_review() -> None:
    malformed = _native_region("210", x=100, y=100, needs_review=True)
    malformed["bbox"].update({"width": 48, "height": 25})
    malformed["assembly_children"] = [
        _native_child("R1", "2", 104, 108, 6, 10),
        _native_child("R2", "1", 114, 108, 4, 10),
        _native_child("R3", "0", 132, 117, 10, 7),
    ]

    result = native_text_segment_result(
        {
            "usable": True,
            "page_size": (500, 300),
            "regions": [malformed],
            "excluded": [],
        },
        page_number=1,
        scope_kind="page",
    )

    candidate = result["candidate_outcomes"][0]
    assert candidate["text"] == "210"
    assert candidate["state"] == "review"
    assert candidate["native_structure_ambiguous"] is True
    assert candidate["assembly_review_reason"].startswith("Native stacked")


def test_section_crop_translates_candidate_and_native_provenance() -> None:
    region = _native_region("R2+0.2", x=110, y=70)
    region["recognition_evidence"] = {
        "native_bbox": dict(region["bbox"]),
        "sources": ["native_pdf"],
    }
    cropped = crop_text_layer_result(
        {
            "usable": True,
            "page_size": (500, 300),
            "regions": [region],
            "excluded": [],
        },
        {"x": 100, "y": 50, "width": 160, "height": 100},
    )

    assert cropped["regions"][0]["bbox"] == _bbox(10, 20)
    assert cropped["regions"][0]["recognition_evidence"]["native_bbox"] == _bbox(
        10, 20
    )


def test_hybrid_agreement_keeps_one_native_candidate_with_both_sources() -> None:
    native = _scan_result(_outcome("PDF:P1:T00001", "R1+0.25", "eligible"))
    native["regions"][0]["recognition_source"] = "native_pdf"
    ocr = _scan_result(_outcome("C0001", "R1+0.25", "eligible", x=12))

    merged = merge_native_and_ocr_results(native, ocr)

    assert merged["eligible_count"] == 1
    assert merged["detected_count"] == 1
    assert merged["regions"][0]["candidate_id"] == "PDF:P1:T00001"
    assert merged["regions"][0]["recognition_source"] == "native_pdf+ocr"
    assert merged["regions"][0]["recognition_evidence"]["agreement"] == 1.0


def test_hybrid_numeric_conflict_moves_native_candidate_to_review() -> None:
    native = _scan_result(_outcome("PDF:P1:T00001", "R1+0.25", "eligible"))
    native["regions"][0]["recognition_source"] = "native_pdf"
    ocr = _scan_result(_outcome("C0001", "R1+0.26", "eligible", x=12))

    merged = merge_native_and_ocr_results(native, ocr)

    assert merged["eligible_count"] == 0
    assert merged["review_count"] == 1
    assert merged["review_candidates"][0]["source_conflict"] is True
    assert merged["candidate_outcomes"][0]["rule"] == "native_ocr_numeric_conflict"


def test_excluded_ocr_disagreement_cannot_demote_exact_native_text() -> None:
    native = _scan_result(_outcome("PDF:P1:T00001", "R1+0.25", "eligible"))
    native["regions"][0]["recognition_source"] = "native_pdf"
    ocr = _scan_result(_outcome("C0001", "R1+0.26", "excluded", x=12))

    merged = merge_native_and_ocr_results(native, ocr)

    assert merged["eligible_count"] == 1
    assert merged["review_count"] == 0
    assert merged["regions"][0]["source_conflict"] is False


def test_hybrid_keeps_non_overlapping_ocr_candidate() -> None:
    native = _scan_result(_outcome("PDF:P1:T00001", "R1+0.25", "eligible"))
    native["regions"][0]["recognition_source"] = "native_pdf"
    ocr = _scan_result(_outcome("C0002", "Ø20", "eligible", x=220))

    merged = merge_native_and_ocr_results(native, ocr)

    assert merged["eligible_count"] == 2
    assert {item["text"] for item in merged["regions"]} == {"R1+0.25", "Ø20"}
    assert merged["source_profile"]["text_layer_strategy"] == (
        "native_text_first_hybrid_ocr"
    )


def test_vector_native_result_remains_primary_when_graphics_are_present() -> None:
    native = _scan_result(_outcome("PDF:P1:T00001", "45.3", "eligible"))
    native["structured_symbol_stats"] = {
        "object_count": 2,
        "complete_count": 1,
        "review_count": 1,
    }

    assert native_vector_result_is_primary(
        native,
        evidence_profile="vector",
    )


def test_structured_ocr_evidence_never_replaces_exact_native_text() -> None:
    native = _scan_result(_outcome("PDF:P1:T00001", "0.5 A B C", "review"))
    native["review_candidates"][0]["recognition_source"] = "native_pdf"
    ocr_outcome = _outcome("C0001", "⟂ | 0.5AC | 0.5AC", "review", x=12)
    ocr_outcome["engineering_symbol"] = {
        "kind": "feature_control_frame",
        "complete": True,
        "completed_symbol": "⟂",
    }
    ocr = _scan_result(ocr_outcome)

    merged = merge_native_and_ocr_results(native, ocr)

    assert merged["candidate_outcomes"][0]["text"] == "0.5 A B C"
    assert merged["candidate_outcomes"][0]["recognition_evidence"][
        "selected_source"
    ] == "native_pdf"
