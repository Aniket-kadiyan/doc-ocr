"""Production-path contract for additive graphical GD&T detection."""

from __future__ import annotations

from copy import deepcopy

from PIL import Image

import gdt_scan_integration as integration
from candidate_lifecycle import final_candidate_counts
from gdt_config import GdtConfig


def _item(
    candidate_id: str,
    text: str,
    bbox: dict[str, float],
    *,
    state: str = "eligible",
    source: str = "ocr",
    category: str = "Linear",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "object_id": candidate_id,
        "bbox": dict(bbox),
        "text": text,
        "confidence": 0.99,
        "recognized": True,
        "recognition_source": source,
        "recognition_evidence": {
            "selected_source": source,
            "sources": [source],
        },
        "category": category,
        "type": category,
        "state": state,
        "reason": "existing result",
        "rule": "existing_result",
        "page_filter_rule": "existing_result",
        "page_filter_reason": "existing result",
    }


def _segment(*items: dict[str, object]) -> dict[str, object]:
    regions = [deepcopy(item) for item in items if item["state"] == "eligible"]
    reviews = [deepcopy(item) for item in items if item["state"] == "review"]
    outcomes = [deepcopy(item) for item in items]
    return {
        "count": len(regions),
        "detected_count": len(outcomes),
        "recognized_count": len(outcomes),
        "eligible_count": len(regions),
        "review_count": len(reviews),
        "excluded_count": 0,
        "unread_count": 0,
        "skipped_existing_count": 0,
        "filter_rule_counts": {"existing_result": len(outcomes)},
        "regions": regions,
        "review_candidates": reviews,
        "candidate_outcomes": outcomes,
    }


def _frame(
    *,
    tolerance: float | None = 0.25,
    text: str = "⌖ Ø0.25 A",
) -> dict[str, object]:
    tolerance_payload = {
        "value": tolerance,
        "state": "present" if tolerance is not None else "unreadable",
        "confidence": 0.9 if tolerance is not None else 0.0,
        "raw": str(tolerance) if tolerance is not None else "",
    }
    return {
        "candidate_id": "fcf-test",
        "bbox": {"x": 100.0, "y": 100.0, "width": 120.0, "height": 30.0},
        "text": text,
        "confidence": 0.92,
        "type": "GD&T",
        "category": "GD&T",
        "subtype": "Position",
        "label": "GD&T Position",
        "needs_review": tolerance is None,
        "recognized": True,
        "page_filter_rule": "feature_control_frame",
        "page_filter_reason": "ruled feature control frame",
        "review_reason": "the tolerance could not be read" if tolerance is None else "",
        "gdt": {
            "frames": [
                {
                    "type": "feature_control_frame",
                    "id": "test",
                    "symbol": {
                        "symbol_id": "position",
                        "name": "Position",
                        "normalized": "⌖",
                        "status": "confirmed",
                        "confidence": 0.95,
                    },
                    "tolerance": tolerance_payload,
                    "diameter_zone": {
                        "value": True,
                        "state": "present",
                        "confidence": 0.9,
                        "raw": "Ø",
                    },
                    "material_condition": {
                        "value": None,
                        "state": "absent",
                        "confidence": 0.0,
                        "raw": None,
                    },
                    "modifiers": [],
                    "datums": [
                        {
                            "label": "A",
                            "precedence": 1,
                            "material_condition": None,
                            "confidence": 0.9,
                            "raw": "A",
                        }
                    ],
                    "compartments": [
                        {
                            "index": 0,
                            "bounding_box": {
                                "x": 100.0,
                                "y": 100.0,
                                "width": 30.0,
                                "height": 30.0,
                            },
                            "text": "",
                            "confidence": 0.0,
                        },
                        {
                            "index": 1,
                            "bounding_box": {
                                "x": 130.0,
                                "y": 100.0,
                                "width": 50.0,
                                "height": 30.0,
                            },
                            "text": str(tolerance or ""),
                            "confidence": 0.9,
                        },
                        {
                            "index": 2,
                            "bounding_box": {
                                "x": 180.0,
                                "y": 100.0,
                                "width": 40.0,
                                "height": 30.0,
                            },
                            "text": "A",
                            "confidence": 0.9,
                        },
                    ],
                    "warnings": [],
                }
            ]
        },
    }


def _enabled() -> GdtConfig:
    return GdtConfig(page_scan_enabled=True)


def test_flag_off_returns_the_original_result_without_running_detector(monkeypatch) -> None:
    segment = _segment(
        _item("C1", "10", {"x": 10, "y": 10, "width": 20, "height": 10})
    )

    def unexpected(*args, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("detector should not run")

    monkeypatch.setattr(integration, "scan_regions", unexpected)
    result = integration.augment_page_scan_result(
        Image.new("RGB", (300, 200), "white"),
        segment,
        ocr_fn=lambda crop: ("", 0.0),
        config=GdtConfig(page_scan_enabled=False),
    )
    assert result is segment


def test_frame_replaces_only_its_compartment_and_keeps_accounting(monkeypatch) -> None:
    outside = _item(
        "C1",
        "10",
        {"x": 10, "y": 10, "width": 20, "height": 10},
    )
    native_tolerance = _item(
        "C2",
        "0.25",
        {"x": 140, "y": 108, "width": 25, "height": 12},
        source="native_pdf",
    )
    segment = _segment(outside, native_tolerance)
    monkeypatch.setattr(integration, "scan_regions", lambda *args, **kwargs: [_frame()])

    result = integration.augment_page_scan_result(
        Image.new("RGB", (300, 200), "white"),
        segment,
        ocr_fn=lambda crop: ("0.25", 0.9),
        config=_enabled(),
    )

    assert [item["text"] for item in result["regions"]] == ["10", "⌖ Ø0.25 A"]
    frame = result["regions"][1]
    assert frame["recognition_source"] == "native_pdf+gdt"
    assert frame["gdt"]["frames"][0]["tolerance"]["value"] == 0.25
    assert next(item for item in result["candidate_outcomes"] if item["candidate_id"] == "C2")["state"] == "excluded"
    counts = final_candidate_counts(
        result["regions"],
        result["review_candidates"],
        result["candidate_outcomes"],
    )
    assert counts.detected == counts.eligible + counts.review + counts.excluded
    assert result["gdt_integration"]["absorbed_candidate_count"] == 1
    assert segment["regions"][1]["text"] == "0.25"  # input was not mutated


def test_unreadable_frame_is_published_to_review(monkeypatch) -> None:
    monkeypatch.setattr(
        integration,
        "scan_regions",
        lambda *args, **kwargs: [_frame(tolerance=None, text="⌖ Ø? A")],
    )
    result = integration.augment_page_scan_result(
        Image.new("RGB", (300, 200), "white"),
        _segment(),
        ocr_fn=lambda crop: ("", 0.0),
        config=_enabled(),
    )
    assert result["regions"] == []
    assert len(result["review_candidates"]) == 1
    assert "tolerance" in result["review_candidates"][0]["review_reason"]
    assert result["gdt_integration"]["review_frame_count"] == 1


def test_native_full_frame_text_remains_primary_when_ocr_disagrees(monkeypatch) -> None:
    native = _item(
        "N1",
        "POSITION | Ø0.125 | A",
        {"x": 105, "y": 104, "width": 105, "height": 20},
        state="review",
        source="native_pdf",
        category="GD&T",
    )
    monkeypatch.setattr(
        integration,
        "scan_regions",
        lambda *args, **kwargs: [_frame(tolerance=0.5, text="⌖ Ø0.5 A")],
    )
    result = integration.augment_page_scan_result(
        Image.new("RGB", (300, 200), "white"),
        _segment(native),
        ocr_fn=lambda crop: ("0.5", 0.9),
        config=_enabled(),
    )
    assert len(result["review_candidates"]) == 1
    frame = result["review_candidates"][0]
    assert frame["text"] == "POSITION | Ø0.125 | A"
    assert frame["recognition_source"] == "native_pdf+gdt"
    assert frame["source_conflict"] is True
    assert "disagree" in frame["review_reason"]


def test_saved_balloon_frame_is_skipped(monkeypatch) -> None:
    existing = _item(
        "C1",
        "10",
        {"x": 10, "y": 10, "width": 20, "height": 10},
    )
    monkeypatch.setattr(integration, "scan_regions", lambda *args, **kwargs: [_frame()])
    result = integration.augment_page_scan_result(
        Image.new("RGB", (300, 200), "white"),
        _segment(existing),
        ocr_fn=lambda crop: ("0.25", 0.9),
        existing_value_boxes=(
            {"x": 100, "y": 100, "width": 120, "height": 30},
        ),
        config=_enabled(),
    )
    assert [item["candidate_id"] for item in result["regions"]] == ["C1"]
    assert result["gdt_integration"]["published_frame_count"] == 0
    assert result["gdt_integration"]["skipped_existing_count"] == 1
    assert result["skipped_existing_count"] == 1


def test_detector_failure_keeps_original_scan(monkeypatch) -> None:
    segment = _segment(
        _item("C1", "10", {"x": 10, "y": 10, "width": 20, "height": 10})
    )

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic detector failure")

    monkeypatch.setattr(integration, "scan_regions", fail)
    result = integration.augment_page_scan_result(
        Image.new("RGB", (300, 200), "white"),
        segment,
        ocr_fn=lambda crop: ("", 0.0),
        config=_enabled(),
    )
    assert result["regions"] == segment["regions"]
    assert result["candidate_outcomes"] == segment["candidate_outcomes"]
    assert result["gdt_integration"]["status"] == "error"


def test_real_frame_detector_reaches_the_scan_lifecycle() -> None:
    from benchmarks import gdt_synthetic

    scene = gdt_synthetic.build("frames")
    result = integration.augment_page_scan_result(
        scene.image,
        _segment(),
        ocr_fn=lambda crop: ("0.5", 0.9),
        config=_enabled(),
    )
    assert result["gdt_integration"]["status"] == "completed"
    assert result["gdt_integration"]["published_frame_count"] == len(scene.frames)
    published = [*result["regions"], *result["review_candidates"]]
    assert len(published) == len(scene.frames)
    assert all(item["category"] == "GD&T" for item in published)
    final_candidate_counts(
        result["regions"],
        result["review_candidates"],
        result["candidate_outcomes"],
    )
