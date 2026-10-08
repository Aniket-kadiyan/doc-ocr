"""Native text stays authoritative while local pixels restore CAD symbols."""

from __future__ import annotations

from PIL import Image, ImageDraw

from native_symbol_enrichment import enrich_native_result_with_geometry


def _outcome(
    text: str,
    bbox: dict[str, float],
    *,
    state: str = "excluded",
    children: list[dict] | None = None,
) -> dict:
    return {
        "candidate_id": "PDF:P1:T00001",
        "object_id": "PDF:P1:T00001",
        "bbox": bbox,
        "text": text,
        "raw_text": text,
        "confidence": 0.995,
        "recognized": True,
        "state": state,
        "rule": "detail_view_section",
        "reason": "test exclusion",
        "orientation": "horizontal",
        "rotation": 0,
        "recognition_source": "native_pdf",
        "recognition_evidence": {
            "selected_source": "native_pdf",
            "sources": ["native_pdf"],
            "native_text": text,
            "ocr_text": "",
            "agreement": 1.0,
            "conflict": False,
        },
        "assembly_children": children or [],
    }


def _result(*outcomes: dict) -> dict:
    return {
        "count": 0,
        "detected_count": len(outcomes),
        "recognized_count": len(outcomes),
        "eligible_count": 0,
        "excluded_count": len(outcomes),
        "review_count": 0,
        "unread_count": 0,
        "skipped_existing_count": 0,
        "filter_rule_counts": {},
        "regions": [],
        "review_candidates": [],
        "candidate_outcomes": list(outcomes),
        "recognition_source_counts": {"native_pdf": len(outcomes)},
    }


def test_horizontal_native_fcf_uses_frame_geometry_without_replacing_digits():
    image = Image.new("RGB", (260, 110), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 35, 230, 61), outline="black", width=2)
    for x in (62, 142, 172, 202):
        draw.line((x, 35, x, 61), fill="black", width=2)
    draw.ellipse((31, 39, 51, 59), outline="black", width=2)
    draw.line((41, 39, 41, 59), fill="black", width=2)
    draw.line((31, 49, 51, 49), fill="black", width=2)
    children = [
        {"candidate_id": "R1", "text": "0.5", "bbox": {"x": 78, "y": 41, "width": 36, "height": 14}, "confidence": .995},
        {"candidate_id": "R2", "text": "A", "bbox": {"x": 151, "y": 41, "width": 10, "height": 14}, "confidence": .995},
        {"candidate_id": "R3", "text": "B", "bbox": {"x": 181, "y": 41, "width": 10, "height": 14}, "confidence": .995},
        {"candidate_id": "R4", "text": "C", "bbox": {"x": 211, "y": 41, "width": 10, "height": 14}, "confidence": .995},
    ]
    source = _outcome(
        "Ø0.5A B C",
        {"x": 70, "y": 36, "width": 160, "height": 24},
        children=children,
    )

    enriched = enrich_native_result_with_geometry(image, _result(source))
    candidate = enriched["candidate_outcomes"][0]

    assert candidate["state"] == "eligible"
    assert candidate["text"] == "⌖ | Ø0.5 | A | B | C"
    assert candidate["recognition_source"] == "native_pdf"
    assert candidate["recognition_evidence"]["native_text"] == "Ø0.5A B C"
    assert candidate["engineering_symbol"]["subtype"] == "Position"
    assert candidate["engineering_symbol"]["source"] == "frame_geometry+native_pdf"


def test_fcf_like_native_text_without_a_frame_is_sent_to_review():
    image = Image.new("RGB", (260, 110), "white")
    source = _outcome(
        "Ø0.5A B C",
        {"x": 70, "y": 36, "width": 160, "height": 24},
        children=[
            {"candidate_id": "R1", "text": "0.5", "bbox": {"x": 78, "y": 41, "width": 36, "height": 14}, "confidence": .995},
            {"candidate_id": "R2", "text": "A", "bbox": {"x": 151, "y": 41, "width": 10, "height": 14}, "confidence": .995},
            {"candidate_id": "R3", "text": "B", "bbox": {"x": 181, "y": 41, "width": 10, "height": 14}, "confidence": .995},
        ],
    )

    enriched = enrich_native_result_with_geometry(image, _result(source))

    assert enriched["review_count"] == 1
    assert enriched["candidate_outcomes"][0]["rule"] == (
        "native_structured_symbol_unresolved"
    )


def test_explicit_surface_finish_syntax_overrides_nearby_section_text():
    image = Image.new("RGB", (160, 80), "white")
    source = _outcome(
        "Rz12.5",
        {"x": 60, "y": 25, "width": 50, "height": 18},
    )

    enriched = enrich_native_result_with_geometry(image, _result(source))
    candidate = enriched["candidate_outcomes"][0]

    assert candidate["state"] == "eligible"
    assert candidate["category"] == "Surface Finish"
    assert candidate["engineering_symbol"]["kind"] == "surface_finish"


def test_local_degree_ring_enriches_native_tolerance_without_ocr():
    image = Image.new("RGB", (260, 100), "white")
    draw = ImageDraw.Draw(image)
    # Main-glyph-height ink establishes the baseline; the small hollow circle
    # above it is the candidate-local degree mark.
    for x in (35, 50, 65, 95, 110):
        draw.rectangle((x, 42, x + 4, 64), fill="black")
    draw.ellipse((82, 34, 88, 40), outline="black", width=1)
    source = _outcome(
        "22.9 ±1",
        {"x": 25, "y": 30, "width": 105, "height": 36},
        state="eligible",
    )

    enriched = enrich_native_result_with_geometry(image, _result(source))
    candidate = enriched["candidate_outcomes"][0]

    assert candidate["text"] == "22.9°±1°"
    assert candidate["state"] == "eligible"
    assert candidate["recognition_source"] == "native_pdf"
    assert candidate["recognition_evidence"]["ocr_text"] == ""
    assert candidate["recognition_evidence"]["native_text_before_geometry"] == (
        "22.9 ±1"
    )
