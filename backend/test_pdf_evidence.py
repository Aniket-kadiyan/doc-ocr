"""Fast native-PDF evidence tests; no OCR models are loaded."""

from __future__ import annotations

from io import BytesIO

from PIL import Image
import pytest

from pdf_evidence import (
    NativePdfSpan,
    PdfPageEvidence,
    crop_pdf_page_evidence,
    extract_pdf_page_evidence,
    fuse_native_with_ocr,
    match_native_spans,
    native_match_is_authoritative,
    native_result,
)


def _pdf_with_text(text: str = "R1 ±0.25") -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 40), text, fontsize=12)
    payload = document.tobytes()
    document.close()
    return payload


def _image_only_pdf() -> bytes:
    import fitz

    image_buffer = BytesIO()
    Image.new("RGB", (200, 100), "white").save(image_buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    payload = document.tobytes()
    document.close()
    return payload


def _span(
    text: str,
    *,
    x: float = 10,
    y: float = 20,
    width: float = 100,
    height: float = 20,
) -> NativePdfSpan:
    return NativePdfSpan(
        span_id="P1:S00001",
        text=text,
        bbox={"x": x, "y": y, "width": width, "height": height},
    )


def _tight_match(text: str):
    span = _span(text)
    match = match_native_spans(dict(span.bbox), (span,))
    assert match is not None
    return match


def test_extracts_and_scales_positioned_text_from_vector_pdf() -> None:
    evidence = extract_pdf_page_evidence(
        _pdf_with_text(),
        page_number=1,
        target_size=(400, 200),
    )

    assert evidence.profile == "vector"
    assert evidence.page_count == 1
    assert evidence.native_text_available
    assert "R1" in " ".join(span.text for span in evidence.spans)
    assert evidence.spans[0].bbox["x"] == pytest.approx(40, abs=2)
    assert evidence.spans[0].orientation == "horizontal"


def test_image_only_pdf_is_profiled_as_raster_without_native_text() -> None:
    evidence = extract_pdf_page_evidence(
        _image_only_pdf(),
        page_number=1,
        target_size=(400, 200),
    )

    assert evidence.profile == "raster"
    assert not evidence.native_text_available
    assert evidence.spans == ()
    assert evidence.raster_coverage == pytest.approx(1.0)


def test_section_crop_clips_and_translates_native_spans() -> None:
    evidence = PdfPageEvidence(
        page_number=1,
        page_count=1,
        target_width=400,
        target_height=200,
        profile="vector",
        spans=(_span("25", x=110, y=70, width=30, height=12),),
        native_character_count=2,
    )

    cropped = crop_pdf_page_evidence(
        evidence,
        {"x": 100, "y": 50, "width": 120, "height": 80},
    )

    assert cropped.target_width == 120
    assert cropped.target_height == 80
    assert cropped.spans[0].bbox == {
        "x": 10.0,
        "y": 20.0,
        "width": 30.0,
        "height": 12.0,
    }


def test_section_crop_does_not_assign_full_text_to_a_clipped_fragment() -> None:
    evidence = PdfPageEvidence(
        page_number=1,
        page_count=1,
        target_width=400,
        target_height=200,
        profile="vector",
        spans=(_span("Rz 12.5", x=90, y=70, width=60, height=12),),
        native_character_count=6,
    )

    cropped = crop_pdf_page_evidence(
        evidence,
        {"x": 100, "y": 50, "width": 120, "height": 80},
    )

    assert cropped.spans == ()
    assert cropped.native_character_count == 0


def test_small_raster_fragment_cannot_inherit_a_larger_native_callout() -> None:
    span = _span("Rz 12.5", width=100, height=18)

    assert (
        match_native_spans(
            {"x": 75, "y": 20, "width": 12, "height": 18},
            (span,),
        )
        is None
    )
    assert match_native_spans(dict(span.bbox), (span,)) is not None


def test_only_tight_complete_matches_can_bypass_ocr() -> None:
    tight = _tight_match("R1±0.25")
    span = _span("R1±0.25")
    loose = match_native_spans(
        {"x": 0, "y": 20, "width": 140, "height": 20},
        (span,),
    )

    assert native_match_is_authoritative(tight, validator=lambda _text: True)
    assert loose is not None
    assert not native_match_is_authoritative(loose, validator=lambda _text: True)


def test_parenthesized_native_value_requires_independent_ocr() -> None:
    # The benchmark's custom font exposes ``(67.1°)`` as ``(67.1)`` in its
    # PDF text layer, so tight geometry alone cannot establish completeness.
    match = _tight_match("(67.1)")

    assert not native_match_is_authoritative(match, validator=lambda _text: True)


def test_equivalent_native_and_ocr_readings_merge_symbol_evidence() -> None:
    fused = fuse_native_with_ocr(
        _tight_match("50±0.2"),
        {"text": "Ø500.2", "confidence": 0.91},
        native_authoritative=False,
    )

    assert fused["text"] == "Ø50±0.2"
    assert fused["recognition_source"] == "native_pdf+ocr"
    assert not fused["source_conflict"]
    assert fused["recognition_evidence"]["native_text"] == "50±0.2"
    assert fused["recognition_evidence"]["ocr_text"] == "Ø500.2"


def test_disagreement_retains_both_readings_and_requires_review() -> None:
    fused = fuse_native_with_ocr(
        _tight_match("R1±0.25"),
        {"text": "R1±0.26", "confidence": 0.93},
        native_authoritative=False,
        final=True,
    )

    assert fused["text"] == "R1±0.26"
    assert fused["source_conflict"]
    assert fused["needs_review"]
    assert fused["authoritative_review_required"]
    assert fused["recognition_evidence"]["native_text"] == "R1±0.25"
    assert fused["recognition_evidence"]["ocr_text"] == "R1±0.26"


def test_native_only_result_records_its_span_provenance() -> None:
    result = native_result(_tight_match("25"))

    assert result["text"] == "25"
    assert result["recognition_source"] == "native_pdf"
    assert result["recognition_evidence"]["sources"] == ["native_pdf"]
    assert result["recognition_evidence"]["native_span_ids"] == ["P1:S00001"]
