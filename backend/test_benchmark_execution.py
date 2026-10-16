"""Fast tests that keep the benchmark on the production source-selection path."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from benchmarks import execution
from pdf_evidence import PdfPageEvidence


@dataclass(frozen=True)
class _Mask:
    x: float = 1.0

    def to_dict(self):
        return {"x": self.x, "y": 2.0, "width": 3.0, "height": 4.0}


@dataclass(frozen=True)
class _Layout:
    table_masks: tuple[_Mask, ...] = (_Mask(),)


class _Pipeline:
    def __init__(self, result=None):
        self.result = result or {"candidate_outcomes": []}
        self.calls = []

    def segment_page(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return dict(self.result)


def _evidence(profile: str) -> PdfPageEvidence:
    return PdfPageEvidence(
        page_number=1, page_count=1, target_width=100, target_height=80,
        profile=profile,
    )


def test_vector_native_result_bypasses_raster_ocr(monkeypatch):
    native = {"eligible_count": 1, "review_count": 0, "candidate_outcomes": []}
    monkeypatch.setattr(execution, "analyze_page_layout", lambda image: _Layout())
    monkeypatch.setattr(
        execution, "_native_pdf_inputs",
        lambda *args, **kwargs: ({"usable": True}, _evidence("vector")),
    )
    monkeypatch.setattr(
        execution, "native_text_segment_result",
        lambda *args, **kwargs: dict(native),
    )
    monkeypatch.setattr(
        execution, "enrich_native_result_with_geometry",
        lambda image, result: result,
    )
    pipeline = _Pipeline()
    result = execution.segment_production_page(
        Image.new("RGB", (100, 80), "white"), page_number=1,
        pipeline=pipeline, source_pdf_bytes=b"pdf",
    )
    assert pipeline.calls == []
    assert result["eligible_count"] == 1
    assert result["coordinate_space"] == "page"


def test_untrusted_pdf_text_uses_raster_ocr_with_fallback_evidence(monkeypatch):
    monkeypatch.setattr(execution, "analyze_page_layout", lambda image: _Layout())
    monkeypatch.setattr(
        execution, "_native_pdf_inputs",
        lambda *args, **kwargs: (
            {"usable": False},
            PdfPageEvidence.raster_fallback(
                page_number=1, target_size=(100, 80), error="untrusted"
            ),
        ),
    )
    pipeline = _Pipeline({"candidate_outcomes": [], "eligible_count": 0})
    execution.segment_production_page(
        Image.new("RGB", (100, 80), "white"), page_number=1,
        pipeline=pipeline, source_pdf_bytes=b"pdf",
    )
    assert len(pipeline.calls) == 1
    kwargs = pipeline.calls[0][1]
    assert kwargs["native_evidence"].profile == "raster"
    assert kwargs["layout"] == _Layout()


def test_hybrid_pdf_keeps_native_result_and_merges_ocr_additively(monkeypatch):
    native = {
        "eligible_count": 1, "review_count": 0,
        "candidate_outcomes": [{"text": "25", "state": "eligible"}],
    }
    ocr = {
        "eligible_count": 1,
        "candidate_outcomes": [{"text": "R2", "state": "eligible"}],
    }
    merged = {"eligible_count": 2, "candidate_outcomes": []}
    monkeypatch.setattr(execution, "analyze_page_layout", lambda image: _Layout())
    monkeypatch.setattr(
        execution, "_native_pdf_inputs",
        lambda *args, **kwargs: ({"usable": True}, _evidence("hybrid")),
    )
    monkeypatch.setattr(
        execution, "native_text_segment_result", lambda *args, **kwargs: dict(native)
    )
    monkeypatch.setattr(
        execution, "enrich_native_result_with_geometry",
        lambda image, result: result,
    )
    seen = {}

    def merge(native_result, ocr_result):
        seen["native"] = native_result
        seen["ocr"] = ocr_result
        return dict(merged)

    monkeypatch.setattr(execution, "merge_native_and_ocr_results", merge)
    pipeline = _Pipeline(ocr)
    result = execution.segment_production_page(
        Image.new("RGB", (100, 80), "white"), page_number=1,
        pipeline=pipeline, source_pdf_bytes=b"pdf",
    )
    assert len(pipeline.calls) == 1
    assert seen == {"native": native, "ocr": ocr}
    assert result["eligible_count"] == 2
