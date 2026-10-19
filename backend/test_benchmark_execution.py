"""Fast tests that keep the benchmark on the production source-selection path."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from benchmarks import execution
from gdt_config import GdtConfig
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


def test_gdt_flag_off_does_not_change_established_page_result(monkeypatch):
    established = {
        "eligible_count": 1,
        "review_count": 0,
        "candidate_outcomes": [{"text": "R2", "state": "eligible"}],
    }
    monkeypatch.setattr(execution, "analyze_page_layout", lambda image: _Layout())
    monkeypatch.setattr(
        execution,
        "augment_page_scan_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled GD&T hook was called")
        ),
    )
    pipeline = _Pipeline(established)
    result = execution.segment_production_page(
        Image.new("RGB", (100, 80), "white"),
        page_number=1,
        pipeline=pipeline,
        gdt_config=GdtConfig(page_scan_enabled=False),
    )
    assert result == {**established, "coordinate_space": "page"}
    assert len(pipeline.calls) == 1


def test_gdt_hook_runs_after_native_selection_without_full_page_ocr(monkeypatch):
    native = {
        "eligible_count": 1,
        "review_count": 0,
        "candidate_outcomes": [{"text": "25", "state": "eligible"}],
    }
    monkeypatch.setattr(execution, "analyze_page_layout", lambda image: _Layout())
    monkeypatch.setattr(
        execution, "_native_pdf_inputs",
        lambda *args, **kwargs: ({"usable": True}, _evidence("vector")),
    )
    monkeypatch.setattr(
        execution, "native_text_segment_result", lambda *args, **kwargs: dict(native)
    )
    monkeypatch.setattr(
        execution, "enrich_native_result_with_geometry",
        lambda image, result: result,
    )
    seen = {}

    def augment(image, result, **kwargs):
        seen["result"] = result
        seen["page"] = kwargs["page"]
        seen["config"] = kwargs["config"]
        return {
            **result,
            "gdt_integration": {
                "status": "completed",
                "published_frame_count": 1,
            },
        }

    monkeypatch.setattr(execution, "augment_page_scan_result", augment)
    pipeline = _Pipeline()
    result = execution.segment_production_page(
        Image.new("RGB", (100, 80), "white"),
        page_number=2,
        pipeline=pipeline,
        source_pdf_bytes=b"pdf",
        gdt_config=GdtConfig(page_scan_enabled=True),
    )
    assert pipeline.calls == []
    assert seen["result"] is native or seen["result"] == native
    assert seen["page"] == 2
    assert seen["config"].page_scan_enabled is True
    assert result["gdt_integration"]["published_frame_count"] == 1
    assert result["coordinate_space"] == "page"


def test_forced_gdt_policy_is_recorded_in_document_snapshot(monkeypatch, tmp_path):
    source = tmp_path / "drawing.png"
    source.write_bytes(b"not-read-by-mocked-renderer")
    monkeypatch.setattr(
        execution,
        "render_pages",
        lambda path, dpi: iter((Image.new("RGB", (100, 80), "white"),)),
    )
    seen = {}

    def segment(image, **kwargs):
        seen["config"] = kwargs["gdt_config"]
        return {
            "detected_count": 0,
            "recognized_count": 0,
            "eligible_count": 0,
            "review_count": 0,
            "excluded_count": 0,
            "unread_count": 0,
            "skipped_existing_count": 0,
            "regions": [],
            "review_candidates": [],
            "candidate_outcomes": [],
            "gdt_integration": {"status": "completed"},
        }

    monkeypatch.setattr(execution, "segment_production_page", segment)
    snapshot = execution.segment_document(
        source,
        250,
        pipeline=_Pipeline(),
        gdt_page_scan=True,
    )
    assert seen["config"].page_scan_enabled is True
    assert snapshot["gdt_page_scan"] == {
        "policy": "forced_on",
        "enabled": True,
        "configured_enabled": True,
        "gdt_enabled": True,
        "page_scan_enabled": True,
    }
