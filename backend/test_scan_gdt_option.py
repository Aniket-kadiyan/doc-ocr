"""
The scan's GD&T option: off skips the feature control frame pass, which costs
3-20 s a page, and the route, the page scan and the text-layer read all honour
it. On stays the default, so a client that never sends it scans as before.

    python -m pytest -q test_scan_gdt_option.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
import pdf_text_layer
from auth_routes import require_protected_request

DRAWING = Path("/Users/chitrajain/Desktop/doc-ocr-box/Docs/BS1801006.020.pdf")
EMPTY_SEGMENT = {
    "regions": [],
    "review_candidates": [],
    "detected_count": 0,
    "recognized_count": 0,
    "eligible_count": 0,
    "excluded_count": 0,
    "review_count": 0,
}


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def segment_page(self, image: Any, **options: Any) -> dict[str, Any]:
        self.calls.append(options)
        return dict(EMPTY_SEGMENT)


@pytest.fixture
def scan(monkeypatch):
    pipeline = _Pipeline()
    monkeypatch.setattr(main, "get_pipeline", lambda: pipeline)
    # Run the job inline, so the test sees what the scan was asked to do.
    monkeypatch.setattr(
        main.scan_job_manager,
        "submit",
        lambda run, metadata: {"metadata": metadata, "result": run(lambda **_: None)},
    )
    main.app.dependency_overrides[require_protected_request] = lambda: None
    client = TestClient(main.app)

    def post(**form: str) -> dict[str, Any]:
        png = io.BytesIO()
        Image.new("RGB", (200, 100), "white").save(png, format="PNG")
        response = client.post(
            "/ocr/scan-jobs",
            files={"file": ("page.png", png.getvalue(), "image/png")},
            data={"scope_kind": "page", "scope_width": "200", "scope_height": "100", **form},
        )
        assert response.status_code == 202, response.text
        return response.json()

    yield post, pipeline
    main.app.dependency_overrides.pop(require_protected_request, None)
    client.close()


def test_a_scan_reads_frames_unless_told_not_to(scan):
    post, pipeline = scan
    assert post()["metadata"]["gdt"] is True
    assert post(gdt="false")["metadata"]["gdt"] is False
    assert [call["read_frames"] for call in pipeline.calls] == [True, False]


def test_a_section_scan_honours_the_option_too(scan):
    post, pipeline = scan
    post(scope_kind="section", gdt="false")
    assert pipeline.calls[-1]["read_frames"] is False


@pytest.mark.skipif(not DRAWING.exists(), reason="sample drawing not present")
def test_the_text_layer_read_skips_frames_when_told(monkeypatch):
    called = []
    monkeypatch.setattr(
        pdf_text_layer,
        "_native_frames",
        lambda *args, **kwargs: called.append(1) or args[4],
    )
    monkeypatch.setattr(pdf_text_layer, "_image_area_notes", lambda *a, **k: [])
    monkeypatch.setattr(pdf_text_layer, "_levelled_crop", lambda *a, **k: None)

    pdf_text_layer.text_layer_regions(DRAWING, read_frames=False)
    assert called == []
    pdf_text_layer.text_layer_regions(DRAWING)
    assert called == [1]
