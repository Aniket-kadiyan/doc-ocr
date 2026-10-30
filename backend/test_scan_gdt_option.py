"""The queued scan job carries the optional GD&T frame-read flag end to end."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
import pdf_text_layer
from auth_routes import require_protected_request


EMPTY_SEGMENT = {
    "regions": [],
    "review_candidates": [],
    "candidate_outcomes": [],
    "detected_count": 0,
    "recognized_count": 0,
    "eligible_count": 0,
    "excluded_count": 0,
    "review_count": 0,
    "unread_count": 0,
    "filter_rule_counts": {},
}


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def segment_page(self, image: Any, **options: Any) -> dict[str, Any]:
        self.calls.append(options)
        return dict(EMPTY_SEGMENT)


class _InlineManager:
    def submit(
        self,
        run: Any,
        *,
        owner_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "owner_id": owner_id,
            "metadata": metadata,
            "result": run(lambda **_: None),
        }


@pytest.fixture
def scan(monkeypatch: pytest.MonkeyPatch):
    pipeline = _Pipeline()
    monkeypatch.setattr(main, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(main, "_get_scan_job_manager", lambda: _InlineManager())
    main.app.dependency_overrides[require_protected_request] = lambda: SimpleNamespace(
        user=SimpleNamespace(id="test-user")
    )
    client = TestClient(main.app)

    def post(
        *,
        scope_kind: str = "page",
        source_pdf: bytes | None = None,
        **form: str,
    ) -> dict[str, Any]:
        png = io.BytesIO()
        Image.new("RGB", (200, 100), "white").save(png, format="PNG")
        files: dict[str, tuple[str, bytes, str]] = {
            "file": ("page.png", png.getvalue(), "image/png")
        }
        if source_pdf is not None:
            files["source_pdf"] = ("drawing.pdf", source_pdf, "application/pdf")
        response = client.post(
            "/ocr/scan-jobs",
            files=files,
            data={
                "scope_kind": scope_kind,
                "scope_width": "200",
                "scope_height": "100",
                **form,
            },
        )
        assert response.status_code == 202, response.text
        return response.json()

    yield post, pipeline
    main.app.dependency_overrides.pop(require_protected_request, None)
    client.close()


def test_page_and_section_scans_honour_the_gdt_option(scan: Any) -> None:
    post, pipeline = scan

    assert post()["metadata"]["gdt"] is True
    assert post(gdt="false")["metadata"]["gdt"] is False
    post(scope_kind="section", gdt="false")

    assert [call["read_frames"] for call in pipeline.calls] == [
        True,
        False,
        False,
    ]


def test_pdf_primary_pass_honours_the_gdt_option(
    scan: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post, pipeline = scan
    calls: list[dict[str, Any]] = []

    def fake_text_layer(*args: Any, **options: Any) -> dict[str, Any]:
        calls.append(options)
        return {
            "usable": True,
            "page_size": (200, 100),
            "regions": [],
            "excluded": [],
            "text_blocks": [],
            "fonts": [],
        }

    monkeypatch.setattr(pdf_text_layer, "text_layer_regions", fake_text_layer)
    result = post(source_pdf=b"%PDF-test", gdt="false")

    assert result["metadata"]["gdt"] is False
    assert calls == [{"dpi": 108.0, "refine": True, "read_frames": False}]
    assert pipeline.calls == []

