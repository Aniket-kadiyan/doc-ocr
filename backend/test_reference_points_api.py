"""Contract tests for the reference-point backend endpoint."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

import main
from page_layout import LayoutBox
from reference_table import ReferencePoint, ReferenceTable


def _upload() -> UploadFile:
    payload = io.BytesIO()
    Image.new("RGB", (80, 60), "white").save(payload, format="PNG")
    payload.seek(0)
    return UploadFile(payload, filename="page.png")


def _table() -> ReferenceTable:
    return ReferenceTable(
        bbox=LayoutBox(40, 20, 200, 120),
        axes=("X", "Y", "Z"),
        name_header="POINT",
        header_box=LayoutBox(40, 20, 200, 30),
        points=(
            ReferencePoint(
                symbol="a",
                coordinates=(("X", "0.0"), ("Y", "35.6"), ("Z", "6.9")),
                name_box=LayoutBox(40, 50, 40, 30),
                row_box=LayoutBox(40, 50, 200, 30),
            ),
            ReferencePoint(
                symbol="b",
                coordinates=(("X", "10.0"), ("Y", "72.2"), ("Z", "9.3")),
                name_box=LayoutBox(40, 80, 40, 30),
                row_box=LayoutBox(40, 80, 200, 30),
            ),
        ),
    )


class _Pipeline:
    def __init__(self, result):
        self.result = result

    def reference_points(self, image):
        assert image.mode == "RGB"
        return self.result


def test_response_scales_geometry_and_keeps_unmatched_rows() -> None:
    marker = SimpleNamespace(
        symbol="a",
        bbox=LayoutBox(20, 10, 10, 12),
        score=0.91,
        margin=0.24,
    )
    result = {"table": _table(), "markers": [marker], "table_count": 4}
    with patch.object(main, "get_pipeline", return_value=_Pipeline(result)):
        payload = asyncio.run(
            main.read_reference_points(file=_upload(), display_scale=2.0)
        )

    assert payload["found"] is True
    assert payload["tableCount"] == 4
    assert payload["table"]["bbox"] == {
        "x": 20.0,
        "y": 10.0,
        "width": 100.0,
        "height": 60.0,
    }
    assert payload["table"]["points"][0]["markers"][0]["bbox"]["x"] == 10.0
    assert payload["unmatchedSymbols"] == ["b"]


def test_no_coordinate_table_is_an_explicit_empty_result() -> None:
    result = {"table": None, "markers": [], "table_count": 3}
    with patch.object(main, "get_pipeline", return_value=_Pipeline(result)):
        # FastAPI resolves Form(1.0) before calling the endpoint. Direct
        # contract tests call the function themselves, so pass that resolved
        # value explicitly instead of handing math.isfinite a Form field.
        payload = asyncio.run(
            main.read_reference_points(file=_upload(), display_scale=1.0)
        )

    assert payload == {
        "found": False,
        "tableCount": 3,
        "table": None,
        "unmatchedSymbols": [],
    }


def test_non_positive_display_scale_is_rejected() -> None:
    try:
        asyncio.run(main.read_reference_points(file=_upload(), display_scale=0))
    except HTTPException as error:
        assert error.status_code == 422
    else:
        raise AssertionError("zero display scale was accepted")
