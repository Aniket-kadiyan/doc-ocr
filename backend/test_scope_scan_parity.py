"""Section mode reuses page panels, coordinates, OCR and value rules."""

from __future__ import annotations

from typing import Any

from PIL import Image

from ocr_pipeline import OcrPipeline
from page_layout import LayoutBox, LayoutPanel, PageLayout


def _ready_result(text: str) -> dict[str, Any]:
    return {
        "text": text,
        "raw_ocr": text,
        "confidence": 0.99,
        "confusable_corrected": False,
        "orientation_confidence": 0.99,
        "agreement": 1.0,
        "needs_review": False,
        "type": "Linear",
        "orientation": "horizontal",
        "rotation": 0,
        "engine": "paddleocr",
        "symbols_detected": {},
        "ocr_profile": "batch_recognition",
    }


def test_section_and_page_return_the_same_value_from_the_same_panel() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    pipeline._page_batch_recognition_available = True
    detector_sizes: list[tuple[int, int]] = []

    def fake_detector(image, *, target_long_edge):
        assert target_long_edge >= 1000
        detector_sizes.append(image.size)
        if image.size == (120, 100):
            return [{"x": 70, "y": 30, "w": 40, "h": 16, "conf": 0.95}]
        if image.size == (100, 120):
            return [{"x": 54, "y": 70, "w": 16, "h": 40, "conf": 0.95}]
        return []

    pipeline._detector_only_boxes = fake_detector
    pipeline._recognize_page_batch = lambda crops, **_kwargs: [
        _ready_result("25.0") for _crop in crops
    ]

    def authoritative(crop, **_kwargs):
        result = _ready_result("25.0")
        result["text_bbox"] = {
            "x": 2,
            "y": 2,
            "width": max(1, crop.width - 4),
            "height": max(1, crop.height - 4),
        }
        return result

    pipeline.recognize = authoritative
    pipeline.detect_regions = lambda *_args, **_kwargs: []
    pipeline._detect_angled_regions = lambda *_args, **_kwargs: []
    pipeline._notes_regions = lambda *_args, **_kwargs: []
    pipeline._frame_regions = lambda *_args, **_kwargs: []

    layout = PageLayout(
        width=200,
        height=100,
        table_masks=(),
        panels=(
            LayoutPanel("P1", LayoutBox(0, 0, 80, 100)),
            LayoutPanel("P2", LayoutBox(80, 0, 120, 100)),
        ),
        overlaps=(),
    )
    image = Image.new("RGB", (200, 100), "white")

    page_result = pipeline.segment_page(
        image,
        layout=layout,
        read_frames=False,
    )
    page_detector_calls = len(detector_sizes)
    detector_sizes.clear()

    section_result = pipeline.segment_page(
        image,
        layout=layout,
        scope_bbox={"x": 140, "y": 10, "width": 55, "height": 80},
        read_frames=False,
    )

    assert section_result["coordinate_space"] == "page"
    assert section_result["regions"] == page_result["regions"]
    assert section_result["candidate_outcomes"] == page_result["candidate_outcomes"]
    assert section_result["regions"][0]["bbox"]["x"] == 150.0
    assert len(detector_sizes) < page_detector_calls
    assert (80, 100) not in detector_sizes

