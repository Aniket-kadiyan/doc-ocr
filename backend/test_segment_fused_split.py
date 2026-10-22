"""
A section scan must split a fused cluster before judging it as a whole.

On a real sheet "R10.0" shared a cluster with the "8.0 TO INT" printed beside
it, and "Ø0.210 [Ø5.33]" with the "CHAMFER BOTH SIDES" note under it. Read as
one string neither is a value, and the old order of operations threw the whole
cluster away at the grammar or prose gate without ever trying to take it apart.
These tests pin the new order: a multi-box cluster whose read is not one
complete value is split first, and one surviving value is enough.
"""

from __future__ import annotations

from PIL import Image

from ocr_pipeline import OcrPipeline


def _result(text: str) -> dict:
    return {
        "text": text,
        "raw_ocr": text,
        "confidence": 0.97,
        "type": "linear",
        "orientation": "horizontal",
        "rotation": 0,
        "needs_review": False,
        "agreement": 1.0,
        "engine": "paddleocr",
        "symbols_detected": {},
    }


def _pipeline(fused_text: str, boxes: list[dict]) -> tuple[OcrPipeline, list[int]]:
    pipeline = object.__new__(OcrPipeline)
    pipeline.detect_regions = lambda *_a, **_k: boxes
    pipeline._expand_clusters = lambda _image, clusters, **_k: clusters
    pipeline.recognize = lambda _image, **_k: _result(fused_text)
    pipeline._complete_angle_regions = lambda _image, regions: regions
    pipeline._detect_angled_regions = lambda *_a, **_k: []
    pipeline._notes_regions = lambda *_a, **_k: []
    seen: list[int] = []

    def split(_image, coords, *, min_values=2, **_k):
        seen.append(min_values)
        x0, y0, _x1, _y1 = coords
        return [
            {
                "bbox": {"x": x0 + 4.0, "y": y0 + 4.0, "width": 60.0, "height": 14.0},
                **{k: v for k, v in _result("R10.0").items() if k != "raw_ocr"},
                "raw_text": "R10.0",
            }
        ]

    pipeline._split_stacked_cluster = split
    return pipeline, seen


# Two boxes on one row, close enough for the section clustering to join them.
_TWO_BOXES = [
    {"x": 10.0, "y": 10.0, "w": 60.0, "h": 14.0, "text": "", "conf": 0.9},
    {"x": 74.0, "y": 9.0, "w": 90.0, "h": 16.0, "text": "", "conf": 0.9},
]


def test_a_fused_read_that_is_not_a_value_is_split_and_one_value_is_kept() -> None:
    pipeline, seen = _pipeline("R10.08.0TO INT", _TWO_BOXES)

    result = pipeline.segment(Image.new("RGB", (1000, 300), "white"))

    assert seen == [1], "the split must be asked for a single peeled value"
    assert [r["text"] for r in result["regions"]] == ["R10.0"]
    assert result["regions"][0]["raw_text"] == "R10.0"


def test_a_fused_read_is_split_even_when_it_reads_as_prose() -> None:
    # Four stacked lines: a radius, a tag, and two note lines. The whole read
    # fails the prose test; before, that ended the cluster.
    pipeline, seen = _pipeline("R1.4SC-235POS ON60.8PCD", _TWO_BOXES)

    result = pipeline.segment(Image.new("RGB", (1000, 300), "white"))

    assert seen == [1]
    assert [r["text"] for r in result["regions"]] == ["R10.0"]


def test_a_single_complete_value_is_not_split() -> None:
    pipeline, seen = _pipeline("1.00[25.40]", _TWO_BOXES)

    result = pipeline.segment(Image.new("RGB", (1000, 300), "white"))

    assert seen == []
    assert [r["text"] for r in result["regions"]] == ["1.00[25.40]"]
    assert result["regions"][0]["raw_text"] == "1.00[25.40]"


def test_a_lone_box_that_is_not_a_value_is_not_split() -> None:
    # One detector box cannot be a fusion of two callouts; the gates decide.
    pipeline, seen = _pipeline("ALUMINIUM6061-T6", _TWO_BOXES[:1])

    result = pipeline.segment(Image.new("RGB", (1000, 300), "white"))

    assert seen == []
    assert result["regions"] == []


def test_a_low_confidence_section_read_is_excluded_not_ballooned() -> None:
    pipeline, _seen = _pipeline("8/5.5", _TWO_BOXES[:1])
    low = _result("8/5.5")
    low["confidence"] = 0.54
    pipeline.recognize = lambda _image, **_k: low

    result = pipeline.segment(Image.new("RGB", (1000, 300), "white"))

    assert result["regions"] == []
    assert result["filter_rule_counts"] == {"section_low_confidence": 1}
    outcome = result["candidate_outcomes"][0]
    assert outcome["state"] == "excluded"
    assert outcome["rule"] == "section_low_confidence"
    assert outcome["raw_text"] == "8/5.5"
