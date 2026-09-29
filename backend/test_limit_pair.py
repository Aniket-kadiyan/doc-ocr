"""
Tests for the limit-dimension helpers in ocr_pipeline.

A limit dimension is drawn as its upper limit over its lower limit —
``Ø0.620`` over ``0.612`` on 47630.pdf — and is ONE callout. The page route
used to balloon it as two (``620612`` and ``Ø1.0011``, the detector's column
cuts). These tests pin the text rules that join it back together, and the
rules that keep two independent stacked callouts apart.

    python -m pytest -q test_limit_pair.py
"""

from __future__ import annotations

from ocr_pipeline import (
    OcrPipeline,
    join_dual_unit_limits,
    limit_pair_from_read,
    parse_limit_pair,
)


def test_two_rows_of_a_limit_dimension_join():
    assert parse_limit_pair("Ø0.620", "0.612") == "Ø0.620/0.612"
    assert parse_limit_pair("0.620", "0.612") == "0.620/0.612"
    assert parse_limit_pair("20.05", "20.00") == "20.05/20.00"
    # The Ø may be credited to either row (it is drawn between them).
    assert parse_limit_pair("0.620", "Ø0.612") == "Ø0.620/0.612"


def test_two_independent_stacked_callouts_do_not_join():
    # 8.00 [203.20] over 9.00 [228.60] on the same sheet: two lengths.
    assert parse_limit_pair("8.00", "9.00") is None
    assert parse_limit_pair("0.20", "0.25") is None
    # Bracketed or toleranced rows are not a plain limit pair.
    assert parse_limit_pair("8.00[203.20]", "9.00[228.60]") is None
    assert parse_limit_pair("0.620±0.01", "0.612") is None


def test_precision_and_identity_rules():
    assert parse_limit_pair("0.620", "0.62") is None
    assert parse_limit_pair("620", "612") is None
    assert parse_limit_pair("0.620", "0.620") is None


def test_limit_stack_read_flat_is_recovered():
    assert limit_pair_from_read("Ø0.620φ0.612") == "Ø0.620/0.612"
    assert limit_pair_from_read("0.620/0.612") == "0.620/0.612"
    assert limit_pair_from_read("Ø0.620 0.612") == "Ø0.620/0.612"
    assert limit_pair_from_read("6200.612±0.7") is None
    assert limit_pair_from_read("Ø0.210[05.33]") is None


def _region(text, x, y, w, h, conf=1.0):
    return {"text": text, "bbox": {"x": x, "y": y, "width": w, "height": h}, "confidence": conf}


def test_metric_limits_join_their_inch_limits():
    inch = _region("Ø0.620/0.612", 1301, 810, 148, 73)
    metric = _region("15.75/15.54", 1515, 810, 114, 73, conf=0.9)
    other = _region("1.13[28.70]REF.", 366, 1463, 363, 62)
    kept, removed = join_dual_unit_limits([other, inch, metric])
    assert removed == [metric]
    assert [r["text"] for r in kept] == ["1.13[28.70]REF.", "Ø0.620/0.612[15.75/15.54]"]
    joined = kept[1]
    assert joined["bbox"] == {"x": 1301, "y": 810, "width": 328, "height": 73}
    assert joined["confidence"] == 0.9


def test_metric_limits_that_do_not_convert_stay_apart():
    inch = _region("Ø0.620/0.612", 1301, 810, 148, 73)
    wrong = _region("16.75/15.54", 1515, 810, 114, 73)
    kept, removed = join_dual_unit_limits([inch, wrong])
    assert removed == [] and len(kept) == 2
    far = _region("15.75/15.54", 1700, 810, 114, 73)
    kept, removed = join_dual_unit_limits([inch, far])
    assert removed == [] and len(kept) == 2


def test_limit_group_grows_over_the_stacks_other_cuts():
    # 47630: the page objects the detector made of ONE limit stack.
    records = [
        {"bbox": {"x": 1374, "y": 793, "width": 89, "height": 109}},   # "620612" column
        {"bbox": {"x": 1316, "y": 796, "width": 140, "height": 55}},   # "0.620" top row
        {"bbox": {"x": 1286, "y": 798, "width": 86, "height": 98}},    # "Ø1.00" column
        {"bbox": {"x": 1275, "y": 805, "width": 189, "height": 110}},  # the whole read flat
        {"bbox": {"x": 1500, "y": 792, "width": 146, "height": 59}},   # 15.75, the next stack
        {"bbox": {"x": 1010, "y": 136, "width": 291, "height": 61}},   # elsewhere
    ]
    seed = {"x": 1374, "y": 793, "width": 89, "height": 109}
    group = OcrPipeline._limit_group_bbox(seed, records, 60.0)
    assert group == {"x": 1275.0, "y": 793.0, "width": 189.0, "height": 122.0}


def test_note_markers_are_put_back_on_their_lines():
    boxes = [
        {"x": 1345, "y": 1541, "w": 131, "h": 43, "text": "NOTES:", "conf": 1.0},
        {"x": 1393, "y": 1578, "w": 1245, "h": 47, "text": "MATERIAL: GRADE 5 EQUIV", "conf": 1.0},
        {"x": 1352, "y": 1591, "w": 31, "h": 28, "text": "1.", "conf": 1.0},
        {"x": 1397, "y": 1620, "w": 998, "h": 43, "text": "PIN TO BE STRAIGHT", "conf": 1.0},
        {"x": 1353, "y": 1626, "w": 25, "h": 29, "text": "2", "conf": 1.0},
        # Two markers fused into one tall object, beside two lines.
        {"x": 1348, "y": 1696, "w": 39, "h": 78, "text": "45.", "conf": 0.9},
        {"x": 1401, "y": 1703, "w": 1104, "h": 34, "text": "FINISH: ZINC YELLOW", "conf": 1.0},
        {"x": 1399, "y": 1740, "w": 1189, "h": 35, "text": "LANYARD RING MUST", "conf": 1.0},
    ]
    joined = OcrPipeline._join_note_markers(boxes)
    texts = sorted(b["text"] for b in joined)
    assert texts == [
        "1. MATERIAL: GRADE 5 EQUIV",
        "2. PIN TO BE STRAIGHT",
        "4. FINISH: ZINC YELLOW",
        "5. LANYARD RING MUST",
        "NOTES:",
    ]
    region, _ = OcrPipeline._detect_notes_block(joined)
    assert region is not None
    assert region["text"].split("\n")[0] == "1. MATERIAL: GRADE 5 EQUIV"


def test_ink_touching_the_left_edge_of_a_levelled_box_is_noticed():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (200, 60), "white")
    draw = ImageDraw.Draw(image)
    # A glyph stem cut through by the box edge at x=80.
    draw.rectangle((72, 15, 84, 45), fill="black")
    assert OcrPipeline._ink_at_left_edge(image, 80, 10, 50, 30.0)
    # A blank margin, or a single speck, is not a cut glyph.
    assert not OcrPipeline._ink_at_left_edge(image, 150, 10, 50, 30.0)
    draw.point((148, 30), fill="black")
    assert not OcrPipeline._ink_at_left_edge(image, 150, 10, 50, 30.0)
