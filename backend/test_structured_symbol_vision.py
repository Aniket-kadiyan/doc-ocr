"""Synthetic checks for geometry-backed engineering symbols."""

from PIL import Image, ImageDraw
from structured_symbol_vision import plan_structured_engineering_symbols


def _record(text: str, bbox: dict[str, float]) -> dict:
    return {"candidate_id": "C1", "bbox": bbox, "text": text,
            "recognized": bool(text), "table_excluded": False,
            "result": {"text": text, "confidence": .95}}


def test_feature_control_frame_groups_cells_and_completes_perpendicularity():
    image = Image.new("RGB", (220, 100), "white"); draw = ImageDraw.Draw(image)
    draw.rectangle((30, 35, 170, 55), outline="black", width=2)
    draw.line((60, 35, 60, 55), fill="black", width=2)
    draw.line((120, 35, 120, 55), fill="black", width=2)
    draw.line((44, 39, 44, 51), fill="black", width=2)
    draw.line((38, 50, 50, 50), fill="black", width=2)
    records = [_record("0.3", {"x": 72, "y": 38, "width": 32, "height": 14}),
               {**_record("A", {"x": 137, "y": 38, "width": 10, "height": 14}), "candidate_id": "C2"}]
    plans = plan_structured_engineering_symbols(image, records)
    assert len(plans) == 1
    assert plans[0].evidence.kind == "feature_control_frame"
    assert plans[0].evidence.complete
    assert plans[0].evidence.subtype == "Perpendicularity"
    assert plans[0].text.startswith("⟂ | 0.3 | A")
    assert plans[0].member_indexes == (0, 1)


def test_boxed_single_letter_is_a_datum_object():
    image = Image.new("RGB", (120, 90), "white"); draw = ImageDraw.Draw(image)
    draw.rectangle((45, 30, 67, 52), outline="black", width=2)
    plans = plan_structured_engineering_symbols(
        image, [_record("B", {"x": 50, "y": 34, "width": 10, "height": 14})])
    assert len(plans) == 1 and plans[0].evidence.kind == "datum"
    assert plans[0].evidence.subtype == "B"


def test_explicit_surface_finish_text_is_preserved():
    image = Image.new("RGB", (160, 80), "white")
    plans = plan_structured_engineering_symbols(
        image, [_record("Ra 1.6", {"x": 70, "y": 30, "width": 45, "height": 15})])
    assert len(plans) == 1 and plans[0].evidence.kind == "surface_finish"
    assert plans[0].evidence.complete and plans[0].text == "Ra 1.6"


def test_plain_numeric_without_texture_geometry_is_not_relabelled():
    image = Image.new("RGB", (160, 80), "white")
    assert plan_structured_engineering_symbols(
        image, [_record("1.6", {"x": 70, "y": 30, "width": 30, "height": 15})]) == []
