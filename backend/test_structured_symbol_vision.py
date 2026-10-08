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


def test_two_cell_box_with_horizontal_stroke_does_not_invent_straightness():
    image = Image.new("RGB", (180, 90), "white"); draw = ImageDraw.Draw(image)
    draw.rectangle((30, 32, 130, 54), outline="black", width=2)
    draw.line((66, 32, 66, 54), fill="black", width=2)
    draw.line((40, 43, 57, 43), fill="black", width=2)

    assert plan_structured_engineering_symbols(
        image, [_record("0.2", {"x": 78, "y": 36, "width": 28, "height": 14})]
    ) == []


def test_position_crosshair_is_distinguished_from_perpendicularity():
    image = Image.new("RGB", (240, 100), "white"); draw = ImageDraw.Draw(image)
    draw.rectangle((25, 34, 205, 58), outline="black", width=2)
    for x in (65, 125, 155, 180):
        draw.line((x, 34, x, 58), fill="black", width=2)
    draw.ellipse((36, 38, 54, 56), outline="black", width=2)
    draw.line((45, 38, 45, 56), fill="black", width=2)
    draw.line((36, 47, 54, 47), fill="black", width=2)
    records = [
        _record("0.5", {"x": 80, "y": 38, "width": 30, "height": 14}),
        {**_record("A", {"x": 135, "y": 38, "width": 10, "height": 14}),
         "candidate_id": "C2"},
        {**_record("B", {"x": 162, "y": 38, "width": 10, "height": 14}),
         "candidate_id": "C3"},
        {**_record("C", {"x": 187, "y": 38, "width": 10, "height": 14}),
         "candidate_id": "C4"},
    ]

    plans = plan_structured_engineering_symbols(image, records)

    assert len(plans) == 1
    assert plans[0].evidence.subtype == "Position"
    assert plans[0].text.startswith("⌖ | 0.5 | A | B | C")


def test_large_nearby_diagonals_do_not_relabel_a_dimension_as_surface_finish():
    image = Image.new("RGB", (260, 180), "white"); draw = ImageDraw.Draw(image)
    draw.line((20, 160, 130, 20), fill="black", width=2)
    draw.line((30, 20, 145, 160), fill="black", width=2)

    assert plan_structured_engineering_symbols(
        image, [_record("45.3", {"x": 150, "y": 75, "width": 40, "height": 16})]
    ) == []
