"""
Tests for feature control frame parsing (backend/fcf_parse.py).

Frames are drawn by ``benchmarks.gdt_synthetic`` and read back, with OCR
supplied as a stub. That is the point of injecting ``ocr_fn``: the geometry —
finding the rectangle, splitting composite segments, splitting compartments,
erasing the rules — is the part most likely to break, and it is tested here
with no model loaded.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_fcf_parse.py
"""

from __future__ import annotations

import pytest
from PIL import Image

import fcf_parse as fcf
import gdt_registry as reg

pytest.importorskip("cv2")

from benchmarks import gdt_synthetic as syn  # noqa: E402


def stub_ocr(mapping: dict[str, str] | None = None, default: str = ""):
    """An OCR hook that returns canned text by call order."""
    calls: list[str] = []
    sequence = list((mapping or {}).values())

    def ocr_fn(image: Image.Image) -> tuple[str, float]:
        index = len(calls)
        calls.append("called")
        if index < len(sequence):
            return (sequence[index], 0.95)
        return (default, 0.5)

    ocr_fn.calls = calls  # type: ignore[attr-defined]
    return ocr_fn


# --- finding frames ----------------------------------------------------------


def test_every_drawn_frame_is_found():
    scene = syn.build("frames")
    boxes = fcf.find_frames(scene.image)
    assert len(boxes) == len(scene.frames)


def test_a_blank_page_yields_no_frames():
    assert fcf.find_frames(Image.new("RGB", (400, 300), "white")) == []


def test_a_plain_rectangle_with_no_characteristic_is_not_published():
    """A title-block cell and a basic dimension are both boxes."""
    import numpy as np
    import cv2

    canvas = np.full((160, 400), 255, np.uint8)
    cv2.rectangle(canvas, (40, 40), (340, 100), 0, 2)
    image = Image.fromarray(canvas).convert("RGB")
    frames = fcf.parse_frames(image, ocr_fn=stub_ocr(default="50"))
    assert frames == []


# --- compartments ------------------------------------------------------------


@pytest.mark.parametrize("scene_name", ["frames", "composite_frame"])
def test_compartment_counts_match_the_drawn_frames(scene_name):
    scene = syn.build(scene_name)
    counts = []
    for x, y, w, h in _frame_sized(fcf.find_frames(scene.image), scene):
        crop = scene.image.crop((int(x), int(y), int(x + w), int(y + h)))
        for bx, by, bw, bh in fcf.split_segments(crop):
            band = crop.crop((bx, by, bx + bw, by + bh))
            counts.append(len(fcf.split_compartments(band)))
    expected = sorted(2 + len(f.datums) for f in scene.frames)
    assert sorted(counts) == expected


def test_a_digit_stroke_is_not_mistaken_for_a_compartment_divider():
    """The "1" of a 0.1 tolerance split that compartment in two."""
    import numpy as np

    canvas = np.full((60, 300), 255, np.uint8)
    frame, _ = syn.draw_frame(
        canvas, 5, 5, 48, symbol_id="position", tolerance=0.1, datums=("A",)
    )
    image = Image.fromarray(canvas).convert("RGB")
    box = fcf.find_frames(image)[0]
    crop = image.crop(
        (int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3]))
    )
    assert len(fcf.split_compartments(crop)) == 3


def test_erase_rules_removes_borders_but_keeps_a_straightness_stroke():
    """The stroke symbol is a line too — it must survive rule removal."""
    import numpy as np

    canvas = np.full((70, 300), 255, np.uint8)
    syn.draw_frame(canvas, 5, 5, 56, symbol_id="straightness", tolerance=0.5)
    image = Image.fromarray(canvas).convert("RGB")
    box = fcf.find_frames(image)[0]
    crop = image.crop(
        (int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3]))
    )
    cleaned = fcf.erase_rules(crop)

    import gdt_detect as detect

    before = float((detect.binarize(detect.to_gray(crop)) > 0).mean())
    after = float((detect.binarize(detect.to_gray(cleaned)) > 0).mean())
    assert after < before, "no rules were erased"
    # The characteristic compartment still has its stroke.
    cells = fcf.split_compartments(crop)
    cell = cleaned.crop((cells[0][0] + 2, 2, cells[0][0] + cells[0][2] - 2, crop.height - 2))
    assert (detect.binarize(detect.to_gray(cell)) > 0).any(), "the stroke was erased"


# --- composite frames --------------------------------------------------------


def _frame_sized(boxes, scene):
    """The detected boxes big enough to be one of the scene's frames.

    The finder over-produces by design: the legs and crossbar of a datum
    letter pair with the rule above them into a small box of their own, and
    it is the parser that turns those away.
    """
    height = min(f.bbox[3] for f in scene.frames) * 0.9
    width = min(f.bbox[2] for f in scene.frames) * 0.9
    return [b for b in boxes if b[3] >= height and b[2] >= width]


def test_a_composite_frame_splits_into_its_segments():
    scene = syn.build("composite_frame")
    detected = _frame_sized(fcf.find_frames(scene.image), scene)
    assert len(detected) == 1, "stacked segments share rules and detect as one"
    x, y, w, h = detected[0]
    crop = scene.image.crop((int(x), int(y), int(x + w), int(y + h)))
    assert len(fcf.split_segments(crop)) == 2


def test_composite_segments_are_grouped_together():
    boxes = [(10.0, 10.0, 200.0, 40.0), (10.0, 50.0, 200.0, 40.0)]
    assert fcf.group_composite(boxes) == [[0, 1]]


def test_frames_far_apart_are_not_grouped():
    boxes = [(10.0, 10.0, 200.0, 40.0), (10.0, 400.0, 200.0, 40.0)]
    assert fcf.group_composite(boxes) == [[0], [1]]


def test_frames_of_different_width_are_not_grouped():
    boxes = [(10.0, 10.0, 200.0, 40.0), (10.0, 50.0, 60.0, 40.0)]
    assert fcf.group_composite(boxes) == [[0], [1]]


# --- field parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "text,value,diameter,material",
    [
        ("0.5", 0.5, False, None),
        ("Ø0.5", 0.5, True, None),
        ("Ø0.05(M)", 0.05, True, "mmc"),
        ("0.2 (L)", 0.2, False, "lmc"),
        ("Ø0,5", 0.5, True, None),      # comma decimal
        ("", None, False, None),
    ],
)
def test_tolerance_compartment_parsing(text, value, diameter, material):
    parsed = fcf.parse_tolerance_compartment(text, 0.9)
    assert parsed["value"] == value
    assert parsed["diameter_zone"] is diameter
    assert parsed["material_condition"] == material


def test_a_bare_letter_is_not_read_as_a_modifier():
    """A lone M is more often a stray mark; AIGDT writes Ø as a bare P."""
    parsed = fcf.parse_tolerance_compartment("0.5 P", 0.9)
    assert parsed["material_condition"] is None
    assert "projected_tolerance_zone" not in parsed["modifiers"]


@pytest.mark.parametrize(
    "text,label,material",
    [
        ("A", "A", None),
        ("B(M)", "B", "mmc"),
        ("[C]", "C", None),
        ("A-B", "AB", None),
    ],
)
def test_datum_compartment_parsing(text, label, material):
    ref, _ = fcf.parse_datum_compartment(text, 0.9, 1)
    assert ref is not None
    assert ref.label == label
    assert ref.material_condition == material


def test_a_reserved_datum_letter_is_flagged():
    """ASME Y14.5 does not use I, O or Q — they read as 1 and 0."""
    ref, warnings = fcf.parse_datum_compartment("O", 0.9, 1)
    assert ref is not None and ref.label == "O"
    assert any("reserved" in w for w in warnings)


def test_an_unreadable_datum_compartment_yields_nothing_not_a_guess():
    ref, _ = fcf.parse_datum_compartment("###", 0.2, 1)
    assert ref is None


# --- end to end --------------------------------------------------------------


def test_a_frame_reads_into_structured_fields_with_datum_precedence():
    scene = syn.build("frames")
    target = next(f for f in scene.frames if len(f.datums) == 3)
    ocr = stub_ocr({"0": "", "1": "Ø0.5(M)", "2": "A", "3": "B", "4": "C"})

    box = min(
        fcf.find_frames(scene.image), key=lambda b: abs(b[1] - target.bbox[1])
    )
    x, y, w, h = box
    crop = scene.image.crop((int(x), int(y), int(x + w), int(y + h)))
    frame = fcf.parse_frame(crop, bbox=box, ocr_fn=ocr)

    assert frame.symbol is not None and frame.symbol.symbol_id == target.symbol_id
    assert frame.tolerance.state == fcf.PRESENT and frame.tolerance.value == 0.5
    assert frame.diameter_zone.value is True
    assert frame.material_condition.value == "mmc"
    assert [d.label for d in frame.datums] == ["A", "B", "C"]
    assert [d.precedence for d in frame.datums] == [1, 2, 3]


def test_precedence_follows_position_even_when_a_datum_is_unreadable():
    """A dropped middle datum must not promote the third to second."""
    scene = syn.build("frames")
    target = next(f for f in scene.frames if len(f.datums) == 3)
    ocr = stub_ocr({"0": "", "1": "0.5", "2": "A", "3": "###", "4": "C"})
    box = min(
        fcf.find_frames(scene.image), key=lambda b: abs(b[1] - target.bbox[1])
    )
    x, y, w, h = box
    crop = scene.image.crop((int(x), int(y), int(x + w), int(y + h)))
    frame = fcf.parse_frame(crop, bbox=box, ocr_fn=ocr)

    labels = {d.label: d.precedence for d in frame.datums}
    assert labels.get("A") == 1
    assert labels.get("C") == 3, "the third datum was promoted to second"


def test_a_form_control_with_no_datum_says_so_rather_than_reporting_a_gap():
    scene = syn.build("frames")
    target = next(f for f in scene.frames if f.symbol_id == "flatness")
    ocr = stub_ocr({"0": "", "1": "0.5"})
    box = min(
        fcf.find_frames(scene.image), key=lambda b: abs(b[1] - target.bbox[1])
    )
    x, y, w, h = box
    crop = scene.image.crop((int(x), int(y), int(x + w), int(y + h)))
    frame = fcf.parse_frame(crop, bbox=box, ocr_fn=ocr)
    assert frame.datums == []
    assert any("takes no datum" in w for w in frame.warnings)


def test_ocr_failure_on_one_cell_does_not_lose_the_frame():
    def exploding_ocr(image):
        raise RuntimeError("model unavailable")

    scene = syn.build("frames")
    box = fcf.find_frames(scene.image)[0]
    x, y, w, h = box
    crop = scene.image.crop((int(x), int(y), int(x + w), int(y + h)))
    frame = fcf.parse_frame(crop, bbox=box, ocr_fn=exploding_ocr)
    assert frame.symbol is not None
    assert any("OCR failed" in w for w in frame.warnings)


def test_an_ambiguous_characteristic_frame_is_retained_for_review():
    """Uncertainty must downgrade a frame, not silently remove it."""
    from gdt_config import GdtConfig
    from gdt_detect import STATUS_AMBIGUOUS

    scene = syn.build("frames")
    config = GdtConfig(accept_confidence=0.999)
    frames = fcf.parse_frames(
        scene.image,
        ocr_fn=stub_ocr(default="0.5"),
        config=config,
    )

    assert frames
    assert any(
        frame.symbol is not None and frame.symbol.status == STATUS_AMBIGUOUS
        for frame in frames
    )


def test_an_unreadable_modifier_is_not_reported_as_absent():
    """Absent means RFS. Unreadable means nobody knows. They differ."""
    tolerance_states = {fcf.PRESENT, fcf.ABSENT, fcf.UNREADABLE, fcf.NOT_APPLICABLE}
    assert fcf.Field.unreadable().state == fcf.UNREADABLE
    assert fcf.Field.unreadable().state in tolerance_states
    assert fcf.Field().state == fcf.ABSENT
    assert fcf.Field.unreadable().state != fcf.Field().state


# --- schema ------------------------------------------------------------------


def _valid_payload() -> dict:
    return {
        "type": "feature_control_frame",
        "id": "abc123",
        "page": 1,
        "bounding_box": {"x": 1.0, "y": 2.0, "width": 10.0, "height": 4.0},
        "symbol": {"symbol_id": "position", "name": "Position"},
        "tolerance": {"value": 0.5, "state": "present", "confidence": 0.9, "raw": "0.5"},
        "diameter_zone": {"value": True, "state": "present", "confidence": 0.9, "raw": ""},
        "material_condition": {"value": "mmc", "state": "present", "confidence": 0.9, "raw": ""},
        "modifiers": [],
        "datums": [
            {"label": "A", "precedence": 1},
            {"label": "B", "precedence": 2},
        ],
        "compartments": [],
        "segment_index": 0,
        "composite_id": None,
        "confidence": 0.9,
        "warnings": [],
        "tolerance_unit": "drawing_unit",
    }


def test_a_well_formed_frame_validates():
    assert fcf.validate(_valid_payload()) == []


def test_validation_rejects_an_unknown_symbol_id():
    payload = _valid_payload()
    payload["symbol"]["symbol_id"] = "not_a_symbol"
    assert any("unknown symbol id" in p for p in fcf.validate(payload))


def test_validation_rejects_out_of_order_datum_precedence():
    payload = _valid_payload()
    payload["datums"] = [
        {"label": "A", "precedence": 2},
        {"label": "B", "precedence": 1},
    ]
    assert any("precedence must increase" in p for p in fcf.validate(payload))


def test_validation_rejects_a_present_field_with_no_value():
    payload = _valid_payload()
    payload["tolerance"] = {"value": None, "state": "present", "confidence": 0.0, "raw": None}
    assert any("marked present but carries no value" in p for p in fcf.validate(payload))


def test_validation_rejects_an_invented_unit():
    payload = _valid_payload()
    payload["tolerance_unit"] = "mm"
    assert any("drawing_unit" in p for p in fcf.validate(payload))


def test_a_real_parsed_frame_validates():
    scene = syn.build("frames")
    frames = fcf.parse_frames(scene.image, ocr_fn=stub_ocr(default="0.5"))
    assert frames
    for frame in frames:
        assert fcf.validate(frame.to_dict()) == []


# --- the diametral zone marker, read from pixels -----------------------------


def test_the_diameter_zone_is_found_without_any_ocr_text():
    """A ⌀ is a glyph, and OCR does not read glyphs.

    On a real drawing the tolerance compartment comes back as "0.30" with the
    ⌀ silently dropped, and a text-only parse then reports a parallel-plane
    zone for a callout specifying a cylindrical one — a different control.
    """
    scene = syn.build("composite_frame")  # position frames, diametral zones
    frames = fcf.parse_frames(scene.image, ocr_fn=lambda image: ("", 0.0))
    assert frames
    for frame in frames:
        assert frame.diameter_zone.value is True
        assert frame.diameter_zone.state == fcf.PRESENT


def test_a_plain_tolerance_is_not_given_a_zone_marker():
    """The digit zero of "0.5" is a ring too; it must not read as a ⌀."""
    scene = syn.build("frames")
    frames = fcf.parse_frames(scene.image, ocr_fn=lambda image: ("", 0.0))
    assert frames
    for frame in frames:
        truth = min(
            scene.frames,
            key=lambda t: abs(t.bbox[1] - frame.bbox[1]) + abs(t.bbox[0] - frame.bbox[0]),
        )
        assert frame.diameter_zone.value == truth.diameter_zone


# --- composite frames are not rejected before they can be split --------------


def test_the_aspect_gate_is_applied_per_segment_not_per_stack():
    """The gate assumes one frame; a composite is a stack of them.

    A real three-segment control (⊥0.02 A / —0.02 / //0.02 D) measures about
    70x53 — raw aspect 1.32 against a 1.6 floor — so the gate rejected it
    before `split_segments`, the function written to handle stacks, ever ran.
    Per segment the same rectangle is about 3.9, comfortably inside.
    """
    import numpy as np

    import gdt_detect

    from gdt_config import get_config

    cfg = get_config()
    # A squat stack: three segments, wider than one segment is tall but
    # barely wider than the stack is tall.
    canvas = np.full((60, 80), 255, np.uint8)
    import cv2

    cv2.rectangle(canvas, (4, 4), (74, 54), 0, 2)
    for y in (21, 38):
        cv2.line(canvas, (4, y), (74, y), 0, 2)
    binary = gdt_detect.binarize(gdt_detect.to_gray(Image.fromarray(canvas).convert("RGB")))

    segments = fcf._count_segments(binary[4:55, 4:75])
    assert segments == 3, f"counted {segments} segments, expected 3"

    raw_aspect = 70 / 50
    assert raw_aspect < cfg.frame_min_aspect, "fixture is not squat enough"
    assert raw_aspect * segments >= cfg.frame_min_aspect, (
        "per-segment aspect must clear the floor the stack cannot"
    )


def test_a_single_frame_counts_as_one_segment():
    """The per-segment rule must not inflate an ordinary frame's aspect."""
    import gdt_detect

    scene = syn.build("frames")
    x, y, w, h = fcf.find_frames(scene.image)[0]
    crop = scene.image.crop((int(x), int(y), int(x + w), int(y + h)))
    binary = gdt_detect.binarize(gdt_detect.to_gray(crop))
    assert fcf._count_segments(binary) == 1


def test_segment_counting_agrees_with_segment_splitting():
    """Both answers come from one helper so they cannot disagree."""
    import numpy as np

    import gdt_detect

    scene = syn.build("composite_frame")
    x, y, w, h = fcf.find_frames(scene.image)[0]
    crop = scene.image.crop((int(x), int(y), int(x + w), int(y + h)))
    binary = gdt_detect.binarize(gdt_detect.to_gray(crop))
    assert fcf._count_segments(binary) == len(fcf.split_segments(crop))


# --- finding frames on a real sheet ------------------------------------------
#
# Measured on a scanned drawing with 13 feature control frames, the contour
# finder this replaced found 2 at 2x and 0 at 4x. These pin the three things
# that lost the rest: a frame tied to the part by a leader, a frame next to a
# large outline, and tolerance digits resting on the bottom rule.


def _frame_canvas(width: int = 900, height: int = 500):
    import numpy as np

    canvas = np.full((height, width), 255, np.uint8)
    frame, _ = syn.draw_frame(
        canvas, 400, 200, 44, symbol_id="position", tolerance=0.3,
        diameter_zone=True, datums=("A",),
    )
    return canvas, frame


def _found(boxes, frame) -> bool:
    fx, fy, fw, fh = frame.bbox
    cx, cy = fx + fw / 2, fy + fh / 2
    return any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in boxes)


def test_a_frame_joined_to_the_part_by_a_leader_is_still_found():
    """The leader's horizontal run is a rule to morphology, so the outer
    contour of the frame was the outer contour of the whole part."""
    import cv2

    canvas, frame = _frame_canvas()
    fx, fy, fw, fh = frame.bbox
    # A leader from the frame's left side to a large part outline.
    cv2.line(canvas, (int(fx), int(fy + fh / 2)), (60, int(fy + fh / 2)), 0, 2)
    cv2.rectangle(canvas, (40, 40), (320, 460), 0, 3)
    cv2.line(canvas, (40, 250), (320, 250), 0, 3)
    image = Image.fromarray(canvas).convert("RGB")
    assert _found(fcf.find_frames(image), frame)


def test_a_frame_beside_a_large_outline_survives_page_despeckling():
    """2% of the largest component is bigger than a frame on a whole page."""
    import cv2

    canvas, frame = _frame_canvas(width=1400, height=1000)
    cv2.rectangle(canvas, (20, 20), (1380, 980), 0, 6)
    cv2.rectangle(canvas, (60, 60), (1340, 940), 0, 6)
    image = Image.fromarray(canvas).convert("RGB")
    assert _found(fcf.find_frames(image), frame)


def test_tolerance_digits_resting_on_the_bottom_rule_do_not_lose_the_frame():
    import cv2

    canvas, frame = _frame_canvas()
    fx, fy, fw, fh = frame.bbox
    # Heavy strokes dropping from the tolerance text onto the bottom rule.
    for x in (int(fx) + 80, int(fx) + 96, int(fx) + 112):
        cv2.line(canvas, (x, int(fy) + 12), (x, int(fy + fh) - 1), 0, 4)
    image = Image.fromarray(canvas).convert("RGB")
    assert _found(fcf.find_frames(image), frame)


def test_a_box_formed_inside_a_frame_by_its_contents_is_not_a_frame():
    """A frame does not contain a frame."""
    scene = syn.build("composite_frame")
    frames = fcf.parse_frames(scene.image, ocr_fn=stub_ocr(default="0.5"))
    assert len(frames) == 2, [(f.symbol.symbol_id, f.bbox) for f in frames]
    assert {f.composite_id for f in frames} == {frames[0].composite_id}
    assert all(f.bbox[3] > 30 for f in frames)


def test_a_box_with_no_characteristic_costs_one_ocr_call():
    """The finder over-produces by design, so the parser must turn a boxed
    basic dimension away after reading its first compartment, not all."""
    import cv2
    import numpy as np

    canvas = np.full((80, 400), 255, np.uint8)
    cv2.rectangle(canvas, (20, 20), (360, 64), 0, 2)
    cv2.line(canvas, (60, 20), (60, 64), 0, 2)
    cv2.line(canvas, (200, 20), (200, 64), 0, 2)
    for x in (28, 80, 230):
        cv2.putText(canvas, "50", (x, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    image = Image.fromarray(canvas).convert("RGB")
    ocr = stub_ocr(default="50")
    frame = fcf.parse_frame(
        image.crop((20, 20, 362, 66)), bbox=(20, 20, 342, 46), ocr_fn=ocr
    )
    assert frame.symbol is None or frame.symbol.symbol_id is None
    assert len(ocr.calls) == 1
