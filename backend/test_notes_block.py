"""
Tests for OcrPipeline._detect_notes_block and _lines_from_boxes.

The box geometry is taken from real PaddleOCR output for two drawings:

  47630.pdf    the title block starts TEN pixels under the last note line, and
               its numbered markers are nearly twice the height of the line
               pitch. Both of those broke earlier versions of this code.
  GEP5A34.pdf  one printed note line arrives as three boxes whose y values
               differ by a few pixels ("2" / "HOT" / "DIP GALVANIZE...").

    python test_notes_block.py
"""

from __future__ import annotations

from ocr_pipeline import OcrPipeline


def _b(text, x, y, w, h, conf=0.9):
    return {"x": x, "y": y, "w": w, "h": h, "text": text, "conf": conf}


# 47630.pdf: notes end at y=768, the title block begins at y=778.
NOTES_ABOVE_TITLE_BLOCK = [
    _b("NOTES:", 581, 666, 58, 21),
    _b("1. MATERIAL: GRADE 5 EQUIV; 85KPSI YIELD MIN.", 579, 682, 457, 25),
    _b("TENSILE MIN.", 1035, 683, 103, 24),
    _b('2. PIN TO BE STRAIGHT WITHIN .O04"', 576, 696, 309, 29),
    _b("OF ENTIRE LENGTH", 875, 700, 160, 20),
    _b("5. LANYARD RING MUST WITHSTAND 50 LBS MIN.", 581, 748, 539, 20),
    # The title block, ten pixels below the last note.
    _b("TOLERANCES", 630, 778, 76, 14),
    _b("DWG NO.", 736, 778, 33, 11),
    _b("47630", 766, 779, 61, 25),
    _b("DATE: 9-14-09", 640, 800, 90, 20),
    _b("DRAWN: TZ", 740, 800, 60, 20),
    _b("SCALE: N.T.S.", 640, 820, 80, 20),
    _b("SHEET 1 OF 1", 840, 820, 80, 20),
    _b("CHECKED BY:", 640, 840, 90, 20),
    _b("APPROVED BY:", 640, 860, 95, 20),
    _b("UNITS INCHES[MILLIMETERS]", 840, 860, 140, 20),
]

TITLE_BLOCK_WORDS = (
    "TOLERANCES", "DWG", "DATE", "DRAWN", "SCALE",
    "SHEET", "CHECKED", "APPROVED", "UNITS", "47630",
)


def test_notes_do_not_swallow_the_title_block():
    # The whole title block used to be appended to the notes paragraph,
    # ten pixels being far inside any distance-based bound.
    region, _ = OcrPipeline._detect_notes_block(NOTES_ABOVE_TITLE_BLOCK)
    assert region is not None
    leaked = [w for w in TITLE_BLOCK_WORDS if w in region["text"]]
    assert leaked == [], f"title block leaked into the notes: {leaked}"


def test_each_note_stays_on_its_own_line():
    region, _ = OcrPipeline._detect_notes_block(NOTES_ABOVE_TITLE_BLOCK)
    lines = region["text"].splitlines()
    assert len(lines) == 3, lines
    assert lines[0].startswith("1. MATERIAL")
    assert lines[1].startswith("2. PIN TO BE STRAIGHT")
    assert lines[2].startswith("5. LANYARD RING")


def test_wrapped_tail_rejoins_its_own_note():
    region, _ = OcrPipeline._detect_notes_block(NOTES_ABOVE_TITLE_BLOCK)
    lines = region["text"].splitlines()
    assert lines[0].endswith("TENSILE MIN.")
    assert lines[1].endswith("OF ENTIRE LENGTH")


def test_fragments_of_one_printed_line_are_rejoined():
    # "2" / "HOT" / "DIP GALVANIZE..." are one line whose boxes differ in y.
    frags = [
        _b("2", 43, 326, 12, 22),
        _b("HOT", 84, 328, 40, 22),
        _b("DIP GALVANIZE PER ASTM A153", 158, 319, 400, 22),
    ]
    assert OcrPipeline._lines_from_boxes(frags, 12.0) == [
        "2. HOT DIP GALVANIZE PER ASTM A153"
    ]


def test_separate_printed_lines_are_not_fused():
    # Real spacing: boxes about as tall as the gap between baselines. (Boxes
    # taller than the pitch would mean printed lines overlapping each other,
    # which is why this uses h=12 on a 16px pitch.)
    lines = [
        _b("2. HOT DIP GALVANIZE", 60, 300, 200, 12),
        _b("PER ASTM A153", 60, 316, 150, 12),
        _b("3. ALL DIMENSIONS", 60, 332, 180, 12),
    ]
    assert len(OcrPipeline._lines_from_boxes(lines, 12.0)) == 3


def test_a_dimension_under_the_notes_is_not_absorbed():
    boxes = NOTES_ABOVE_TITLE_BLOCK[:6] + [_b("3.20[81.28]", 600, 772, 100, 20)]
    region, _ = OcrPipeline._detect_notes_block(boxes)
    assert "3.20" not in region["text"]


def test_numbered_table_rows_are_not_a_notes_block():
    rows = [_b("4. REV D", 60, 40, 200, 20), _b("9. REV E", 60, 70, 200, 20)]
    assert OcrPipeline._detect_notes_block(rows)[0] is None


def test_a_single_point_is_not_a_notes_block():
    one = [_b("1. SOMETHING HERE", 60, 40, 200, 20)]
    assert OcrPipeline._detect_notes_block(one)[0] is None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed")
