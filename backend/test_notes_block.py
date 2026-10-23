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

from ocr_pipeline import OcrPipeline, _centre_inside


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


# ── Markers with no "." or ")" separator ──────────────────────────────────
# ballwoon drawing.pdf numbers its notes with a bare digit and a space, and
# prints the same list twice, once in Japanese and once in English. Both
# blocks were missed entirely, and every line of both came back as its own
# balloon.

UNPUNCTUATED_NOTES = [
    _b("NOTES", 628, 500, 50, 16),
    _b("1 UNSPECIFIED BENDING RADIUS OF PIPE IS", 645, 519, 420, 18),
    _b("R15 ON A CENTER LINE.", 659, 536, 220, 18),
    _b("2 BOTH SIDES OF PIPE END SHALL BE DOUBLE", 637, 554, 430, 18),
    _b("FLARED AFTER INSERTED FLARE NUT.", 659, 575, 330, 18),
    _b("3 DETAILED SHAPE OF THE FLARE NUT", 639, 595, 350, 18),
    _b("4 MUST BE FREE FROM BURRS AND SHARP EDGES.", 639, 615, 440, 18),
]

# The CJK block: each marker is its own box, with the note's text beside it.
DETACHED_MARKER_NOTES = [
    _b("注記", 639, 88, 40, 18),
    _b("1", 662, 116, 8, 18),
    _b("指示無き曲げRは全て2軸を含む平面内における", 698, 116, 430, 18),
    _b("2", 662, 173, 8, 18),
    _b("PIPE両側はフレアナット挿入後ダブルフレアのこと", 698, 168, 430, 18),
    _b("3", 662, 198, 8, 18),
    _b("フレアナット及びダブルフレアの詳細形状は", 697, 196, 400, 18),
    _b("4", 661, 251, 8, 18),
    _b("有害なバリ及びシャープエッジ等無きこと", 698, 249, 390, 18),
]


def test_notes_numbered_without_a_separator_are_recognised():
    region, _ = OcrPipeline._detect_notes_block(UNPUNCTUATED_NOTES)
    assert region is not None
    assert "UNSPECIFIED BENDING RADIUS" in region["text"]
    assert "MUST BE FREE FROM BURRS" in region["text"]


def test_a_count_callout_is_not_an_unpunctuated_note_marker():
    # "2 PLACES" and "4 HOLES" are callouts. Two of them must not read as a
    # numbered list just because a digit is followed by a word.
    callouts = [
        _b("2 PLACES", 60, 40, 90, 20),
        _b("4 HOLES", 60, 90, 80, 20),
    ]
    assert OcrPipeline._detect_notes_block(callouts)[0] is None


def test_a_marker_detected_apart_from_its_text_still_opens_a_note():
    region, _ = OcrPipeline._detect_notes_block(DETACHED_MARKER_NOTES)
    assert region is not None
    assert "指示無き曲げ" in region["text"]
    # The block spans the text column, not just the narrow marker column.
    assert region["bbox"]["width"] > 300


def test_both_language_blocks_are_reported_separately():
    blocks = OcrPipeline._detect_notes_blocks(
        DETACHED_MARKER_NOTES + UNPUNCTUATED_NOTES
    )
    assert len(blocks) == 2
    texts = [region["text"] for region, _ in blocks]
    assert any("指示無き曲げ" in text for text in texts)
    assert any("UNSPECIFIED BENDING RADIUS" in text for text in texts)


def test_a_misread_marker_does_not_reject_the_whole_block():
    # PaddleOCR reads the "6" of note 6 as "0" on this sheet. The list must
    # survive it; the growth pass picks the line up as ordinary prose.
    boxes = list(UNPUNCTUATED_NOTES) + [
        _b("0 THE RESTRICTED SUBSTANCES SHALL BE", 639, 635, 400, 18),
    ]
    region, _ = OcrPipeline._detect_notes_block(boxes)
    assert region is not None
    assert "UNSPECIFIED BENDING RADIUS" in region["text"]
    assert "RESTRICTED SUBSTANCES" in region["text"]


def test_numbers_scattered_across_the_drawing_are_not_a_numbered_list():
    # Page-scan candidates are atomic, so every lone dimension value is its
    # own box and pairs with whatever word is printed near it. Across a sheet
    # that reads as "1 RUBBER / 2 SLIT AREA / 3 DETAIL AT …" and once
    # assembled into one region it was a balloon 1730 pixels wide.
    scattered = [
        _b("2", 300, 400, 12, 18),   _b("RUBBER", 340, 398, 70, 18),
        _b("3", 900, 410, 12, 18),   _b("SLIT AREA", 940, 408, 90, 18),
        _b("4", 1500, 420, 12, 18),  _b("DETAIL AT", 1540, 418, 95, 18),
        _b("5", 1900, 430, 12, 18),  _b("DELIVERY", 1940, 428, 90, 18),
    ]

    assert OcrPipeline._detect_notes_blocks(scattered) == []


def test_a_real_block_keeps_its_markers_in_one_column():
    region, _ = OcrPipeline._detect_notes_block(DETACHED_MARKER_NOTES)
    assert region is not None
    assert "指示無き曲げ" in region["text"]


# ── What a notes block suppresses ─────────────────────────────────────────


def _box(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


def test_a_fragment_of_the_paragraph_is_covered_by_its_block():
    block = _box(2198, 417, 520, 301)
    assert _centre_inside(_box(2373, 600, 67, 22), block)


def test_a_line_poking_past_the_block_edge_is_still_covered():
    # Boxes routinely overhang the paragraph the block assembled by a pixel
    # or two, which is why the test is on the centre and not on overlap.
    block = _box(2198, 417, 520, 301)
    assert _centre_inside(_box(2650, 700, 80, 22), block)


def test_a_dimension_outside_the_block_is_not_covered():
    block = _box(2198, 417, 520, 301)
    assert not _centre_inside(_box(1317, 962, 25, 67), block)
    assert not _centre_inside(_box(2700, 417, 80, 22), block)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed")


# ── Notes in the bottom-left corner, beside the title block ───────────────
# LP58.pdf and TS2: the notes sit at the left of the bottom border and the
# title block sits to their RIGHT, at the same height. The title-block floor
# used to take any title-block word below the first markers, anywhere across
# the sheet, and cut the list off at its height.

BOTTOM_LEFT_NOTES = [
    _b("NOTES:", 237, 1485, 167, 40),
    _b("1. MATERIAL: PER ANSI C135.1", 225, 1554, 800, 40),
    _b("2. FINISH: HOT DIP GALVANIZED PER ASTM A-153", 222, 1600, 1200, 40),
    _b("3. DIMENSION L PER PURCHASE ORDER SHALL BE MEASURED", 221, 1650, 1300, 40),
    _b("OF THE HEAD TO THE LAST THREAD", 287, 1700, 885, 40),
    # The title block beside them, starting above the last two notes.
    _b("TOLERANCES", 1455, 1730, 178, 28),
    _b("UNLESS SPECIFIED", 1411, 1760, 261, 30),
    _b("DWG NO.", 1709, 1730, 69, 22),
    _b("4. SPECIAL POINT SHALL BE MEASURED FROM", 218, 1750, 1000, 40),
    _b("THE LAST THREAD TO THE END AND IS", 287, 1800, 957, 40),
    _b("CONSIDERED A CRITICAL FEATURE.", 285, 1850, 835, 40),
]


def test_notes_beside_the_title_block_keep_every_point():
    region, _ = OcrPipeline._detect_notes_block(BOTTOM_LEFT_NOTES)
    assert region is not None
    lines = region["text"].splitlines()
    assert lines[-1] == "CONSIDERED A CRITICAL FEATURE."
    assert any(line.startswith("4. SPECIAL POINT") for line in lines)
    assert "TOLERANCES" not in region["text"]


def test_a_comma_read_for_the_marker_dot_still_opens_a_point():
    # A heavy scanned "3." is read as "3," often enough to lose a list.
    boxes = [
        _b("1, MATERIAL: PER ANSI C135.1", 60, 100, 500, 30),
        _b("2, FINISH: HOT DIP GALVANIZED", 60, 140, 500, 30),
    ]
    assert OcrPipeline._detect_notes_block(boxes)[0] is not None
    # A thousands figure is not a marker.
    assert OcrPipeline._detect_notes_block([_b("1,326 [33.68]", 60, 100, 200, 30)] * 2)[0] is None


def test_the_block_box_covers_a_wrapped_line_wider_than_its_markers():
    region, _ = OcrPipeline._detect_notes_block(NOTES_ABOVE_TITLE_BLOCK[:6])
    right = region["bbox"]["x"] + region["bbox"]["width"]
    # "TENSILE MIN." runs to x=1138, past every numbered line.
    assert right >= 1138


# ── The whole notes pass, on detector boxes ───────────────────────────────

from PIL import Image, ImageDraw  # noqa: E402


def _notes_from(boxes, size=(1600, 1200), image=None):
    """Run _notes_regions with the page detector replaced by fixed boxes."""
    pipeline = OcrPipeline.__new__(OcrPipeline)
    page = image or Image.new("RGB", size, "white")

    def detect(img):
        # The page gets the fixed boxes; crop re-reads find nothing, so the
        # page reading is what is published.
        return [dict(b) for b in boxes] if img.size == page.size else []

    pipeline._detect_regions_ocr = detect
    return pipeline._notes_regions(page)


def test_a_single_note_in_the_bottom_left_corner_is_published():
    # TS2: "NOTE:" over one numbered point, against the bottom border. The
    # numbered-list detector needs two points, so it was never found.
    regions = _notes_from(
        [_b("NOTE:", 45, 1100, 90, 37, 1.0), _b("1. MATERIAL MUST BE FREE FROM CRACK.", 46, 1137, 632, 37, 1.0)]
    )
    assert [r["text"] for r in regions] == ["1. MATERIAL MUST BE FREE FROM CRACK."]
    region = regions[0]
    assert region["type"] == "General Note" and region["heading"] == "NOTE:"
    assert region["bbox"]["x"] <= 46 and region["bbox"]["y"] + region["bbox"]["height"] >= 1170


def test_an_unnumbered_note_is_published_as_one_region():
    regions = _notes_from(
        [
            _b("THIS DRAWING IS CUSTOMIZED TO PROCURE FROM INDIA", 400, 600, 900, 40, 0.99),
            _b("* MARKS ARE DIFFERENT FROM ORIGINAL DRAWING", 400, 646, 880, 40, 0.99),
        ]
    )
    assert len(regions) == 1
    assert regions[0]["text"].splitlines() == [
        "THIS DRAWING IS CUSTOMIZED TO PROCURE FROM INDIA",
        "* MARKS ARE DIFFERENT FROM ORIGINAL DRAWING",
    ]
    assert [line["text"] for line in regions[0]["lines"]] == regions[0]["text"].splitlines()


def test_a_ruled_table_is_not_published_as_a_note():
    image = Image.new("RGB", (1600, 1200), "white")
    draw = ImageDraw.Draw(image)
    for y in (295, 325, 355, 385):
        draw.line([(80, y), (700, y)], fill="black", width=2)
    for x in (80, 400, 700):
        draw.line([(x, 295), (x, 385)], fill="black", width=2)
    rows = []
    for i, (label, value) in enumerate([("MODULE OF THE GEAR", "1.75"), ("NUMBER OF TEETH", "18"), ("PRESSURE ANGLE", "20")]):
        rows += [_b(label, 100, 300 + 30 * i, 280, 22, 1.0), _b(value, 420, 300 + 30 * i, 60, 22, 1.0)]
    assert _notes_from(rows, image=image) == []


def test_vertical_text_is_read_upright_and_mapped_back_onto_the_page():
    # A margin note printed up the left edge. The page pass reads it as tall
    # scraps; the turned read must land back on the same ink.
    image = Image.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 300, 50, 700], fill="black")  # the "text" column
    scraps = [_b("oughess", 28, 300, 24, 120, 0.8), _b("imens", 28, 440, 24, 110, 0.8), _b("va", 28, 580, 24, 110, 0.8)]

    pipeline = OcrPipeline.__new__(OcrPipeline)

    def detect(img):
        import numpy as np

        ys, xs = np.nonzero(np.asarray(img.convert("L")) < 128)
        if not len(xs):
            return []
        return [{"x": float(xs.min()), "y": float(ys.min()), "w": float(xs.max() - xs.min() + 1),
                 "h": float(ys.max() - ys.min() + 1), "text": "DO NOT SCALE THE DRAWING", "conf": 0.99}]

    pipeline._detect_regions_ocr = detect
    turned, replaced = pipeline._rotated_text_boxes(image, scraps)
    assert replaced == {0, 1, 2}
    (box,) = turned
    assert box["angle"] in (90.0, 270.0)
    assert abs(box["x"] - 30) <= 2 and abs(box["y"] - 300) <= 2
    assert abs(box["x"] + box["w"] - 51) <= 2 and abs(box["y"] + box["h"] - 701) <= 2
