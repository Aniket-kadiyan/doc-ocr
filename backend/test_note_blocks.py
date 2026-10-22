"""
Tests for note_blocks: rebuilding notes from loose text fragments.

The fragments are shaped like what the two readers hand over: native PDF
runs (one per word, exact boxes, source "pdf-text") and OCR detection boxes
(one per line segment, slightly padded). Geometry is in page pixels.

    python -m pytest -q test_note_blocks.py
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from note_blocks import (
    Fragment,
    InkMap,
    build_blocks,
    build_lines,
    normalized_note_text,
    note_blocks,
)


def ocr(text, x, y, w, h=20, conf=0.98):
    return Fragment(text, x, y, w, h, source="ocr", confidence=conf)


def words(line, x, y, size=12.0, char_w=7.0, space=4.0):
    """One printed line as native word runs, the way pdfium splits them."""
    out = []
    for word in line.split():
        width = char_w * len(word)
        out.append(Fragment(word, x, y, width, size * 0.75, size=size, source="pdf-text"))
        x += width + space
    return out


def notes(blocks):
    return [b for b in blocks if b.category == "note"]


# ── continuous and multiline notes ────────────────────────────────────────


def test_a_numbered_note_with_wrapped_lines_is_one_block_in_order():
    # The task's own example: one note over three printed lines.
    frags = (
        words("1. REMOVE ALL BURRS AND SHARP EDGES.", 100, 500)
        + words("BREAK ALL EDGES 0.2 MAX.", 118, 516)
        + words("UNLESS OTHERWISE SPECIFIED.", 118, 532)
    )
    blocks = notes(note_blocks(frags))
    assert len(blocks) == 1
    assert blocks[0].text.splitlines() == [
        "1. REMOVE ALL BURRS AND SHARP EDGES.",
        "BREAK ALL EDGES 0.2 MAX.",
        "UNLESS OTHERWISE SPECIFIED.",
    ]


def test_native_words_are_joined_with_their_spaces_and_glued_runs_without():
    # "70" written as two runs a hair apart is one number; words a space
    # apart are words.
    frags = words("PART TO BE DRY", 50, 50) + [
        Fragment("7", 143, 50, 7, 9, size=12, source="pdf-text"),
        Fragment("0", 150.5, 50, 7, 9, size=12, source="pdf-text"),
    ]
    (line,) = [l for l in build_lines(frags) if "PART" in l.text]
    assert line.text == "PART TO BE DRY 70"


def test_a_long_sentence_across_the_sheet_stays_one_line():
    text = "DIMENSIONS IN ( ) ARE FOR REFERENCE ONLY AND DO NOT NEED TO BE MEASURED"
    blocks = notes(note_blocks(words(text, 900, 60, size=18, char_w=10, space=6)))
    assert [b.text for b in blocks] == [text]


def test_a_headed_list_keeps_the_heading_out_of_its_text():
    frags = [
        ocr("NOTES:", 60, 100, 70),
        ocr("1. MATERIAL MUST BE FREE FROM CRACK.", 60, 126, 420),
    ]
    (block,) = notes(note_blocks(frags))
    assert block.heading is not None and block.heading.text == "NOTES:"
    assert block.text == "1. MATERIAL MUST BE FREE FROM CRACK."


def test_a_single_unheaded_numbered_point_is_still_a_note():
    (block,) = notes(note_blocks([ocr("1. MATERIAL MUST BE FREE FROM CRACK.", 60, 126, 420)]))
    assert block.reason == "numbered note"


def test_an_unnumbered_paragraph_is_a_note():
    frags = [
        ocr("ALL DRAWINGS OF CAST /", 1490, 880, 200),
        ocr("MACHINED COMPONENTS TO BE", 1492, 900, 230),
        ocr("SUPPLIED WITH 3D MODEL", 1492, 920, 200),
        ocr("FOR IDENTIFICATION OF ALL", 1493, 940, 228),
        ocr("PROFILE DIMENSIONS.", 1493, 960, 170),
    ]
    (block,) = notes(note_blocks(frags))
    assert len(block.lines) == 5 and block.reason == "prose paragraph"


# ── what must NOT be joined ───────────────────────────────────────────────


def test_two_notes_side_by_side_stay_two_blocks():
    left = [ocr("ALL DIMENSIONS ARE IN MM", 20, 100, 300), ocr("DO NOT SCALE THIS DRAWING", 20, 122, 300)]
    right = [ocr("BREAK ALL SHARP EDGES AND", 700, 100, 300), ocr("REMOVE BURRS FROM EVERY EDGE", 700, 122, 330)]
    blocks = notes(note_blocks(left + right))
    assert sorted(b.text for b in blocks) == [
        "ALL DIMENSIONS ARE IN MM\nDO NOT SCALE THIS DRAWING",
        "BREAK ALL SHARP EDGES AND\nREMOVE BURRS FROM EVERY EDGE",
    ]


def test_a_wide_gap_on_one_baseline_is_a_column_break():
    frags = [ocr("ALL DIMENSIONS ARE IN MM", 20, 100, 300), ocr("SECTION A-A", 600, 100, 120)]
    texts = sorted(l.text for l in build_lines(frags))
    assert texts == ["ALL DIMENSIONS ARE IN MM", "SECTION A-A"]


def test_rows_a_box_height_apart_do_not_chain_into_one_line():
    # Table rows whose padded boxes overlap their neighbours used to band
    # into one "line" ("MODULE NO.OF SHIFT FACTOR TEETH").
    rows = [ocr(t, 150, 400 + 22 * i, 120, h=26) for i, t in enumerate(["MODULE", "NO. OF TEETH", "SHIFT FACTOR"])]
    assert len(build_lines(rows)) == 3


def test_a_callout_above_a_note_is_not_swallowed_by_it():
    frags = [ocr("R0.2 +0.3", 552, 880, 90), ocr("All around the teeth profile", 552, 904, 180)]
    blocks = note_blocks(frags)
    (note,) = notes(blocks)
    assert note.text == "All around the teeth profile"
    assert any(b.text == "R0.2 +0.3" and b.category != "note" for b in blocks)


def test_a_value_finishing_an_unfinished_sentence_joins_it():
    frags = [ocr("BREAK ALL SHARP EDGES TO A RADIUS OF", 50, 50, 400), ocr("0.2 MAX", 50, 72, 70)]
    (block,) = notes(note_blocks(frags))
    assert block.text.endswith("0.2 MAX")


def test_repeated_notes_at_different_places_are_both_kept():
    text = "REMOVE ALL BURRS AND SHARP EDGES"
    blocks = notes(note_blocks([ocr(text, 50, 100, 360), ocr(text, 50, 900, 360)]))
    assert [b.text for b in blocks] == [text, text]


# ── classification ────────────────────────────────────────────────────────


def test_labels_and_title_block_text_are_kept_but_not_notes():
    blocks = note_blocks(
        [
            ocr("SECTION A-A", 50, 50, 120),
            ocr("Designed by", 600, 900, 100),
            ocr("THE INFORMATION CONTAINED IN THIS DRAWING IS THE SOLE PROPERTY OF ACME", 900, 1000, 700),
            ocr("PERFORM cold forging B.V. Drawing Name : ICE Connection Shaft", 900, 1100, 600),
        ]
    )
    assert notes(blocks) == []
    assert {b.category for b in blocks} == {"label", "title_block"}


def test_table_rows_with_value_cells_are_a_table():
    rows = []
    for i, (label, value) in enumerate([("NUMBER OF TEETH", "34"), ("PITCH CIRCLE DIA", "34"), ("BASE DIAMETER VALUE", "29.445")]):
        rows += [ocr(label, 100, 300 + 24 * i, 200), ocr(value, 420, 300 + 24 * i, 40)]
    assert notes(note_blocks(rows)) == []


def test_a_sentence_in_a_table_cell_is_still_a_note():
    rows = [
        ocr("ADDENDUM SHALL NOT RISE SHARPLY", 100, 300, 300),
        ocr("REMARK", 20, 300, 60),
        ocr("ALL VALUES ARE GIVEN IN MICRONS", 100, 324, 300),
        ocr("NOTE", 20, 324, 50),
    ]
    assert len(notes(note_blocks(rows))) == 1


def test_a_revision_entry_is_not_a_note():
    rows = [ocr("2", 20, 300, 14), ocr("HRC MODIFIED AS PER ECN18980", 60, 300, 300), ocr("15.07.2024", 420, 300, 90)]
    blocks = note_blocks(rows)
    assert notes(blocks) == []
    assert any(b.category == "revision" for b in blocks)


def test_misread_scraps_are_not_a_note():
    assert notes(note_blocks([ocr("drain ha esi t 7 C a n S m lues va B", 18, 444, 400)])) == []


# ── rules on the page ─────────────────────────────────────────────────────


def _ruled_page(vertical=None, horizontal=None, underline=None):
    image = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(image)
    if vertical is not None:
        draw.line([(vertical, 0), (vertical, 400)], fill="black", width=2)
    if horizontal is not None:
        # A cell border: the rule runs into the table's side borders.
        draw.line([(40, horizontal), (760, horizontal)], fill="black", width=2)
        draw.line([(40, 60), (40, 200)], fill="black", width=2)
        draw.line([(760, 60), (760, 200)], fill="black", width=2)
    if underline is not None:
        x0, x1, y = underline
        draw.line([(x0, y), (x1, y)], fill="black", width=2)
    return InkMap(image)


def test_a_vertical_rule_between_two_fragments_splits_them_into_cells():
    frags = [ocr("ALL DIMENSIONS ARE", 20, 100, 200), ocr("TOLERANCES UNLESS", 240, 100, 200)]
    assert len(build_lines(frags)) == 1
    assert len(build_lines(frags, _ruled_page(vertical=230))) == 2


def test_a_cell_border_between_two_lines_keeps_them_apart():
    frags = [ocr("RELEASED FOR PRODUCTION NOW", 100, 100, 300), ocr("ADDED SC DIMS AND MORE TEXT", 100, 126, 300)]
    assert len(build_blocks(build_lines(frags))) == 1
    ink = _ruled_page(horizontal=123)
    assert len(build_blocks(build_lines(frags, ink), ink)) == 2


def test_an_underline_does_not_split_a_paragraph():
    # Underlined note lines (a Japanese notes block) start the underline
    # where the text starts; a table rule reaches past it on both sides.
    frags = [ocr("FIRST LINE OF THE NOTE TEXT", 100, 100, 300), ocr("SECOND LINE OF THE SAME NOTE", 100, 126, 300)]
    ink = _ruled_page(underline=(100, 700, 123))
    assert len(build_blocks(build_lines(frags, ink), ink)) == 1


# ── rotation and normalisation ────────────────────────────────────────────


def test_vertical_text_groups_in_its_own_reading_frame():
    # A margin note printed up the left edge: each line is a tall box, the
    # lines stacked left to right, reading bottom to top (angle 270, as
    # pdfium measures it: clockwise).
    lines = ["DO NOT SCALE THE DRAWING", "REMOVE BURRS AND SHARP CORNERS", "DIMENSIONS IN MM"]
    frags = [
        Fragment(t, 20 + 22 * i, 700 - 12 * len(t), 18, 12 * len(t), angle=270.0, size=18, source="ocr")
        for i, t in enumerate(lines)
    ]
    (block,) = notes(note_blocks(frags))
    assert block.angle == 270
    # Reading order in the levelled frame: the line nearest the left edge of
    # the page is the top line of the note.
    assert block.text.splitlines()[0] == "DO NOT SCALE THE DRAWING"


def test_markers_are_written_one_way_inside_a_numbered_list():
    frags = (
        [ocr("1. CLEANLINESS REQUIREMENT TO BE FREE FROM DIRT", 60, 100, 420)]
        + [ocr("2 PART TO BE DRY", 60, 124, 160)]
        + [ocr("3, MATERIAL STAINLESS STEEL", 60, 148, 300)]
    )
    (block,) = notes(note_blocks(frags))
    assert normalized_note_text(block).splitlines() == [
        "1. CLEANLINESS REQUIREMENT TO BE FREE FROM DIRT",
        "2. PART TO BE DRY",
        "3. MATERIAL STAINLESS STEEL",
    ]
    # The raw lines are untouched.
    assert block.text.splitlines()[1] == "2 PART TO BE DRY"


def test_undecodable_native_glyphs_are_flagged():
    assert Fragment("MAT�RIAL", 0, 0, 10, 10, source="pdf-text").suspect
    assert not Fragment("MATERIAL", 0, 0, 10, 10, source="pdf-text").suspect
