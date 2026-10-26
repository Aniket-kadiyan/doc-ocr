"""
Tests for reading a drawing's callouts from the PDF's own text layer.

The geometry cases use hand-built runs so they need no PDF; the end-to-end
case uses BS1801006.020, the one sample drawing that carries a text layer,
and skips when that file is not present.

    python -m pytest -q test_pdf_text_layer.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pdf_text_layer import (
    TextRun,
    rescue_geometry_marks,
    _decode_symbols,
    _join_runs,
    _title_line_boxes,
    group_runs,
    has_usable_text_layer,
    native_title_fields,
    page_text_runs,
    text_layer_regions,
)

DOCS = Path("/Users/chitrajain/Desktop/doc-ocr-box/Docs")
DRAWING = DOCS / "BS1801006.020.pdf"
# A drawing whose Ø and ° are vector outlines, not text.
SYMBOLS_AS_GEOMETRY = DOCS / "3015-100-2232F.pdf"
# A sheet of 90 callouts, carrying every way a value went missing: a label
# banded with the deviation of the callout beside it, a caption stacked under
# a value, a lone signed deviation, and a diameter written in a text font.
CROWDED = DOCS / "503645_DES001_CF BALLOON DRAWING.pdf"
PAGE = (2065, 2923)


def run(text, x, y, width, height, angle=0.0, font="ArialMT", size=None):
    """A run and its levelled box, which is the page box when upright."""
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = x + width / 2, y + height / 2
    lw = abs(cos_t) * width + abs(sin_t) * height
    lh = abs(sin_t) * width + abs(cos_t) * height
    return TextRun(
        text=text, x=x, y=y, width=width, height=height, angle=angle, font=font,
        lx=(cx * cos_t + cy * sin_t) - lw / 2,
        ly=(-cx * sin_t + cy * cos_t) - lh / 2,
        lw=lw, lh=lh, size=size if size is not None else height,
    )


def test_symbol_glyphs_are_decoded_only_in_a_symbol_font():
    # AutoCAD writes Ø as the glyph "P" of its AIGDT font.
    assert _decode_symbols("P", "AIGDT") == "Ø"
    assert _decode_symbols("B", "AIGDT") == "±"
    # The same letters in a text font are letters: the part name is PUNCH.
    assert _decode_symbols("P", "ArialMT") == "P"
    assert _decode_symbols("B", "Tahoma") == "B"
    # A CAD encoding leaks degree in as a C1 control code.
    assert _decode_symbols("\x83", "ArialMT") == "°"


def test_a_number_split_across_runs_is_joined_without_a_space():
    # "70" arrives as "7" then "0"; a space would rewrite the value as 7 0.
    parts = [run("7", 100, 100, 16, 25), run("0", 118, 100, 16, 25),
             run("±0.2", 141, 103, 60, 20)]
    assert _join_runs(parts, 25.0) == "70 ±0.2"


def test_a_real_gap_keeps_its_space():
    parts = [run("SHARP", 100, 100, 120, 25), run("po", 300, 100, 40, 25)]
    assert _join_runs(parts, 25.0) == "SHARP po"


def test_runs_at_different_angles_are_never_grouped_together():
    # A vertical dimension crosses the horizontal ones; their axis-aligned
    # boxes overlap while the writing never touches.
    groups = group_runs(
        [run("15", 500, 500, 40, 25), run("Ø30", 505, 480, 60, 25, angle=270.0)],
        PAGE,
    )
    assert len(groups) == 2
    assert {g["angle"] for g in groups} == {0.0, 270.0}


def test_a_value_and_the_deviations_stacked_beside_it_are_one_callout():
    # BS1801006.020 writes Ø33 at 34pt with its two deviations at 27pt, the
    # upper one above the lower, and the value's box overlaps both rows.
    groups = group_runs(
        [
            run("Ø33", 1158, 238, 63, 36, size=34.4),
            run("-", 1233, 241, 7, 2, size=27.5),
            run("0.05", 1248, 232, 51, 20, size=27.5),
            run("-", 1233, 269, 7, 2, size=27.5),
            run("0.1", 1248, 260, 32, 20, size=27.5),
        ],
        PAGE,
    )
    assert len(groups) == 1
    assert groups[0]["text"] == "Ø33 - 0.05 - 0.1"


def test_brackets_stay_around_the_number_they_enclose():
    # A slanted "( 67.1° )": every run is the same size, so none of them is a
    # value the others qualify, and reading order is simply left to right.
    groups = group_runs(
        [
            run("67.1", 1538, 1117, 56, 71, angle=303.5, size=28.4),
            run("(", 1523, 1189, 25, 19, angle=303.5, size=28.4),
            run(")", 1595, 1081, 25, 19, angle=303.5, size=28.4),
        ],
        PAGE,
    )
    assert len(groups) == 1
    assert groups[0]["text"] == "( 67.1 )"


def test_two_callouts_a_line_apart_stay_apart():
    groups = group_runs(
        [run("R2 ±0.2", 100, 100, 107, 25), run("R10 ±2", 100, 400, 103, 25)],
        PAGE,
    )
    assert len(groups) == 2


def test_a_page_with_no_text_is_not_usable():
    assert not has_usable_text_layer([], set())
    # A "make searchable" tool hides its own OCR behind the image; that text
    # is worse than our pipeline's, so the page counts as a scan.
    assert not has_usable_text_layer(
        [run("25", 10, 10, 30, 12)], {"GlyphLessFont"}
    )
    assert has_usable_text_layer([run("25", 10, 10, 30, 12)], {"ArialMT"})


@pytest.mark.skipif(not DRAWING.exists(), reason="sample drawing not present")
def test_every_callout_on_a_real_vector_drawing_is_read_exactly():
    result = text_layer_regions(DRAWING)

    assert result["usable"] is True
    assert result["page_size"] == PAGE
    assert "AIGDT" in result["fonts"]

    by_text = {region["text"]: region for region in result["regions"]}
    # The twelve callouts this drawing carries, exactly as the benchmark
    # fixture lists them.
    assert {
        "60°±0°30'", "Ø33-0.05-0.1", "15-0.2", "R2±0.2", "70±0.2",
        "114.5±0.05", "R2±0.5", "R10±2", "Ø30±0.2", "5°±0°30'",
        "Ø23.5-0.05", "Ø30-0.2",
    } <= set(by_text)

    assert by_text["Ø33-0.05-0.1"]["type"] == "diameter"
    assert by_text["60°±0°30'"]["type"] == "angle"
    assert by_text["R10±2"]["category"] == "Radius"
    # Every dimension is exact, so none of them is sent for review. Only a
    # bare number with no mark saying what it measures is, and on this sheet
    # those are title-block codes, not callouts.
    dimensions = [
        "60°±0°30'", "Ø33-0.05-0.1", "15-0.2", "R2±0.2", "70±0.2",
        "114.5±0.05", "R2±0.5", "R10±2", "Ø30±0.2", "5°±0°30'",
        "Ø23.5-0.05", "Ø30-0.2",
    ]
    assert not any(by_text[value]["needs_review"] for value in dimensions)

    # A rotated callout carries the rectangle the balloon is drawn on.
    rotated = by_text["114.5±0.05"]
    assert rotated["orientation"] == "vertical"
    assert "oriented_box" in rotated
    assert abs(abs(rotated["oriented_box"]["rotation"]) - 90.0) < 2.0


@pytest.mark.skipif(not DRAWING.exists(), reason="sample drawing not present")
def test_extraction_reports_the_fonts_and_page_size():
    runs, size, fonts = page_text_runs(DRAWING)

    assert size == PAGE
    assert runs and all(r.text.strip() for r in runs)
    assert {"AIGDT", "ArialMT"} <= fonts
    # Three writing directions on one sheet.
    assert len({round(r.angle) for r in runs}) >= 3


def test_a_raster_drawing_with_a_few_vector_labels_falls_back_to_scanning():
    # The dimensions of a scanned drawing are in the picture, so a caption or
    # a stamp laid over it must not be mistaken for a readable text layer.
    labels = [run("DRAWN BY", 10, 10, 90, 12)]
    assert has_usable_text_layer(labels, {"ArialMT"}, 0.0)
    assert has_usable_text_layer(labels, {"ArialMT"}, 0.05)
    assert not has_usable_text_layer(labels, {"ArialMT"}, 0.91)


@pytest.mark.skipif(not DRAWING.exists(), reason="sample drawing not present")
def test_a_vector_drawings_only_image_is_a_logo():
    from pdf_text_layer import page_raster_coverage

    assert page_raster_coverage(DRAWING) < 0.1


@pytest.mark.skipif(
    not SYMBOLS_AS_GEOMETRY.exists(), reason="sample drawing not present"
)
def test_a_diameter_drawn_as_geometry_is_recovered_from_the_picture():
    # This drawing's text layer holds every digit and no symbol at all: its
    # Ø and ° are vector outlines. The digits are still exact, and the
    # diameter is read back off the drawing beside them.
    values = {
        region["text"]: region
        for region in text_layer_regions(SYMBOLS_AS_GEOMETRY)["regions"]
    }

    assert "Ø50±0.2" in values
    assert values["Ø50±0.2"]["type"] == "diameter"
    # Its deviations are a column of smaller text beside it, written the way
    # the OCR route writes one.
    assert "Ø32.3+0.2/0" in values
    assert "Ø8+0.2/0" in values
    # A plain length keeps its own reading; the recovery must not decorate
    # every value it cannot explain.
    assert "45.3" in values and "34.2" in values
    # Nor may it turn a radius or a surface finish into a diameter.
    assert "R1±0.25" in values
    assert "Rz12.5" in values


@pytest.mark.skipif(not DRAWING.exists(), reason="sample drawing not present")
def test_symbol_recovery_adds_nothing_when_the_text_already_has_it():
    values = [
        region["text"] for region in text_layer_regions(DRAWING)["regions"]
    ]

    assert "Ø33-0.05-0.1" in values
    # Every radius and angle on this sheet stays what it is.
    assert "R2±0.2" in values and "R10±2" in values
    assert "60°±0°30'" in values
    assert not any(value.startswith("ØR") for value in values)


@pytest.mark.skipif(
    not SYMBOLS_AS_GEOMETRY.exists(), reason="sample drawing not present"
)
def test_short_exact_values_are_callouts_not_fragments():
    # These were all dropped while the OCR route's fragment rule was applied
    # to exact text: a plain 7.3, a minimum, a bracketed reference, and the
    # value of a feature control frame.
    values = {
        region["text"]: region
        for region in text_layer_regions(SYMBOLS_AS_GEOMETRY)["regions"]
    }

    for value in ("7.3", "3MIN", "(25)", "(0.2)"):
        assert value in values, value
    # The values of this sheet's feature control frames are published inside
    # their frames, with the symbols and modifiers the text layer cannot hold,
    # rather than as bare numbers.
    assert "⌓ 0.2 Ⓤ" in values and "⌖ Ø0.5 A B C" in values
    assert "0.2" not in values and "Ø0.5A B C" not in values
    # A numbered note leaves lone digits behind; they are not callouts.
    assert "1" not in values and "2" not in values and "3" not in values


def test_a_lone_digit_is_never_a_callout_but_a_short_value_is():
    from pdf_text_layer import is_bare_number, is_callout

    assert is_callout("7.3") and is_callout("3MIN") and is_callout("(25)")
    assert is_callout("21") and is_callout("R1")
    assert not is_callout("1") and not is_callout("2")
    assert not is_callout("SECTION VIEW A-A")
    # A number with nothing saying what it measures may have lost a symbol
    # the drawing wrote as geometry, so it is published for confirmation.
    assert is_bare_number("45") and is_bare_number("21")
    assert not is_bare_number("7.3") and not is_bare_number("Ø50±0.2")
    assert not is_bare_number("3MIN") and not is_bare_number("R1")


def test_only_a_read_that_says_the_same_number_may_replace_exact_text():
    # "1 × 45°": the drawing writes the × and the ° as geometry, so the text
    # layer holds "1" and "45" and the recogniser supplies the rest.
    assert rescue_geometry_marks("45", "1×45°") == "1×45°"
    # A different number is the recogniser disagreeing with a certainty.
    assert rescue_geometry_marks("45", "1×46°") is None
    # Nothing gained means nothing to replace.
    assert rescue_geometry_marks("45", "45") is None
    # A value that already carries a mark keeps its own reading.
    assert rescue_geometry_marks("Ø45", "Ø45°") is None
    # So does one with a structure the recogniser would flatten: the text
    # layer got the deviation pair right and OCR returns "21°+1°0".
    assert rescue_geometry_marks("21+1/0", "21°+1°0") is None


@pytest.mark.skipif(
    not SYMBOLS_AS_GEOMETRY.exists(), reason="sample drawing not present"
)
def test_a_chamfer_whose_times_and_degree_are_geometry_is_read_whole():
    values = {
        region["text"]: region
        for region in text_layer_regions(SYMBOLS_AS_GEOMETRY)["regions"]
    }

    assert "1×45°" in values
    assert values["1×45°"]["needs_review"] is False
    # And it is no longer published as the bare "45" it used to be.
    assert "45" not in values


def test_a_label_does_not_band_with_the_deviation_of_the_callout_beside_it():
    # "View I" is set at 21pt and the "+0.2" of the R0.3 callout beside it at
    # 12pt. The small run falls inside the tall one's vertical span, so they
    # banded together and the whole group was then dropped as prose, taking
    # an exact R0.3 +0.2 with it.
    groups = group_runs(
        [
            run("View", 1574, 51, 45, 16, size=21.1),
            run("I", 1627, 51, 2, 15, size=21.1),
            run("R0.3", 1640, 66, 30, 11, size=14.8),
            run("+0.2", 1674, 61, 23, 9, size=11.9),
        ],
        PAGE,
    )
    texts = {group["text"] for group in groups}
    assert "View I" in texts
    assert "R0.3 +0.2" in texts


def test_a_caption_under_a_value_is_not_swallowed_as_a_deviation():
    # A label printed under a callout stacks exactly like a deviation does,
    # and absorbing it turned an exact value into prose that the callout gate
    # then threw away whole.
    groups = group_runs(
        [
            run("2.8", 672, 558, 20, 11, size=14.8),
            run("Spline", 672, 575, 40, 14, size=14.8),
            run("Run-off", 718, 575, 48, 11, size=14.8),
        ],
        PAGE,
    )
    assert {group["text"] for group in groups} == {"2.8", "Spline Run-off"}


def test_a_value_still_takes_the_deviations_stacked_under_it():
    # The gate above must not cost the case the merge exists for.
    groups = group_runs(
        [
            run("70", 500, 500, 34, 25, size=25.0),
            run("+0.2", 503, 528, 40, 18, size=19.0),
        ],
        PAGE,
    )
    assert len(groups) == 1
    assert groups[0]["text"] == "70 +0.2"


def test_two_note_numbers_in_a_column_are_not_stacked_into_a_value():
    # "2." over "3." down the margin of a note list. Each is a lone digit,
    # and nothing a deviation could qualify sits above either.
    groups = group_runs(
        [
            run("2", 69, 1161, 14, 11, size=12.3),
            run("3", 69, 1176, 14, 11, size=12.3),
        ],
        PAGE,
    )
    assert {group["text"] for group in groups} == {"2", "3"}


def test_a_lone_signed_deviation_is_published_for_review():
    from pdf_text_layer import is_bare_number, is_callout

    # has_dimension_value drops these as "a tolerance with nothing to apply
    # it to", which is right for an OCR read and wrong for exact text.
    assert is_callout("+0.5") and is_callout("-0.5") and is_callout("-0.8")
    # It says how much, never of what, so it always goes to the inspector.
    assert is_bare_number("+0.5") and is_bare_number("-0.8")


def test_a_diameter_written_in_a_text_font_is_decoded():
    # AutoCAD's "%%c" reaches the text layer as 0x91 when the sheet writes it
    # in ArialMT rather than in AIGDT, and "Ø37.5" was published opening with
    # a control character.
    assert _decode_symbols("\x91", "ArialMT") == "Ø"
    assert _decode_symbols("\x83", "ArialMT") == "°"


@pytest.mark.skipif(not CROWDED.exists(), reason="sample drawing not present")
def test_no_value_on_a_crowded_sheet_is_lost_to_its_neighbours():
    result = text_layer_regions(CROWDED, recover_symbols=False)
    by_text = {region["text"]: region for region in result["regions"]}

    # Each of these was excluded, fused into the label printed next to it.
    assert "2.8" in by_text
    assert "(15.9)" in by_text
    assert any(text.startswith("R0.3+0.2") for text in by_text)
    # A lone deviation against a step: published, and flagged.
    assert by_text["+0.5"]["needs_review"] is True
    # Diameters the sheet writes in a text font, once read as "\x9137.5+0.2".
    assert "Ø37.5+0.2" in by_text and "Ø41+0.3" in by_text
    # No published value may still carry an undecoded glyph.
    assert not [
        text for text in by_text if any(0x7F <= ord(c) <= 0x9F for c in text)
    ]


# ── Characteristic marks, stacked values and diagonal text ───────────────
# BS1801025.048b writes each pressed diameter twice, "*Ø37.65 +0.01" before
# pressing and "●Ø37.50" after, mostly as vertical text, and every one of the
# "*" values was lost while the drawing still showed it.
MARKED = DOCS / "BS1801025.048b.pdf"


def test_a_marked_value_is_a_callout():
    from pdf_text_layer import is_bare_number, is_callout

    for text in ["*Ø37.65+0.01", "(*Ø35.77)", "●Ø37.50", "(●Ø37.90)", "*Ø33.21-0.01"]:
        assert is_callout(text), text
        assert not is_bare_number(text), text
    # The mark alone, or on a word, is still not a value.
    assert not is_callout("*")
    assert not is_callout("*BEFORE PRESSING IN")


def test_a_bullet_mismapped_to_a_letter_is_read_as_the_bullet():
    from pdf_text_layer import _decode_bullets

    assert _decode_bullets("łØ37.9") == "●Ø37.9"
    assert _decode_bullets("(ł") == "(●"
    assert _decode_bullets("ł") == "●"
    # A word that really spells with it is left alone.
    assert _decode_bullets("złoty") == "złoty"


def test_two_marked_values_in_a_column_stay_two_callouts():
    # "*Ø33.21 -0.01" over "●Ø33.08": each a whole value with its own Ø,
    # once fused into "*●Ø3Ø3/3.21/3.08/-0.01" and thrown away.
    groups = group_runs(
        [
            run("*Ø38.05", 3507, 1735, 134, 37, size=41.3),
            run("-0.01", 3653, 1746, 57, 20, size=27.5),
            run("●Ø37.9", 3509, 1790, 115, 36, size=41.3),
            run("0", 3626, 1796, 16, 25, size=34.4),
        ],
        PAGE,
    )
    assert sorted(group["text"] for group in groups) == ["*Ø38.05 -0.01", "●Ø37.90"]


def test_each_deviation_follows_the_value_it_qualifies():
    groups = group_runs(
        [
            run("R0.3", 772, 2523, 69, 25, size=34.4),
            run("+0.05", 849, 2512, 68, 20, size=27.5),
            run("(34x)", 930, 2523, 74, 32, size=34.4),
        ],
        PAGE,
    )
    assert [group["text"] for group in groups] == ["R0.3 +0.05 (34x)"]


def test_a_diagonal_glyph_is_levelled_to_its_own_height():
    from pdf_text_layer import _level_glyph

    # A 25px digit at 45° has a 35px square hull. Levelling the hull again
    # made it 50px tall; the glyph itself is what lines are banded by.
    _, _, _, height = _level_glyph(100, 100, 35, 35, 45.0, 34.4)
    assert height == pytest.approx(0.73 * 34.4, abs=1.0)
    # Off 45° the hull is solved exactly: a 20x25 glyph at 30°.
    a, b = math.cos(math.radians(30)), math.sin(math.radians(30))
    hull_w, hull_h = a * 20 + b * 25, b * 20 + a * 25
    _, _, width, height = _level_glyph(0, 0, hull_w, hull_h, 30.0, 34.4)
    assert (width, height) == (pytest.approx(20), pytest.approx(25))


def test_a_rejected_value_that_still_measures_goes_to_review_not_away():
    from pdf_text_layer import _still_a_measurement

    # A mark the gate has never seen in front of a diameter.
    assert _still_a_measurement([run("§Ø10", 0, 0, 60, 25)], "§Ø10")
    # Title-block tolerance notes carry a word and stay out.
    assert not _still_a_measurement([run("ANGLES", 0, 0, 90, 25), run("±2°", 95, 0, 40, 25)], "ANGLES ±2°")
    assert not _still_a_measurement([run("BS1801025.048b", 0, 0, 200, 25)], "BS1801025.048b")


def test_a_plus_minus_pdfium_returns_no_text_for_is_still_read():
    # 0x93 is "%%p" in a text font. pdfium gives it no text, and the empty
    # string passed the line-break test, so the ± vanished and split the run.
    assert _decode_symbols("\x93", "ArialMT") == "±"


def test_a_general_tolerance_table_is_not_a_callout():
    from pdf_text_layer import _still_a_measurement, is_callout

    for text in [
        "X.XXX±.005[.012]", "±NA±.1±.03[.12]", ".03[.75]ANG±1°0.500",
        ".X±.060FRAC±1/16",
    ]:
        assert not is_callout(text), text
        assert not _still_a_measurement([run(text, 0, 0, 200, 25)], text), text
    assert is_callout("L±.09") and is_callout("6×60°±30'")
    assert is_callout("ØA±.008") and is_callout("0.5×45°") and is_callout("2X Ø6.5")


TOLERANCE_IN_C1 = DOCS / "Dwg" / "11-2-21_rev.L_P600-2325_redacted.pdf"
SHIFTED_TOLERANCE = DOCS / "Dwg" / "560-391h_redacted.pdf"


@pytest.mark.skipif(not TOLERANCE_IN_C1.exists(), reason="sample drawing not present")
def test_values_whose_plus_minus_has_no_text_keep_it():
    texts = {region["text"] for region in text_layer_regions(TOLERANCE_IN_C1, refine=False)["regions"]}
    assert {"L±.09", "ØA±.008", "ØB±.008"} <= texts


@pytest.mark.skipif(not SHIFTED_TOLERANCE.exists(), reason="sample drawing not present")
def test_shifted_symbols_are_decoded_and_the_tolerance_table_stays_out():
    result = text_layer_regions(SHIFTED_TOLERANCE, refine=False)
    excluded = {item["text"] for item in result["excluded"]}
    assert "FRAC±.03[.75]ANG±1°0.500" in excluded
    published = [region["text"] for region in result["regions"]]
    assert not [text for text in published if "X.X" in text or "NA" in text]
    assert not [text for text in published if any(ord(c) < 0x20 for c in text)]


@pytest.mark.skipif(not MARKED.exists(), reason="sample drawing not present")
def test_every_value_before_and_after_pressing_is_read():
    result = text_layer_regions(MARKED, refine=False)
    by_text = {region["text"]: region for region in result["regions"]}

    expected = [
        "*Ø37.65+0.01", "●Ø37.50", "*Ø41.16+0.01", "●Ø41.00",
        "*Ø35.77-0.01", "●Ø35.63", "*Ø33.21-0.01", "●Ø33.08",
        "*Ø38.05-0.01", "●Ø37.90",
        "(*Ø35.77)", "(●Ø35.63)", "(*Ø37.65)", "(●Ø37.50)",
        "(*Ø38.05)", "(●Ø37.90)",
        "90°±0°30'", "70°±0°30'", "42°±0°30'", "5°±0°30'",
        "R0.5±0.05", "0.5±0.1", "R0.3+0.05(34x)",
        # The diagonal chamfer, two callouts once fused into one.
        "Ø0.2+0.1", "6×60°±30'",
    ]
    assert [value for value in expected if value not in by_text] == []
    for value in expected[:16]:
        assert by_text[value]["category"] == "Diameter", value
        assert by_text[value]["needs_review"] is False, value
    # A bracketed reference is typed like any other bracket; the mark in
    # front of a plain one must not hide its Ø from the composer.
    for value in expected[:10]:
        assert by_text[value]["type"] == "diameter", value
    assert not any("ł" in text for text in by_text)


# ── Notes, page geometry and edges ────────────────────────────────────────
# Hand-built PDFs (standard Helvetica, no fonts embedded) so the CropBox,
# MediaBox and /Rotate are exactly what the test says. Every box is checked
# against where the text actually renders, which is the ground truth the
# client draws balloons on.


def _tiny_pdf(texts, media=(0, 0, 600, 400), crop=None, rotate=0):
    """One page; ``texts`` is a list of (x, y, size, text) in PDF user space."""
    crop = crop or media
    stream = "".join(
        f"BT /F1 {size} Tf {x} {y} Td ({text}) Tj ET\n" for x, y, size, text in texts
    ).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [{' '.join(map(str, media))}] "
            f"/CropBox [{' '.join(map(str, crop))}] /Rotate {rotate} "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode(),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.7\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def _ink_box(pdf, dpi):
    import numpy as np
    import pypdfium2 as pdfium

    image = pdfium.PdfDocument(pdf)[0].render(scale=dpi / 72).to_pil().convert("L")
    ys, xs = np.nonzero(np.asarray(image) < 128)
    return xs.min(), ys.min(), xs.max(), ys.max()


@pytest.mark.parametrize(
    "media, crop, rotate",
    [
        ((0, 0, 600, 400), None, 0),
        # A CropBox that does not start at the origin.
        ((0, 0, 800, 600), (100, 150, 700, 550), 0),
        # A MediaBox below zero, as 129E01-13300-ID.pdf has: every box landed
        # 1197pt too low, the lower ones off the page entirely.
        ((0, -400, 600, 0), None, 0),
        ((0, 0, 600, 400), None, 90),
        ((0, 0, 600, 400), None, 180),
        ((0, 0, 800, 600), (100, 150, 700, 550), 270),
    ],
)
def test_text_boxes_sit_on_their_ink_whatever_the_page_geometry(media, crop, rotate):
    origin = crop or media
    pdf = _tiny_pdf([(origin[0] + 10, origin[1] + 10, 12, "CORNER NOTE")], media, crop, rotate)
    (run_,) = page_text_runs(pdf, 0, dpi=144)[0]
    x0, y0, x1, y1 = _ink_box(pdf, 144)
    assert abs(run_.x - x0) <= 2 and abs(run_.y - y0) <= 2
    assert abs(run_.x + run_.width - x1) <= 2 and abs(run_.y + run_.height - y1) <= 2
    # The reading direction on the rendered page, clockwise like pdfium.
    assert run_.angle == rotate


@pytest.mark.parametrize("corner", ["bottom-left", "bottom-right", "top-left", "top-right"])
def test_a_note_in_any_corner_is_published_with_its_box_on_the_page(corner):
    width, height = 842, 595
    note = ["REMOVE ALL BURRS AND SHARP EDGES.", "BREAK ALL EDGES 0.2 MAX.", "UNLESS OTHERWISE SPECIFIED."]
    left = 6 if "left" in corner else width - 230
    top = height - 8 if "top" in corner else 8 + 2 * 11
    texts = [(left, top - 11 * i - 8, 9, line) for i, line in enumerate(note)]
    pdf = _tiny_pdf(texts, media=(0, 0, width, height))

    result = text_layer_regions(pdf, dpi=108, recover_symbols=False)
    notes = [r for r in result["regions"] if r["category"] == "General Note"]
    assert [n["text"] for n in notes] == ["\n".join(note)]
    box = notes[0]["bbox"]
    page_w, page_h = result["page_size"]
    assert box["x"] >= 0 and box["y"] >= 0
    assert box["x"] + box["width"] <= page_w and box["y"] + box["height"] <= page_h
    # The box is the note's, not the page's.
    assert box["width"] * box["height"] < 0.05 * page_w * page_h
    # Nothing inside the note is ballooned as a dimension on its own.
    assert not any("0.2" in r["text"] for r in result["regions"] if r is not notes[0])


def test_note_lines_keep_their_line_boxes_for_audit():
    pdf = _tiny_pdf([(40, 60, 9, "1. PART TO BE LEAK TESTED."), (40, 49, 9, "2. PART TO BE DRY.")])
    (note,) = [r for r in text_layer_regions(pdf, dpi=108, recover_symbols=False)["regions"] if r["category"] == "General Note"]
    assert [line["text"] for line in note["lines"]] == ["1. PART TO BE LEAK TESTED.", "2. PART TO BE DRY."]
    assert note["raw_text"] == note["text"]
    assert note["provenance"][0]["engine"] == "pdf-text-layer"


def test_labels_are_accounted_for_but_not_ballooned():
    pdf = _tiny_pdf([(40, 300, 10, "SECTION VIEW A-A"), (300, 200, 10, "25.4")])
    result = text_layer_regions(pdf, dpi=108, recover_symbols=False)
    assert [r["text"] for r in result["regions"]] == ["25.4"]
    assert [b["text"] for b in result["text_blocks"]] == ["SECTION VIEW A-A"]


@pytest.mark.skipif(not SYMBOLS_AS_GEOMETRY.exists(), reason="sample drawing not present")
def test_every_note_on_a_real_vector_drawing_is_read_whole():
    # 3015-100-2232F carries three notes: a numbered block in the bottom-left
    # corner, a five-line paragraph on the right and a sentence across the
    # top. None reached the client before: prose was excluded, and only the
    # OCR route, which never runs on a vector PDF, looked for notes at all.
    result = text_layer_regions(SYMBOLS_AS_GEOMETRY, dpi=108, recover_symbols=False)
    notes = {r["text"]: r for r in result["regions"] if r["category"] == "General Note"}
    assert set(notes) == {
        "DIMENSIONS IN ( ) ARE FOR REFERENCE ONLY AND DO NOT NEED TO BE MEASURED",
        "ALL DRAWINGS OF CAST /\nMACHINED COMPONENTS TO BE\nSUPPLIED WITH 3D MODEL\n"
        "FOR IDENTIFICATION OF ALL\nPROFILE DIMENSIONS.",
        "1. CLEANLINESS REQUIREMENT TO BE FREE FROM DIRT AND\nOILS.\n"
        "2. PART TO BE LEAK TESTED @3Bar FOR 30Secs\n2. PART TO BE DRY\n"
        "3. MATERIAL STAINLESS STEEL 304L THICKNESS 3mm MIN",
    }
    corner = next(r for t, r in notes.items() if t.startswith("1."))
    # The drawing itself writes "2 PART" and "3 MATERIAL" with no dot.
    assert "\n2 PART TO BE DRY\n3 MATERIAL" in corner["raw_text"]
    page_w, page_h = result["page_size"]
    assert corner["bbox"]["x"] < 0.1 * page_w and corner["bbox"]["y"] > 0.85 * page_h


def _pdf_with_image(texts, image_rect, size=(600, 400)):
    """A vector page with text, plus one embedded grey picture at ``image_rect`` (x, y, w, h in pt)."""
    width, height = 40, 20
    pixels = bytes([90] * (width * height))
    x, y, w, h = image_rect
    content = (
        f"q {w} 0 0 {h} {x} {y} cm /Im1 Do Q\n"
        + "".join(f"BT /F1 {s} Tf {tx} {ty} Td ({t}) Tj ET\n" for tx, ty, s, t in texts)
    ).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {size[0]} {size[1]}] "
            "/Resources << /Font << /F1 5 0 R >> /XObject << /Im1 6 0 R >> >> /Contents 4 0 R >>"
        ).encode(),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length {len(pixels)} >>\nstream\n"
        ).encode()
        + pixels
        + b"\nendstream",
    ]
    out, offsets = bytearray(b"%PDF-1.7\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_a_picture_on_a_vector_page_is_read_for_the_notes_inside_it(monkeypatch):
    # A mixed page: vector dimensions, and a pasted picture of a notes block
    # that the text layer cannot see into. Only the picture is OCR'd, and
    # what it finds comes back at its place on the page.
    import ocr_pipeline
    from pdf_text_layer import _image_area_notes, page_image_boxes

    pdf = _pdf_with_image([(400, 300, 10, "25.4")], image_rect=(50, 40, 200, 100))
    (box,) = page_image_boxes(pdf, 0, dpi=144)
    # 50pt from the left, 40pt above the bottom of a 400pt page, at 2px/pt.
    assert box == pytest.approx({"x": 100, "y": 520, "width": 400, "height": 200}, abs=1)

    seen = []

    class FakePipeline:
        def _notes_regions(self, crop):
            seen.append(crop.size)
            return [{"text": "1. PICTURED NOTE", "bbox": {"x": 10, "y": 20, "width": 100, "height": 30},
                     "lines": [{"text": "1. PICTURED NOTE", "bbox": {"x": 10, "y": 20, "width": 100, "height": 30}}],
                     "category": "General Note", "type": "General Note", "confidence": 0.97}]

    monkeypatch.setattr(ocr_pipeline, "get_pipeline", lambda: FakePipeline())
    runs = page_text_runs(pdf, 0, dpi=144)[0]
    (note,) = _image_area_notes(pdf, 0, runs, dpi=144)
    assert seen == [(400, 200)]
    assert note["bbox"]["x"] == 110 and note["bbox"]["y"] == 540
    assert note["lines"][0]["bbox"]["x"] == 110
    assert note["provenance"][0]["engine"] == "paddleocr"

    # A picture the native text already sits on is not read again.
    covered = _pdf_with_image([(60, 80, 10, "NATIVE LABEL")], image_rect=(50, 40, 200, 100))
    seen.clear()
    assert _image_area_notes(covered, 0, page_text_runs(covered, 0, dpi=144)[0], dpi=144) == []
    assert seen == []


def test_title_line_boxes_join_a_label_split_into_words():
    # "Drawing" and "nr.:" are two runs; seen right-to-left they must still
    # join, or the matcher reads "nr.:" as the drawing number.
    lines = _title_line_boxes(
        [
            run("nr.:", 659, 1158, 16, 6),
            run("Drawing", 613, 1156, 40, 10),
            run("BS1801006.020", 720, 1156, 80, 10),
        ]
    )
    assert [line["text"] for line in lines] == ["Drawing nr.:", "BS1801006.020"]


@pytest.mark.skipif(not CROWDED.exists(), reason="sample drawing not present")
def test_native_title_fields_read_the_revision_from_the_text_layer():
    fields = native_title_fields(str(CROWDED), 0, ["REV"], dpi=108.0)
    assert [(field["label"], field["value"]) for field in fields] == [("Rev.", "R2")]


@pytest.mark.skipif(not SYMBOLS_AS_GEOMETRY.exists(), reason="sample drawing not present")
def test_fast_read_shares_region_ids_with_the_refined_one():
    data = SYMBOLS_AS_GEOMETRY.read_bytes()
    fast = text_layer_regions(data, 0, dpi=108.0, refine=False)["regions"]
    full = text_layer_regions(data, 0, dpi=108.0)["regions"]

    fast_ids = [region["candidate_id"] for region in fast]
    full_ids = [region["candidate_id"] for region in full]
    assert len(set(fast_ids)) == len(fast_ids)
    assert len(set(full_ids)) == len(full_ids)
    # The refined read only adds frames; everything else it matches by id.
    assert all(
        candidate in fast_ids for candidate in full_ids if not candidate.startswith("fcf-")
    )
    assert any(candidate.startswith("fcf-") for candidate in full_ids)
    assert not any(candidate.startswith("fcf-") for candidate in fast_ids)
