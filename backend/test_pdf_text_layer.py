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
    group_runs,
    has_usable_text_layer,
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
        "60°±0°30", "Ø33-0.05-0.1", "15-0.2", "R2±0.2", "70±0.2",
        "114.5±0.05", "R2±0.5", "R10±2", "Ø30±0.2", "5°±0°30",
        "Ø23.5-0.05", "Ø30-0.2",
    } <= set(by_text)

    assert by_text["Ø33-0.05-0.1"]["type"] == "diameter"
    assert by_text["60°±0°30"]["type"] == "angle"
    assert by_text["R10±2"]["category"] == "Radius"
    # Every dimension is exact, so none of them is sent for review. Only a
    # bare number with no mark saying what it measures is, and on this sheet
    # those are title-block codes, not callouts.
    dimensions = [
        "60°±0°30", "Ø33-0.05-0.1", "15-0.2", "R2±0.2", "70±0.2",
        "114.5±0.05", "R2±0.5", "R10±2", "Ø30±0.2", "5°±0°30",
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
    assert "60°±0°30" in values
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

    for value in ("7.3", "3MIN", "(25)", "(0.2)", "0.2", "Ø0.5A B C"):
        assert value in values, value
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
