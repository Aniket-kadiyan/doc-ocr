"""
Tests for title_fields.extract_title_fields.

The box coordinates below are the real PaddleOCR detection output for the title
block of GEP5A34.pdf, so the layouts exercised here are the ones that actually
occur: a value to the right of its label, a label and value fused in one box, a
colon-terminated label with an empty field, and a label whose value sits in the
row below.

    python test_title_fields.py
"""

from __future__ import annotations

from title_fields import extract_title_fields


def _b(text, x, y, w, h, conf=1.0):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": conf}


# The title block of GEP5A34.pdf as the detector reads it.
TITLE_BLOCK = [
    _b("DWG NO.", 403, 36, 109, 33),
    _b("GEP5A34", 522, 53, 261, 68),
    _b("2", 1018, 26, 74, 104),  # the REV value; its label is never detected
    _b("DATE:", 402, 136, 70, 30),
    _b("1/3/14", 474, 135, 122, 35),
    _b("SIZE", 788, 133, 52, 24),
    _b("SCALE:", 884, 137, 84, 33),
    _b("DRAWN: DX", 401, 178, 144, 28),
    _b("SHEET 1 OF 4", 852, 175, 214, 35),
    _b("CHECKED BY:", 403, 221, 166, 29),
    _b("UNITS", 791, 217, 75, 31),
    _b("INCHES[MILLIMETERS]", 803, 254, 279, 41),
    _b("APPROVED BY:", 402, 270, 173, 28),
]


def _values(keywords):
    return {f["keyword"]: f["value"] for f in extract_title_fields(TITLE_BLOCK, keywords)}


def test_value_to_the_right_of_its_label():
    # "DWG NO." with the number beside it — the label chrome ("NO.") is not
    # mistaken for the value.
    assert _values(["DWG"])["DWG"] == "GEP5A34"


def test_label_and_value_in_one_box():
    assert _values(["DRAWN"])["DRAWN"] == "DX"
    # No colon: the keyword itself splits the box.
    assert _values(["SHEET"])["SHEET"] == "1 OF 4"


def test_colon_label_with_value_in_its_own_box():
    assert _values(["DATE"])["DATE"] == "1/3/14"


def test_value_below_its_label():
    assert _values(["UNITS"])["UNITS"] == "INCHES[MILLIMETERS]"


def test_empty_colon_field_does_not_steal_the_row_below():
    # "SCALE:" is blank on this drawing; "SHEET 1 OF 4" is printed under it and
    # must not be read as the scale.
    assert _values(["SCALE"])["SCALE"] == ""
    assert _values(["CHECKED BY"])["CHECKED BY"] == ""
    assert _values(["APPROVED BY"])["APPROVED BY"] == ""


def test_missing_keyword_reports_an_empty_value():
    # The REV label is not in the detector's output at any resolution, so the
    # field is reported empty rather than guessed from the nearby "2".
    fields = extract_title_fields(TITLE_BLOCK, ["REV"])
    assert len(fields) == 1
    assert fields[0]["keyword"] == "REV"
    assert fields[0]["value"] == ""
    assert fields[0]["bbox"] is None


def test_every_keyword_gets_exactly_one_row_in_order():
    keywords = ["DWG", "REV", "DATE"]
    fields = extract_title_fields(TITLE_BLOCK, keywords)
    assert [f["keyword"] for f in fields] == keywords


def test_no_keywords_yields_nothing():
    assert extract_title_fields(TITLE_BLOCK, []) == []


def test_matching_is_case_insensitive():
    assert _values(["dwg"])["dwg"] == "GEP5A34"


def test_unknown_keyword_is_reported_empty():
    fields = extract_title_fields(TITLE_BLOCK, ["CUSTOMER"])
    assert fields[0]["value"] == ""




# ---------------------------------------------------------------------------
# Regressions from a real full-page run of GEP5A34.pdf
# ---------------------------------------------------------------------------

# The general-notes paragraph, which sits in the MIDDLE of the sheet and
# contains the word REVISION.
NOTES_ON_THE_SHEET = [
    _b("2. HOT DIP GALVANIZE PER ASTM A153 (LATEST REVISION)", 60, 300, 420, 20),
    _b("3. ALL DIMENSIONS ARE AFTER GALVANIZING", 60, 325, 380, 20),
]


def test_rev_keyword_does_not_match_the_word_revision():
    # "REV" once matched inside "(LATEST REVISION)" in a note and took
    # "GALVANIZING" from the line below as the revision number.
    fields = extract_title_fields(NOTES_ON_THE_SHEET + TITLE_BLOCK, ["REV"])
    assert fields[0]["value"] != "GALVANIZING"
    assert "GALVANIZ" not in fields[0]["value"].upper()


def test_keywords_prefer_the_bottom_right_table():
    # A DATE in a revision block higher up the sheet must not displace the
    # title block's own DATE.
    elsewhere = [_b("DATE: 9/9/99", 60, 100, 120, 20)]
    got = extract_title_fields(elsewhere + TITLE_BLOCK, ["DATE"])[0]
    assert got["value"] == "1/3/14"


def test_falls_back_to_the_whole_image_when_the_corner_has_nothing():
    # A crop of just the title block: every label is in the upper-left of the
    # crop, so the bottom-right pass finds nothing and the fallback must run.
    top_left_only = [
        _b("DWG NO.", 0, 0, 109, 33),
        _b("GEP5A34", 119, 17, 261, 68),
    ]
    assert extract_title_fields(top_left_only, ["DWG"])[0]["value"] == "GEP5A34"


def test_a_label_must_start_with_the_keyword():
    # 47630.pdf has no REV cell in its title block; the only text containing
    # "REV" is note 4, "FINISH: ZINC YELLOW PER ASTM B633-LATEST REV. TYPE II".
    # Matching that produced a row labelled with the whole note. An empty REV
    # is the honest answer.
    sheet = [
        _b("DWG NO.", 736, 778, 33, 11),
        _b("47630", 766, 779, 61, 25),
        _b("B633-LATEST REV. TYPE II", 861, 735, 218, 18),
        _b("TYPE II", 861, 755, 60, 18),
    ]
    fields = {f["keyword"]: f for f in extract_title_fields(sheet, ["DWG", "REV"])}
    assert fields["DWG"]["value"] == "47630"
    assert fields["REV"]["value"] == ""
    assert fields["REV"]["label"] == "REV"


def test_revisions_table_reports_the_latest_row():
    # 47630.pdf carries its revisions in a table at the top right rather than a
    # REV cell in the title block. The table is appended downward, so the
    # drawing's current revision is its LAST row, not the first one under the
    # column heading.
    rev_table = [
        _b("REV", 715, 60, 21, 14),
        _b("1", 722, 81, 7, 8),
        _b("2", 721, 99, 11, 10),
    ]
    assert extract_title_fields(rev_table, ["REV"])[0]["value"] == "2"


def test_a_single_value_cell_is_unaffected():
    # A plain "REV" cell with one number under it must still read that number.
    cell = [_b("REV", 1018, 10, 30, 14), _b("2", 1018, 26, 74, 104)]
    assert extract_title_fields(cell, ["REV"])[0]["value"] == "2"


def test_rev_cell_does_not_take_the_row_below_it():
    # 47630.pdf: REV sits directly over its number, and the SCALE cell sits
    # under that. Following the column too far returned "N.T.S." as the
    # revision, so the walk now stops when the next box stops looking like
    # another row of the same column.
    title_block = [
        _b("REV", 791, 357, 32, 19),
        _b("2", 791, 374, 33, 39),
        _b("SCALE: N.T.S.", 697, 419, 105, 24),
        _b("SIZE", 637, 416, 39, 19),
    ]
    assert extract_title_fields(title_block, ["REV"])[0]["value"] == "2"


def test_label_chrome_is_not_taken_as_the_value():
    # At some scales "DWG NO." arrives as two boxes. Without a guard the "NO."
    # beside "DWG" was published as the drawing number.
    split_label = [
        _b("DWG", 170, 155, 29, 14),
        _b("NO.", 200, 155, 20, 14),
        _b("47630", 232, 157, 57, 24),
    ]
    assert extract_title_fields(split_label, ["DWG"])[0]["value"] == "47630"



if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed")
