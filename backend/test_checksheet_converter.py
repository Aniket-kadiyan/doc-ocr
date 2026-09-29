"""Regression checks for inspection JSON -> Digital Checksheet conversion.

Run from ``backend/``:
    python test_checksheet_converter.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from checksheet_converter import (
    convert_ts2_json_to_checksheet,
    is_notes_text,
    make_template_id_from_filename,
    parse_range,
    save_checksheet_template,
)

# 47630.pdf: the ballooned NOTES block, flattened onto one line the way the
# inspection sheet sends it.
NOTES_BLOCK = (
    "1. MATERIAL: GRADE 5 EQUIV; 85KPSI YIELD MIN.;120KPSI TENSILE MIN. "
    '2. PIN TO BE STRAIGHT WITHIN .004" OF ENTIRE LENGTH '
    "3. PIN TO ASSEMBLED WITH CABLE AND HAIR PIN COTTER AS SHOWN "
    "4. FINISH: ZINC YELLOW PER ASTM B633-LATEST REV. TYPE II "
    "5. LANYARD RING MUST WITHSTAND 50 LBS MIN. WITHOUT OPENING."
)


SOURCE_JSON = {
    "data": [
        {
            "S.no": "1",
            "Label": "Outer diameter",
            "Value": "58.21±0.05",
            "Tolerance": "+0.05, -0.05",
            "part1": "",
            "part2": "",
            "Method": "Measure",
            "Tool": "Micrometer",
        },
        {
            "S.no": "2",
            "Label": "Face angle",
            "Value": "24°31'16\"",
            "Tolerance": "",
            "part1": "",
            "part2": "",
            "Method": "Inspect",
            "Tool": "Angle gauge",
        },
    ],
    "extra_columns": ["part1", "part2"],
}


def test_conversion_shape_and_ranges() -> None:
    template = convert_ts2_json_to_checksheet(
        SOURCE_JSON,
        template_id="TS2_SAMPLE_INSPECTION",
        template_name="TS2 Sample Inspection",
    )

    assert template["template_id"] == "TS2_SAMPLE_INSPECTION"
    assert len(template["items"]) == 3  # one header plus two data rows

    first_part_input = template["items"][1]["items"][6]["items"][0]
    assert first_part_input["id"] == "row_1_part1"
    assert first_part_input["type"] == "number"
    assert first_part_input["range"] == {"min": 58.16, "max": 58.26}

    angle_part_input = template["items"][2]["items"][6]["items"][0]
    assert angle_part_input["type"] == "number"
    assert angle_part_input["range"] == {
        "min": 24.521111,
        "max": 24.521111,
    }



def test_default_zero_and_supported_tolerances() -> None:
    assert parse_range("25", "") == {"min": 25, "max": 25}
    assert parse_range("25", "0") == {"min": 25, "max": 25}
    assert parse_range("25", "±0.1") == {"min": 24.9, "max": 25.1}
    assert parse_range("25", "+0.1 -0.2") == {"min": 24.8, "max": 25.1}
    assert parse_range("25", "+.1, -.2") == {"min": 24.8, "max": 25.1}
    assert parse_range("25 +0.1 -0.2", "") == {
        "min": 24.8,
        "max": 25.1,
    }
    assert parse_range("58.21±0.05", "") == {
        "min": 58.16,
        "max": 58.26,
    }



def test_angle_ranges_preserve_valid_dms_and_fallback_safely() -> None:
    assert parse_range("90°", "") == {"min": 90, "max": 90}
    assert parse_range("45.5°", "") == {"min": 45.5, "max": 45.5}
    assert parse_range("24°31'", "") == {
        "min": 24.516667,
        "max": 24.516667,
    }
    assert parse_range("24°31′16″", "") == {
        "min": 24.521111,
        "max": 24.521111,
    }
    assert parse_range("24°31'16" + '"', "±1°") == {
        "min": 23.521111,
        "max": 25.521111,
    }
    assert parse_range("24°31'16" + '"', "+2° -1°") == {
        "min": 23.521111,
        "max": 26.521111,
    }

    # Malformed minutes/seconds do not block export; the valid degree portion
    # is used as the nominal value.
    malformed_angles = (
        "24°75'",
        "24°31'80" + '"',
        "24°abc",
        "24°31",
    )
    for malformed_angle in malformed_angles:
        assert parse_range(malformed_angle, "") == {"min": 24, "max": 24}


def test_malformed_tolerance_expressions_block_conversion() -> None:
    malformed_cases = [
        ("25", "+0.1 +0.2"),
        ("25", "+ -0.2"),
        ("25", "±"),
        ("25", "+0.1 -"),
        ("25 +0.1 +0.2", ""),
        ("25 + -0.2", ""),
        ("25 ±", ""),
        ("25 +0.1 -", ""),
    ]
    for value, tolerance in malformed_cases:
        try:
            parse_range(value, tolerance)
        except ValueError as exc:
            assert "Malformed" in str(exc)
        else:
            raise AssertionError(
                f"Expected malformed expression to fail: {value!r}, {tolerance!r}"
            )

def test_a_notes_block_never_blocks_conversion() -> None:
    # "B633-LATEST" after the "1." marker reads as a minus tolerance, which
    # aborted the whole checksheet.
    assert is_notes_text(NOTES_BLOCK)
    assert parse_range(NOTES_BLOCK, "") is None

    template = convert_ts2_json_to_checksheet(
        {
            "data": [
                {"S.no": "21", "Label": "General Notes", "Value": NOTES_BLOCK,
                 "Tolerance": "", "part1": "", "Method": "", "Tool": ""},
                {"S.no": "22", "Label": "", "Value": "25", "Tolerance": "0",
                 "part1": "", "Method": "", "Tool": ""},
            ],
            "extra_columns": ["part1"],
        },
        template_id="notes_test",
        template_name="Notes test",
    )
    # Header plus both rows: the notes row is kept, not dropped.
    assert len(template["items"]) == 3


def test_prose_and_dimensions_are_told_apart() -> None:
    assert not is_notes_text("25 ±0.1")
    assert not is_notes_text("1.5")
    # A lone marker inside a sentence is not a list.
    assert not is_notes_text("1. PIN TO BE STRAIGHT")
    # Markers must start a list and climb.
    assert not is_notes_text("7. ALPHA 4. BETA")
    assert is_notes_text("1. ALPHA 2. BETA")


def test_template_filename_is_safe_and_portable() -> None:
    assert (
        make_template_id_from_filename(
            r"C:\untrusted\path\ts2 32224 cone inspection.json"
        )
        == "TS2_32224_CONE_INSPECTION"
    )


def test_save_creates_then_replaces_complete_json() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        first = save_checksheet_template(
            source_json=SOURCE_JSON,
            source_file_name="ts2_sample_inspection.json",
            target_dir=temporary_directory,
        )
        assert first.path == Path(temporary_directory) / "TS2_SAMPLE_INSPECTION.json"
        assert first.replaced is False

        with first.path.open("r", encoding="utf-8") as template_file:
            assert json.load(template_file) == first.template

        second = save_checksheet_template(
            source_json=SOURCE_JSON,
            source_file_name="ts2_sample_inspection.json",
            target_dir=temporary_directory,
        )
        assert second.path == first.path
        assert second.replaced is True


def test_part_column_is_required() -> None:
    rows_without_parts = [
        {
            key: value
            for key, value in row.items()
            if not key.lower().startswith("part")
        }
        for row in SOURCE_JSON["data"]
    ]
    try:
        convert_ts2_json_to_checksheet(
            {"data": rows_without_parts, "extra_columns": []},
            template_id="TS2_SAMPLE",
            template_name="TS2 Sample",
        )
    except ValueError as exc:
        assert "no extra_columns" in str(exc)
    else:
        raise AssertionError("Expected conversion without part columns to fail")


if __name__ == "__main__":
    test_conversion_shape_and_ranges()
    test_default_zero_and_supported_tolerances()
    test_angle_ranges_preserve_valid_dms_and_fallback_safely()
    test_malformed_tolerance_expressions_block_conversion()
    test_a_notes_block_never_blocks_conversion()
    test_prose_and_dimensions_are_told_apart()
    test_template_filename_is_safe_and_portable()
    test_save_creates_then_replaces_complete_json()
    test_part_column_is_required()
    print("OK")
