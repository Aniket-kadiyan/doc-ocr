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
    make_template_id_from_filename,
    save_checksheet_template,
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
    assert angle_part_input["type"] == "text"
    assert "range" not in angle_part_input


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
    test_template_filename_is_safe_and_portable()
    test_save_creates_then_replaces_complete_json()
    test_part_column_is_required()
    print("OK")
