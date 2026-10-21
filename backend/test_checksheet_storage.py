"""Persistence checks for checksheet revision metadata."""

from __future__ import annotations

import sqlite3
from io import BytesIO
from pathlib import Path

from checksheet_models import ChecksheetSnapshot
from checksheet_storage import ChecksheetStorage


def _snapshot(
    *,
    name: str = "Bracket inspection",
    metadata: dict[str, str] | None = None,
    include_references: bool = False,
    with_candidate: bool = False,
    balloon_style: dict[str, object] | None = None,
) -> ChecksheetSnapshot:
    payload: dict[str, object] = {
        "name": name,
        "drawing_name": "bracket.png",
        "source_project_id": "project-1",
        "source_file_type": "image",
        "pdf_render_scale": 1.5,
        "reading_columns": ["Part 1"],
        "items": [
            {
                "annotation_id": "annotation-1",
                "balloon_number": 1,
                "page": 1,
                "bbox": {"x": 10, "y": 20, "width": 30, "height": 12},
                "rotation": 0,
                "label": "Diameter",
                "specification": "25",
                "tolerance": "+0.1, -0.1",
                "method": "Measure",
                "tool": "Micrometer",
                "dimension_type": "Linear",
            }
        ],
    }
    if include_references:
        items = payload["items"]
        assert isinstance(items, list)
        box = {"x": 40, "y": 50, "width": 8, "height": 9}
        for offset, (axis, value) in enumerate(
            [("X", "1.2"), ("Y", ""), ("Z", "3.4")],
            start=2,
        ):
            items.append(
                {
                    "annotation_id": f"reference:p1:r0:{axis.lower()}",
                    "balloon_number": offset,
                    "item_kind": "reference",
                    "reference": {
                        "symbol": "A",
                        "axis": axis,
                        "point_index": 0,
                        "name_box": box,
                        "row_box": {"x": 35, "y": 45, "width": 80, "height": 14},
                        "coordinates": [
                            {"axis": "X", "value": "1.2"},
                            {"axis": "Z", "value": "3.4"},
                        ],
                        "markers": [
                            {"bbox": box, "score": 0.93, "margin": 0.18}
                        ],
                    },
                    "page": 1,
                    "bbox": box,
                    "rotation": 0,
                    "label": f"A_{axis}",
                    "specification": value,
                    "tolerance": "",
                    "method": "",
                    "tool": "",
                    "dimension_type": "Reference",
                }
            )
    if metadata is not None:
        payload["metadata"] = metadata
    if balloon_style is not None:
        payload["balloon_style"] = balloon_style
    if with_candidate:
        payload["scan_candidates"] = [
            {
                "candidate_id": "candidate-1",
                "source_candidate_id": "C0007",
                "page": 1,
                "order": 0,
                "state": "ignored",
                "restore_state": "other",
                "text": "REV A",
                "raw_text": "REV A",
                "preliminary_text": "REV 4",
                "confidence": 0.81,
                "recognized": True,
                "reason": "Title block text",
                "rule": "title_block",
                "orientation": "horizontal",
                "rotation": 0,
                "bbox": {"x": 80, "y": 90, "width": 40, "height": 12},
                "duplicate_source_ids": ["C0007", "C0008"],
                "duplicate_count": 1,
                "created_at": 100,
                "updated_at": 200,
            }
        ]
    return ChecksheetSnapshot.model_validate(payload)


def _create(
    storage: ChecksheetStorage,
    snapshot: ChecksheetSnapshot,
    source_bytes: bytes = b"drawing-bytes",
) -> dict:
    storage.initialize()
    return storage.create_checksheet(
        snapshot,
        source=BytesIO(source_bytes),
        original_name="bracket.png",
        mime_type="image/png",
    )


def test_revision_metadata_is_stored_and_returned(tmp_path: Path) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(
        storage,
        _snapshot(
            metadata={
                "partName": "  Drive Bracket  ",
                "documentNumber": "DB-1042",
                "revisionNumber": "C",
            }
        ),
    )

    assert created["revision"]["metadata"] == {
        "partName": "Drive Bracket",
        "documentNumber": "DB-1042",
        "revisionNumber": "C",
    }

    checksheet_id = created["checksheet"]["id"]
    detail = storage.get_checksheet(checksheet_id)
    assert detail["revisions"][0]["metadata"] == created["revision"]["metadata"]

    with sqlite3.connect(storage.database_path) as connection:
        rows = connection.execute(
            """
            SELECT field_key, value, position
            FROM checksheet_revision_metadata
            WHERE revision_id = ?
            ORDER BY position
            """,
            (created["revision"]["id"],),
        ).fetchall()
    assert rows == [
        ("partName", "Drive Bracket", 0),
        ("documentNumber", "DB-1042", 1),
        ("revisionNumber", "C", 2),
    ]


def test_revisions_and_duplicates_keep_their_own_metadata(tmp_path: Path) -> None:
    storage = ChecksheetStorage(tmp_path)
    first = _create(
        storage,
        _snapshot(
            metadata={
                "partName": "Bracket A",
                "documentNumber": "DOC-1",
                "revisionNumber": "A",
            }
        ),
    )
    checksheet_id = first["checksheet"]["id"]

    second = storage.create_revision(
        checksheet_id,
        _snapshot(
            metadata={
                "partName": "Bracket A",
                "documentNumber": "DOC-1",
                "revisionNumber": "B",
            }
        ),
        source=BytesIO(b"revised-drawing-bytes"),
        original_name="bracket-rev-b.png",
        mime_type="image/png",
    )

    assert first["revision"]["metadata"]["revisionNumber"] == "A"
    assert second["revision"]["metadata"]["revisionNumber"] == "B"

    duplicate = storage.duplicate_checksheet(checksheet_id)
    assert duplicate["revision"]["revision_number"] == 1
    assert duplicate["revision"]["metadata"] == second["revision"]["metadata"]


def test_missing_metadata_defaults_to_blank_values(tmp_path: Path) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(storage, _snapshot())

    assert created["revision"]["metadata"] == {
        "partName": "",
        "documentNumber": "",
        "revisionNumber": "",
    }
    assert created["revision"]["balloon_style"] == {
        "styleId": "circle",
        "styleVersion": 1,
        "size": "medium",
    }


def test_balloon_style_is_revision_data_and_survives_duplicate(tmp_path: Path) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(
        storage,
        _snapshot(
            balloon_style={
                "styleId": "hexagon",
                "styleVersion": 1,
                "size": "large",
            }
        ),
    )

    assert created["revision"]["balloon_style"] == {
        "styleId": "hexagon",
        "styleVersion": 1,
        "size": "large",
    }
    duplicate = storage.duplicate_checksheet(created["checksheet"]["id"])
    assert duplicate["revision"]["balloon_style"] == created["revision"]["balloon_style"]


def test_candidates_are_revision_data_not_checksheet_rows(tmp_path: Path) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(storage, _snapshot(with_candidate=True))

    assert len(created["rows"]) == 1
    assert created["revision"]["scan_candidates"] == [
        {
            "candidate_id": "candidate-1",
            "source_candidate_id": "C0007",
            "page": 1,
            "order": 0,
            "state": "ignored",
            "restore_state": "other",
            "text": "REV A",
            "raw_text": "REV A",
            "preliminary_text": "REV 4",
            "confidence": 0.81,
            "recognized": True,
            "reason": "Title block text",
            "rule": "title_block",
            "orientation": "horizontal",
            "rotation": 0.0,
            "recovery_attempted": False,
            "authoritative_reread": False,
            "bbox": {"x": 80.0, "y": 90.0, "width": 40.0, "height": 12.0},
            "duplicate_source_ids": ["C0007", "C0008"],
            "duplicate_count": 1,
            "created_at": 100,
            "updated_at": 200,
        }
    ]
    detail = storage.get_checksheet(created["checksheet"]["id"])
    assert detail["revisions"][0]["row_count"] == 1
    assert detail["revisions"][0]["candidate_count"] == 1

    duplicate = storage.duplicate_checksheet(created["checksheet"]["id"])
    assert duplicate["revision"]["scan_candidates"] == created["revision"][
        "scan_candidates"
    ]


def test_existing_database_without_metadata_table_migrates_to_blanks(
    tmp_path: Path,
) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(storage, _snapshot())
    with sqlite3.connect(storage.database_path) as connection:
        connection.execute("DROP TABLE checksheet_revision_metadata")

    reopened = ChecksheetStorage(tmp_path)
    reopened.initialize()
    loaded = reopened.get_run(created["checksheet"]["id"], created["run"]["id"])

    assert loaded["revision"]["metadata"] == {
        "partName": "",
        "documentNumber": "",
        "revisionNumber": "",
    }


def test_reference_rows_and_geometry_survive_storage_and_duplicate(
    tmp_path: Path,
) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(storage, _snapshot(include_references=True))

    reference_rows = created["rows"][1:]
    assert [row["label"] for row in reference_rows] == ["A_X", "A_Y", "A_Z"]
    assert [row["specification"] for row in reference_rows] == ["1.2", "", "3.4"]
    assert all(row["item_kind"] == "reference" for row in reference_rows)
    assert reference_rows[0]["reference"]["symbol"] == "A"
    assert reference_rows[0]["reference"]["markers"][0]["score"] == 0.93

    duplicate = storage.duplicate_checksheet(created["checksheet"]["id"])
    duplicated_references = duplicate["rows"][1:]
    assert [row["reference"]["axis"] for row in duplicated_references] == [
        "X",
        "Y",
        "Z",
    ]


def test_existing_item_table_gains_reference_columns(tmp_path: Path) -> None:
    storage = ChecksheetStorage(tmp_path)
    storage.initialize()
    with sqlite3.connect(storage.database_path) as connection:
        connection.execute("ALTER TABLE checksheet_items DROP COLUMN reference_json")
        connection.execute("ALTER TABLE checksheet_items DROP COLUMN item_kind")

    reopened = ChecksheetStorage(tmp_path)
    reopened.initialize()
    with sqlite3.connect(reopened.database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(checksheet_items)"
            ).fetchall()
        }
    assert {"item_kind", "reference_json"}.issubset(columns)
