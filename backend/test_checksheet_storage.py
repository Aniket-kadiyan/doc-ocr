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
    if metadata is not None:
        payload["metadata"] = metadata
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
