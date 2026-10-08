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
    with_candidate: bool = False,
    with_assembly: bool = False,
    with_engineering_parse: bool = False,
    with_engineering_disposition: bool = False,
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
                "oriented_box": {
                    "x": 11,
                    "y": 21,
                    "width": 29,
                    "height": 9,
                    "rotation": 17,
                },
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
    if with_assembly:
        assembly = {
            "objectId": "C0002",
            "assemblyId": "C0002",
            "rule": "inline_prefix",
            "conflict": False,
            "children": [
                {
                    "candidateId": "C0001",
                    "text": "4X",
                    "rawText": "4X",
                    "bbox": {"x": 10, "y": 20, "width": 12, "height": 10},
                    "confidence": 0.95,
                    "orientation": "horizontal",
                    "rotation": 0,
                    "role": "multiplier",
                }
            ],
        }
        payload["items"][0]["assembly"] = assembly  # type: ignore[index]
    engineering_parse = {
        "schemaVersion": 1,
        "rawText": "25+0.1",
        "normalizedText": "25+0.1",
        "status": "complete",
        "kind": "linear",
        "complete": True,
        "components": {
            "nominal": "25",
            "tolerance": {
                "mode": "single_deviation",
                "upper": "+0.1",
                "lower": None,
            },
        },
        "tokens": [],
        "unparsedFragments": [],
        "warnings": ["single_bound_tolerance"],
        "normalizationSteps": [],
    }
    if with_engineering_parse:
        payload["items"][0]["engineering_parse"] = engineering_parse  # type: ignore[index]
    engineering_disposition = {
        "schemaVersion": 1,
        "state": "eligible",
        "rule": "complete_engineering_object",
        "reason": "Complete assembled engineering value",
        "parseStatus": "complete",
        "parseKind": "linear",
        "hardContext": False,
    }
    if with_engineering_disposition:
        payload["items"][0]["engineering_disposition"] = (  # type: ignore[index]
            engineering_disposition
        )
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
                **({"assembly": assembly} if with_assembly else {}),
                **(
                    {"engineering_parse": engineering_parse}
                    if with_engineering_parse
                    else {}
                ),
                **(
                    {"engineering_disposition": engineering_disposition}
                    if with_engineering_disposition
                    else {}
                ),
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
    assert created["rows"][0]["oriented_box"] == {
        "x": 11.0,
        "y": 21.0,
        "width": 29.0,
        "height": 9.0,
        "rotation": 17.0,
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


def test_engineering_object_evidence_survives_storage_and_duplicate(
    tmp_path: Path,
) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(
        storage,
        _snapshot(with_candidate=True, with_assembly=True),
    )

    assert created["rows"][0]["assembly"]["children"][0]["text"] == "4X"
    assert created["revision"]["scan_candidates"][0]["assembly"]["rule"] == (
        "inline_prefix"
    )

    duplicate = storage.duplicate_checksheet(created["checksheet"]["id"])
    assert duplicate["rows"][0]["assembly"] == created["rows"][0]["assembly"]


def test_engineering_parse_survives_storage_and_duplicate(tmp_path: Path) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(
        storage,
        _snapshot(with_candidate=True, with_engineering_parse=True),
    )

    item_parse = created["rows"][0]["engineering_parse"]
    candidate_parse = created["revision"]["scan_candidates"][0][
        "engineering_parse"
    ]
    assert item_parse["components"]["nominal"] == "25"
    assert candidate_parse["warnings"] == ["single_bound_tolerance"]

    duplicate = storage.duplicate_checksheet(created["checksheet"]["id"])
    assert duplicate["rows"][0]["engineering_parse"] == item_parse


def test_engineering_disposition_survives_storage_and_duplicate(
    tmp_path: Path,
) -> None:
    storage = ChecksheetStorage(tmp_path)
    created = _create(
        storage,
        _snapshot(with_candidate=True, with_engineering_disposition=True),
    )

    item_disposition = created["rows"][0]["engineering_disposition"]
    candidate_disposition = created["revision"]["scan_candidates"][0][
        "engineering_disposition"
    ]
    assert item_disposition["rule"] == "complete_engineering_object"
    assert candidate_disposition["state"] == "eligible"

    duplicate = storage.duplicate_checksheet(created["checksheet"]["id"])
    assert duplicate["rows"][0]["engineering_disposition"] == item_disposition


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
