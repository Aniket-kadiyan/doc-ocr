"""Idempotent management of the Digital Checksheet template index."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Sequence


class ChecksheetIndexError(Exception):
    """Raised when index.json is missing, invalid, or has a conflict."""


# Prevent two requests in this backend process from updating index.json
# simultaneously.
_INDEX_LOCK = threading.Lock()


def _read_index(index_path: Path) -> list[dict[str, Any]]:
    """Load and validate the existing Digital Checksheet index."""

    if not index_path.exists():
        raise ChecksheetIndexError(
            f"Template index was not found: {index_path}"
        )

    try:
        raw_index = json.loads(
            index_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ChecksheetIndexError(
            f"Template index contains invalid JSON: {index_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ChecksheetIndexError(
            f"Template index could not be read: {index_path}: {exc}"
        ) from exc

    if not isinstance(raw_index, list):
        raise ChecksheetIndexError(
            "Template index must contain a JSON array."
        )

    if not all(isinstance(entry, dict) for entry in raw_index):
        raise ChecksheetIndexError(
            "Every template index entry must be a JSON object."
        )

    return raw_index


def _write_index_atomically(
    index_path: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Replace index.json without exposing a partially written file."""

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=index_path.parent,
            prefix=".index.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(
                entries,
                temporary_file,
                ensure_ascii=False,
                indent=4,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, index_path)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def register_template_if_missing(
    *,
    index_path: str | Path,
    template_id: str,
    file_name: str,
    template_name: str,
    users: Sequence[str] = ("admin_m",),
) -> bool:
    """Append a template only when no matching index entry exists.

    Returns:
        True when a new entry was appended.
        False when the exact id/file entry already existed.

    Existing entries are never modified.
    """

    template_id = template_id.strip()
    file_name = file_name.strip()
    template_name = template_name.strip()

    if not template_id:
        raise ValueError("Template ID cannot be empty.")

    if not file_name:
        raise ValueError("Template filename cannot be empty.")

    if not template_name:
        raise ValueError("Template name cannot be empty.")

    resolved_index_path = Path(index_path)

    with _INDEX_LOCK:
        entries = _read_index(resolved_index_path)

        entry_with_id = next(
            (
                entry
                for entry in entries
                if entry.get("id") == template_id
            ),
            None,
        )

        entry_with_file = next(
            (
                entry
                for entry in entries
                if entry.get("file") == file_name
            ),
            None,
        )

        # The exact template is already registered. Do not rewrite index.json.
        if (
            entry_with_id is not None
            and entry_with_file is not None
            and entry_with_id is entry_with_file
        ):
            return False

        # An ID or filename collision should not be silently overwritten.
        if entry_with_id is not None:
            raise ChecksheetIndexError(
                f"Template ID {template_id!r} already exists in index.json "
                f"with file {entry_with_id.get('file')!r}."
            )

        if entry_with_file is not None:
            raise ChecksheetIndexError(
                f"Template file {file_name!r} already exists in index.json "
                f"with ID {entry_with_file.get('id')!r}."
            )

        new_entry: dict[str, Any] = {
            "id": template_id,
            "file": file_name,
            "name": template_name,
            "restrictions": {
                "users": list(users),
            },
        }

        entries.append(new_entry)
        _write_index_atomically(
            resolved_index_path,
            entries,
        )

        return True
