"""SQLite persistence and source-document storage for internal checksheets."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, Iterator, Sequence
from uuid import uuid4

from checksheet_models import ChecksheetSnapshot, ReadingPatch


class ChecksheetStorageError(RuntimeError):
    """Base error translated to an HTTP response by the route layer."""


class ChecksheetNotFoundError(ChecksheetStorageError):
    pass


class ChecksheetConflictError(ChecksheetStorageError):
    pass


class ChecksheetValidationError(ChecksheetStorageError):
    pass


@dataclass(frozen=True)
class StoredDocument:
    id: str
    original_name: str
    mime_type: str
    file_type: str
    path: Path
    size_bytes: int
    sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _configured_root() -> Path:
    configured = os.environ.get("CHECKSHEET_DATA_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        return path.resolve()
    return (Path(__file__).resolve().parent / "data" / "checksheets").resolve()


def _max_document_bytes() -> int:
    raw = os.environ.get("CHECKSHEET_MAX_DOCUMENT_MB", "200").strip()
    try:
        megabytes = max(1, min(2048, int(raw)))
    except ValueError:
        megabytes = 200
    return megabytes * 1024 * 1024


class ChecksheetStorage:
    """Owns all checksheet mutations and keeps browser state out of the model."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or _configured_root()).resolve()
        self.database_path = self.root / "checksheets.sqlite3"
        self.documents_dir = self.root / "documents"
        self._initialize_lock = Lock()
        self._initialized = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.initialize()
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=30)
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS source_documents (
                        id TEXT PRIMARY KEY,
                        sha256 TEXT NOT NULL,
                        original_name TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        file_type TEXT NOT NULL CHECK (file_type IN ('pdf', 'image')),
                        storage_name TEXT NOT NULL UNIQUE,
                        size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_source_documents_hash
                        ON source_documents (sha256, size_bytes);

                    CREATE TABLE IF NOT EXISTS checksheet_definitions (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        drawing_name TEXT NOT NULL,
                        source_project_id TEXT,
                        archived_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_checksheet_definitions_updated
                        ON checksheet_definitions (updated_at DESC);

                    CREATE TABLE IF NOT EXISTS checksheet_revisions (
                        id TEXT PRIMARY KEY,
                        checksheet_id TEXT NOT NULL
                            REFERENCES checksheet_definitions(id) ON DELETE CASCADE,
                        revision_number INTEGER NOT NULL CHECK (revision_number >= 1),
                        source_document_id TEXT NOT NULL
                            REFERENCES source_documents(id),
                        pdf_render_scale REAL NOT NULL CHECK (pdf_render_scale > 0),
                        created_at TEXT NOT NULL,
                        UNIQUE (checksheet_id, revision_number)
                    );

                    CREATE INDEX IF NOT EXISTS idx_checksheet_revisions_definition
                        ON checksheet_revisions (checksheet_id, revision_number DESC);

                    CREATE TABLE IF NOT EXISTS checksheet_columns (
                        id TEXT PRIMARY KEY,
                        revision_id TEXT NOT NULL
                            REFERENCES checksheet_revisions(id) ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        position INTEGER NOT NULL CHECK (position >= 0),
                        UNIQUE (revision_id, position)
                    );

                    CREATE TABLE IF NOT EXISTS checksheet_items (
                        id TEXT PRIMARY KEY,
                        revision_id TEXT NOT NULL
                            REFERENCES checksheet_revisions(id) ON DELETE CASCADE,
                        annotation_id TEXT NOT NULL,
                        balloon_number INTEGER NOT NULL CHECK (balloon_number >= 1),
                        page INTEGER NOT NULL CHECK (page >= 1),
                        bbox_x REAL NOT NULL,
                        bbox_y REAL NOT NULL,
                        bbox_width REAL NOT NULL CHECK (bbox_width > 0),
                        bbox_height REAL NOT NULL CHECK (bbox_height > 0),
                        rotation REAL NOT NULL,
                        label TEXT NOT NULL,
                        specification TEXT NOT NULL,
                        tolerance TEXT NOT NULL,
                        method TEXT NOT NULL,
                        tool TEXT NOT NULL,
                        dimension_type TEXT NOT NULL,
                        position INTEGER NOT NULL CHECK (position >= 0),
                        UNIQUE (revision_id, annotation_id),
                        UNIQUE (revision_id, balloon_number)
                    );

                    CREATE INDEX IF NOT EXISTS idx_checksheet_items_revision
                        ON checksheet_items (revision_id, position);

                    CREATE TABLE IF NOT EXISTS checksheet_runs (
                        id TEXT PRIMARY KEY,
                        checksheet_id TEXT NOT NULL
                            REFERENCES checksheet_definitions(id) ON DELETE CASCADE,
                        revision_id TEXT NOT NULL
                            REFERENCES checksheet_revisions(id) ON DELETE CASCADE,
                        run_number INTEGER NOT NULL CHECK (run_number >= 1),
                        status TEXT NOT NULL CHECK (status IN ('draft', 'completed')),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        UNIQUE (checksheet_id, run_number)
                    );

                    CREATE INDEX IF NOT EXISTS idx_checksheet_runs_definition
                        ON checksheet_runs (checksheet_id, updated_at DESC);

                    CREATE TABLE IF NOT EXISTS checksheet_readings (
                        run_id TEXT NOT NULL
                            REFERENCES checksheet_runs(id) ON DELETE CASCADE,
                        item_id TEXT NOT NULL
                            REFERENCES checksheet_items(id) ON DELETE CASCADE,
                        column_id TEXT NOT NULL
                            REFERENCES checksheet_columns(id) ON DELETE CASCADE,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, item_id, column_id)
                    );
                    """
                )
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    def _stage_document(self, source: BinaryIO) -> tuple[Path, str, int]:
        temporary = self.documents_dir / f".upload-{uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        maximum = _max_document_bytes()
        try:
            with temporary.open("wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > maximum:
                        raise ChecksheetValidationError(
                            "The source drawing exceeds the configured "
                            f"{maximum // (1024 * 1024)} MB limit."
                        )
                    digest.update(chunk)
                    destination.write(chunk)
            if size == 0:
                raise ChecksheetValidationError("The source drawing is empty.")
            return temporary, digest.hexdigest(), size
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _persist_staged_document(
        self,
        connection: sqlite3.Connection,
        *,
        temporary: Path,
        sha256: str,
        size_bytes: int,
        original_name: str,
        mime_type: str,
        file_type: str,
    ) -> tuple[StoredDocument, Path | None]:
        existing = connection.execute(
            """
            SELECT * FROM source_documents
            WHERE sha256 = ? AND size_bytes = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (sha256, size_bytes),
        ).fetchone()
        if existing is not None:
            temporary.unlink(missing_ok=True)
            return self._document_from_row(existing), None

        document_id = str(uuid4())
        safe_name = Path(original_name).name or "drawing"
        suffix = Path(safe_name).suffix.lower()
        if len(suffix) > 12 or not suffix.replace(".", "").isalnum():
            suffix = ".pdf" if file_type == "pdf" else ".img"
        storage_name = f"{document_id}{suffix}"
        final_path = self.documents_dir / storage_name
        temporary.replace(final_path)
        connection.execute(
            """
            INSERT INTO source_documents (
                id, sha256, original_name, mime_type, file_type,
                storage_name, size_bytes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                sha256,
                safe_name,
                mime_type or "application/octet-stream",
                file_type,
                storage_name,
                size_bytes,
                _utc_now(),
            ),
        )
        return (
            StoredDocument(
                id=document_id,
                original_name=safe_name,
                mime_type=mime_type or "application/octet-stream",
                file_type=file_type,
                path=final_path,
                size_bytes=size_bytes,
                sha256=sha256,
            ),
            final_path,
        )

    def _document_from_row(self, row: sqlite3.Row) -> StoredDocument:
        return StoredDocument(
            id=str(row["id"]),
            original_name=str(row["original_name"]),
            mime_type=str(row["mime_type"]),
            file_type=str(row["file_type"]),
            path=self.documents_dir / str(row["storage_name"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
        )

    def _insert_snapshot(
        self,
        connection: sqlite3.Connection,
        *,
        checksheet_id: str,
        revision_id: str,
        revision_number: int,
        document_id: str,
        snapshot: ChecksheetSnapshot,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO checksheet_revisions (
                id, checksheet_id, revision_number, source_document_id,
                pdf_render_scale, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                checksheet_id,
                revision_number,
                document_id,
                snapshot.pdf_render_scale,
                created_at,
            ),
        )
        for position, name in enumerate(snapshot.reading_columns):
            connection.execute(
                """
                INSERT INTO checksheet_columns (id, revision_id, name, position)
                VALUES (?, ?, ?, ?)
                """,
                (str(uuid4()), revision_id, name, position),
            )
        for position, item in enumerate(snapshot.items):
            connection.execute(
                """
                INSERT INTO checksheet_items (
                    id, revision_id, annotation_id, balloon_number, page,
                    bbox_x, bbox_y, bbox_width, bbox_height, rotation,
                    label, specification, tolerance, method, tool,
                    dimension_type, position
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    revision_id,
                    item.annotation_id,
                    item.balloon_number,
                    item.page,
                    item.bbox.x,
                    item.bbox.y,
                    item.bbox.width,
                    item.bbox.height,
                    item.rotation,
                    item.label,
                    item.specification,
                    item.tolerance,
                    item.method,
                    item.tool,
                    item.dimension_type,
                    position,
                ),
            )

    def _insert_run(
        self,
        connection: sqlite3.Connection,
        *,
        checksheet_id: str,
        revision_id: str,
        created_at: str,
    ) -> tuple[str, int]:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(run_number), 0) + 1 AS next_number
            FROM checksheet_runs
            WHERE checksheet_id = ?
            """,
            (checksheet_id,),
        ).fetchone()
        run_number = int(row["next_number"])
        run_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO checksheet_runs (
                id, checksheet_id, revision_id, run_number, status,
                created_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, 'draft', ?, ?, NULL)
            """,
            (
                run_id,
                checksheet_id,
                revision_id,
                run_number,
                created_at,
                created_at,
            ),
        )
        return run_id, run_number

    def create_checksheet(
        self,
        snapshot: ChecksheetSnapshot,
        *,
        source: BinaryIO,
        original_name: str,
        mime_type: str,
    ) -> dict[str, Any]:
        temporary, sha256, size_bytes = self._stage_document(source)
        created_document_path: Path | None = None
        checksheet_id = str(uuid4())
        revision_id = str(uuid4())
        run_id = ""
        now = _utc_now()
        try:
            with self._connect() as connection, connection:
                document, created_document_path = self._persist_staged_document(
                    connection,
                    temporary=temporary,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    original_name=original_name,
                    mime_type=mime_type,
                    file_type=snapshot.source_file_type,
                )
                connection.execute(
                    """
                    INSERT INTO checksheet_definitions (
                        id, name, drawing_name, source_project_id,
                        archived_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        checksheet_id,
                        snapshot.name,
                        snapshot.drawing_name,
                        snapshot.source_project_id,
                        now,
                        now,
                    ),
                )
                self._insert_snapshot(
                    connection,
                    checksheet_id=checksheet_id,
                    revision_id=revision_id,
                    revision_number=1,
                    document_id=document.id,
                    snapshot=snapshot,
                    created_at=now,
                )
                run_id, _ = self._insert_run(
                    connection,
                    checksheet_id=checksheet_id,
                    revision_id=revision_id,
                    created_at=now,
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            if created_document_path is not None:
                created_document_path.unlink(missing_ok=True)
            raise
        return self.get_run(checksheet_id, run_id)

    def create_revision(
        self,
        checksheet_id: str,
        snapshot: ChecksheetSnapshot,
        *,
        source: BinaryIO,
        original_name: str,
        mime_type: str,
    ) -> dict[str, Any]:
        temporary, sha256, size_bytes = self._stage_document(source)
        created_document_path: Path | None = None
        revision_id = str(uuid4())
        run_id = ""
        now = _utc_now()
        try:
            with self._connect() as connection, connection:
                definition = connection.execute(
                    "SELECT * FROM checksheet_definitions WHERE id = ?",
                    (checksheet_id,),
                ).fetchone()
                if definition is None:
                    raise ChecksheetNotFoundError("Checksheet not found.")
                if definition["archived_at"] is not None:
                    raise ChecksheetConflictError(
                        "Restore the checksheet before creating a revision."
                    )
                document, created_document_path = self._persist_staged_document(
                    connection,
                    temporary=temporary,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    original_name=original_name,
                    mime_type=mime_type,
                    file_type=snapshot.source_file_type,
                )
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(revision_number), 0) + 1 AS next_number
                    FROM checksheet_revisions
                    WHERE checksheet_id = ?
                    """,
                    (checksheet_id,),
                ).fetchone()
                revision_number = int(row["next_number"])
                self._insert_snapshot(
                    connection,
                    checksheet_id=checksheet_id,
                    revision_id=revision_id,
                    revision_number=revision_number,
                    document_id=document.id,
                    snapshot=snapshot,
                    created_at=now,
                )
                run_id, _ = self._insert_run(
                    connection,
                    checksheet_id=checksheet_id,
                    revision_id=revision_id,
                    created_at=now,
                )
                connection.execute(
                    """
                    UPDATE checksheet_definitions
                    SET name = ?, drawing_name = ?, source_project_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        snapshot.name,
                        snapshot.drawing_name,
                        snapshot.source_project_id,
                        now,
                        checksheet_id,
                    ),
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            if created_document_path is not None:
                created_document_path.unlink(missing_ok=True)
            raise
        return self.get_run(checksheet_id, run_id)

    def list_checksheets(
        self,
        *,
        search: str = "",
        archived: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = [
            "d.archived_at IS NOT NULL" if archived else "d.archived_at IS NULL"
        ]
        parameters: list[Any] = []
        if search.strip():
            clauses.append("(d.name LIKE ? OR d.drawing_name LIKE ?)")
            query = f"%{search.strip()}%"
            parameters.extend((query, query))
        where = " AND ".join(clauses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    d.*,
                    r.id AS revision_id,
                    r.revision_number,
                    (SELECT COUNT(*) FROM checksheet_items i
                        WHERE i.revision_id = r.id) AS row_count,
                    (SELECT COUNT(*) FROM checksheet_runs run
                        WHERE run.checksheet_id = d.id) AS run_count,
                    (SELECT COUNT(*) FROM checksheet_runs run
                        WHERE run.checksheet_id = d.id AND run.status = 'draft')
                        AS draft_count,
                    (SELECT COUNT(*) FROM checksheet_runs run
                        WHERE run.checksheet_id = d.id AND run.status = 'completed')
                        AS completed_count,
                    (SELECT run.id FROM checksheet_runs run
                        WHERE run.checksheet_id = d.id
                        ORDER BY run.updated_at DESC LIMIT 1) AS latest_run_id,
                    (SELECT run.status FROM checksheet_runs run
                        WHERE run.checksheet_id = d.id
                        ORDER BY run.updated_at DESC LIMIT 1) AS latest_run_status
                    ,(SELECT run.id FROM checksheet_runs run
                        WHERE run.checksheet_id = d.id AND run.status = 'draft'
                        ORDER BY run.updated_at DESC LIMIT 1) AS latest_draft_run_id
                FROM checksheet_definitions d
                JOIN checksheet_revisions r
                  ON r.checksheet_id = d.id
                 AND r.revision_number = (
                    SELECT MAX(r2.revision_number)
                    FROM checksheet_revisions r2
                    WHERE r2.checksheet_id = d.id
                 )
                WHERE {where}
                ORDER BY d.updated_at DESC
                """,
                parameters,
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def _summary_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "drawing_name": str(row["drawing_name"]),
            "source_project_id": row["source_project_id"],
            "archived": row["archived_at"] is not None,
            "archived_at": row["archived_at"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "revision_id": str(row["revision_id"]),
            "revision_number": int(row["revision_number"]),
            "row_count": int(row["row_count"]),
            "run_count": int(row["run_count"]),
            "draft_count": int(row["draft_count"]),
            "completed_count": int(row["completed_count"]),
            "latest_run_id": row["latest_run_id"],
            "latest_run_status": row["latest_run_status"],
            "latest_draft_run_id": row["latest_draft_run_id"],
        }

    def get_checksheet(self, checksheet_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            definition = connection.execute(
                "SELECT * FROM checksheet_definitions WHERE id = ?",
                (checksheet_id,),
            ).fetchone()
            if definition is None:
                raise ChecksheetNotFoundError("Checksheet not found.")
            revisions = connection.execute(
                """
                SELECT r.*,
                    (SELECT COUNT(*) FROM checksheet_items i
                        WHERE i.revision_id = r.id) AS row_count
                FROM checksheet_revisions r
                WHERE r.checksheet_id = ?
                ORDER BY r.revision_number DESC
                """,
                (checksheet_id,),
            ).fetchall()
            runs = connection.execute(
                """
                SELECT run.*, r.revision_number
                FROM checksheet_runs run
                JOIN checksheet_revisions r ON r.id = run.revision_id
                WHERE run.checksheet_id = ?
                ORDER BY run.run_number DESC
                """,
                (checksheet_id,),
            ).fetchall()
        return {
            "id": str(definition["id"]),
            "name": str(definition["name"]),
            "drawing_name": str(definition["drawing_name"]),
            "source_project_id": definition["source_project_id"],
            "archived": definition["archived_at"] is not None,
            "archived_at": definition["archived_at"],
            "created_at": str(definition["created_at"]),
            "updated_at": str(definition["updated_at"]),
            "revisions": [
                {
                    "id": str(row["id"]),
                    "revision_number": int(row["revision_number"]),
                    "row_count": int(row["row_count"]),
                    "created_at": str(row["created_at"]),
                }
                for row in revisions
            ],
            "runs": [
                {
                    "id": str(row["id"]),
                    "run_number": int(row["run_number"]),
                    "revision_id": str(row["revision_id"]),
                    "revision_number": int(row["revision_number"]),
                    "status": str(row["status"]),
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                    "completed_at": row["completed_at"],
                }
                for row in runs
            ],
        }

    def get_run(self, checksheet_id: str, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    run.*,
                    d.name AS checksheet_name,
                    d.drawing_name,
                    d.archived_at,
                    r.revision_number,
                    r.pdf_render_scale,
                    doc.id AS document_id,
                    doc.original_name,
                    doc.mime_type,
                    doc.file_type,
                    doc.size_bytes
                FROM checksheet_runs run
                JOIN checksheet_definitions d ON d.id = run.checksheet_id
                JOIN checksheet_revisions r ON r.id = run.revision_id
                JOIN source_documents doc ON doc.id = r.source_document_id
                WHERE run.id = ? AND run.checksheet_id = ?
                """,
                (run_id, checksheet_id),
            ).fetchone()
            if row is None:
                raise ChecksheetNotFoundError("Checksheet run not found.")
            columns = connection.execute(
                """
                SELECT id, name, position
                FROM checksheet_columns
                WHERE revision_id = ?
                ORDER BY position ASC
                """,
                (row["revision_id"],),
            ).fetchall()
            items = connection.execute(
                """
                SELECT * FROM checksheet_items
                WHERE revision_id = ?
                ORDER BY position ASC
                """,
                (row["revision_id"],),
            ).fetchall()
            reading_rows = connection.execute(
                """
                SELECT item_id, column_id, value
                FROM checksheet_readings
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
        readings: dict[str, dict[str, str]] = {}
        for reading in reading_rows:
            readings.setdefault(str(reading["item_id"]), {})[
                str(reading["column_id"])
            ] = str(reading["value"])
        return {
            "checksheet": {
                "id": checksheet_id,
                "name": str(row["checksheet_name"]),
                "drawing_name": str(row["drawing_name"]),
                "archived": row["archived_at"] is not None,
            },
            "revision": {
                "id": str(row["revision_id"]),
                "revision_number": int(row["revision_number"]),
                "pdf_render_scale": float(row["pdf_render_scale"]),
            },
            "run": {
                "id": str(row["id"]),
                "run_number": int(row["run_number"]),
                "status": str(row["status"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "completed_at": row["completed_at"],
            },
            "document": {
                "id": str(row["document_id"]),
                "file_name": str(row["original_name"]),
                "mime_type": str(row["mime_type"]),
                "file_type": str(row["file_type"]),
                "size_bytes": int(row["size_bytes"]),
            },
            "columns": [
                {
                    "id": str(column["id"]),
                    "name": str(column["name"]),
                    "position": int(column["position"]),
                }
                for column in columns
            ],
            "rows": [
                {
                    "id": str(item["id"]),
                    "annotation_id": str(item["annotation_id"]),
                    "balloon_number": int(item["balloon_number"]),
                    "page": int(item["page"]),
                    "bbox": {
                        "x": float(item["bbox_x"]),
                        "y": float(item["bbox_y"]),
                        "width": float(item["bbox_width"]),
                        "height": float(item["bbox_height"]),
                    },
                    "rotation": float(item["rotation"]),
                    "label": str(item["label"]),
                    "specification": str(item["specification"]),
                    "tolerance": str(item["tolerance"]),
                    "method": str(item["method"]),
                    "tool": str(item["tool"]),
                    "dimension_type": str(item["dimension_type"]),
                    "readings": readings.get(str(item["id"]), {}),
                }
                for item in items
            ],
        }

    def start_run(
        self,
        checksheet_id: str,
        revision_id: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._connect() as connection, connection:
            definition = connection.execute(
                "SELECT * FROM checksheet_definitions WHERE id = ?",
                (checksheet_id,),
            ).fetchone()
            if definition is None:
                raise ChecksheetNotFoundError("Checksheet not found.")
            if definition["archived_at"] is not None:
                raise ChecksheetConflictError(
                    "Restore the checksheet before starting an inspection."
                )
            if revision_id is None:
                revision = connection.execute(
                    """
                    SELECT id FROM checksheet_revisions
                    WHERE checksheet_id = ?
                    ORDER BY revision_number DESC LIMIT 1
                    """,
                    (checksheet_id,),
                ).fetchone()
            else:
                revision = connection.execute(
                    """
                    SELECT id FROM checksheet_revisions
                    WHERE id = ? AND checksheet_id = ?
                    """,
                    (revision_id, checksheet_id),
                ).fetchone()
            if revision is None:
                raise ChecksheetNotFoundError("Checksheet revision not found.")
            run_id, _ = self._insert_run(
                connection,
                checksheet_id=checksheet_id,
                revision_id=str(revision["id"]),
                created_at=now,
            )
            connection.execute(
                "UPDATE checksheet_definitions SET updated_at = ? WHERE id = ?",
                (now, checksheet_id),
            )
        return self.get_run(checksheet_id, run_id)

    def update_readings(
        self,
        checksheet_id: str,
        run_id: str,
        patches: Sequence[ReadingPatch],
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._connect() as connection, connection:
            run = connection.execute(
                """
                SELECT run.*, definition.archived_at
                FROM checksheet_runs run
                JOIN checksheet_definitions definition
                  ON definition.id = run.checksheet_id
                WHERE run.id = ? AND run.checksheet_id = ?
                """,
                (run_id, checksheet_id),
            ).fetchone()
            if run is None:
                raise ChecksheetNotFoundError("Checksheet run not found.")
            if run["archived_at"] is not None:
                raise ChecksheetConflictError(
                    "Restore the checksheet before editing its draft."
                )
            if run["status"] != "draft":
                raise ChecksheetConflictError(
                    "Completed inspection runs cannot be edited."
                )
            item_rows = connection.execute(
                """
                SELECT id, annotation_id FROM checksheet_items
                WHERE revision_id = ?
                """,
                (run["revision_id"],),
            ).fetchall()
            column_rows = connection.execute(
                """
                SELECT id FROM checksheet_columns
                WHERE revision_id = ?
                """,
                (run["revision_id"],),
            ).fetchall()
            item_ids = {
                str(item["annotation_id"]): str(item["id"])
                for item in item_rows
            }
            column_ids = {str(column["id"]) for column in column_rows}
            resolved: dict[tuple[str, str], str] = {}
            for patch in patches:
                item_id = item_ids.get(patch.annotation_id)
                if item_id is None:
                    raise ChecksheetValidationError(
                        f"Annotation {patch.annotation_id!r} is not part of this revision."
                    )
                if patch.column_id not in column_ids:
                    raise ChecksheetValidationError(
                        f"Reading column {patch.column_id!r} is not part of this revision."
                    )
                resolved[(item_id, patch.column_id)] = patch.value
            for (item_id, column_id), value in resolved.items():
                connection.execute(
                    """
                    INSERT INTO checksheet_readings (
                        run_id, item_id, column_id, value, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (run_id, item_id, column_id)
                    DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (run_id, item_id, column_id, value, now),
                )
            connection.execute(
                "UPDATE checksheet_runs SET updated_at = ? WHERE id = ?",
                (now, run_id),
            )
            connection.execute(
                "UPDATE checksheet_definitions SET updated_at = ? WHERE id = ?",
                (now, checksheet_id),
            )
        return {"status": "saved", "updated_at": now}

    def complete_run(self, checksheet_id: str, run_id: str) -> dict[str, Any]:
        now = _utc_now()
        with self._connect() as connection, connection:
            run = connection.execute(
                """
                SELECT run.*, definition.archived_at
                FROM checksheet_runs run
                JOIN checksheet_definitions definition
                  ON definition.id = run.checksheet_id
                WHERE run.id = ? AND run.checksheet_id = ?
                """,
                (run_id, checksheet_id),
            ).fetchone()
            if run is None:
                raise ChecksheetNotFoundError("Checksheet run not found.")
            if run["archived_at"] is not None:
                raise ChecksheetConflictError(
                    "Restore the checksheet before completing its draft."
                )
            if run["status"] == "draft":
                connection.execute(
                    """
                    UPDATE checksheet_runs
                    SET status = 'completed', updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (now, now, run_id),
                )
                connection.execute(
                    "UPDATE checksheet_definitions SET updated_at = ? WHERE id = ?",
                    (now, checksheet_id),
                )
        return self.get_run(checksheet_id, run_id)

    def update_checksheet(
        self,
        checksheet_id: str,
        *,
        name: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        if name is None and archived is None:
            return self.get_checksheet(checksheet_id)
        now = _utc_now()
        assignments = ["updated_at = ?"]
        parameters: list[Any] = [now]
        if name is not None:
            assignments.append("name = ?")
            parameters.append(name)
        if archived is not None:
            assignments.append("archived_at = ?")
            parameters.append(now if archived else None)
        parameters.append(checksheet_id)
        with self._connect() as connection, connection:
            cursor = connection.execute(
                f"UPDATE checksheet_definitions SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
            if cursor.rowcount == 0:
                raise ChecksheetNotFoundError("Checksheet not found.")
        return self.get_checksheet(checksheet_id)

    def duplicate_checksheet(
        self,
        checksheet_id: str,
        *,
        name: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        new_checksheet_id = str(uuid4())
        new_revision_id = str(uuid4())
        with self._connect() as connection, connection:
            definition = connection.execute(
                "SELECT * FROM checksheet_definitions WHERE id = ?",
                (checksheet_id,),
            ).fetchone()
            if definition is None:
                raise ChecksheetNotFoundError("Checksheet not found.")
            revision = connection.execute(
                """
                SELECT * FROM checksheet_revisions
                WHERE checksheet_id = ?
                ORDER BY revision_number DESC LIMIT 1
                """,
                (checksheet_id,),
            ).fetchone()
            if revision is None:
                raise ChecksheetNotFoundError("Checksheet revision not found.")
            duplicate_name = name or f"{definition['name']} Copy"
            connection.execute(
                """
                INSERT INTO checksheet_definitions (
                    id, name, drawing_name, source_project_id,
                    archived_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    new_checksheet_id,
                    duplicate_name,
                    definition["drawing_name"],
                    definition["source_project_id"],
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO checksheet_revisions (
                    id, checksheet_id, revision_number, source_document_id,
                    pdf_render_scale, created_at
                ) VALUES (?, ?, 1, ?, ?, ?)
                """,
                (
                    new_revision_id,
                    new_checksheet_id,
                    revision["source_document_id"],
                    revision["pdf_render_scale"],
                    now,
                ),
            )
            columns = connection.execute(
                """
                SELECT * FROM checksheet_columns
                WHERE revision_id = ? ORDER BY position
                """,
                (revision["id"],),
            ).fetchall()
            for column in columns:
                connection.execute(
                    """
                    INSERT INTO checksheet_columns (id, revision_id, name, position)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        new_revision_id,
                        column["name"],
                        column["position"],
                    ),
                )
            items = connection.execute(
                """
                SELECT * FROM checksheet_items
                WHERE revision_id = ? ORDER BY position
                """,
                (revision["id"],),
            ).fetchall()
            for item in items:
                connection.execute(
                    """
                    INSERT INTO checksheet_items (
                        id, revision_id, annotation_id, balloon_number, page,
                        bbox_x, bbox_y, bbox_width, bbox_height, rotation,
                        label, specification, tolerance, method, tool,
                        dimension_type, position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        new_revision_id,
                        item["annotation_id"],
                        item["balloon_number"],
                        item["page"],
                        item["bbox_x"],
                        item["bbox_y"],
                        item["bbox_width"],
                        item["bbox_height"],
                        item["rotation"],
                        item["label"],
                        item["specification"],
                        item["tolerance"],
                        item["method"],
                        item["tool"],
                        item["dimension_type"],
                        item["position"],
                    ),
                )
            run_id, _ = self._insert_run(
                connection,
                checksheet_id=new_checksheet_id,
                revision_id=new_revision_id,
                created_at=now,
            )
        return self.get_run(new_checksheet_id, run_id)

    def delete_checksheet_permanently(self, checksheet_id: str) -> None:
        document_ids: list[str] = []
        removable_documents: list[Path] = []
        with self._connect() as connection, connection:
            definition = connection.execute(
                "SELECT * FROM checksheet_definitions WHERE id = ?",
                (checksheet_id,),
            ).fetchone()
            if definition is None:
                raise ChecksheetNotFoundError("Checksheet not found.")
            if definition["archived_at"] is None:
                raise ChecksheetConflictError(
                    "Archive the checksheet before permanently deleting it."
                )
            document_ids = [
                str(row["source_document_id"])
                for row in connection.execute(
                    """
                    SELECT DISTINCT source_document_id
                    FROM checksheet_revisions
                    WHERE checksheet_id = ?
                    """,
                    (checksheet_id,),
                ).fetchall()
            ]
            connection.execute(
                "DELETE FROM checksheet_definitions WHERE id = ?",
                (checksheet_id,),
            )
            for document_id in document_ids:
                referenced = connection.execute(
                    """
                    SELECT 1 FROM checksheet_revisions
                    WHERE source_document_id = ? LIMIT 1
                    """,
                    (document_id,),
                ).fetchone()
                if referenced is not None:
                    continue
                document = connection.execute(
                    "SELECT * FROM source_documents WHERE id = ?",
                    (document_id,),
                ).fetchone()
                if document is not None:
                    removable_documents.append(
                        self.documents_dir / str(document["storage_name"])
                    )
                    connection.execute(
                        "DELETE FROM source_documents WHERE id = ?",
                        (document_id,),
                    )
        for path in removable_documents:
            path.unlink(missing_ok=True)

    def get_document(self, document_id: str) -> StoredDocument:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            raise ChecksheetNotFoundError("Source document not found.")
        document = self._document_from_row(row)
        if not document.path.is_file():
            raise ChecksheetStorageError(
                "The checksheet source document is missing from backend storage."
            )
        return document


_storage: ChecksheetStorage | None = None
_storage_lock = Lock()


def get_checksheet_storage() -> ChecksheetStorage:
    global _storage
    if _storage is None:
        with _storage_lock:
            if _storage is None:
                _storage = ChecksheetStorage()
    return _storage


def initialize_checksheet_storage() -> None:
    get_checksheet_storage().initialize()
