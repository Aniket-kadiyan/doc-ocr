"""
Durable record of operator corrections to GD&T detections.

What this is for
----------------
An operator who sees the recognizer name a glyph wrongly is the best labelling
source the system will ever have: they are looking at the drawing, they know
the part, and they are already in the loop. Capturing that correction turns a
one-off fix into an evaluation set and, eventually, training data —
``docs/GDT_DATASET.md`` describes the workflow this feeds.

Append-only, JSON Lines
-----------------------
A correction is an observation, not a record to be edited. Appending means a
crash mid-write loses at most the last line, concurrent writers interleave
safely at line granularity, and the history of what a detection was corrected
*to* and *from* survives — which is what makes it useful later. Reading back a
corrupt line skips it rather than failing the file.

Retention
---------
Each line carries only what the correction is about: the symbol, the box, the
page, and an optional document hash. **No image data and no file contents.**
The document is identified by a hash the caller supplies, so a correction can
be tied back to a drawing without this store holding the drawing — which keeps
it inside the same retention posture as the checksheet store rather than
creating a second copy of customer data.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config_env import get_str

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()

#: What the operator did.
ACTION_RECLASSIFY = "reclassify"   # it is a different symbol
ACTION_REJECT = "reject"           # it is not a symbol at all
ACTION_CONFIRM = "confirm"         # the detection was right
ACTION_ADD = "add"                 # a symbol was missed entirely

ACTIONS = frozenset(
    {ACTION_RECLASSIFY, ACTION_REJECT, ACTION_CONFIRM, ACTION_ADD}
)


class CorrectionError(ValueError):
    """Raised for a correction the store will not accept."""


def corrections_dir() -> Path:
    """Where corrections are written.

    Follows the ``CHECKSHEET_DATA_DIR`` convention: an absolute path is used
    as given, a relative one resolves against the repository root.
    """
    configured = get_str("GDT_CORRECTIONS_DIR", "backend/data/gdt_corrections")
    path = Path(configured)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _log_path() -> Path:
    return corrections_dir() / "corrections.jsonl"


@dataclass
class Correction:
    """One operator judgement about one detection."""

    action: str
    #: What the recognizer said. ``None`` when the operator added a symbol it
    #: missed entirely.
    detected_symbol_id: str | None = None
    #: What it should have been. ``None`` for a rejection.
    corrected_symbol_id: str | None = None
    page: int = 1
    bounding_box: dict[str, float] | None = None
    #: Confidence the recognizer published, kept so the threshold behaviour
    #: can be analysed against outcomes later.
    detected_confidence: float | None = None
    detected_status: str | None = None
    #: Opaque identifier for the drawing — a content hash the caller supplies.
    #: Never a file path, never the file.
    document_hash: str | None = None
    #: Free-text note from the operator.
    note: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def validate(self) -> None:
        import gdt_registry as reg

        if self.action not in ACTIONS:
            raise CorrectionError(
                f"action must be one of {sorted(ACTIONS)}, got {self.action!r}"
            )
        for name, symbol_id in (
            ("detected_symbol_id", self.detected_symbol_id),
            ("corrected_symbol_id", self.corrected_symbol_id),
        ):
            if symbol_id is not None and symbol_id not in reg.BY_ID:
                raise CorrectionError(f"{name}: unknown symbol id {symbol_id!r}")
        if self.action == ACTION_RECLASSIFY and not self.corrected_symbol_id:
            raise CorrectionError("a reclassification needs corrected_symbol_id")
        if self.action == ACTION_ADD and not self.corrected_symbol_id:
            raise CorrectionError("an addition needs corrected_symbol_id")
        if self.action == ACTION_REJECT and not self.detected_symbol_id:
            raise CorrectionError("a rejection needs the detection it rejects")
        if self.page < 1:
            raise CorrectionError("page must be 1 or greater")
        if self.bounding_box is not None:
            missing = {"x", "y", "width", "height"} - set(self.bounding_box)
            if missing:
                raise CorrectionError(f"bounding_box is missing {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record(correction: Correction) -> Correction:
    """Validate and append one correction. Returns it with its id filled in."""
    correction.validate()
    path = _log_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(correction.to_dict(), ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    return correction


def read_all(*, document_hash: str | None = None) -> list[dict[str, Any]]:
    """Every correction on file, newest last.

    A line that will not parse is skipped rather than failing the read: a
    truncated final line from an interrupted write must not make the whole
    history unreadable.
    """
    path = _log_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if document_hash and entry.get("document_hash") != document_hash:
                continue
            out.append(entry)
    return out


def summary() -> dict[str, Any]:
    """Counts by action and by symbol, for the dataset and for triage.

    The per-symbol confusion counts are the directly useful part: they say
    which pairs the recognizer actually gets wrong in service, which is a
    better guide to what to improve than any synthetic benchmark.
    """
    entries = read_all()
    by_action: dict[str, int] = {}
    confusions: dict[str, int] = {}
    for entry in entries:
        action = entry.get("action", "unknown")
        by_action[action] = by_action.get(action, 0) + 1
        if action == ACTION_RECLASSIFY:
            key = f"{entry.get('detected_symbol_id')}->{entry.get('corrected_symbol_id')}"
            confusions[key] = confusions.get(key, 0) + 1
    return {
        "total": len(entries),
        "by_action": by_action,
        "confusions": dict(
            sorted(confusions.items(), key=lambda kv: -kv[1])
        ),
        "path": str(_log_path()),
    }
