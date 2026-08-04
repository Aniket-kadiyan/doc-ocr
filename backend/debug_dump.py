"""
Step-by-step pipeline dump for inspection.

Set ``DEBUG_DUMP=1`` on the **server process** to write every stage under
``backend/debug_output/<sha1-of-upload>/``.

Other controls:
- ``DEBUG_DUMP=0`` or unset → off (default)
- ``DEBUG_DUMP_FORCE=1`` → re-dump even if ``.done`` exists for that crop
- Per-request: ``?debug_dump=1`` or header ``X-Debug-Dump: 1`` (no restart needed)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from debug_steps_catalog import GLOSSARY, lookup_step

logger = logging.getLogger(__name__)

DUMP_ENV = "DEBUG_DUMP"
FORCE_ENV = "DEBUG_DUMP_FORCE"
DEFAULT_ROOT = Path(__file__).resolve().parent / "debug_output"
_DONE = ".done"
_STEP_FILE_RE = re.compile(r"^(\d{2})_(.+)\.(png|json)$")
_TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in _TRUTHY


def dump_enabled() -> bool:
    """True when DEBUG_DUMP env is truthy on the server process."""
    return _is_truthy(os.environ.get(DUMP_ENV))


def dump_force() -> bool:
    """True when DEBUG_DUMP_FORCE=1 — overwrite existing .done folders."""
    return _is_truthy(os.environ.get(FORCE_ENV))


def should_dump(*, request_override: bool = False) -> bool:
    """Env flag OR per-request ?debug_dump=1 / X-Debug-Dump: 1."""
    return dump_enabled() or request_override


def dump_status() -> dict[str, Any]:
    return {
        "enabled": dump_enabled(),
        "force": dump_force(),
        "env": DUMP_ENV,
        "force_env": FORCE_ENV,
        "root": str(DEFAULT_ROOT),
        "per_request": "POST /ocr/recognize?debug_dump=1 or header X-Debug-Dump: 1",
    }


def dump_skip_reason(
    *,
    enabled: bool,
    already_done: bool,
    force: bool,
) -> str | None:
    if not enabled:
        return (
            "DEBUG_DUMP not set on server. Restart with: DEBUG_DUMP=1 npm run ocr-api:debug "
            "— or add ?debug_dump=1 to the recognize request."
        )
    if already_done and not force:
        return (
            "This crop was already dumped (.done exists). Delete its folder under "
            "backend/debug_output/ or set DEBUG_DUMP_FORCE=1 to re-dump."
        )
    return None


def hash_key(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def _parse_step_filename(fname: str) -> tuple[int, str, str]:
    m = _STEP_FILE_RE.match(fname)
    if m:
        return int(m.group(1)), m.group(2), m.group(3)
    return 0, fname, "other"


class StepDumper:
    """Collects ordered stage artifacts for a single input image."""

    def __init__(
        self,
        key: str,
        root: Path | None = None,
        enabled: bool = True,
        force: bool = False,
    ) -> None:
        self._enabled = enabled
        self._force = force
        self.dir = (root or DEFAULT_ROOT) / key
        self.already_done = (self.dir / _DONE).exists()
        self.skip_reason = dump_skip_reason(
            enabled=enabled,
            already_done=self.already_done,
            force=force,
        )
        self._index: list[str] = []
        self._records: list[dict[str, Any]] = []
        self._step = 0
        if self.active:
            self.dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        root: Path | None = None,
        enabled: bool = True,
        force: bool = False,
    ) -> "StepDumper":
        return cls(hash_key(data), root=root, enabled=enabled, force=force)

    @property
    def active(self) -> bool:
        if not self._enabled:
            return False
        if self._force:
            return True
        return not self.already_done

    def _record(
        self,
        fname: str,
        name: str,
        kind: str,
        summary: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        info = lookup_step(name)
        step_no, _, file_kind = _parse_step_filename(fname)
        rec: dict[str, Any] = {
            "file": fname,
            "step": step_no,
            "name": name,
            "type": file_kind if file_kind != "other" else kind,
            "module": info.get("module", ""),
            "use_case": info.get("use_case", ""),
            "values": info.get("values", {}),
        }
        if summary:
            rec["value_summary"] = summary
        if extra:
            rec.update(extra)
        self._records.append(rec)

    def _ensure_dir(self) -> bool:
        """(Re)create the dump dir before a write.

        The folder can vanish mid-request when it lives under a cloud-synced
        path (iCloud/Google Drive on ``Desktop``) that evicts freshly created
        directories. Recreate it on demand so dumping never 500s the caller.
        """
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as exc:
            logger.warning("debug dump: could not create %s: %s", self.dir, exc)
            return False

    def image(
        self,
        name: str,
        img: Image.Image,
        summary: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        self._step += 1
        fname = f"{self._step:02d}_{name}.png"
        if not self._ensure_dir():
            return
        try:
            img.convert("RGB").save(self.dir / fname)
        except OSError as exc:
            logger.warning("debug dump: failed writing %s: %s", fname, exc)
            return
        self._index.append(fname)
        base_summary = {"width": img.width, "height": img.height}
        if summary:
            base_summary.update(summary)
        self._record(fname, name, "image", summary=base_summary)

    def json(self, name: str, data: Any) -> None:
        if not self.active:
            return
        fname = f"{name}.json"
        if not self._write_text(fname, json.dumps(data, indent=2, ensure_ascii=False, default=str)):
            return
        if name == "result":
            info = lookup_step("result")
            self._records.append(
                {
                    "file": fname,
                    "step": None,
                    "name": "result",
                    "type": "json",
                    "module": info.get("module", ""),
                    "use_case": info.get("use_case", ""),
                    "values": info.get("values", {}),
                    "value_summary": {
                        "text": data.get("text") if isinstance(data, dict) else None,
                        "raw_ocr": data.get("raw_ocr") if isinstance(data, dict) else None,
                        "type": data.get("type") if isinstance(data, dict) else None,
                    },
                }
            )

    def _write_text(self, fname: str, text: str) -> bool:
        """Best-effort text write; returns False if the dir/file is unwritable."""
        if not self._ensure_dir():
            return False
        try:
            # Always UTF-8: dumps use ensure_ascii=False and can contain
            # engineering/GD&T symbols (Ø, ±, ≥) that crash on Windows'
            # default cp1252 codec.
            (self.dir / fname).write_text(text, encoding="utf-8")
            return True
        except OSError as exc:
            logger.warning("debug dump: failed writing %s: %s", fname, exc)
            return False

    def stage(self, name: str, data: Any, summary: dict[str, Any] | None = None) -> None:
        if not self.active:
            return
        self._step += 1
        fname = f"{self._step:02d}_{name}.json"
        if not self._write_text(fname, json.dumps(data, indent=2, ensure_ascii=False, default=str)):
            return
        self._index.append(fname)
        self._record(fname, name, "json", summary=summary or _summarize_stage(name, data))

    def finalize(self, result: dict[str, Any] | None = None) -> str | None:
        if not self.active:
            return None
        if result is not None:
            self.json("result", result)
        self._write_text(
            "steps.json",
            json.dumps(
                {"order": self._index, "steps": self._records, "glossary": GLOSSARY},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
        )
        self._write_text(_DONE, "")  # _write_text already forces UTF-8
        return str(self.dir)


def dump_segment(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    *,
    enabled: bool,
    force: bool = False,
) -> str | None:
    """
    Capture the whole-selection view of auto-segment for diagnosis.

    Writes, under ``debug_output/segment_<sha1>/``:
      - ``input.png``     — the exact image segmentation received
      - ``detect.png``    — every detect_regions proposal (cyan, numbered)
      - ``clusters.png``  — final per-dimension clusters (red, numbered)
      - ``regions.json``  — coordinates + text for both, so a *missed* value
                            (no box at all) is distinguishable from a *misread*
                            one (box present, empty/garbled text).

    Unlike the per-crop StepDumper this runs once per selection, so a value that
    detection drops still leaves a trace (its absence from detect.png).
    """
    if not enabled:
        return None
    key = "segment_" + hash_key(image.tobytes())
    out_dir = DEFAULT_ROOT / key
    done = out_dir / _DONE
    if done.exists() and not force:
        return str(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(out_dir / "input.png")

        det = image.convert("RGB").copy()
        dd = ImageDraw.Draw(det)
        for i, b in enumerate(boxes):
            dd.rectangle(
                [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]],
                outline=(0, 200, 220),
                width=2,
            )
            dd.text((b["x"] + 1, max(0, b["y"] - 10)), str(i), fill=(0, 150, 170))
        det.save(out_dir / "detect.png")

        clu = image.convert("RGB").copy()
        cd = ImageDraw.Draw(clu)
        for i, c in enumerate(clusters):
            cd.rectangle(
                [c["x"], c["y"], c["x"] + c["width"], c["y"] + c["height"]],
                outline=(220, 30, 30),
                width=2,
            )
            cd.text((c["x"] + 1, max(0, c["y"] - 10)), str(i + 1), fill=(180, 20, 20))
        clu.save(out_dir / "clusters.png")

        (out_dir / "regions.json").write_text(
            json.dumps(
                {
                    "image_size": list(image.size),
                    "detect_count": len(boxes),
                    "cluster_count": len(clusters),
                    "detect_boxes": boxes,
                    "clusters": clusters,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        done.write_text("", encoding="utf-8")
        return str(out_dir)
    except OSError as exc:
        logger.warning("segment dump: failed under %s: %s", out_dir, exc)
        return None


def _summarize_stage(name: str, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    if name == "symbol_vision":
        return {
            "phi_detected": data.get("phi_detected"),
            "phi_score": data.get("phi_score"),
            "plus_minus_detected": (data.get("symbols_merged") or {}).get("plus_minus"),
            "diameter_score": (data.get("symbols_merged") or {}).get("diameter_score"),
            "prefix_ocr": data.get("prefix_ocr"),
        }
    if name == "paddle_winner":
        return {k: data.get(k) for k in ("text", "confidence", "agreement", "corrected")}
    if name == "compose_output":
        return {k: data.get(k) for k in ("text", "kind", "applied")}
    if name == "compose_input":
        return {
            "raw_ocr": data.get("raw_ocr"),
            "prefix_ocr": data.get("prefix_ocr"),
            "vertical": data.get("vertical"),
        }
    if name == "00_meta":
        return {k: data.get(k) for k in ("image_size", "vertical", "dump_dir")}
    return {}
