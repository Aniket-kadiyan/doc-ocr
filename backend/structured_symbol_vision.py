"""Geometry-backed completion of structured engineering symbols.

OCR reads the letters and numbers inside a callout, while the thin lines that
give those characters meaning are often omitted.  This module combines source
pixels with OCR records to recover feature-control frames, boxed datums, and
surface-finish values.  Unresolved frames are returned as incomplete evidence
so the disposition policy sends them to Review rather than dropping them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from PIL import Image


BBox = Mapping[str, float]
StructuredSymbolKind = Literal["feature_control_frame", "datum", "surface_finish"]
_NUMBER_RE = re.compile(r"\d")
_DATUM_RE = re.compile(r"^[A-Z]$")
_SURFACE_RE = re.compile(
    r"^(?:(?:RA|RZ|RMAX|RQ)\s*)?\d+(?:\.\d+)?(?:\s*(?:µM|UM))?$"
    r"|^N\s?(?:1[0-2]|[1-9])$",
    re.IGNORECASE,
)
_EXPLICIT_SURFACE_RE = re.compile(
    r"^(?:RA|RZ|RMAX|RQ)\s*\d|^N\s?(?:1[0-2]|[1-9])$", re.IGNORECASE
)
_GDT_GLYPHS: dict[str, tuple[str, str]] = {
    "⏤": ("⏤", "Straightness"), "⏥": ("⏥", "Flatness"),
    "▱": ("⏥", "Flatness"), "○": ("○", "Circularity"),
    "◯": ("○", "Circularity"), "⌭": ("⌭", "Cylindricity"),
    "⌒": ("⌒", "Profile of a Line"), "⌓": ("⌓", "Profile of a Surface"),
    "∥": ("∥", "Parallelism"), "//": ("∥", "Parallelism"),
    "⟂": ("⟂", "Perpendicularity"), "⊥": ("⟂", "Perpendicularity"),
    "∠": ("∠", "Angularity"), "⌖": ("⌖", "Position"),
    "◎": ("◎", "Concentricity"), "⌯": ("⌯", "Symmetry"),
    "↗": ("↗", "Runout"), "⌰": ("⌰", "Total Runout"),
}
_GDT_KEYWORDS: dict[str, tuple[str, str]] = {
    "STRAIGHTNESS": ("⏤", "Straightness"), "FLATNESS": ("⏥", "Flatness"),
    "CIRCULARITY": ("○", "Circularity"), "ROUNDNESS": ("○", "Circularity"),
    "CYLINDRICITY": ("⌭", "Cylindricity"), "PARALLELISM": ("∥", "Parallelism"),
    "PERPENDICULARITY": ("⟂", "Perpendicularity"),
    "SQUARENESS": ("⟂", "Perpendicularity"), "ANGULARITY": ("∠", "Angularity"),
    "POSITION": ("⌖", "Position"), "CONCENTRICITY": ("◎", "Concentricity"),
    "SYMMETRY": ("⌯", "Symmetry"), "RUNOUT": ("↗", "Runout"),
}


@dataclass(frozen=True)
class StructuredSymbolEvidence:
    kind: StructuredSymbolKind
    bbox: dict[str, float]
    confidence: float
    complete: bool
    subtype: str | None = None
    completed_symbol: str | None = None
    source: str = "geometry"
    cells: tuple[dict[str, float], ...] = ()
    warnings: tuple[str, ...] = ()
    visual_features: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "kind": self.kind, "bbox": dict(self.bbox),
            "confidence": self.confidence, "complete": self.complete,
            "subtype": self.subtype, "completed_symbol": self.completed_symbol,
            "source": self.source, "cells": [dict(cell) for cell in self.cells],
            "warnings": list(self.warnings), "visual_features": dict(self.visual_features),
        }


@dataclass(frozen=True)
class StructuredObjectPlan:
    member_indexes: tuple[int, ...]
    anchor_index: int
    object_id: str
    text: str
    bbox: dict[str, float]
    confidence: float
    evidence: StructuredSymbolEvidence
    needs_review: bool
    review_reason: str


@dataclass(frozen=True)
class _Segment:
    position: int
    start: int
    end: int
    thickness: int = 1

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class _Frame:
    bbox: dict[str, float]
    cells: tuple[dict[str, float], ...]


def _box(record: Mapping[str, Any]) -> dict[str, float]:
    value = record.get("bbox") if isinstance(record.get("bbox"), Mapping) else record
    return {key: float(value[key]) for key in ("x", "y", "width", "height")}  # type: ignore[index]


def _text(record: Mapping[str, Any]) -> str:
    result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
    return str(record.get("text") or result.get("text") or "").strip()


def _confidence(record: Mapping[str, Any]) -> float:
    result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
    return float(result.get("confidence") or record.get("confidence") or 0.0)


def _center(box: BBox) -> tuple[float, float]:
    return float(box["x"]) + float(box["width"]) / 2, float(box["y"]) + float(box["height"]) / 2


def _inside(box: BBox, point: tuple[float, float], pad: float = 0.0) -> bool:
    x, y = point
    return (float(box["x"]) - pad <= x <= float(box["x"]) + float(box["width"]) + pad
            and float(box["y"]) - pad <= y <= float(box["y"]) + float(box["height"]) + pad)


def _ink(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if not gray.size:
        return np.zeros((0, 0), dtype=bool)
    threshold = min(225, max(120, int(np.percentile(gray, 10)) + 70))
    return gray < threshold


def _runs(values: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(values)
    if not idx.size:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]
    ends = np.r_[starts[1:] - 1, idx.size - 1]
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)
            if int(idx[b] - idx[a] + 1) >= minimum]


def _segments(ink: np.ndarray, *, horizontal: bool, minimum: int) -> list[_Segment]:
    raw: list[_Segment] = []
    for pos in range(ink.shape[0] if horizontal else ink.shape[1]):
        values = ink[pos, :] if horizontal else ink[:, pos]
        raw.extend(_Segment(pos, a, b) for a, b in _runs(values, minimum))
    merged: list[_Segment] = []
    for segment in sorted(raw, key=lambda value: (value.position, value.start)):
        found = next((i for i, current in enumerate(merged)
                      if segment.position <= current.position + current.thickness + 1
                      and min(segment.end, current.end) - max(segment.start, current.start)
                      >= 0.65 * min(segment.length, current.length)), None)
        if found is None:
            merged.append(segment)
        else:
            current = merged[found]
            merged[found] = _Segment(
                min(current.position, segment.position), min(current.start, segment.start),
                max(current.end, segment.end),
                max(current.position + current.thickness, segment.position + segment.thickness)
                - min(current.position, segment.position),
            )
    return merged


def _text_height(records: Sequence[Mapping[str, Any]], page_height: int) -> float:
    heights = []
    for record in records:
        if record.get("table_excluded") or not _text(record):
            continue
        try:
            height = _box(record)["height"]
        except (KeyError, TypeError, ValueError):
            continue
        if 3 <= height <= 0.25 * max(page_height, 1):
            heights.append(height)
    heights.sort()
    return max(6.0, heights[len(heights) // 2]) if heights else max(8.0, page_height * 0.012)


def _detect_frames(binary: np.ndarray, text_height: float) -> list[_Frame]:
    if not binary.size:
        return []
    horizontal = _segments(binary, horizontal=True, minimum=max(12, round(1.5 * text_height)))
    vertical = _segments(binary, horizontal=False, minimum=max(5, round(0.5 * text_height)))
    tolerance = max(2, round(0.10 * text_height))
    frames: list[_Frame] = []
    for ti, top in enumerate(horizontal):
        for bottom in horizontal[ti + 1:]:
            height = bottom.position - top.position
            if height < 0.55 * text_height:
                continue
            if height > 2.4 * text_height:
                break
            x0, x1 = max(top.start, bottom.start), min(top.end, bottom.end)
            if not 1.5 * text_height <= x1 - x0 <= 24 * text_height:
                continue
            positions = [line.position for line in vertical
                         if x0 - tolerance <= line.position <= x1 + tolerance
                         and line.start <= top.position + tolerance
                         and line.end >= bottom.position - tolerance]
            boundaries: list[int] = []
            for pos in sorted(positions):
                if not boundaries or pos - boundaries[-1] > tolerance:
                    boundaries.append(pos)
                else:
                    boundaries[-1] = round((boundaries[-1] + pos) / 2)
            lefts = [x for x in boundaries if abs(x - x0) <= 2 * tolerance]
            rights = [x for x in boundaries if abs(x - x1) <= 2 * tolerance]
            if not lefts or not rights:
                continue
            left, right = lefts[0], rights[-1]
            dividers = [x for x in boundaries if left <= x <= right]
            cells = tuple({"x": float(a), "y": float(top.position), "width": float(b-a),
                           "height": float(height)} for a, b in zip(dividers, dividers[1:])
                          if b-a >= max(3, 0.25 * text_height))
            if cells:
                frames.append(_Frame({"x": float(left), "y": float(top.position),
                                      "width": float(right-left), "height": float(height)}, cells))
    deduped: list[_Frame] = []
    for frame in sorted(frames, key=lambda item: (-len(item.cells), item.bbox["width"])):
        if any(_inside(old.bbox, _center(frame.bbox), 0.2 * text_height)
               and abs(old.bbox["height"] - frame.bbox["height"]) <= 0.3 * text_height
               for old in deduped):
            continue
        deduped.append(frame)
    return deduped


def _characteristic_text(text: str) -> tuple[str, str] | None:
    upper = text.upper()
    for glyph, result in _GDT_GLYPHS.items():
        if glyph in text:
            return result
    for keyword, result in _GDT_KEYWORDS.items():
        if re.search(r"(?<![A-Z0-9])" + re.escape(keyword) + r"(?![A-Z0-9])", upper):
            return result
    return None


def _longest(values: np.ndarray) -> int:
    return max((b-a+1 for a, b in _runs(values, 1)), default=0)


def _infer_characteristic(image: Image.Image, cell: BBox) -> tuple[str, str, float] | None:
    x0 = max(0, round(float(cell["x"]) + .12 * float(cell["width"])))
    y0 = max(0, round(float(cell["y"]) + .12 * float(cell["height"])))
    x1 = min(image.width, round(float(cell["x"]) + .88 * float(cell["width"])))
    y1 = min(image.height, round(float(cell["y"]) + .88 * float(cell["height"])))
    if x1-x0 < 4 or y1-y0 < 4:
        return None
    binary = _ink(image.crop((x0, y0, x1, y1)))
    height, width = binary.shape
    strong_h = [(row, _longest(binary[row, :])) for row in range(height)
                if _longest(binary[row, :]) >= .45 * width]
    strong_v = [(col, _longest(binary[:, col])) for col in range(width)
                if _longest(binary[:, col]) >= .45 * height]
    if strong_h and strong_v:
        return "⟂", "Perpendicularity", .82
    positions: list[int] = []
    for pos, _ in strong_v:
        if not positions or pos - positions[-1] >= .2 * width:
            positions.append(pos)
    if len(positions) >= 2:
        return "∥", "Parallelism", .78
    if strong_h and not strong_v:
        return "⏤", "Straightness", .76
    return None


def _members(frame: _Frame, records: Sequence[Mapping[str, Any]], pad: float) -> tuple[int, ...]:
    found = []
    for index, record in enumerate(records):
        if record.get("table_excluded"):
            continue
        try:
            if _inside(frame.bbox, _center(_box(record)), pad):
                found.append(index)
        except (KeyError, TypeError, ValueError):
            pass
    return tuple(found)


def _frame_plan(image: Image.Image, frame: _Frame, records: Sequence[Mapping[str, Any]],
                text_height: float) -> StructuredObjectPlan | None:
    if len(frame.cells) < 2:
        return None
    members = _members(frame, records, .12 * text_height)
    if not members:
        return None
    cell_texts: list[str] = []
    for cell in frame.cells:
        indexes = [i for i in members if _inside(cell, _center(_box(records[i])), 1)]
        indexes.sort(key=lambda i: _box(records[i])["x"])
        cell_texts.append(" ".join(_text(records[i]) for i in indexes if _text(records[i])))
    combined = " ".join(filter(None, cell_texts))
    if not _NUMBER_RE.search(combined):
        return None
    characteristic = _characteristic_text(cell_texts[0] if cell_texts else "")
    source, visual_conf = "frame_geometry+ocr", .96
    if characteristic is None:
        inferred = _infer_characteristic(image, frame.cells[0])
        if inferred:
            characteristic = inferred[:2]
            visual_conf = inferred[2]
            source = "frame_geometry+symbol_completion"
        else:
            visual_conf = 0
    symbol = characteristic[0] if characteristic else None
    subtype = characteristic[1] if characteristic else None
    if symbol and (not cell_texts or _characteristic_text(cell_texts[0]) is None):
        cell_texts[0] = symbol
    text = " | ".join(filter(None, cell_texts)) or combined
    complete = bool(symbol and _NUMBER_RE.search(text))
    evidence = StructuredSymbolEvidence(
        "feature_control_frame", dict(frame.bbox),
        round(min(.98, .70 + .04 * min(len(frame.cells), 4) + .10 * visual_conf), 4),
        complete, subtype, symbol, source, frame.cells,
        () if complete else ("unresolved_gdt_characteristic",),
        {"cell_count": len(frame.cells)},
    )
    anchor = next((i for i in members if _NUMBER_RE.search(_text(records[i]))), members[0])
    return StructuredObjectPlan(
        members, anchor, str(records[anchor].get("candidate_id") or f"C{anchor+1:04d}"),
        text, dict(frame.bbox), min((_confidence(records[i]) for i in members), default=0),
        evidence, not complete,
        "Feature-control frame characteristic could not be resolved" if not complete else "",
    )


def _datum_plan(frame: _Frame, records: Sequence[Mapping[str, Any]], assigned: set[int],
                text_height: float) -> StructuredObjectPlan | None:
    if len(frame.cells) != 1:
        return None
    ratio = frame.bbox["width"] / max(frame.bbox["height"], 1)
    if not .65 <= ratio <= 1.8 or not .55 * text_height <= frame.bbox["height"] <= 2.4 * text_height:
        return None
    members = tuple(i for i in _members(frame, records, .08 * text_height) if i not in assigned)
    letters = [i for i in members if _DATUM_RE.fullmatch(_text(records[i]).upper())]
    if len(letters) != 1:
        return None
    anchor = letters[0]
    letter = _text(records[anchor]).upper()
    evidence = StructuredSymbolEvidence(
        "datum", dict(frame.bbox), .88, True, letter, source="boxed_letter_geometry+ocr",
        cells=frame.cells, visual_features={"boxed_letter": True},
    )
    return StructuredObjectPlan((anchor,), anchor,
        str(records[anchor].get("candidate_id") or f"C{anchor+1:04d}"), letter,
        dict(frame.bbox), _confidence(records[anchor]), evidence, False, "")


def _diagonal_support(points: np.ndarray, slope: float) -> int:
    if not points.size:
        return 0
    bins = np.round((points[:, 0] - slope * points[:, 1]) / 1.5).astype(int)
    return int(np.unique(bins, return_counts=True)[1].max())


def _surface_mark(image: Image.Image, box: BBox) -> tuple[dict[str, float], float] | None:
    height = max(float(box["height"]), 1)
    x0 = max(0, round(float(box["x"]) - 3.2 * height))
    x1 = max(0, round(float(box["x"]) + .15 * height))
    y0 = max(0, round(float(box["y"]) - 1.4 * height))
    y1 = min(image.height, round(float(box["y"]) + float(box["height"]) + 1.4 * height))
    if x1-x0 < 6 or y1-y0 < 6:
        return None
    binary = _ink(image.crop((x0, y0, x1, y1)))
    points = np.argwhere(binary)
    if len(points) < 8:
        return None
    pos = max((_diagonal_support(points, s) for s in (.5, .75, 1, 1.5, 2)), default=0)
    neg = max((_diagonal_support(points, s) for s in (-.5, -.75, -1, -1.5, -2)), default=0)
    required = max(4, round(.35 * min(binary.shape)))
    if pos < required or neg < required:
        return None
    ys, xs = points[:, 0], points[:, 1]
    return ({"x": float(x0+xs.min()), "y": float(y0+ys.min()),
             "width": float(xs.max()-xs.min()+1), "height": float(ys.max()-ys.min()+1)},
            round(min(.9, .68 + .02 * min(pos, neg)), 4))


def _surface_plan(image: Image.Image, index: int, record: Mapping[str, Any]) -> StructuredObjectPlan | None:
    if record.get("table_excluded") or not _SURFACE_RE.fullmatch(_text(record)):
        return None
    text, box = _text(record), _box(record)
    explicit = bool(_EXPLICIT_SURFACE_RE.search(text))
    mark = _surface_mark(image, box)
    if not explicit and mark is None:
        return None
    mark_box, score = mark if mark else (dict(box), 0)
    x0, y0 = min(box["x"], mark_box["x"]), min(box["y"], mark_box["y"])
    union = {"x": x0, "y": y0,
             "width": max(box["x"]+box["width"], mark_box["x"]+mark_box["width"])-x0,
             "height": max(box["y"]+box["height"], mark_box["y"]+mark_box["height"])-y0}
    prefix = re.match(r"(?:RA|RZ|RMAX|RQ|N)", text, re.IGNORECASE)
    evidence = StructuredSymbolEvidence(
        "surface_finish", {k: round(float(v), 1) for k, v in union.items()},
        round(max(.94 if explicit else 0, score), 4), True,
        prefix.group(0).upper() if prefix else "Surface texture",
        source=("surface_text+texture_geometry" if explicit and mark else
                "surface_text" if explicit else "texture_geometry+numeric_value"),
        visual_features={"texture_mark": mark is not None,
                         "mark_bbox": mark_box if mark else None},
    )
    return StructuredObjectPlan((index,), index,
        str(record.get("candidate_id") or f"C{index+1:04d}"), text,
        dict(evidence.bbox), _confidence(record), evidence, False, "")


def plan_structured_engineering_symbols(
    image: Image.Image, records: Sequence[Mapping[str, Any]]
) -> list[StructuredObjectPlan]:
    if not records:
        return []
    height = _text_height(records, image.height)
    frames = _detect_frames(_ink(image), height)
    plans: list[StructuredObjectPlan] = []
    assigned: set[int] = set()
    for frame in sorted(frames, key=lambda item: -len(item.cells)):
        plan = _frame_plan(image, frame, records, height)
        if plan and not assigned.intersection(plan.member_indexes):
            plans.append(plan); assigned.update(plan.member_indexes)
    for frame in frames:
        plan = _datum_plan(frame, records, assigned, height)
        if plan:
            plans.append(plan); assigned.update(plan.member_indexes)
    for index, record in enumerate(records):
        if index in assigned:
            continue
        try:
            plan = _surface_plan(image, index, record)
        except (KeyError, TypeError, ValueError):
            plan = None
        if plan:
            plans.append(plan); assigned.add(index)
    return sorted(plans, key=lambda item: min(item.member_indexes))


def structured_symbol_statistics(
    plans: Iterable[StructuredObjectPlan | Mapping[str, Any]],
) -> dict[str, Any]:
    kinds: Counter[str] = Counter(); count = complete = 0
    for item in plans:
        count += 1
        if isinstance(item, StructuredObjectPlan):
            kind, done = item.evidence.kind, item.evidence.complete
        else:
            evidence = item.get("evidence") or item.get("engineering_symbol") or {}
            kind, done = str(evidence.get("kind") or "unknown"), bool(evidence.get("complete"))
        kinds[kind] += 1; complete += int(done)
    return {"schema_version": 1, "object_count": count, "complete_count": complete,
            "review_count": count-complete, "kind_counts": dict(sorted(kinds.items()))}
