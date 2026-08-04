"""
Filters for auto-segment output — drop partial/noise reads.
"""

from __future__ import annotations

import re

_MIN_DIGITS = 3
_PARTIAL_RE = re.compile(
    r"^[\s]*(?:[±+\-]?\d*[.,]?|\d+[.,]?|\d{1,2}[±+\-]?\d{0,2})$"
)


def is_segment_worthy(text: str) -> bool:
    """
    Return False for fragment reads like ``.05``, ``215.``, or ``05``.

    Full dimensions, angles, and tolerances should pass.
    """
    t = (text or "").strip()
    if len(t) < 3:
        return False

    if "°" in t or "Ø" in t or "ø" in t or "φ" in t or "Φ" in t:
        return True
    if re.search(r"['\"′″]", t):
        return True
    if "±" in t or "+/-" in t:
        return len(re.findall(r"\d", t)) >= 2
    if re.match(r"^[Rr]\d", t):
        return True

    digits = re.findall(r"\d", t)
    if len(digits) < _MIN_DIGITS:
        return False
    if _PARTIAL_RE.match(t):
        return False
    if t.endswith(".") and len(digits) < 4:
        return False
    return True


_RICH_SYMBOLS = ("Ø", "ø", "φ", "Φ", "°", "±")


def _region_quality(region: dict) -> tuple:
    """Rank by completeness: dimension symbols, then digit count, length, conf."""
    t = region.get("text") or ""
    symbols = sum(t.count(s) for s in _RICH_SYMBOLS)
    digits = sum(c.isdigit() for c in t)
    return (symbols, digits, len(t.strip()), float(region.get("confidence") or 0.0))


def _overlap_frac(a: dict, b: dict) -> float:
    """Intersection as a fraction of the *smaller* box (catches containment)."""
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = ax0 + a["width"], ay0 + a["height"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = bx0 + b["width"], by0 + b["height"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1.0)
    area_b = max((bx1 - bx0) * (by1 - by0), 1.0)
    return inter / min(area_a, area_b)


def dedupe_regions(regions: list[dict], *, overlap_thresh: float = 0.5) -> list[dict]:
    """
    Suppress overlapping duplicate segments, keeping the richest read.

    When morphology splits one dimension into both a proper box and a partial
    sliver (e.g. ``Ø175,32 REF.`` and a stray ``175REF``), their boxes overlap.
    We greedily keep the most complete read and drop anything that overlaps it,
    preserving the input (reading) order of the survivors.
    """
    if len(regions) <= 1:
        return regions
    best_first = sorted(range(len(regions)), key=lambda i: _region_quality(regions[i]), reverse=True)
    keep = [False] * len(regions)
    kept_boxes: list[dict] = []
    for i in best_first:
        box = regions[i]["bbox"]
        if any(_overlap_frac(box, kb) >= overlap_thresh for kb in kept_boxes):
            continue
        keep[i] = True
        kept_boxes.append(box)
    return [r for i, r in enumerate(regions) if keep[i]]
