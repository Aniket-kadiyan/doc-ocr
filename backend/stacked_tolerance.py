"""
Assemble a callout written as a value with its deviations stacked beside it.

A fit or limit dimension is drawn as one value followed by two smaller numbers
stacked vertically — the upper and lower deviation::

        Ø20H10 +0.084     →  Ø20H10 +0.084/0
                   0

Flat reading order destroys this. Sorting the parts into rows puts the upper
deviation on the value's own row (it straddles the row boundary) and the lower
one after it, so the recogniser emits ``+0.08420H100``. The parts are easy to
tell apart geometrically, so we read them by position instead: the big box is
the value, and the two smaller numeric boxes to its right, one above the other,
are its deviations.

Works in whatever frame the boxes are given in, so the caller passes the boxes
of a callout that has already been levelled along its leader line.
"""

from __future__ import annotations

import re
from typing import Any

Box = dict[str, Any]

# Every part must be read confidently: the assembled string is trusted over the
# recogniser's own (scrambled) read, so a shaky part would be worse than useless.
_PART_CONFIDENCE_MIN = 0.85
# A deviation is smaller than its value but not a speck (a stray tick, a comma).
_MIN_DEVIATION_FRAC = 0.25


def _has_digit(text: str) -> bool:
    return any(c.isdigit() for c in text or "")


def _center(box: Box) -> tuple[float, float]:
    return (box["x"] + box["w"] / 2.0, box["y"] + box["h"] / 2.0)


def _vertical_overlap(a: Box, b: Box) -> float:
    top = max(a["y"], b["y"])
    bottom = min(a["y"] + a["h"], b["y"] + b["h"])
    return max(0.0, bottom - top)


def stacked_deviation_text(boxes: list[Box]) -> str | None:
    """
    Read ``boxes`` as ``value upper/lower``, or return ``None``.

    ``None`` means the parts do not form that pattern, and the caller should
    keep whatever the recogniser produced. The pattern is deliberately narrow:
    exactly one clear value box and exactly two numeric boxes stacked to its
    right, all read confidently, all overlapping the value's own line — so a
    neighbouring callout that crept into the crop is excluded rather than
    glued on.
    """
    usable = [
        b
        for b in boxes
        if b.get("w", 0) > 1
        and b.get("h", 0) > 1
        and _has_digit(b.get("text", ""))
        and float(b.get("conf") or 0.0) >= _PART_CONFIDENCE_MIN
    ]
    if len(usable) < 3:
        return None

    main = max(usable, key=lambda b: b["w"] * b["h"])
    main_cx, _ = _center(main)
    min_height = main["h"] * _MIN_DEVIATION_FRAC

    candidates = [
        b
        for b in usable
        if b is not main
        and _center(b)[0] > main_cx
        and b["h"] >= min_height
        and _vertical_overlap(b, main) > 0
    ]
    if len(candidates) != 2:
        return None
    # Nothing may sit left of the value: that would make it a middle fragment,
    # not the start of the callout.
    if any(_center(b)[0] < main_cx for b in usable if b is not main and b not in candidates):
        return None

    candidates.sort(key=lambda b: _center(b)[1])
    upper, lower = candidates
    # The two deviations are stacked, not side by side.
    if _center(lower)[1] - _center(upper)[1] < min(upper["h"], lower["h"]) * 0.5:
        return None

    value = (main.get("text") or "").strip()
    up = (upper.get("text") or "").strip()
    low = (lower.get("text") or "").strip()
    if not value or not up or not low:
        return None
    # A deviation is a signed decimal; anything else means we mis-grouped.
    if not all(re.fullmatch(r"[+\-−–]?\d*[.,]?\d+", t) for t in (up, low)):
        return None
    return f"{value} {up}/{low}"
