"""
Filters for auto-segment output — drop partial/noise reads.
"""

from __future__ import annotations

import re

from geom_utils import overlap_frac as geom_overlap_frac

_MIN_DIGITS = 3
_PARTIAL_RE = re.compile(
    r"^[\s]*(?:[±+\-]?\d*[.,]?|\d+[.,]?|\d{1,2}[±+\-]?\d{0,2})$"
)


# Free-text annotation notes / process callouts that clutter dimension callouts
# (e.g. the "(BOTH SIDES)" beside a chamfer, or "SHAVING" over a face). Matched
# against the *detection*-stage text, still clean before the composer injects a
# spurious Ø/R prefix.
_NOTE_RE = re.compile(
    r"BOTH\s*SIDES|\bSIDES\b|\bTYP(?:ICAL)?\b|\bPLACES\b|\bSHAVING\b", re.I
)
# Surface-texture marks (▽ ▷ △ ∇ and lookalikes) detect as standalone symbol
# boxes with no value; they bridge neighbouring callouts if left in.
_FINISH_SYMS = set("▽▷◁△▲▼∇▹◃")
# The same marks as PaddleOCR mis-reads them: a run of triangles comes back as
# a short all-caps string of V / W / N / Y ("W", "VV", "NV"). No dimension is a
# bare 1–3 letter word from this set, so it is safe to drop at detection.
_FINISH_LETTERS = set("VWNYvwny")


def is_annotation_note(text: str) -> bool:
    """
    True when a detected box is a pure annotation note, not a dimension.

    Such notes ("(BOTH SIDES)", "SHAVING") and bare surface-finish symbols carry
    no measurable value but fuse into neighbouring callouts and get a bogus Ø
    injected by the composer. Dropped at the detection stage so they never
    cluster or recognise. Guarded on digit count so a fused ``0.5×45 BOTH SIDES``
    box (which still holds a real value) is kept.
    """
    t = text or ""
    if sum(c.isdigit() for c in t) > 2:
        return False
    stripped = t.strip()
    if stripped and all(c in _FINISH_SYMS or c.isspace() for c in stripped):
        return True
    compact = "".join(stripped.split())
    if 0 < len(compact) <= 3 and all(c in _FINISH_LETTERS for c in compact):
        return True
    return bool(_NOTE_RE.search(t))


def strip_foreign_glyphs(text: str) -> str:
    """
    Remove CJK / other non-engineering code points from a read.

    PaddleOCR's multilingual recogniser emits a Chinese character for a stroke
    cluster it cannot resolve (``四1`` for a leader-adjacent ``R1``). Nothing
    on an engineering drawing lives above U+2E80 (Ø ° ± × ⌀ ⟂ ∥ ∠ ◎ ⌖ ▽ Ⓜ are
    all below it), so those glyphs are noise and are dropped.
    """
    return "".join(c for c in (text or "") if ord(c) < 0x2E80).strip()


def contains_annotation_note(text: str) -> bool:
    """
    True when a read carries note wording at all, fused values included.

    ``is_annotation_note`` deliberately keeps a box whose note is glued to a
    real value, because the value is still wanted. A *rotated* re-read has no
    such obligation: the upright pass already reported whatever values are
    there, so a rotated read mixing a note into digits
    (``14(BOTH SIDES)5×45.20×19×1``) is a fused blob and nothing is lost by
    dropping it.
    """
    return bool(_NOTE_RE.search(text or ""))


# A drawing carries plenty of text that reads like a number but is not a
# dimension: table rows, material codes, dates, revision notes. Ballooning those
# clutters the sheet and buries the dimensions among them.
# A date needs a four-digit year and the same separator twice, so a deviation
# pair like Ø33-0.05-0.1 is not mistaken for one.
_DATE_RE = re.compile(r"\d{1,2}\s*([.\-/])\s*\d{1,2}\s*\1\s*\d{4}")
# No dimension carries a seven-digit run; a drawing or part number does.
_LONG_NUMBER_RE = re.compile(r"\d{7,}")
# A range written with a tilde is a specification (hardness, case depth), and a
# colon marks a labelled table row (``Mat.:``) or a scale (``2:1``).
_SPEC_RANGE_RE = re.compile(r"\d\s*~\s*\d")
# A part or material code: a few letters run straight into a long number.
_CODE_RE = re.compile(r"[A-Za-z]{3,}\d{4,}")
# Text opening with a word rather than a value or a symbol.
_LEADING_WORD_RE = re.compile(r"^\s*[A-Za-z]{4,}")
# A tolerance with nothing to apply it to.
_BARE_TOLERANCE_RE = re.compile(r"^\s*[±+\-−]\s*\d+(?:[.,]\d+)?\s*$")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
# Characters no dimension contains. "?" in particular is the recogniser saying
# it could not resolve a glyph, which makes the whole read untrustworthy.
_JUNK_CHARS = set("?@#&\\|%*_<>{}")
_DIMENSION_SYMBOLS = "Ø⌀øφΦ°±×"


def has_dimension_value(text: str) -> bool:
    """
    True when a read is an actual dimension rather than text that merely has
    digits in it.

    Balloons are for things a person measures. A revision date, a material code
    and a specification row all carry digits, and boxing them puts a numbered
    balloon on something nobody will inspect. The tests here are deliberately
    narrow, because a wrongly dropped dimension is far worse than a stray
    balloon: anything carrying a dimension symbol is kept unless it is plainly
    a date or a code.
    """
    t = (text or "").strip()
    if not any(c.isdigit() for c in t):
        return False
    if _DATE_RE.search(t):
        return False
    if _CODE_RE.search(t):
        return False
    if _LONG_NUMBER_RE.search(t):
        return False
    if _SPEC_RANGE_RE.search(t) or ":" in t:
        return False
    if any(c in _JUNK_CHARS for c in t):
        return False
    if _BARE_TOLERANCE_RE.match(t):
        return False
    if _LEADING_WORD_RE.match(t):
        return False
    # Prose: several words and nothing that marks a measurement.
    if len(_WORD_RE.findall(t)) >= 2 and not any(c in _DIMENSION_SYMBOLS for c in t):
        return False
    return True


def is_segment_worthy(text: str) -> bool:
    """
    Return False for fragment reads like ``.05``, ``215.``, or ``05``.

    Full dimensions, angles, and tolerances should pass.
    """
    t = (text or "").strip()
    # A short radius callout (R1, R2) is a complete value; check it before the
    # length floor that rejects other 2-character fragments.
    if re.match(r"^[Rr]\d", t):
        return True
    if len(t) < 3:
        return False

    if "°" in t or "Ø" in t or "ø" in t or "φ" in t or "Φ" in t:
        return True
    if re.search(r"['\"′″]", t):
        return True
    if "±" in t or "+/-" in t:
        return len(re.findall(r"\d", t)) >= 2

    digits = re.findall(r"\d", t)
    if len(digits) < _MIN_DIGITS:
        return False
    if _PARTIAL_RE.match(t):
        return False
    if t.endswith(".") and len(digits) < 4:
        return False
    return True


# A complete dual-unit dimension token: primary value + bracketed conversion,
# e.g. "1.28 [32.51]". Two or more of these in a single recognized region means
# the cluster fused separate stacked callouts and should be split by row.
_DUAL_VALUE_RE = re.compile(r"\d*\.?\d+\s*[\[(]\s*\d*\.?\d+\s*[\])]")
# A diameter/radius callout start, e.g. "Ø20", "⌀18", "R2.70". Two or more in
# one read means several distinct feature callouts fused into one cluster
# (e.g. two Ø leaders: "Ø20H10 … Ø18H10 …").
_FEATURE_START_RE = re.compile(r"[Ø⌀øφΦ]\s*\d|(?:^|\s)[Rr]\d")


def count_dimension_values(text: str) -> int:
    """
    Number of complete, independent dimension values in a recognized string.

    Counts two unambiguous markers of a fused multi-callout cluster: dual-unit
    pairs (``1.28 [32.51]`` over ``1.10 [27.94]``) and repeated Ø/R feature
    starts (``Ø20H10 … Ø18H10 …``). Single values, tolerances (``215.37±0.05``)
    and angles return 1, so a split is only triggered when there is genuinely
    more than one value present.
    """
    t = text or ""
    pairs = len(_DUAL_VALUE_RE.findall(t))
    features = len(_FEATURE_START_RE.findall(t))
    n = max(pairs, features)
    return n if n >= 2 else 1


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

    Overlap is measured between each region's *tight* rectangle — the oriented
    box when the region has one. Two callouts drawn along parallel leaders
    (``Ø20H10`` above ``Ø18H10`` on 45° lines) have axis-aligned hulls that
    overlap by more than half while their actual ink never touches, and the
    axis-aligned test used to delete one of them.
    """
    if len(regions) <= 1:
        return regions
    best_first = sorted(range(len(regions)), key=lambda i: _region_quality(regions[i]), reverse=True)
    keep = [False] * len(regions)
    kept: list[dict] = []
    for i in best_first:
        region = regions[i]
        if any(geom_overlap_frac(region, k) >= overlap_thresh for k in kept):
            continue
        keep[i] = True
        kept.append(region)
    return [r for i, r in enumerate(regions) if keep[i]]
