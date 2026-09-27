"""
Dual-unit (inch [mm]) cross-validation and digit repair.

Many CAD sheets dimension in a primary unit with the converted value in
brackets, e.g. ``.30 [7.62]`` or ``1.08 [27.43]`` (inch, then mm = inch x 25.4).
The bracket is a *redundant* encoding of the same measurement, so it is a free
consistency check: a correct read satisfies ``bracket ~= primary * 25.4``.

This is the single strongest signal for catching the digit confusions the raw
OCR makes on these drawings — a ``1`` misread as ``7`` (or vice versa) breaks the
25.4 ratio, and the *other* half of the pair pins down what the digit must have
been. When the ratio is violated we treat it as an OCR error and try to repair
it: enumerate bounded single/double digit edits (confusable substitutions and
one deletion) on each side and pick the hypothesis whose ratio snaps back to
25.4 with the fewest, most-likely edits. Unrepairable pairs are flagged for
human review instead of being silently accepted.

Direction is auto-detected per pair (inch-first or mm-first), and values with no
bracket pair are returned untouched — nothing here fires unless a dual pair is
actually present, so single-unit sheets are unaffected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INCH_TO_MM = 25.4

# A dual pair: an optional prefix (Ø / R / ±), a decimal primary value, a gap,
# then the bracketed converted value. Both numbers are plain decimals — a pair
# carrying a ± tolerance is left to the tolerance path and skipped here.
_DUAL_RE = re.compile(
    r"^(?P<pre>[^\d\[\(.]*)"
    r"(?P<prim>\d*\.?\d+)"
    r"(?P<gap>\s*)"
    r"(?P<open>[\[(])"
    r"(?P<lgap>\s*)"
    r"(?P<brk>\d*\.?\d+)"
    r"(?P<rgap>\s*)"
    r"(?P<close>[\])])"
    r"(?P<post>.*)$"
)

# Digit confusions PaddleOCR makes on thin CAD strokes, ordered loosely by how
# often each is the real culprit. Used only to *propose* repairs; every proposal
# still has to satisfy the 25.4 ratio, so a wrong guess can never win.
_DIGIT_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "1": ("7", "4", "9"),
    "7": ("1", "2"),
    "4": ("1", "9"),
    "9": ("4", "8", "0", "3", "1"),
    "3": ("8", "5", "9"),
    "8": ("3", "6", "0", "9"),
    "5": ("6", "8", "3"),
    "6": ("5", "8", "0"),
    "0": ("8", "6", "9"),
    "2": ("7", "3"),
}


def _tol(expected_mm: float) -> float:
    """
    Ratio tolerance. Rounding each side to 2 decimals moves the product by at
    most ~0.13 mm — and, crucially, that error is *independent of magnitude*, so
    the band stays flat rather than scaling with the value. A tiny relative term
    only guards pathologically large dimensions. Kept tight so a genuine
    significant-digit error (the confusions we want to catch) always breaks it.
    """
    return max(0.2, 0.004 * expected_mm)


@dataclass
class DualResult:
    text: str
    # absent      — no bracket pair, nothing to check
    # consistent  — pair already satisfies the 25.4 ratio
    # repaired    — pair violated the ratio but a confident digit fix restored it
    # inconsistent — pair violates the ratio and no bounded fix restores it
    status: str
    direction: str | None = None  # "inch_mm" | "mm_inch"
    primary: str | None = None
    bracket: str | None = None
    expected_mm: float | None = None
    edits: int = 0

    @property
    def needs_review(self) -> bool:
        return self.status == "inconsistent"

    @property
    def corrected(self) -> bool:
        return self.status == "repaired"


def parse_dual_pair(text: str):
    """Return the regex match for a ``primary [bracket]`` pair, or ``None``."""
    if not text or "[" not in text and "(" not in text:
        return None
    return _DUAL_RE.match(text.strip())


def _consistent(prim: str, brk: str) -> tuple[bool, str | None, float]:
    """True when ``prim``/``brk`` satisfy the 25.4 ratio in either direction."""
    try:
        vp, vb = float(prim), float(brk)
    except ValueError:
        return (False, None, 0.0)
    if vp <= 0 or vb <= 0:
        return (False, None, 0.0)
    # inch-first: bracket is the mm value; mm-first: primary is the mm value.
    for direction, inch, mm in (("inch_mm", vp, vb), ("mm_inch", vb, vp)):
        expected = inch * INCH_TO_MM
        if abs(mm - expected) <= _tol(expected):
            return (True, direction, expected)
    return (False, None, 0.0)


def _edit_variants(num: str) -> list[tuple[str, int]]:
    """
    Bounded edit candidates for one decimal string: the original (0 edits),
    every single confusable-digit substitution, and every single-digit deletion.
    Only edits that keep a parseable decimal are returned.
    """
    out: list[tuple[str, int]] = [(num, 0)]
    chars = list(num)
    for i, ch in enumerate(chars):
        # Substitutions from the confusion map.
        for repl in _DIGIT_CONFUSIONS.get(ch, ()):
            cand = "".join(chars[:i] + [repl] + chars[i + 1:])
            out.append((cand, 1))
        # Deletion of a spurious digit (a stroke hallucinated into the number).
        if ch.isdigit() and len(num) > 1:
            cand = num[:i] + num[i + 1:]
            if cand and cand != "." and _is_number(cand):
                out.append((cand, 2))  # deletions cost more than substitutions
    return out


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def repair_dual(text: str) -> DualResult:
    """
    Validate a dual-unit reading and repair digit misreads that break the ratio.

    No-op (``absent``) for any text without a numeric ``[bracket]`` pair, so this
    is safe to call on every read regardless of whether the sheet is dual-unit.
    """
    m = parse_dual_pair(text)
    if not m:
        return DualResult(text, "absent")

    prim, brk = m.group("prim"), m.group("brk")
    ok, direction, expected = _consistent(prim, brk)
    if ok:
        return DualResult(
            text, "consistent", direction=direction,
            primary=prim, bracket=brk, expected_mm=expected,
        )

    # Search bounded edits on both sides for the cheapest ratio-satisfying fix.
    best: tuple[int, str, str, str, float] | None = None
    for p_cand, p_cost in _edit_variants(prim):
        for b_cand, b_cost in _edit_variants(brk):
            cost = p_cost + b_cost
            if cost == 0 or cost > 2:
                continue
            okc, dirc, expc = _consistent(p_cand, b_cand)
            if not okc:
                continue
            if best is None or cost < best[0]:
                best = (cost, p_cand, b_cand, dirc or "", expc)

    if best is None:
        return DualResult(
            text, "inconsistent", primary=prim, bracket=brk,
        )

    _, p_fix, b_fix, dirc, expc = best
    repaired = (
        m.group("pre") + p_fix + m.group("gap") + m.group("open")
        + m.group("lgap") + b_fix + m.group("rgap") + m.group("close")
        + m.group("post")
    )
    return DualResult(
        repaired, "repaired", direction=dirc,
        primary=p_fix, bracket=b_fix, expected_mm=expc, edits=best[0],
    )
