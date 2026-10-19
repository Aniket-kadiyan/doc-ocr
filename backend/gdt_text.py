"""
Textual GD&T recognition — spec section 16, with the guards it was missing.

Why this module exists
----------------------
The earlier section-16 classifier accepted a GD&T word by itself. Its failure
mode was established by measurement over 170 OCR strings from four real
drawings:

* **true positives: 0**
* **false positives: 1**

The one hit was this, from the cup-turning drawing's accuracy note::

    BEARING NO ... ROUNDNESS(2PT./180)50µm MAX. ROUNDNESS(3PT./120)50µm
    MAX. TAPER 50µm MAX. SQUARENESS WITH FACE 50µm MAX. ... ECCENTRICITY
    WITH 500µm MAX. ... SURFACE ROUGHNESS ON 6.3Ra µm ...

The old rule matched ``SQUARENESS`` and would have published a 400-character
general note as a GD&T perpendicularity control. That is the failure the
section was switched off to avoid, and it is the one this module has to
prevent.

What it is *for*
----------------
Not duplication of the graphical recognizer. ``gdt_detect`` finds glyphs; it
cannot see a drawing that spells a control out in words, because there is no
glyph to find. ``FLATNESS 0.05`` and ``POSITION Ø0.1 A B`` are real callouts
with nothing for a template to match, and they are invisible to every other
path in the system. That gap is what this closes.

The rule it follows
-------------------
**A word is not evidence.** A GD&T term appearing in text means nothing on its
own — engineers write "roundness" and "squareness" in measurement notes all
day. Recognition requires the term *plus* the shape of a callout: short text,
a tolerance value, and no sign of being prose. Each accepted match reports the
evidence that carried it, so a wrong one can be argued with rather than
guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import feature_dictionary as fd

# ── Evidence kinds ───────────────────────────────────────────────────────────

#: An unambiguous GD&T glyph. Strong enough on its own.
E_GLYPH = "unambiguous_glyph"
#: A pipe-delimited frame, "| Ø0.1 | A | B |". Strong on its own.
E_PIPE_FRAME = "pipe_frame"
#: A spelled-out control name, with a tolerance beside it.
E_KEYWORD_WITH_VALUE = "keyword_with_value"
#: A datum reference list following a control name.
E_DATUM_LIST = "datum_list"

#: Why a candidate was refused.
R_NOTE_BLOCK = "reads as prose, not a callout"
R_TOO_LONG = "too long to be a feature control frame"
R_MEASUREMENT_SPEC = "the term is part of a measurement specification"
R_NO_VALUE = "the control name carries no tolerance value"
R_AMBIGUOUS_GLYPH = "the glyph has a non-GD&T meaning on a drawing"


# ── Which glyphs may speak alone ─────────────────────────────────────────────
#
# ``feature_dictionary.GDT_SYMBOLS`` is a display table and includes marks that
# are not safely diagnostic in text:
#
#   ○   a circle, the letter O, a bullet
#   //  a double slash — appears in URLs and in "1/2 // 3"
#   ∥   same shape, same problem
#   ⌒   also the arc-length symbol
#   ∠   also an ordinary angle mark
#   ↗   an arrow, used for many things
#
# The rest have no other meaning on an engineering drawing, so finding one in
# OCR text is strong evidence by itself.
_AMBIGUOUS_GLYPHS = frozenset({"○", "◯", "//", "∥", "⌒", "∠", "↗"})

UNAMBIGUOUS_GLYPHS: dict[str, str] = {
    glyph: name
    for glyph, name in fd.GDT_SYMBOLS.items()
    if glyph not in _AMBIGUOUS_GLYPHS
}


# ── Shape of a callout ───────────────────────────────────────────────────────

#: Longest text that can plausibly be one feature control frame. A real one is
#: a symbol, a tolerance and up to three datum letters; the note block that
#: triggered the original false positive is over 400 characters.
MAX_CALLOUT_CHARS = 64

#: A frame rendered as text is a handful of tokens, not a paragraph.
MAX_CALLOUT_WORDS = 10

#: A tolerance value: a bare decimal, optionally with a Ø zone marker.
_VALUE = re.compile(r"[Ø⌀φ]?\s*\d+(?:[.,]\d+)?")

#: Datum references trailing a control: "A", "A B", "A B C".
_DATUMS = re.compile(r"(?:(?<=\s)|^)([A-HJ-NP-Z])(?:\s+([A-HJ-NP-Z]))?(?:\s+([A-HJ-NP-Z]))?\s*$")

#: Marks a term as part of a *measurement specification* rather than a control.
#: "ROUNDNESS(2PT./180)" states how roundness is to be measured; "TAPER 50µm
#: MAX." states a limit in a note. Neither is a feature control frame.
_MEASUREMENT_SPEC = re.compile(
    r"\(\s*\d+\s*PT|MAX\.?(?:\s|$)|MIN\.?(?:\s|$)|µM|UM\b|MICRON|\bRA\b|ROUGHNESS",
    re.IGNORECASE,
)


@dataclass
class GdtTextMatch:
    """One textual GD&T recognition, with the evidence behind it."""

    symbol_id: str | None
    name: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    #: The text this was read from, preserved verbatim.
    source_text: str = ""
    #: Tolerance value, when one was read.
    value: float | None = None
    #: Datum letters in precedence order.
    datums: list[str] = field(default_factory=list)
    #: True when the tolerance zone is diametral.
    diameter_zone: bool = False
    method: str = "text_rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "evidence": list(self.evidence),
            "source_text": self.source_text,
            "value": self.value,
            "datums": list(self.datums),
            "diameter_zone": self.diameter_zone,
            "method": self.method,
        }


@dataclass
class GdtTextRejection:
    """A candidate that looked like GD&T and was refused, and why."""

    reason: str
    matched: str
    source_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "matched": self.matched,
            "source_text": self.source_text,
        }


# ── Recognition ──────────────────────────────────────────────────────────────


def _name_to_symbol_id(name: str) -> str | None:
    """Map a dictionary display name onto the shared GD&T registry."""

    import gdt_registry as reg

    direct = reg.by_name(name)
    if direct is not None:
        return direct.id

    # These dictionary labels name a family rather than one exact control.
    # Preserve SI-3.1's established result until graphical/context evidence can
    # safely distinguish the members of each family.
    return {
        "Runout": "circular_runout",
        "Profile": "profile_of_a_surface",
        "Total Runout": "total_runout",
    }.get(name)


#: ``looks_like_note_block`` is only consulted above this many words.
#: It counts any word containing a letter as an "alpha word", and in Python
#: ``"Ø".isalpha()`` is true — so ``POSITION Ø0.1 A B`` scores four alpha words
#: (the two bare datum letters among them) and reads as a paragraph. It is a
#: good paragraph test and a bad callout test; the length bounds above already
#: cover the short end.
_NOTE_BLOCK_MIN_WORDS = 7


def _looks_like_prose(text: str, upper: str) -> bool:
    """Prose, a note, or a specification paragraph — never a frame."""
    words = text.split()
    if len(text) > MAX_CALLOUT_CHARS:
        return True
    if len(words) > MAX_CALLOUT_WORDS:
        return True
    if len(words) >= _NOTE_BLOCK_MIN_WORDS and fd.looks_like_note_block(text):
        return True
    return bool(set(fd.NOTE_KEYWORDS) & set(upper.split()))


def recognize(
    text: str, *, collect_rejections: bool = False
) -> tuple[GdtTextMatch | None, list[GdtTextRejection]]:
    """Read a GD&T control out of OCR text, or refuse and say why.

    Returns ``(match, rejections)``. ``match`` is ``None`` whenever the
    evidence does not clear the bar, which is the common case — this rule is
    meant to be quiet.
    """
    raw = (text or "").strip()
    rejections: list[GdtTextRejection] = []
    if not raw:
        return None, rejections

    upper = raw.upper()
    prose = _looks_like_prose(raw, upper)

    # ── Strong evidence 1: an unambiguous glyph ──────────────────────────────
    for glyph, name in UNAMBIGUOUS_GLYPHS.items():
        if glyph in raw:
            if prose:
                rejections.append(GdtTextRejection(R_NOTE_BLOCK, glyph, raw))
                continue
            return (
                _build(raw, upper, name, 0.9, [E_GLYPH]),
                rejections,
            )

    # An ambiguous glyph is noted but never decides.
    if collect_rejections:
        for glyph in _AMBIGUOUS_GLYPHS:
            if glyph in raw:
                rejections.append(GdtTextRejection(R_AMBIGUOUS_GLYPH, glyph, raw))

    # ── Strong evidence 2: a pipe-delimited frame ────────────────────────────
    if raw.count("|") >= 2 and re.search(r"\d", raw) and not prose:
        name = None
        for keyword, candidate in fd.GDT_KEYWORDS.items():
            if _whole_word(upper, keyword):
                name = candidate
                break
        if name or any(g in raw for g in fd.DIAMETER_SYMBOLS):
            return (
                _build(raw, upper, name or "", 0.78, [E_PIPE_FRAME]),
                rejections,
            )

    # ── Medium evidence: a control name plus the shape of a callout ──────────
    for keyword, name in fd.GDT_KEYWORDS.items():
        if not _whole_word(upper, keyword):
            continue
        if prose:
            rejections.append(GdtTextRejection(R_NOTE_BLOCK, keyword, raw))
            continue
        if _MEASUREMENT_SPEC.search(raw):
            rejections.append(GdtTextRejection(R_MEASUREMENT_SPEC, keyword, raw))
            continue
        # The term has to be qualified by a tolerance to be a control.
        tail = upper.split(keyword, 1)[1] if keyword in upper else ""
        if not _VALUE.search(tail):
            rejections.append(GdtTextRejection(R_NO_VALUE, keyword, raw))
            continue
        evidence = [E_KEYWORD_WITH_VALUE]
        confidence = 0.72
        if _DATUMS.search(tail.strip()):
            evidence.append(E_DATUM_LIST)
            confidence = 0.8
        return _build(raw, upper, name, confidence, evidence), rejections

    return None, rejections


def _whole_word(upper: str, keyword: str) -> bool:
    return (
        re.search(r"(?<![A-Z0-9])" + re.escape(keyword) + r"(?![A-Z0-9])", upper)
        is not None
    )


def _build(
    raw: str, upper: str, name: str, confidence: float, evidence: list[str]
) -> GdtTextMatch:
    value = None
    diameter_zone = any(g in raw for g in fd.DIAMETER_SYMBOLS)
    number = re.search(r"\d+(?:[.,]\d+)?", raw)
    if number:
        try:
            value = float(number.group(0).replace(",", "."))
        except ValueError:
            value = None
    datums: list[str] = []
    match = _DATUMS.search(upper.strip())
    if match:
        datums = [g for g in match.groups() if g]
    return GdtTextMatch(
        symbol_id=_name_to_symbol_id(name) if name else None,
        name=name or "Feature Control Frame",
        confidence=confidence,
        evidence=evidence,
        source_text=raw,
        value=value,
        datums=datums,
        diameter_zone=diameter_zone,
    )


def describe_guards() -> str:
    """The guards, for logs and for anyone deciding whether to trust this."""
    return "\n".join(
        [
            "Textual GD&T recognition requires, in order:",
            f"  - text no longer than {MAX_CALLOUT_CHARS} chars / "
            f"{MAX_CALLOUT_WORDS} words",
            "  - not a note block or prose",
            "  - not a measurement specification (MAX./MIN./µm/Ra/(2PT.))",
            "  - an unambiguous glyph, a pipe frame, or a control name with a value",
            f"  - ambiguous glyphs never decide: {sorted(_AMBIGUOUS_GLYPHS)}",
        ]
    )
