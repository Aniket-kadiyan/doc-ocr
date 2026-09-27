"""
Angular dimension parsing — degrees / minutes / seconds (DMS) and decimal degrees.

PaddleOCR reads the digits and (usually) the ° ' " marks, but downstream
number-normalization used to strip ' and ". These helpers detect an angular
dimension *before* any stripping and emit a canonical form:

    32°20'40"   (deg / min / sec)
    32°20'      (deg / min)
    32°         (deg only)
    45.5°       (decimal degrees)
"""

from __future__ import annotations

import re

# Curly / typographic primes that OCR sometimes emits for ' and ".
_PRIME_MAP = {
    "′": "'",  # ′ prime
    "″": '"',  # ″ double prime
    "’": "'",  # ’ right single quote
    "”": '"',  # ” right double quote
    "´": "'",  # ´ acute accent
    "ʺ": '"',  # ʺ modifier double prime
}

_DEG = "[°˚⁰oO]"  # degree glyph + common OCR confusions for the small circle


def normalize_primes(text: str) -> str:
    if not text:
        return text
    for src, dst in _PRIME_MAP.items():
        text = text.replace(src, dst)
    return text


# OCR routinely reads the degree ° as a double-quote ". Recover it only when the
# quote sits in the degree slot of a DMS triple (digits " digits ') so a genuine
# seconds/inch mark like 40" is left alone.
_DEG_QUOTE_RE = re.compile(r"^(\s*\d{1,3})\"(?=\s*\d{1,2}\s*')")


def _recover_degree_quote(text: str) -> str:
    return _DEG_QUOTE_RE.sub(r"\1°", text)


# Full / partial DMS. The degree mark is optional so we still recover
# 32 20'40" (vision/OCR dropped the °) when minute/second ticks survive.
_DMS_RE = re.compile(
    r"^\s*(\d{1,3})\s*(?:°|˚|⁰)?\s*"  # degrees (° optional)
    r"(?:(\d{1,2})\s*'\s*"  # minutes '
    r"(?:(\d{1,2}(?:\.\d+)?)\s*\"?)?)?\s*$"  # seconds "
)

# PaddleOCR routinely drops the ' and " ticks altogether, reading 32°20'40" as
# "32°20 40" (often with a trailing "|" / "1" from the extension line it sits
# on). When the degree mark IS present, two following 1–2 digit groups can only
# be minutes/seconds, so recover them without ticks.
_DMS_TICKLESS_RE = re.compile(
    r"^\s*(\d{1,3})\s*(?:°|˚|⁰)\s*"
    r"(\d{1,2})\s*'?\s*"
    r"(?:(\d{1,2})\s*\"?)?\s*$"
)
# Stray stroke glyphs OCR appends when an extension/leader line touches the
# end of the text: 1 l I | and a stray bullet / dot.
_TRAILING_STROKE_RE = re.compile(r"[\s1lI|●•·.]{1,2}$")

# Decimal degrees: 45.5° / 90°
_DEC_DEG_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*(?:°|˚|⁰)\s*$")


def has_angle_marks(text: str) -> bool:
    """True if the raw text carries any degree/minute/second evidence."""
    t = normalize_primes(text or "")
    return bool(re.search(r"°|˚|⁰", t) or re.search(r"\d\s*'", t) or re.search(r"\d\s*\"", t))


def parse_dms(text: str, require_marks: bool = True) -> str | None:
    """
    Return a canonical angular string, or None if not an angle.

    require_marks=True (default): only treat as DMS when the text actually
    contains a degree symbol or a minute/second tick — avoids turning a plain
    linear number into a bogus angle. Pass False when an external signal
    (e.g. degree-symbol vision) already established this is an angle.
    """
    t = _recover_degree_quote(normalize_primes((text or "").strip()))
    if not t:
        return None

    # Decimal degrees first (45.5°).
    m = _DEC_DEG_RE.match(t)
    if m:
        return f"{m.group(1)}°"

    if require_marks and not has_angle_marks(t):
        return None

    m = _DMS_RE.match(t)
    if not m and re.search(r"°|˚|⁰", t):
        m = _DMS_TICKLESS_RE.match(t)
        if not m:
            # Retry once with a trailing stray stroke removed ("32°20 40|").
            t2 = _TRAILING_STROKE_RE.sub("", t)
            if t2 != t and re.search(r"\d\s*$", t2):
                m = _DMS_RE.match(t2) or _DMS_TICKLESS_RE.match(t2)
    if not m:
        return None

    deg, minutes, seconds = m.group(1), m.group(2), m.group(3)

    if minutes is not None and not (0 <= int(minutes) < 60):
        return None
    if seconds is not None and not (0 <= float(seconds) < 60):
        return None

    out = f"{deg}°"
    if minutes is not None:
        out += f"{minutes}'"
        if seconds is not None:
            out += f'{seconds}"'
    return out


# An angle with an optional ± tolerance, e.g. 12°±3°, 45.5°, 120°.
_ANGLE_TOL_RE = re.compile(
    r"^(\d{1,3}(?:\.\d+)?°(?:\s*±\s*\d{1,2}(?:\.\d+)?°)?)"
    r"\s*[1lI|●•·.,:;2]{1,2}\s*$"
)


def strip_angle_tail(text: str) -> str:
    """
    Drop stray stroke glyphs after a complete angle: ``12°±3°1`` → ``12°±3°``.

    A digit or bar after a closing ° is never part of the value; it is the
    leader/extension line that touches the end of the text read as a "1"/"|".
    Only the angle-with-optional-tolerance shape is touched, so ``32°20'40"``
    and any non-angle text pass through unchanged.
    """
    m = _ANGLE_TOL_RE.match(text or "")
    return m.group(1) if m else text
