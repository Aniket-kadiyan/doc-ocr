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
