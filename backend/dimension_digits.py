"""
Fix common Paddle misreads on CAD dimensions (comma as ±, glued decimals).
"""

from __future__ import annotations

import re

from angle_utils import normalize_primes, parse_dms
from tolerance_utils import is_tolerance_pair

# 215,370.05  — comma where ± was; 370.05 is 37 + 0.05
_COMMA_PM_RE = re.compile(r"^(\d{2,}),(\d{2})(\d)\.(\d{2})$")
# 215.370.05 — glued nominal + tolerance
_GLUED_TOL_RE = re.compile(r"^(\d+\.\d{1,4})(\d)\.(\d{1,4})$")
# 215,37,05 or 215,37.05
_COMMA_DEC_RE = re.compile(r"^(\d{2,}),(\d{2})[,.](\d{2})$")


# Letters PaddleOCR commonly substitutes for digits. Applied ONLY when the
# character is adjacent to a digit, so we don't corrupt a real Ø / R / unit.
_CONFUSABLES = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "i": "1", "|": "1",
    "S": "5", "s": "5",
    "Z": "2", "z": "2",
    "B": "8",
    "g": "9", "q": "9",
    "G": "6",
}


def correct_numeric_confusables(t: str) -> tuple[str, bool]:
    """
    Replace letter-for-digit misreads that sit next to digits. Returns the
    corrected string and whether any substitution was made (which callers use
    to flag the read for manual review — these corrections are best-effort).
    """
    if not t:
        return t, False
    chars = list(t)
    changed = False
    for i, ch in enumerate(chars):
        if ch not in _CONFUSABLES:
            continue
        prev_d = i > 0 and chars[i - 1].isdigit()
        next_d = i + 1 < len(chars) and chars[i + 1].isdigit()
        if prev_d or next_d:
            chars[i] = _CONFUSABLES[ch]
            changed = True
    return "".join(chars), changed


def _strip_dimension_spaces(t: str) -> str:
    """
    Drop spaces inside a number/symbol run ("215 , 37" → "215,37",
    "Ø 174" → "Ø174") while keeping spaces *between words* so label text such as
    "FINISH TURNED DIMENSIONS" is not glued into one token. A space is preserved
    only when it has a letter on both sides.
    """
    out: list[str] = []
    for i, ch in enumerate(t):
        if ch == " ":
            prev = t[i - 1] if i else ""
            nxt = t[i + 1] if i + 1 < len(t) else ""
            if prev.isalpha() and nxt.isalpha():
                out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def normalize_cad_number_string(t: str) -> str:
    """Normalize separators before tolerance / phi compose."""
    if not t:
        return t

    s = normalize_primes(t.strip())

    # Angular dimensions (32°20'40") must keep their minute/second ticks.
    dms = parse_dms(s, require_marks=True)
    if dms is not None:
        return dms

    s = _strip_dimension_spaces(s).replace("'", "").replace('"', "")

    # European comma decimals in short dims only (215,37 → 215.37)
    m = _COMMA_DEC_RE.match(s)
    if m:
        s = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"

    # Comma as ± : 215,370.05 → 215.37±0.05
    m = _COMMA_PM_RE.match(s)
    if m:
        main = f"{m.group(1)}.{m.group(2)}"
        tol = f"{m.group(3)}.{m.group(4)}"
        if is_tolerance_pair(main, tol):
            return f"{main}±{tol}"

    # Glued: 215.370.05 → 215.37±0.05
    m = _GLUED_TOL_RE.match(s)
    if m:
        main = m.group(1)
        tol = f"{m.group(2)}.{m.group(3)}"
        if is_tolerance_pair(main, tol):
            return f"{main}±{tol}"

    m = re.match(r"^(\d+\.\d+)\+(\d+\.\d+)$", s)
    if m and is_tolerance_pair(m.group(1), m.group(2)):
        return f"{m.group(1)}±{m.group(2)}"

    # 215.37,0.05 or 215.37,05
    m = re.match(r"^(\d+\.\d+)[,.](\d{1,2})(?:\.(\d{1,4}))?$", s)
    if m:
        main = m.group(1)
        if m.group(3):
            tol = f"{m.group(2)}.{m.group(3)}"
        else:
            tol = f"0.{m.group(2).zfill(2)}"
        if is_tolerance_pair(main, tol):
            return f"{main}±{tol}"

    return s
