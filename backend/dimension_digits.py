"""
Fix common Paddle misreads on CAD dimensions (comma as ±, glued decimals).
"""

from __future__ import annotations

import re

from angle_utils import normalize_primes, parse_dms
from page_value_filters import canonical_hole_glyphs
from tolerance_utils import is_tolerance_pair

# 215,370.05  — comma where ± was; 370.05 is 37 + 0.05
_COMMA_PM_RE = re.compile(r"^(\d{2,}),(\d{2})(\d)\.(\d{2})$")
# 215.370.05 — glued nominal + tolerance
_GLUED_TOL_RE = re.compile(r"^(\d+\.\d{1,4})(\d)\.(\d{1,4})$")
# 215,37,05 or 215,37.05
_COMMA_DEC_RE = re.compile(r"^(\d{2,}),(\d{2})[,.](\d{2})$")
# Ø0.62000.612 — a limit dimension drawn as two values one above the other
# ("0.620" over "0.612") and read back on one line with nothing between them.
_GLUED_LIMITS_RE = re.compile(r"^([ØøR]?)(\d+)\.(\d+)\.(\d+)$")


# Letters PaddleOCR commonly substitutes for digits.
_CONFUSABLES = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "i": "1", "|": "1",
    "S": "5", "s": "5",
    "Z": "2", "z": "2",
    "B": "8",
    "G": "6",
}
# Not in the table: the lowercase g and q that older versions mapped to 9. A
# lowercase g straight after a digit is a metric thread tolerance class ("6g",
# "4g6g") far more often than a misread nine on a drawing set in capitals.

# The subset that may be repaired at the EDGE of a digit run, where only one
# neighbour is a digit. These are the zero and one look-alikes: no engineering
# notation puts an O or an I straight after a number (ISO reserves both as
# datum letters for exactly that reason), so "1O" is ten and "O.5" is a half.
# Every other letter in the table is also a real suffix somewhere -- "2B" is a
# UN thread class, "6g" and "6H" are metric ones, "H7" a fit, "3D" a view --
# and turning those into "28", "69", "6H", "37" or "30" invents a different
# dimension. They are repaired only when boxed in by digits on both sides,
# where no suffix reading exists.
_EDGE_CONFUSABLES = {"O": "0", "Q": "0", "l": "1", "I": "1", "i": "1", "|": "1"}

# Characters that may sit inside a numeric token without making it text: the
# digits, separators, signs and the engineering marks a value carries.
_NUMERIC_TOKEN_CHARS = set("0123456789.,+-−±°×x/[]()'\"ØøφΦ⌀∅R ")


def _token_is_numeric(token: str) -> bool:
    return any(ch.isdigit() for ch in token) and all(
        ch in _NUMERIC_TOKEN_CHARS for ch in token
    )


def _digit_beside(
    t: str, index: int, step: int, start: int, end: int, numeric_token: bool
) -> bool:
    """Is there a digit on the ``step`` side of ``index`` within the token?

    One decimal separator may sit between them ("O.59", "1S.5"), and a
    neighbouring letter counts as a digit when it is itself a confusable in a
    token that reads as a number once repaired ("2OO").
    """
    j = index + step
    if j < start or j >= end:
        return False
    ch = t[j]
    if ch.isdigit() or (numeric_token and ch in _CONFUSABLES):
        return True
    if ch in ".,":
        k = j + step
        return start <= k < end and (
            t[k].isdigit() or (numeric_token and t[k] in _CONFUSABLES)
        )
    return False


def correct_numeric_confusables(t: str) -> tuple[str, bool]:
    """
    Repair letter-for-digit misreads, and only those.

    A substitution needs the shape of a number around it:

    * a letter boxed in by digits on both sides ("2l5.37", "1O.0", "R0.2S0")
      can only be a digit, so the whole table applies;
    * a letter with a digit on one side only is repaired just for the zero and
      one look-alikes ("1O", "O.59", "Ø2O"), and only when the token it sits
      in reads as a plain number afterwards -- so "UNC-2B" keeps its thread
      class, "SC-3" its tag and "M54x1.5-6H" its tolerance class, because
      the rest of the token is text;
    * a letter touching another letter is part of a word and is never touched,
      unless that neighbour is itself a confusable inside a numeric token.

    Returns the corrected string and whether anything changed, which callers
    use to flag the read for review -- these repairs are evidence-based but
    still guesses.
    """
    if not t:
        return t, False
    chars = list(t)
    changed = False
    for match in re.finditer(r"\S+", t):
        start, end = match.span()
        token = t[start:end]
        # Does the token read as a number once every candidate is repaired?
        # Decided on the token as a whole so an edge repair never fires on a
        # code, a class or a word fragment.
        repaired_all = "".join(_CONFUSABLES.get(ch, ch) for ch in token)
        # At least one genuine digit is required: a token made only of
        # look-alike letters ("SIO") is a word, not a number with every
        # digit misread.
        numeric_token = any(ch.isdigit() for ch in token) and _token_is_numeric(
            repaired_all
        )
        for i in range(start, end):
            ch = chars[i]
            if ch not in _CONFUSABLES:
                continue
            prev_d = _digit_beside(t, i, -1, start, end, numeric_token)
            next_d = _digit_beside(t, i, 1, start, end, numeric_token)
            if prev_d and next_d:
                chars[i] = _CONFUSABLES[ch]
                changed = True
            elif (prev_d or next_d) and numeric_token and ch in _EDGE_CONFUSABLES:
                chars[i] = _EDGE_CONFUSABLES[ch]
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


def _split_glued_limits(s: str) -> str | None:
    """
    Separate a stacked limit pair that OCR ran together, or return ``None``.

    A limit dimension is two values of the same form, one above the other,
    and a flat read glues them: "0.620" over "0.612" arrives as "0.62000.612".
    The join point is the only one where both halves have the same number of
    decimals, the same integer part (allowing the leading zero OCR adds when
    it reads the lower value's gap as a digit), and values within ten percent
    of each other -- which is what limits of one feature look like and what
    a tolerance glued to its nominal ("215.370.05") does not.
    """
    m = _GLUED_LIMITS_RE.match(s)
    if not m:
        return None
    prefix, head, middle, lower_dec = m.groups()
    # middle = <decimals of upper><integer of lower>: the lower value's
    # decimals fix how many digits the upper one has.
    n = len(lower_dec)
    if len(middle) <= n:
        return None
    upper_dec = middle[:n]
    lower_int = middle[n:].lstrip("0") or "0"
    upper_int = head.lstrip("0") or "0"
    if lower_int != upper_int:
        return None
    upper = f"{upper_int}.{upper_dec}"
    lower = f"{lower_int}.{lower_dec}"
    try:
        a, b = float(upper), float(lower)
    except ValueError:
        return None
    if a == b or abs(a - b) > 0.1 * max(a, b):
        return None
    return f"{prefix}{upper}/{lower}"


def normalize_cad_number_string(t: str) -> str:
    """Normalize separators before tolerance / phi compose."""
    if not t:
        return t

    s = canonical_hole_glyphs(normalize_primes(t.strip()))

    # Angular dimensions (32°20'40") must keep their minute/second ticks.
    dms = parse_dms(s, require_marks=True)
    if dms is not None:
        return dms

    s = _strip_dimension_spaces(s)
    if "°" in s:
        # A toleranced angle is not a bare DMS triple, but its ticks are just
        # as real: "90°±0°30'" lost its minute and published "30" of nothing.
        # Only a tick that closes a number is kept; a stray one is still noise.
        s = re.sub(r"(?<!\d)['\"]", "", s)
    else:
        s = s.replace("'", "").replace('"', "")

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

    # Glued limit pair: 0.62000.612 → 0.620/0.612 (checked before the glued
    # tolerance below, which would otherwise read the same string as
    # 0.6200 ± 0.612 -- a tolerance larger than its nominal).
    limits = _split_glued_limits(s)
    if limits is not None:
        return limits

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
