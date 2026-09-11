"""Conservative rules for ± tolerance pairs — avoid false ± and false Ø."""

from __future__ import annotations

import re


def is_tolerance_pair(main: str, tol: str) -> bool:
    """
    True when second number looks like a tolerance on first (e.g. 174.07 + 0.05).
    Not for unrelated numbers like 50 - 20 or 100 + 200.
    """
    try:
        v_main = float(main)
        v_tol = float(tol)
    except ValueError:
        return False

    if v_tol <= 0 or v_main <= 0:
        return False

    # Tolerance is usually much smaller than nominal
    if v_tol >= v_main * 0.5:
        return False

    # Typical sheet-metal tolerances: 0.0x, 0.1x, 0.5, 1.0
    if v_tol <= 2.0:
        return True

    # Decimal tolerance with fewer digits than main
    if "." in tol and v_tol < v_main:
        return True

    return False


def ocr_has_explicit_plus_minus(text: str) -> bool:
    return bool(re.search(r"±|\+/-|\+/\-", text))


def ocr_has_diameter_marker(text: str) -> bool:
    return bool(re.match(r"^[\s]*[ØøφΦ⌀∅]", text))


def ocr_has_radius_marker(text: str) -> bool:
    return bool(re.match(r"^[\s]*[Rr](?=\d)", text))
