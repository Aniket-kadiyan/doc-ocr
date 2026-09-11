"""
Compose CAD dimensions: fix digits, then Ø / ±.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image

from angle_utils import has_angle_marks, normalize_primes, parse_dms
from dimension_digits import normalize_cad_number_string
from image_preprocess import is_vertical_dimension
from symbol_normalize import fix_engineering_symbols_light
from symbol_vision import DetectedSymbols
from tolerance_utils import (
    is_tolerance_pair,
    ocr_has_diameter_marker,
    ocr_has_explicit_plus_minus,
)

_PHI_SCORE_MIN = 0.32
# Leading O *or* 0 misread for the Ø glyph (e.g. OCR "0174.07" for "Ø174.07").
# Requires 2+ following digits so a genuine "0.5" is never absorbed.
_LEADING_O_PHI_RE = re.compile(r"^[O0](\d{2,})")
_DIA_TOL_RE = re.compile(r"^(\d+\.\d{2})±(\d\.\d{2})$")


@dataclass
class ComposedDimension:
    text: str
    kind: str
    applied: list[str]


def _extract_leading_symbol(t: str) -> tuple[str, str]:
    m = re.match(r"^([\s]*[ØøφΦ⌀Rr])\s*(.*)$", t)
    if m:
        sym = m.group(1).strip()
        if sym.upper().startswith("R"):
            return "R", m.group(2).strip()
        return "Ø", m.group(2).strip()
    return "", t


def _has_phi_evidence(
    symbols: DetectedSymbols,
    raw: str,
    prefix_ocr: str,
) -> bool:
    if ocr_has_diameter_marker(raw) or ocr_has_diameter_marker(prefix_ocr):
        return True
    if re.match(r"^[ØøφΦ⌀]", (prefix_ocr or "").strip()):
        return True
    return symbols.diameter_score >= _PHI_SCORE_MIN


def _is_vertical_diameter_tolerance(body: str, vertical: bool) -> bool:
    """
    Vertical CAD dims like image 2: Ø215.37±0.05 — vision often misses Ø
    but ± + two-decimal tolerance is a strong diameter signal.
    """
    if not vertical:
        return False
    m = _DIA_TOL_RE.match(body)
    if not m:
        return False
    if not is_tolerance_pair(m.group(1), m.group(2)):
        return False
    try:
        nominal = float(m.group(1))
        tol = float(m.group(2))
    except ValueError:
        return False
    return nominal >= 1.0 and tol <= 1.0


def compose_engineering_dimension(
    raw_ocr: str,
    image: Image.Image | None,
    symbols: DetectedSymbols,
    prefix_ocr: str = "",
) -> ComposedDimension:
    applied: list[str] = []
    raw = (raw_ocr or "").strip()
    if not raw and not prefix_ocr.strip():
        return ComposedDimension("", "unknown", applied)

    vertical = image is not None and is_vertical_dimension(image)

    combined = raw
    p = prefix_ocr.strip()
    if p and len(p) <= 3 and re.match(r"^[ØøφΦ⌀Rr]", p):
        if not raw.startswith(p[0]):
            combined = p + raw

    normalized = normalize_cad_number_string(combined)
    if normalized.replace(" ", "") != combined.replace(" ", ""):
        applied.append("cad_digit_fix")

    t = fix_engineering_symbols_light(normalized)

    # --- Angle (DMS / decimal degrees) — resolve before Ø/± heuristics so a
    # degree circle is never mistaken for a diameter (Ø) prefix. ---
    degree_evidence = (
        symbols.degree
        or has_angle_marks(raw)
        or has_angle_marks(prefix_ocr)
        or "°" in t
    )
    angle = parse_dms(t, require_marks=not degree_evidence)
    if angle is None and degree_evidence:
        angle = parse_dms(normalize_primes(t), require_marks=False)
    if angle is not None:
        applied.append("angle_dms")
        return ComposedDimension(angle, "angle", applied)

    sym, rest = _extract_leading_symbol(t)
    t = rest if sym else t
    if sym == "Ø":
        t = "Ø" + t.lstrip()
    elif sym == "R":
        t = "R" + t.lstrip()

    phi_ok = _has_phi_evidence(symbols, raw, prefix_ocr)

    if not t.startswith("Ø") and not t.startswith("R"):
        m = _LEADING_O_PHI_RE.match(t)
        if m and phi_ok:
            t = "Ø" + m.group(1) + t[m.end() :]
            applied.append("leading_o_phi")

    if ocr_has_explicit_plus_minus(raw):
        t = t.replace("+/-", "±").replace("+/−", "±")
        applied.append("explicit_pm")

    body = t.lstrip("ØR")
    m_tol = _DIA_TOL_RE.match(body)

    if not t.startswith("Ø") and not t.startswith("R") and m_tol:
        if _is_vertical_diameter_tolerance(body, vertical):
            t = "Ø" + body
            applied.append("vertical_dia_tol")
        elif phi_ok:
            t = "Ø" + body
            applied.append("phi_prefix")
    elif phi_ok and re.match(r"^\d", t) and not t.startswith("Ø"):
        t = "Ø" + t
        applied.append("phi_prefix")

    if symbols.radius and re.match(r"^\d", t) and not t.startswith("R"):
        t = "R" + t
        applied.append("radius")

    kind = "linear"
    if "±" in t:
        kind = "tolerance"
    if t.startswith("Ø"):
        kind = "diameter"
    if t.startswith("R"):
        kind = "radius"
    if "°" in t:
        kind = "angle"

    return ComposedDimension(fix_engineering_symbols_light(t), kind, applied)

