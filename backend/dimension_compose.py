"""
Compose CAD dimensions: fix digits, then Ø / ±.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image

from angle_utils import has_angle_marks, normalize_primes, parse_dms, strip_angle_tail
from dimension_digits import normalize_cad_number_string
from image_preprocess import is_vertical_dimension
from symbol_normalize import fix_engineering_symbols_light
from symbol_vision import DetectedSymbols
from tolerance_utils import (
    is_tolerance_pair,
    ocr_has_diameter_marker,
    ocr_has_explicit_plus_minus,
)

# Trust OCR first, corroborate with vision. When the OCR text/prefix already
# carries the Ø marker the bar is low; when only the (brittle) template-match
# vision fires, require a much stronger score before injecting a symbol. This
# is the main lever against false-positive Ø/° reads.
_PHI_SCORE_MIN = 0.32          # corroborated by OCR text/prefix
_PHI_SCORE_VISION_ONLY = 0.5   # vision alone, no OCR marker
# Leading O, 0 *or* 8 misread for the Ø glyph (e.g. OCR "0174.07" for
# "Ø174.07", or "80103" for "Ø0103" — the slash through the ring closes the
# upper bowl and the glyph reads as an eight). Requires 2+ following digits so
# a genuine "0.5" is never absorbed, and only ever applies when the vision pass
# has CONFIRMED a diameter, so an ordinary "80.5" is untouched.
_LEADING_O_PHI_RE = re.compile(r"^[O08](\d{2,})")
_DIA_TOL_RE = re.compile(r"^(\d+\.\d{2})±(\d\.\d{2})$")
# A plausible angle number: a bare integer 0–360. Decimals like 50.20 or a ±
# tolerance are NOT angle-like, so a stray vision "degree ring" can't convert a
# linear/diameter dimension into an angle without textual corroboration.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# PaddleOCR frequently reads the leading Ø / R prefix glyph as a dash, and a CAD
# dimension is never negative — so a minus sign in front of the value is a
# misread prefix, not a sign.
_LEADING_MINUS_RE = re.compile(r"^\s*[-−–—~]\s*(?=[\dØøφΦ⌀Rr.])")
# A leading slash is the leader line clipped into the crop, not a glyph. It is
# removed without counting as prefix evidence — unlike the dash above, which is
# how a Ø or R often comes back. (A slash *inside* a value separates stacked
# deviations, so only a leading one is removed.)
_LEADING_STROKE_RE = re.compile(r"^\s*[/\\]\s*(?=[\dØøφΦ⌀Rr.])")
# Letters OCR emits for the Ø glyph itself (W/V/N/Y for a slashed ring, O/Q/C/D
# /G/U/@ for the ring) at the very start of a value. Replaced by Ø only when
# the topology detector has independently confirmed a slashed ring, so a real
# "M8" thread or "C1" chamfer prefix (not in the set / no ring) is untouched.
_LEADING_JUNK_PHI_RE = re.compile(r"^\s*[WVNYOQCDGU@%&#(\[]{1,2}(?=\d{2,})")
_PHI_TOPOLOGY_MIN = 0.8
# Chamfer / lead-in callout: "0.5×45", "0.2-0.3 X 45". The trailing integer is
# an angle whose ° mark OCR dropped; injected only with degree-vision evidence.
_CHAMFER_RE = re.compile(r"^(.*\d)\s*[x×X]\s*(\d{1,3})\s*$")
# "size × angle" with a standard chamfer angle and a small size is a chamfer
# callout whatever OCR dropped: 0.5×45 can only mean 0.5×45°. (A spline
# callout 20×19×1 has two × and a non-standard trailing value, so it is not.)
_CHAMFER_ANGLES = {15, 20, 30, 45, 60}
_X_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*[xX]\s*(?=\d)")


def _max_number(text: str) -> float:
    vals = []
    for tok in _NUMBER_RE.findall(text):
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return max(vals) if vals else 0.0


def _plausible_angle_number(text: str) -> bool:
    """True when the text contains a bare integer (no decimal) in 0–360."""
    if "±" in text:
        return False
    for tok in _NUMBER_RE.findall(text):
        if "." in tok:
            continue
        try:
            if 0 <= int(tok) <= 360:
                return True
        except ValueError:
            continue
    return False


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
    leading_glyph: bool = False,
) -> bool:
    if ocr_has_diameter_marker(raw) or ocr_has_diameter_marker(prefix_ocr):
        return True
    if re.match(r"^[ØøφΦ⌀]", (prefix_ocr or "").strip()):
        return True
    # A stripped leading dash is itself textual evidence that *some* prefix glyph
    # was there, so a modest (corroborated) vision score is enough to call it Ø.
    # Without that hint, vision alone must clear a stronger bar to avoid injecting
    # a false Ø onto a plain linear dimension.
    if leading_glyph:
        # A stripped leading dash is WEAK evidence: a leader line clipped into
        # the crop reads exactly like a misread Ø prefix. So the dash only
        # lowers the bar for a vision pass that actually CONFIRMED a diameter;
        # a score on its own, however high, no longer suffices. Without this a
        # plain "5.50[139.70]" whose leader read as a dash was published as
        # "Ø5.50[139.70]".
        return symbols.diameter and symbols.diameter_score >= _PHI_SCORE_MIN
    return symbols.diameter_score >= _PHI_SCORE_VISION_ONLY



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

    # A leading minus is a misread Ø/R prefix, not a sign — drop it so the value
    # reads correctly and the Ø/R injection below can re-apply the real prefix
    # from symbol evidence. The strip is itself a hint that a prefix glyph was
    # present (see _has_phi_evidence / radius below).
    if _LEADING_STROKE_RE.match(t):
        t = _LEADING_STROKE_RE.sub("", t, count=1)
        applied.append("strip_leading_stroke")

    leading_glyph = bool(_LEADING_MINUS_RE.match(t))
    if leading_glyph:
        t = _LEADING_MINUS_RE.sub("", t, count=1)
        applied.append("strip_leading_minus")

    # --- Angle (DMS / decimal degrees) — resolve before Ø/± heuristics so a
    # degree circle is never mistaken for a diameter (Ø) prefix. ---
    # OCR-read degree marks are trusted directly. A vision-only degree ring must
    # be corroborated by a plausible angle number — otherwise a ring-shaped
    # artifact (part of a 0/6/8/9 digit, or the Ø glyph) would turn a linear or
    # diameter dimension into a false "angle".
    ocr_degree = has_angle_marks(raw) or has_angle_marks(prefix_ocr) or "°" in t
    vision_degree = symbols.degree and _plausible_angle_number(t)
    degree_evidence = ocr_degree or vision_degree
    angle = parse_dms(t, require_marks=not degree_evidence)
    if angle is None and degree_evidence:
        angle = parse_dms(normalize_primes(t), require_marks=False)
    if angle is not None:
        applied.append("angle_dms")
        return ComposedDimension(angle, "angle", applied)

    # "12°±3°1": a stroke read off the leader after a complete angle.
    stripped_tail = strip_angle_tail(t)
    if stripped_tail != t:
        t = stripped_tail
        applied.append("angle_tail")

    # Chamfer "0.5×45" whose ° mark OCR dropped: normalise the multiplier and
    # inject the degree when vision saw the ring after the angle.
    t = _X_BETWEEN_DIGITS_RE.sub("×", t)
    m_ch = _CHAMFER_RE.match(t)
    if m_ch and "°" not in t:
        size, ang = m_ch.group(1), int(m_ch.group(2))
        standard = (
            ang in _CHAMFER_ANGLES
            and "×" not in size
            and _max_number(size) <= 10.0
        )
        if standard or (symbols.degree and 0 < ang <= 90):
            t = f"{size}×{ang}°"
            applied.append("chamfer_degree")

    sym, rest = _extract_leading_symbol(t)
    t = rest if sym else t
    if sym == "Ø":
        t = "Ø" + t.lstrip()
    elif sym == "R":
        t = "R" + t.lstrip()

    phi_ok = _has_phi_evidence(symbols, raw, prefix_ocr, leading_glyph=leading_glyph)

    # "W215.37": the Ø glyph itself came back as a letter. With a confirmed
    # slashed ring at the reading start, the letter IS the Ø.
    if (
        not t.startswith("Ø")
        and not t.startswith("R")
        and symbols.diameter_score >= _PHI_TOPOLOGY_MIN
    ):
        m_junk = _LEADING_JUNK_PHI_RE.match(t)
        if m_junk:
            t = "Ø" + t[m_junk.end() :]
            applied.append("junk_prefix_phi")

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
        # A vertical ±two-decimal value used to be promoted to Ø on layout
        # alone ("vertical_dia_tol"); that mislabelled every vertical linear
        # tolerance (3.81±0.20). The slashed-ring topology detector now
        # supplies the evidence, so the promotion needs it like any other.
        if phi_ok:
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

