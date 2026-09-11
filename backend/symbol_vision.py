"""
Detect ±, °, Ø from pixels. Diameter (phi) uses multi-strip vision.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from image_preprocess import cad_ink_to_gray, is_vertical_dimension, primary_oriented
from phi_detector import detect_phi_multi_strip
from symbol_regions import enlarge_zone, split_symbol_zones

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

PM_THRESHOLD = 0.34


@dataclass
class DetectedSymbols:
    diameter: bool = False
    diameter_score: float = 0.0
    diameter_visual_score: float = 0.0
    diameter_visual_candidate: bool = False
    diameter_ocr_explicit: bool = False
    diameter_ocr_ambiguous: bool = False
    plus_minus: bool = False
    degree: bool = False
    radius: bool = False


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.array(cad_ink_to_gray(img).convert("L"))


def _make_pm_template(size: int) -> np.ndarray:
    t = np.full((size, size), 255, dtype=np.uint8)
    if cv2 is None:
        return t
    c, th = size // 2, max(1, size // 10)
    cv2.line(t, (c, size // 5), (c, 4 * size // 5), 0, th)
    cv2.line(t, (size // 5, c - size // 8), (4 * size // 5, c - size // 8), 0, th)
    cv2.line(t, (size // 5, c + size // 7), (4 * size // 5, c + size // 7), 0, th)
    return t


def _match_pm(gray: np.ndarray) -> float:
    if cv2 is None or gray.size == 0:
        return 0.0
    h = max(gray.shape[0], 1)
    base = max(12, min(h // 2, 72))
    best = 0.0
    for sz in (base, int(base * 1.3)):
        tpl = _make_pm_template(sz)
        if gray.shape[0] < tpl.shape[0] or gray.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(cv2.minMaxLoc(res)[1]))
    return best


def _detect_degree(image: Image.Image) -> bool:
    """
    Degree mark = a *small* high-circularity ring sitting in the upper band,
    with NO diagonal slash through it (that would be a Ø). Conservative on
    purpose: a false positive would turn a linear dimension into an angle.
    """
    if cv2 is None:
        return False
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    if h < 12 or w < 12:
        return False

    # Only look at the top ~55% — the degree glyph is superscript.
    band = gray[: max(6, int(h * 0.55)), :]
    _, binary = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area <= 0 or peri <= 0:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Small, roughly square glyph (a ring), not a tall digit.
        if bh > h * 0.5 or bw > w * 0.5:
            continue
        if min(bw, bh) < 4 or max(bw, bh) / max(min(bw, bh), 1) > 1.8:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.6:
            continue
        # Reject if a diagonal slash crosses the centre (that's a Ø, not °).
        cx, cy = x + bw // 2, y + bh // 2
        r = max(2, min(bw, bh) // 3)
        slash = 0
        for tt in np.linspace(-1, 1, 9):
            px, py = int(cx + tt * r), int(cy - tt * r)
            if 0 <= px < binary.shape[1] and 0 <= py < binary.shape[0] and binary[py, px] > 0:
                slash += 1
        if slash >= 4:
            continue
        return True
    return False


def detect_diameter_symbol(
    image: Image.Image,
) -> tuple[DetectedSymbols, dict]:
    """Run only diameter vision, for both single and batched recognition."""

    vertical = is_vertical_dimension(image)
    has_phi, phi_score, phi_strips = detect_phi_multi_strip(image, vertical)
    visual_candidate = any(
        bool(detail.get("support", {}).get("ring"))
        and (
            bool(detail.get("support", {}).get("template"))
            or bool(detail.get("support", {}).get("contour"))
        )
        for detail in phi_strips
    )
    symbols = DetectedSymbols(
        diameter=has_phi,
        diameter_score=phi_score,
        diameter_visual_score=phi_score,
        diameter_visual_candidate=visual_candidate,
    )
    return symbols, {
        "vertical": vertical,
        "phi_strips": phi_strips,
        "phi_score": phi_score,
        "phi_detected": has_phi,
        "phi_candidate": visual_candidate,
    }


def detect_symbols(image: Image.Image) -> tuple[DetectedSymbols, dict]:
    merged, diameter_debug = detect_diameter_symbol(image)
    vertical = bool(diameter_debug["vertical"])

    oriented = primary_oriented(image)
    merged.degree = _detect_degree(oriented)
    zones = split_symbol_zones(
        oriented,
        from_vertical_rotated=vertical,
    )
    pm_scores: dict[str, float] = {}
    for key in ("prefix", "body"):
        gray = _to_gray(enlarge_zone(zones[key]) if key == "prefix" else zones[key])
        if gray.shape[0] < 6 or gray.shape[1] < 6:
            continue
        sc = _match_pm(gray)
        pm_scores[key] = round(sc, 3)
        if sc >= PM_THRESHOLD:
            merged.plus_minus = True

    debug = {
        **diameter_debug,
        "degree_detected": merged.degree,
        "pm_scores": pm_scores,
        "pm_threshold": PM_THRESHOLD,
    }
    return merged, debug


def detect_prefix_from_ocr_text(prefix_ocr: str) -> DetectedSymbols:
    t = (prefix_ocr or "").strip()
    compact = "".join(t.split())
    out = DetectedSymbols()
    if re.match(r"^[ØøφΦ⌀∅]", compact):
        out.diameter = True
        out.diameter_score = 1.0
        out.diameter_ocr_explicit = True
    elif re.match(r"^[Rr](?=\d)", t):
        out.radius = True
    elif re.match(r"^[O0Q©¢C](?:$|(?=\d{2}))", compact):
        # Useful for deciding to re-read the prefix, but never sufficient to
        # create Ø: these are also legitimate CAD characters and digits.
        out.diameter_ocr_ambiguous = True
        out.diameter_score = 0.20
    if "±" in t or "+/-" in t:
        out.plus_minus = True
    if re.search(r"°|˚|⁰", t):
        out.degree = True
    return out


def symbols_to_dict(s: DetectedSymbols) -> dict:
    return asdict(s)


def merge_symbol_scores(a: DetectedSymbols, b: DetectedSymbols) -> DetectedSymbols:
    score = max(a.diameter_score, b.diameter_score)
    visual_score = max(a.diameter_visual_score, b.diameter_visual_score)
    explicit = a.diameter_ocr_explicit or b.diameter_ocr_explicit
    return DetectedSymbols(
        # A numeric score or O-like OCR character is not a decision. Preserve
        # only a confirmed visual result or an explicit diameter glyph.
        diameter=a.diameter or b.diameter or explicit,
        diameter_score=1.0 if explicit else score,
        diameter_visual_score=visual_score,
        diameter_visual_candidate=(
            a.diameter_visual_candidate or b.diameter_visual_candidate
        ),
        diameter_ocr_explicit=explicit,
        diameter_ocr_ambiguous=(
            a.diameter_ocr_ambiguous or b.diameter_ocr_ambiguous
        ),
        plus_minus=a.plus_minus or b.plus_minus,
        degree=a.degree or b.degree,
        radius=a.radius or b.radius,
    )
