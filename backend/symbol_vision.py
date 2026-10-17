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

# ── ± template matching ──────────────────────────────────────────────────────
#
# Calibrated 2026-10-15 against 86 real regions from four drawings (47630,
# 56103-0182B, BS1801006.020, TS2 32224 cup turning): 23 carrying a ± and 63
# not, cropped from the PDFs at the dpi the benchmark fixtures record and put
# through this module's own preprocessing.
#
# The matcher had two faults, and neither could be fixed alone:
#
#   1. `_make_pm_template` drew black ink on white while `_to_gray` (via
#      `cad_ink_to_gray`) produces white ink on black. TM_CCOEFF_NORMED between
#      inverted images anti-correlates.
#   2. The score was taken over the whole padded zone. `pad_image` adds a 36px
#      white border, which becomes a large black field after the ink
#      conversion, and the correlation was dominated by background agreement
#      rather than by the glyph.
#
# Measured, at each stage (precision / recall over those 86 regions):
#
#   as-written, padded zone   th 0.34   P 1.00  R 0.04   (1 of 23 found)
#   polarity fixed only       th any    P ≤0.26          (negatives outscore
#                                                         positives: the black
#                                                         template background
#                                                         matches the padding)
#   polarity + ink-scoped     th 0.50   P 0.76  R 0.70
#                             th 0.60   P 0.83  R 0.22
#                             th 0.65   P 1.00  R 0.09
#
# 0.60 is the operating point. The reasoning is about what this flag is *for*:
# `detect_symbols`' result is OR-ed with `detect_prefix_from_ocr_text`, and the
# OCR text path reads ± on all 23 real regions — it survives 5px blur, σ=30
# noise and half resolution at 23/23. Vision can therefore only add recall
# where OCR dropped the glyph, and measured on deliberately degraded crops it
# rescued 1 of 16 such cases: when the image is bad enough to defeat the
# recogniser it defeats the template too.
#
# So the marginal recall is near zero and precision is what matters. 0.60
# takes recall from 0.04 to 0.22 — a fivefold improvement on a signal that was
# effectively dead — while adding one false positive across 63 real negative
# regions. 0.50 would reach R 0.70 but costs five, which would pull the merged
# flag's precision from 1.00 to 0.82 in exchange for recall the text path
# already provides.
PM_THRESHOLD = 0.60

#: Template sizes to try, as a fraction of the ink's own height.
_PM_SCALES = (0.7, 0.9, 1.1)


@dataclass
class DetectedSymbols:
    diameter: bool = False
    diameter_score: float = 0.0
    plus_minus: bool = False
    degree: bool = False
    radius: bool = False


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.array(cad_ink_to_gray(img).convert("L"))


def _make_pm_template(size: int) -> np.ndarray:
    """A ± glyph, **white ink on black** to match ``cad_ink_to_gray``.

    The polarity is the whole point. ``_to_gray`` returns ink as white on a
    black field, so a template drawn the other way up anti-correlates with the
    glyph it is meant to find.

    The geometry is deliberately left as it was, and that is worth recording
    because it does not look right: the vertical runs past the lower bar, so
    rendered on its own this reads closer to a ``∓`` than a ``±``. A
    faithful glyph — plus in the upper 60%, bar clear beneath — was built and
    measured against the same 86 real regions, and it was *worse*: the median
    positive-to-negative gap fell from +0.145 to +0.105 and the best F1 from
    0.73 to 0.52.

    The reason is that this template is never matched against an isolated
    glyph. It is matched against a whole ink-cropped value line such as
    ``3,41±0.30`` at roughly the line's own height, so it is scoring a coarse
    pattern across several characters rather than recognising a glyph. Glyph
    fidelity is therefore not what makes it work, and changing the shape to
    look more correct makes it worse. Re-measure before touching it.
    """
    t = np.zeros((size, size), dtype=np.uint8)
    if cv2 is None:
        return t
    c, th = size // 2, max(1, size // 10)
    cv2.line(t, (c, size // 5), (c, 4 * size // 5), 255, th)
    cv2.line(t, (size // 5, c - size // 8), (4 * size // 5, c - size // 8), 255, th)
    cv2.line(t, (size // 5, c + size // 7), (4 * size // 5, c + size // 7), 255, th)
    return t


def _ink_bounds(gray: np.ndarray) -> np.ndarray:
    """Crop a zone to the ink it actually contains.

    Without this the correlation is taken across the white border
    ``pad_image`` adds, which the ink conversion turns into a large black
    field. A template whose own background is black then agrees with the
    padding wherever it is placed, and the peak stops describing the glyph.
    """
    if cv2 is None or gray.size == 0:
        return gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return gray
    return gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _match_pm(gray: np.ndarray) -> float:
    """Best ± correlation in a zone, scored against the ink rather than the pad.

    Sizes come from the ink's own height, so the template is tried at the size
    a glyph on this line would actually be. The previous ``h // 2`` guess was
    taken from the padded zone height and bore no relation to the text.
    """
    if cv2 is None or gray.size == 0:
        return 0.0
    ink = _ink_bounds(gray)
    if ink.shape[0] < 8 or ink.shape[1] < 8:
        return 0.0
    best = 0.0
    for factor in _PM_SCALES:
        size = int(ink.shape[0] * factor)
        if size < 8 or ink.shape[0] < size or ink.shape[1] < size:
            continue
        res = cv2.matchTemplate(ink, _make_pm_template(size), cv2.TM_CCOEFF_NORMED)
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


def detect_symbols(image: Image.Image) -> tuple[DetectedSymbols, dict]:
    merged = DetectedSymbols()
    vertical = is_vertical_dimension(image)
    has_phi, phi_score, phi_strips = detect_phi_multi_strip(image, vertical)
    merged.diameter = has_phi
    merged.diameter_score = phi_score

    oriented = primary_oriented(image)
    merged.degree = _detect_degree(oriented)
    if not merged.degree:
        # A chamfer "0.5×45°" runs along a diagonal leader; its ° ring is not in
        # the upright top band. Check the deskewed view when one exists.
        from image_preprocess import deskew_to_horizontal

        deskewed = deskew_to_horizontal(oriented)
        if deskewed is not None:
            merged.degree = _detect_degree(deskewed)
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
        "vertical": vertical,
        "phi_strips": phi_strips,
        "phi_score": phi_score,
        "phi_detected": has_phi,
        "degree_detected": merged.degree,
        "pm_scores": pm_scores,
        "pm_threshold": PM_THRESHOLD,
    }
    return merged, debug


def detect_prefix_from_ocr_text(prefix_ocr: str) -> DetectedSymbols:
    t = (prefix_ocr or "").strip()
    out = DetectedSymbols()
    if re.match(r"^[ØøφΦ⌀]", t):
        out.diameter = True
        out.diameter_score = 1.0
    elif re.match(r"^[Rr](?=\d)", t):
        out.radius = True
    elif re.match(r"^[O0Q©¢C]$", t):
        # Single-char prefix OCR of Ø. Kept below the vision-only compose bar
        # (a bare "0" in the strip is more often the first digit than a Ø) —
        # it corroborates a stripped leading glyph, it does not decide alone.
        out.diameter_score = 0.4
    if "±" in t or "+/-" in t:
        out.plus_minus = True
    if re.search(r"°|˚|⁰", t):
        out.degree = True
    return out


def symbols_to_dict(s: DetectedSymbols) -> dict:
    return asdict(s)


def merge_symbol_scores(a: DetectedSymbols, b: DetectedSymbols) -> DetectedSymbols:
    """
    Combine visual and OCR-prefix evidence for the symbols.

    Diameter needs *evidence*, not a score. Either side may decide it: the
    vision pass through its own topology decision (a slashed ring, not merely
    something round), or the OCR prefix through an explicit Ø-family glyph.

    A score on its own decides nothing. The previous rule accepted any score
    at or above 0.32, which contradicted this module's own reasoning — an
    ambiguous "O"/"0" prefix is scored 0.4 precisely so that it corroborates
    rather than decides, yet 0.4 cleared the 0.32 bar. The same bar let a
    leader-line ring near a plain length publish it as a diameter, which is
    how "5.50[139.70]" reached the sheet as "Ø5.50[139.70]".
    """
    score = max(a.diameter_score, b.diameter_score)
    return DetectedSymbols(
        diameter=a.diameter or b.diameter,
        diameter_score=score,
        plus_minus=a.plus_minus or b.plus_minus,
        degree=a.degree or b.degree,
        radius=a.radius or b.radius,
    )

