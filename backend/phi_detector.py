"""
Detect diameter symbol (Ø) in the symbol strip of a dimension crop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from image_preprocess import cad_ink_to_gray, is_vertical_dimension

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def _make_phi_template(size: int) -> np.ndarray:
    t = np.full((size, size), 255, dtype=np.uint8)
    if cv2 is None:
        return t
    c, r = size // 2, max(2, size // 3)
    th = max(1, max(1, size // 12))
    cv2.circle(t, (c, c), r, 0, th)
    cv2.line(t, (c - r, c + r), (c + r, c - r), 0, th)
    return t


def _template_score(gray: np.ndarray) -> float:
    if cv2 is None or gray.size == 0:
        return 0.0
    h, w = gray.shape[:2]
    best = 0.0
    for frac in (0.45, 0.6, 0.75, 0.9):
        sz = max(10, int(min(h, w) * frac))
        if sz >= min(h, w):
            continue
        tpl = _make_phi_template(sz)
        if h < tpl.shape[0] or w < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(cv2.minMaxLoc(res)[1]))
    return best


def _hough_circle_score(gray: np.ndarray) -> float:
    if cv2 is None:
        return 0.0
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    min_r = max(3, min(h, w) // 12)
    max_r = max(min_r + 2, min(h, w) // 3)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, min_r),
        param1=80,
        param2=14,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return 0.0

    best = 0.0
    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if cx > w * 0.75:
            continue
        # Slash through circle center
        slash = 0
        for t in np.linspace(-1, 1, 7):
            px, py = int(cx + t * r * 0.9), int(cy - t * r * 0.9)
            if 0 <= px < w and 0 <= py < h and gray[py, px] < 128:
                slash += 1
        if slash >= 3:
            best = max(best, 0.55 + slash * 0.05)
    return min(1.0, best)


def _contour_score(gray: np.ndarray) -> float:
    if cv2 is None or gray.size == 0:
        return 0.0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    area_img = h * w
    best = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_img * 0.015 or area > area_img * 0.6:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.5:
            continue
        x, _, bw, bh = cv2.boundingRect(cnt)
        if x > w * 0.7:
            continue
        cx, cy = x + bw // 2, int(cv2.moments(cnt)["m01"] / max(area, 1))
        r = max(2, min(bw, bh) // 3)
        slash_hits = 0
        for t in np.linspace(-1, 1, 9):
            px, py = int(cx + t * r), int(cy - t * r)
            if 0 <= px < w and 0 <= py < h and binary[py, px] > 0:
                slash_hits += 1
        if slash_hits >= 3:
            best = max(best, min(1.0, circularity * 0.8 + slash_hits * 0.05))
    return best


def detect_phi_in_prefix(prefix_zone: Image.Image) -> tuple[bool, float]:
    gray = np.array(cad_ink_to_gray(prefix_zone).convert("L"))
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return False, 0.0

    scores = [
        _template_score(gray),
        _hough_circle_score(gray),
        _contour_score(gray),
    ]
    score = max(scores)
    return score >= 0.32, round(score, 3)


def detect_phi_multi_strip(
    image: Image.Image, vertical: bool
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Scan every plausible symbol strip (vertical bottom + oriented prefix)."""
    from image_preprocess import orientations_for_ocr, pad_image, upscale_min_edge
    from symbol_regions import enlarge_zone, split_symbol_zones

    prep = upscale_min_edge(pad_image(image))
    w, h = prep.size
    best = 0.0
    details: list[dict[str, Any]] = []
    src_vertical = is_vertical_dimension(prep)

    if src_vertical:
        for frac in (0.30, 0.38, 0.45):
            strip = max(16, int(h * frac))
            for side, crop in (
                ("bottom", prep.crop((0, h - strip, w, h))),
                ("top", prep.crop((0, 0, w, strip))),
            ):
                _, sc = detect_phi_in_prefix(enlarge_zone(crop, 5.0))
                details.append({"strip": f"vertical_{side}", "frac": frac, "score": sc})
                best = max(best, sc)

    for oname, oriented in orientations_for_ocr(prep):
        zones = split_symbol_zones(
            oriented,
            from_vertical_rotated=src_vertical and oname == "cw",
        )
        _, sc = detect_phi_in_prefix(enlarge_zone(zones["prefix"], 5.0))
        details.append({"strip": f"oriented_{oname}_prefix", "score": sc})
        best = max(best, sc)

    return best >= 0.32, round(best, 3), details
