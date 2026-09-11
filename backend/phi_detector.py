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


# A diameter decision needs agreement between independent shape checks.  The
# old implementation accepted the maximum of three scores at 0.32; scanning
# several overlapping strips made an accidental circle/line match enough to
# invent Ø.  These are deliberately method-specific because the score ranges
# are not interchangeable.
PHI_TEMPLATE_SUPPORT = 0.45
PHI_HOUGH_SUPPORT = 0.70
PHI_CONTOUR_SUPPORT = 0.62
PHI_RING_SUPPORT = 0.68
PHI_DIAGONAL_SUPPORT = 0.62


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


def _diagonal_slash_score(gray: np.ndarray) -> float:
    """Measure a continuous, directional slash through the leading glyph.

    Round characters such as ``0``, ``O`` and ``8`` can score highly in circle
    detectors.  A real Ø has an asymmetric diagonal running through the glyph;
    ordinary round digits have similar occupancy on both diagonals.  Work on
    connected components so nearby digits cannot collectively form the slash.
    """

    if cv2 is None or gray.size == 0:
        return 0.0
    _, binary = cv2.threshold(
        gray,
        0,
        1,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    image_h, image_w = gray.shape[:2]
    image_area = image_h * image_w
    best = 0.0

    for index in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[index])
        if area < image_area * 0.002 or x > image_w * 0.72:
            continue
        if height < image_h * 0.25 or width < 4:
            continue

        component = binary[y : y + height, x : x + width]
        diagonal_occupancy: list[float] = []
        for direction in (-1, 1):
            hits = 0
            sample_count = 21
            band = max(1, int(min(width, height) * 0.035))
            for position in np.linspace(0.15, 0.85, sample_count):
                px = int(position * (width - 1))
                diagonal_y = 1.0 - position if direction == -1 else position
                py = int(diagonal_y * (height - 1))
                neighborhood = component[
                    max(0, py - band) : min(height, py + band + 1),
                    max(0, px - band) : min(width, px + band + 1),
                ]
                if neighborhood.any():
                    hits += 1
            diagonal_occupancy.append(hits / sample_count)

        dominant = max(diagonal_occupancy)
        other = min(diagonal_occupancy)
        directionality = dominant - other
        # Continuous occupancy matters, but it must also prefer one diagonal.
        score = dominant * min(1.0, 0.5 + directionality)
        best = max(best, score)

    return min(1.0, best)


def _ring_score(gray: np.ndarray) -> float:
    """Measure closed-loop coverage around Hough circle candidates."""

    if cv2 is None or gray.size == 0:
        return 0.0
    height, width = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    min_radius = max(3, min(height, width) // 12)
    max_radius = max(min_radius + 2, min(height, width) // 3)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, min_radius),
        param1=80,
        param2=14,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return 0.0

    best = 0.0
    for circle in circles[0]:
        cx, cy, radius = (int(value) for value in circle)
        if cx > width * 0.75:
            continue
        quadrant_hits = [0, 0, 0, 0]
        quadrant_samples = [0, 0, 0, 0]
        for index, angle in enumerate(
            np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        ):
            quadrant = index // 12
            quadrant_samples[quadrant] += 1
            hit = False
            for sample_radius in np.linspace(radius * 0.78, radius * 1.22, 9):
                px = int(cx + np.cos(angle) * sample_radius)
                py = int(cy + np.sin(angle) * sample_radius)
                if not (1 <= px < width - 1 and 1 <= py < height - 1):
                    continue
                if gray[py - 1 : py + 2, px - 1 : px + 2].min() < 128:
                    hit = True
                    break
            if hit:
                quadrant_hits[quadrant] += 1

        quadrant_coverage = [
            hits / max(samples, 1)
            for hits, samples in zip(quadrant_hits, quadrant_samples)
        ]
        coverage = sum(quadrant_hits) / max(sum(quadrant_samples), 1)
        # Reject arcs/open shapes by penalizing a missing quadrant.
        balanced_coverage = coverage * min(
            1.0,
            min(quadrant_coverage) / 0.55,
        )
        best = max(best, balanced_coverage)

    return min(1.0, best)


def _prepare_phi_gray(prefix_zone: Image.Image) -> np.ndarray:
    """Return conventional grayscale: white paper and dark CAD ink.

    ``cad_ink_to_gray`` intentionally produces a bright ink mask for OCR
    preprocessing.  The template, Hough slash check, and contour code in this
    module all expect the opposite polarity.  Feeding the mask directly made
    blank background count as a diagonal slash and was the largest source of
    false-positive diameter symbols.
    """

    ink_mask = np.array(cad_ink_to_gray(prefix_zone).convert("L"))
    return np.subtract(255, ink_mask, dtype=np.uint8)


def _fuse_phi_scores(
    template_score: float,
    hough_score: float,
    contour_score: float,
    ring_score: float,
    diagonal_score: float,
) -> dict[str, Any]:
    """Require both a round glyph and an independently measured slash."""

    scores = {
        "template": max(0.0, min(1.0, float(template_score))),
        "hough": max(0.0, min(1.0, float(hough_score))),
        "contour": max(0.0, min(1.0, float(contour_score))),
        "ring": max(0.0, min(1.0, float(ring_score))),
        "diagonal": max(0.0, min(1.0, float(diagonal_score))),
    }
    support = {
        "template": scores["template"] >= PHI_TEMPLATE_SUPPORT,
        "hough": scores["hough"] >= PHI_HOUGH_SUPPORT,
        "contour": scores["contour"] >= PHI_CONTOUR_SUPPORT,
        "ring": scores["ring"] >= PHI_RING_SUPPORT,
        "diagonal": scores["diagonal"] >= PHI_DIAGONAL_SUPPORT,
    }
    support_count = sum(1 for value in support.values() if value)

    confidence = scores["diagonal"] * 0.55 + scores["ring"] * 0.45
    detected = (
        support["diagonal"]
        and support["ring"]
        and (support["template"] or support["contour"])
    )
    return {
        "detected": detected,
        "confidence": round(confidence, 3),
        "scores": {key: round(value, 3) for key, value in scores.items()},
        "support": support,
        "support_count": support_count,
        "decision_rule": "diagonal_and_closed_ring_and_shape",
    }


def _analyze_phi_prefix(prefix_zone: Image.Image) -> dict[str, Any]:
    gray = _prepare_phi_gray(prefix_zone)
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return _fuse_phi_scores(0.0, 0.0, 0.0, 0.0, 0.0)
    return _fuse_phi_scores(
        _template_score(gray),
        _hough_circle_score(gray),
        _contour_score(gray),
        _ring_score(gray),
        _diagonal_slash_score(gray),
    )


def detect_phi_in_prefix(prefix_zone: Image.Image) -> tuple[bool, float]:
    evidence = _analyze_phi_prefix(prefix_zone)
    return bool(evidence["detected"]), float(evidence["confidence"])


def detect_phi_multi_strip(
    image: Image.Image, vertical: bool
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Scan oriented prefix edges and retain every method's evidence."""
    from image_preprocess import orientations_for_ocr, upscale_min_edge
    from symbol_regions import enlarge_zone, split_symbol_zones

    # Split relative to the actual crop contents.  Padding before this step
    # shifted/cut the leading glyph because the prefix fraction included the
    # artificial white border.
    prep = upscale_min_edge(image)
    best = 0.0
    confirmed = False
    details: list[dict[str, Any]] = []
    src_vertical = bool(vertical or is_vertical_dimension(prep))

    for oname, oriented in orientations_for_ocr(prep):
        # A vertical crop can be authored in either reading direction. After
        # rotation, inspect both ends instead of guessing which edge contains
        # the prefix or scanning raw vertical slices that cut digits in half.
        prefix_edges = ("left", "right") if src_vertical else ("left",)
        for edge in prefix_edges:
            zones = split_symbol_zones(
                oriented,
                from_vertical_rotated=edge == "right",
            )
            evidence = _analyze_phi_prefix(enlarge_zone(zones["prefix"], 5.0))
            sc = float(evidence["confidence"])
            details.append(
                {
                    "strip": f"oriented_{oname}_{edge}_prefix",
                    **evidence,
                }
            )
            best = max(best, sc)
            confirmed = confirmed or bool(evidence["detected"])

    return confirmed, round(best, 3), details
