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


# Topology score at/above this is authoritative; below it the legacy strip
# detectors are capped at LEGACY_CAP (< the 0.5 vision-only compose bar, so a
# strip hit alone can never inject a Ø, but it still corroborates OCR hints).
TOPO_ACCEPT = 0.6
LEGACY_CAP = 0.45


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


# --- Topology detector -------------------------------------------------------
# A Ø glyph is a ring cut by a slash: one ink component enclosing TWO holes of
# similar size, neighbours of each other, whose combined extent is one glyph.
# Digits never produce that: 0/6/9/4 enclose one hole, 8/B enclose two holes
# stacked ACROSS the reading axis (lobes), and two touching zeros enclose two
# holes whose union is twice as wide as tall. So a hole-pair with the right
# geometry is near-conclusive evidence, and it does not depend on where the
# prefix strip lands or on template matching (which fires on every digit).

_TOPO_SCORE_START = 0.92     # hole-pair found at the reading start
_TOPO_SCORE_NEAR = 0.8       # one glyph-sized blob precedes it (arrowhead, tick)
_TOPO_SCORE_INSIDE = 0.45    # hole-pair found but not at the reading start


def _glyph_like(bw: int, bh: int, area: float, vertical: bool) -> tuple[int, int]:
    """(perpendicular, along-reading-axis) extents of a component."""
    return (bw, bh) if vertical else (bh, bw)


def detect_phi_topology(image: Image.Image) -> tuple[float, dict[str, Any]]:
    """
    Score Ø presence from ring-with-slash hole topology.

    Returns ``(score, details)`` where score is 0.0 when no slashed-ring pair
    exists (a clean digit run) and ≥ 0.8 when one sits at the reading start.
    Works on the crop in its own orientation: hole topology is rotation
    invariant, only the reading-start test needs the vertical flag.
    """
    if cv2 is None:
        return 0.0, {"reason": "no_cv2"}
    gray = np.array(cad_ink_to_gray(image).convert("L"))
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 0.0, {"reason": "too_small"}
    vertical = h > w * 1.35
    _, binm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hier = cv2.findContours(binm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(contours) == 0:
        return 0.0, {"reason": "no_ink"}
    hier = hier[0]

    outers: list[tuple[int, int, int, int, int, float]] = []
    holes: dict[int, list[tuple[float, float, float, int, int, int, int]]] = {}
    for i, c in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        parent = int(hier[i][3])
        if parent == -1:
            outers.append((i, x, y, bw, bh, area))
            continue
        if area < 2:
            continue
        m = cv2.moments(c)
        m00 = max(m["m00"], 1e-6)
        holes.setdefault(parent, []).append(
            (m["m10"] / m00, m["m01"] / m00, area, x, y, bw, bh)
        )

    # Line height = the LARGEST glyph-sized blob across the reading axis. The
    # median is dragged down by half-height tolerance digits and would reject
    # the (full-height) Ø ring; long geometry is excluded by the aspect test.
    crop_perp = w if vertical else h
    sizes: list[int] = []
    for _i, _x, _y, bw, bh, area in outers:
        perp, along = _glyph_like(bw, bh, area, vertical)
        if perp >= 5 and along <= perp * 2.5 and area >= 6 and perp <= crop_perp * 0.95:
            sizes.append(perp)
    line_h = float(max(sizes)) if sizes else float(crop_perp)

    best = 0.0
    cands: list[dict[str, Any]] = []
    for parent, hl in holes.items():
        if len(hl) < 2:
            continue
        for a in range(len(hl)):
            for b in range(a + 1, len(hl)):
                A, B = hl[a], hl[b]
                big, small = max(A[2], B[2]), min(A[2], B[2])
                if small < big * 0.2:
                    continue
                dx, dy = B[0] - A[0], B[1] - A[1]
                # Lobes of an 8/B are separated ACROSS the reading axis;
                # slashed-ring halves are separated diagonally or ALONG it.
                along_sep, perp_sep = (abs(dy), abs(dx)) if vertical else (abs(dx), abs(dy))
                if perp_sep > along_sep * 2.0:
                    continue
                ux0, uy0 = min(A[3], B[3]), min(A[4], B[4])
                ux1 = max(A[3] + A[5], B[3] + B[5])
                uy1 = max(A[4] + A[6], B[4] + B[6])
                uw, uh = ux1 - ux0, uy1 - uy0
                if uw <= 0 or uh <= 0:
                    continue
                u_perp, u_along = (uw, uh) if vertical else (uh, uw)
                # Both halves live inside ONE glyph cell: not wider than tall
                # (two touching zeros) and one line high.
                if u_along > u_perp * 1.25 or u_perp > u_along * 2.0:
                    continue
                if not (line_h * 0.3 <= u_perp <= line_h * 1.05):
                    continue
                if (dx * dx + dy * dy) ** 0.5 > max(uw, uh):
                    continue
                fill_a = A[2] / max(A[5] * A[6], 1)
                fill_b = B[2] / max(B[5] * B[6], 1)
                if min(fill_a, fill_b) < 0.25:
                    continue
                # Halves of ONE ring: each spans most of the ring in at least
                # one axis and together they fill the ring interior. Holes of
                # two different letters (an "e" and an "a" of a watermark) are
                # small islands in a large union and fail both.
                if max(A[5], B[5]) < uw * 0.55 or max(A[6], B[6]) < uh * 0.55:
                    continue
                if (A[2] + B[2]) < uw * uh * 0.3:
                    continue

                # Reading-start test: count glyph-sized blobs before/after the
                # pair along the reading axis within its line band.
                before = after = 0
                for i, x, y, bw, bh, area in outers:
                    if i == parent or area < 6:
                        continue
                    perp_i, along_i = _glyph_like(bw, bh, area, vertical)
                    if perp_i < line_h * 0.45 or along_i > perp_i * 2.5:
                        continue
                    if vertical:
                        ov = min(x + bw, ux1) - max(x, ux0)
                        if ov < min(bw, uw) * 0.4:
                            continue
                        if y + bh <= uy0:
                            before += 1
                        elif y >= uy1:
                            after += 1
                    else:
                        ov = min(y + bh, uy1) - max(y, uy0)
                        if ov < min(bh, uh) * 0.4:
                            continue
                        if x + bw <= ux0:
                            before += 1
                        elif x >= ux1:
                            after += 1
                if vertical:
                    lead = min(before, after)  # reading direction unknown
                else:
                    lead = before
                if lead == 0:
                    sc = _TOPO_SCORE_START
                elif lead == 1:
                    sc = _TOPO_SCORE_NEAR
                else:
                    sc = _TOPO_SCORE_INSIDE
                cands.append(
                    {
                        "union": [int(ux0), int(uy0), int(uw), int(uh)],
                        "lead": lead,
                        "score": sc,
                    }
                )
                best = max(best, sc)
    return round(best, 3), {"line_h": line_h, "vertical": vertical, "cands": cands}


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

    # Topology is decisive (see below): when it confirms a Ø its score is used
    # outright and the strip detectors cannot change the answer, so they are
    # skipped. On a vertical crop the six enlarged strips cost ~0.7 s, ten
    # times the rest of the symbol pass.
    topo_score, topo = detect_phi_topology(image)
    if topo_score >= TOPO_ACCEPT:
        details.append(
            {"strip": "topology", "score": topo_score, "cands": topo.get("cands", [])}
        )
        return True, round(topo_score, 3), details

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

    # Topology is the decisive signal. The strip detectors above (template /
    # Hough / contour) fire on 0/4/8 loops nearly as often as on a real Ø, so
    # without a slashed-ring hole pair their score is capped below the
    # vision-only compose threshold; with one, its score is used outright.
    details.append({"strip": "topology", "score": topo_score, "cands": topo.get("cands", [])})
    topology_confirmed = topo_score >= TOPO_ACCEPT
    if topology_confirmed:
        best = topo_score
    else:
        best = min(best, LEGACY_CAP)
    # The boolean must honour the cap this module documents: "a strip hit alone
    # can never inject a Ø, but it still corroborates OCR hints". Reporting
    # True for a capped legacy-only hit made it decisive after all, which is
    # how a plain length whose leader read as a dash became "Ø5.50[139.70]".
    # The score is still returned so callers can use it as corroboration.
    return topology_confirmed, round(best, 3), details

