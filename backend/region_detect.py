"""
Vision-based text-region proposal for CAD drawings.

PaddleOCR's text *detection* is unreliable as a region proposer on engineering
sheets: dimension text is thin, blue, often rotated 90°, and spread across a
large area between geometry lines. Instead we find candidate text blobs with
classic morphology:

  1. ink mask  — blue/dark strokes stand out (reuses `cad_ink_to_gray`)
  2. line kill — open with long 1-D kernels to find the long straight geometry
                 lines (part profiles, centrelines) and subtract them; text
                 strokes are too short to survive, so only geometry is removed
  3. connect  — dilate so each dimension's characters merge into one blob while
                separate dimensions (far apart) stay distinct
  4. components — connected components, filtered by size / fill ratio

Each proposed box is then read by the existing single-value `recognize`
pipeline, which already handles vertical orientation well. Falls back to an
empty list when OpenCV is unavailable so the caller can use its OCR-based path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from image_preprocess import cad_ink_to_gray

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only without OpenCV
    cv2 = None  # type: ignore


def opencv_available() -> bool:
    return cv2 is not None


def dominant_slant_angles(
    image: Any,
    *,
    max_magnitudes: int = 1,
    axis_guard_deg: float = 12.0,
    min_slant_share: float = 0.04,
    min_weight_frac: float = 0.15,
) -> list[float]:
    """
    Cheap detector for the dominant *slanted* text direction(s) in a CAD crop.

    Chamfer / lead-in callouts (``0.5×45°``, ``0.2-0.3×45°``) and angled fit
    dimensions run along diagonal leaders that PaddleOCR's horizontal-biased
    detector misses. This finds the magnitude(s) of that slant so the caller can
    rotate the crop upright and re-detect. Returns **acute magnitudes in degrees**
    (e.g. ``[45.0]``); the caller rotates by both ``+a`` and ``-a`` since a bucket
    is sign-ambiguous. Empty list = no strong slant, so the caller pays only this
    (~1 Hough call) and skips the expensive angled OCR passes entirely.

    The vote is length-weighted Hough over the ink mask with long axis-aligned
    geometry removed; near-horizontal/near-vertical buckets are suppressed
    (``axis_guard_deg``) because those are already covered by the 0°/90° passes.
    """
    if cv2 is None:
        return []

    gray = np.array(cad_ink_to_gray(image).convert("L"))
    h, w = gray.shape
    if h < 16 or w < 16:
        return []

    _, binm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove long straight geometry (profiles, centrelines) so their strong
    # axis-aligned vote doesn't drown out short diagonal text strokes.
    line_len = max(28, int(max(h, w) * 0.30))
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(binm, cv2.MORPH_OPEN, vert_k),
        cv2.morphologyEx(binm, cv2.MORPH_OPEN, horiz_k),
    )
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    text = cv2.subtract(binm, lines)

    segs = cv2.HoughLinesP(
        text, 1, np.pi / 180, threshold=30, minLineLength=25, maxLineGap=4
    )
    if segs is None:
        return []

    # Length-weighted histogram over folded 0..180° buckets.
    hist = np.zeros(180, dtype=np.float64)
    for x1, y1, x2, y2 in segs[:, 0]:
        ang = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))) % 180.0
        hist[int(ang) % 180] += float(np.hypot(x2 - x1, y2 - y1))

    grand_total = float(hist.sum())
    if grand_total <= 0:
        return []

    # Suppress near-axis buckets (already handled by upright / 90° passes).
    for a in range(180):
        acute = min(a, abs(a - 90), abs(a - 180))
        if acute < axis_guard_deg:
            hist[a] = 0.0

    # Long axis-aligned geometry dominates the length total, so gate the whole
    # feature on the diagonal share: sheets with only stray diagonal leaders
    # (no diagonal text) fall below this and skip the expensive angled passes.
    slant_total = float(hist.sum())
    if slant_total < grand_total * min_slant_share:
        return []

    # Fold each 0..180 bucket to its acute magnitude (a and 180-a are the same
    # slant magnitude, opposite sign) and accumulate, so both signs reinforce.
    mag_hist: dict[int, float] = {}
    for a in range(180):
        if hist[a] <= 0:
            continue
        mag = a if a <= 90 else 180 - a
        mag_hist[mag] = mag_hist.get(mag, 0.0) + hist[a]

    floor = slant_total * min_weight_frac
    ranked = sorted(
        (m for m, wsum in mag_hist.items() if wsum >= floor),
        key=lambda m: mag_hist[m],
        reverse=True,
    )
    return [float(m) for m in ranked[:max_magnitudes]]


def detect_leader_lines(
    image: Any,
    *,
    axis_guard_deg: float = 12.0,
    min_length_frac: float = 0.05,
    merge_angle_deg: float = 3.0,
    merge_offset_px: float = 18.0,
    max_lines: int = 6,
) -> list[dict[str, float]]:
    """
    Find the diagonal leader lines a callout can be written along.

    A chamfer or fit callout is lettered along its leader, so the leader *is*
    the text's baseline. A long straight stroke is the easiest thing in the
    drawing to measure precisely, which makes this a far better angle source
    than voting on the text strokes themselves: on the GPD 18T sheet the two
    fit leaders measure -34.5° and -57.0°, and those are exactly the angles at
    which their callouts read correctly.

    Near-horizontal and near-vertical lines are skipped — they are dimension and
    extension lines, whose text is already read by the upright and 90° passes.
    Collinear segments of one leader are merged. Returns the longest lines
    first, each as ``{angle, x1, y1, x2, y2, length}`` with ``angle`` in
    (-90, 90], clockwise positive.
    """
    if cv2 is None:
        return []

    gray = np.array(cad_ink_to_gray(image).convert("L"))
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return []
    _, binm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    min_len = max(60.0, max(h, w) * min_length_frac)
    segments = cv2.HoughLinesP(
        binm,
        1,
        np.pi / 360,
        threshold=60,
        minLineLength=int(min_len),
        maxLineGap=6,
    )
    if segments is None:
        return []

    # Group collinear segments: same angle, and the same perpendicular offset
    # from the origin (Hough normal form), so one dashed or broken leader does
    # not become several lines.
    groups: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in segments[:, 0]:
        angle = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        acute = abs(angle)
        if acute < axis_guard_deg or acute > 90.0 - axis_guard_deg:
            continue
        normal = np.radians(angle + 90.0)
        offset = float(x1) * np.cos(normal) + float(y1) * np.sin(normal)
        for group in groups:
            if (
                abs(group["angle"] - angle) <= merge_angle_deg
                and abs(group["offset"] - offset) <= merge_offset_px
            ):
                group["points"].extend([(float(x1), float(y1)), (float(x2), float(y2))])
                break
        else:
            groups.append(
                {
                    "angle": angle,
                    "offset": offset,
                    "points": [(float(x1), float(y1)), (float(x2), float(y2))],
                }
            )

    lines: list[dict[str, float]] = []
    for group in groups:
        points = group["points"]
        # The merged line runs between the two most distant endpoints.
        direction = np.radians(group["angle"])
        along = [p[0] * np.cos(direction) + p[1] * np.sin(direction) for p in points]
        start = points[int(np.argmin(along))]
        end = points[int(np.argmax(along))]
        length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if length < min_len:
            continue
        lines.append(
            {
                "angle": round(group["angle"], 1),
                "x1": round(start[0], 1),
                "y1": round(start[1], 1),
                "x2": round(end[0], 1),
                "y2": round(end[1], 1),
                "length": round(length, 1),
            }
        )

    lines.sort(key=lambda line: line["length"], reverse=True)
    return lines[:max_lines]


def propose_text_regions(
    image: Any,
    *,
    min_area_frac: float = 0.00025,
    line_frac: float = 0.45,
    connect_px: int = 5,
) -> list[dict[str, float]]:
    """
    Propose text-region boxes (received-image pixels) via ink-mask morphology.

    ``line_frac`` sets the long-line kernel length as a fraction of the image's
    larger side — strokes shorter than that survive as text. ``connect_px``
    controls how aggressively neighbouring characters merge into one blob.
    Returns dicts: {x, y, w, h}.
    """
    if cv2 is None:
        return []

    gray = np.array(cad_ink_to_gray(image).convert("L"))
    h, w = gray.shape
    if h < 4 or w < 4:
        return []

    # Ink (blue/dark strokes) -> white on black.
    _, binm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove long straight geometry lines (vertical profiles + centrelines).
    line_len = max(28, int(max(h, w) * line_frac))
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(binm, cv2.MORPH_OPEN, vert_k),
        cv2.morphologyEx(binm, cv2.MORPH_OPEN, horiz_k),
    )
    # Dilate the line mask slightly so the line's full thickness is removed.
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    text_mask = cv2.subtract(binm, lines)

    # Merge characters within one dimension; keep dilation moderate.
    k = max(3, connect_px)
    merged = cv2.dilate(
        text_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=2,
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    min_area = max(40.0, min_area_frac * h * w)

    boxes: list[dict[str, float]] = []
    for i in range(1, n):
        x, y, bw, bh, area = (float(v) for v in stats[i])
        if area < min_area:
            continue
        if bw >= w * 0.92 and bh >= h * 0.92:
            continue  # spans the whole crop -> not a single value
        fill = area / max(bw * bh, 1.0)
        if fill < 0.04:
            continue  # line-like leftover, not a text blob

        candidates = _refine_blob(
            x,
            y,
            bw,
            bh,
            text_mask,
            min_area=min_area * 0.2,
        )
        for box in candidates:
            if _is_plausible_dimension_box(box, w, h, text_mask):
                boxes.append(box)

    return boxes


def _is_plausible_dimension_box(
    box: dict[str, float],
    img_w: int,
    img_h: int,
    text_mask: Any,
) -> bool:
    """Drop geometry leftovers and interior shading blobs."""
    bw, bh = box["w"], box["h"]
    short = min(bw, bh)
    if short < 7 or bw * bh < 90:
        return False
    if bw >= img_w * 0.48 or bh >= img_h * 0.48:
        return False
    if max(bw, bh) > max(img_w, img_h) * 0.36:
        return False

    ix0, iy0 = int(box["x"]), int(box["y"])
    ix1 = min(img_w, ix0 + int(bw))
    iy1 = min(img_h, iy0 + int(bh))
    roi = text_mask[iy0:iy1, ix0:ix1]
    if roi.size == 0:
        return False

    ink = float(np.sum(roi > 0))
    fill = ink / max(bw * bh, 1.0)
    if fill < 0.055:
        return False
    # Solid interior regions (hatching / profile fill) are not one dimension.
    if fill > 0.72 and min(bw, bh) > 18:
        return False
    area = bw * bh
    aspect = bw / max(bh, 1.0)
    if area > 2200 and 0.4 < aspect < 2.5 and fill > 0.14:
        return False
    return True


def _refine_blob(
    x: float,
    y: float,
    bw: float,
    bh: float,
    text_mask: Any,
    *,
    min_area: float,
) -> list[dict[str, float]]:
    """
    Split an over-merged morphology blob into one box per ink band.

    CAD dimensions stacked on one leader column leave clear horizontal gaps
    between values; row-projection finds those gaps even when connected
    components still bridge the whole column.
    """
    single = [{"x": round(x, 1), "y": round(y, 1), "w": round(bw, 1), "h": round(bh, 1)}]
    if cv2 is None:
        return single

    ix0, iy0 = int(x), int(y)
    ix1, iy1 = int(x + bw), int(y + bh)
    roi = text_mask[iy0:iy1, ix0:ix1]
    if roi.size == 0:
        return single

    char = max(4.0, min(bw, bh) * 0.35)
    # Only split clearly tall stacks — not every elongated blob.
    if bh < bw * 2.6 and bw < bh * 2.6:
        return single

    by_rows = _bands_from_projection(
        roi, along="rows", char=char, min_area=min_area, min_gap=char * 1.4
    )
    if len(by_rows) > 1:
        return [
            {
                "x": round(x + bx, 1),
                "y": round(y + by, 1),
                "w": round(bw_, 1),
                "h": round(bh_, 1),
            }
            for by, bh_, bx, bw_ in by_rows
        ]

    by_cols = _bands_from_projection(
        roi, along="cols", char=char, min_area=min_area, min_gap=char * 1.4
    )
    if len(by_cols) > 1:
        return [
            {
                "x": round(x + bx, 1),
                "y": round(y + by, 1),
                "w": round(bw_, 1),
                "h": round(bh_, 1),
            }
            for by, bh_, bx, bw_ in by_cols
        ]

    return single


def _bands_from_projection(
    roi: Any,
    *,
    along: str,
    char: float,
    min_area: float,
    min_gap: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """
    Find ink bands split by whitespace gaps.

    ``along="rows"`` splits stacked horizontal text (gaps between y-bands).
    ``along="cols"`` splits side-by-side text (gaps between x-bands).

    Returns ``(y, h, x, w)`` tuples relative to the ROI origin.
    """
    if along == "rows":
        proj = np.sum(roi > 0, axis=1)
        span = float(roi.shape[1])
    else:
        proj = np.sum(roi > 0, axis=0)
        span = float(roi.shape[0])

    thresh = max(2.0, span * 0.07)
    active = proj >= thresh

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for i, on in enumerate(active):
        if on and not in_band:
            start = i
            in_band = True
        elif not on and in_band:
            bands.append((start, i))
            in_band = False
    if in_band:
        bands.append((start, len(active)))

    gap_merge = max(2, int(char * 0.35))
    merged: list[tuple[int, int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= gap_merge:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    if min_gap > 0 and len(merged) > 1:
        kept = [merged[0]]
        for band in merged[1:]:
            if band[0] - kept[-1][1] < min_gap:
                kept[-1] = (kept[-1][0], band[1])
            else:
                kept.append(band)
        merged = kept

    out: list[tuple[float, float, float, float]] = []
    for b0, b1 in merged:
        if b1 - b0 < max(3, int(char * 0.55)):
            continue
        if along == "rows":
            band_roi = roi[b0:b1, :]
            cols = np.any(band_roi > 0, axis=0)
            if not cols.any():
                continue
            cx0 = int(np.argmax(cols))
            cx1 = len(cols) - int(np.argmax(cols[::-1]))
            area = float(np.sum(band_roi > 0))
            if area < min_area:
                continue
            out.append((float(b0), float(b1 - b0), float(cx0), float(cx1 - cx0)))
        else:
            band_roi = roi[:, b0:b1]
            rows = np.any(band_roi > 0, axis=1)
            if not rows.any():
                continue
            ry0 = int(np.argmax(rows))
            ry1 = len(rows) - int(np.argmax(rows[::-1]))
            area = float(np.sum(band_roi > 0))
            if area < min_area:
                continue
            out.append((float(ry0), float(ry1 - ry0), float(b0), float(b1 - b0)))

    return out
