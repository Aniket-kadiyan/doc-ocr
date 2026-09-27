"""
Reject stray-line reads that PaddleOCR hallucinates as digits.

Leader lines, extension lines, witness lines and dimension ticks are thin
straight strokes. Cropped and fed to OCR they frequently come back as ``1``,
``l``, ``|``, ``i`` or ``11`` — a phantom dimension with no value behind it.
A human never confuses a ruled line with a printed ``1``: the line is far too
long and thin, and its ink is a single straight stroke.

This module encodes exactly that judgement. It only fires when the *read itself*
is one-like (so a real multi-digit value is never at risk) AND the crop's ink is
a single, extremely elongated, well-filled straight component. Both conditions
must hold, and the thresholds are deliberately conservative — dropping a real
dimension is worse than keeping a stray, so ambiguous cases are kept.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from image_preprocess import cad_ink_to_gray

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is available in this backend
    cv2 = None  # type: ignore

# Characters a straight stroke gets read as. If a token is made only of these
# (plus separators) it is "one-like" and eligible for the geometry check.
_ONE_LIKE = set("1Iil|/\\")

# A single ink stroke this elongated (long axis / short axis) is a ruled line,
# not a glyph. A printed "1" in an engineering font sits around 3–5.
_LINE_ELONGATION = 7.0
# A ruled line fills its (thin) oriented box densely; a glyph leaves whitespace.
_LINE_FILL_MIN = 0.55
# The line must be essentially one connected stroke, not several glyph parts.
_DOMINANT_CC_FRAC = 0.85


def _is_one_like(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    core = [c for c in compact if not c.isspace()]
    # Every character is a stroke-confusable, and it's a short token (a genuine
    # value like "111.5" carries a '.' or more structure and is excluded).
    return len(core) <= 3 and all(c in _ONE_LIKE for c in core)


def _line_metrics(mask: np.ndarray) -> tuple[float, float, float] | None:
    """
    (elongation, fill, dominant_cc_fraction) of the largest ink component, or
    ``None`` when there is too little ink to judge.
    """
    if cv2 is None:
        return None
    total_ink = int((mask > 0).sum())
    if total_ink < 12:
        return None

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    comp_area = int(stats[idx, cv2.CC_STAT_AREA])
    dominant = comp_area / max(total_ink, 1)

    ys, xs = np.where(labels == idx)
    if xs.size < 5:
        return None
    pts = np.column_stack([xs, ys]).astype(np.int32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    long_side, short_side = max(w, h), max(min(w, h), 1.0)
    elongation = long_side / short_side
    fill = comp_area / max(long_side * short_side, 1.0)
    return elongation, fill, dominant


def is_stray_line(image: Image.Image, text: str) -> bool:
    """
    True when ``text`` is a one-like read backed by straight-line (not glyph) ink.

    Safe to call on every segmented region: returns False immediately unless the
    read is one-like, so real values are never touched.
    """
    if not _is_one_like(text):
        return False
    try:
        gray = np.asarray(cad_ink_to_gray(image).convert("L"))
    except Exception:  # noqa: BLE001 - best-effort geometry gate
        return False
    if cv2 is None:
        return False
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    metrics = _line_metrics(mask)
    if metrics is None:
        return False
    elongation, fill, dominant = metrics
    return (
        elongation >= _LINE_ELONGATION
        and fill >= _LINE_FILL_MIN
        and dominant >= _DOMINANT_CC_FRAC
    )
