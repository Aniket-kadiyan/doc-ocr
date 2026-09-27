"""Preprocessing for CAD drawings."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def pad_image(img: Image.Image, px: int = 36) -> Image.Image:
    w, h = img.size
    padded = Image.new("RGB", (w + 2 * px, h + 2 * px), (255, 255, 255))
    padded.paste(img, (px, px))
    return padded


def upscale_min_edge(img: Image.Image, min_edge: int = 280) -> Image.Image:
    w, h = img.size
    edge = max(w, h)
    if edge >= min_edge:
        return img
    factor = min(12.0, min_edge / max(edge, 1))
    nw, nh = int(w * factor), int(h * factor)
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def cad_ink_to_gray(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    ink = 255.0 - np.maximum(np.maximum(r, g), b)
    blue_bias = np.clip(b - r, 0, 255) * 0.75
    gray = np.clip(ink + blue_bias, 0, 255).astype(np.uint8)
    if cv2 is not None:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(gray).convert("RGB")


def _suppress_long_lines(mask: np.ndarray) -> np.ndarray:
    """
    Zero out connected components that are long thin straight strokes.

    A leader/extension/dimension line is one hugely elongated component spanning
    much of the crop; glyph strokes are short by comparison. Removing those
    components leaves the text ink, so a downstream axis/skew estimate reflects
    the value, not the rule it sits on. Best-effort: returns the input unchanged
    if cv2 is missing.
    """
    if cv2 is None:
        return mask
    h, w = mask.shape[:2]
    span = max(h, w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    out = mask.copy()
    for i in range(1, n):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        long_side = max(cw, ch)
        short_side = max(min(cw, ch), 1)
        # Long, thin, and stroke-like fill (not a big filled blob) → a rule.
        if (
            long_side >= span * 0.5
            and long_side / short_side >= 6.0
            and area <= long_side * 4.0
        ):
            out[labels == i] = 0
    return out


def estimate_skew_angle(img: Image.Image) -> float | None:
    """
    Estimate the slant of a text crop in degrees via PCA on its ink pixels.

    CAD values aligned to a slanted leader read poorly because PaddleOCR expects
    horizontal text. The principal axis of the ink cloud follows the text
    baseline, so its angle to the x-axis is the skew. Returns the angle in
    degrees (positive = counter-clockwise correction needed) or ``None`` when
    there is too little ink to be confident. Near-vertical crops are handled by
    the existing 90° rotation path, so this focuses on moderate slants.
    """
    if cv2 is None:
        return None
    try:
        gray = np.asarray(cad_ink_to_gray(img).convert("L"))
    except Exception:  # noqa: BLE001
        return None
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Drop long straight strokes (leader / extension / dimension lines) before
    # measuring text skew — otherwise the leader a slanted callout sits on
    # dominates the PCA and the estimate follows the line, not the text.
    mask = _suppress_long_lines(mask)

    ys, xs = np.where(mask > 0)
    if xs.size < 40:
        return None
    x = xs.astype(np.float64) - xs.mean()
    y = ys.astype(np.float64) - ys.mean()
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return None
    evals, evecs = np.linalg.eigh(cov)
    # Reject nearly-isotropic ink (a compact blob has no meaningful axis).
    if evals[1] <= 0 or evals[0] / evals[1] > 0.65:
        return None
    major = evecs[:, int(np.argmax(evals))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    # Fold into (-90, 90]; a text line's major axis is horizontal-ish.
    if angle <= -90:
        angle += 180
    if angle > 90:
        angle -= 180
    if abs(angle) > 45:  # near-vertical -> leave to the rotation path
        return None
    return angle


def deskew_to_horizontal(
    img: Image.Image, *, min_angle: float = 4.0, max_angle: float = 45.0
) -> Image.Image | None:
    """
    Rotate a slanted crop so its text baseline is horizontal, or ``None``.

    Only fires for a meaningful slant (``min_angle``..``max_angle``); tiny skews
    aren't worth an extra OCR pass and near-vertical text is handled elsewhere.
    """
    angle = estimate_skew_angle(img)
    if angle is None or not (min_angle <= abs(angle) <= max_angle):
        return None
    # PIL rotates counter-clockwise for positive angles; rotating by +angle
    # levels a baseline whose major axis sits at +angle to the x-axis.
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255),
                      resample=Image.Resampling.BICUBIC)


def clahe_rgb(img: Image.Image) -> Image.Image:
    base = cad_ink_to_gray(img)
    if cv2 is None:
        g = base.convert("L")
        g = ImageOps.autocontrast(g, cutoff=0)
        return g.convert("RGB")
    arr = np.array(base.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return Image.fromarray(clahe.apply(arr)).convert("RGB")


def sharpen_rgb(img: Image.Image) -> Image.Image:
    g = cad_ink_to_gray(img).convert("L")
    g = ImageEnhance.Contrast(g).enhance(1.8)
    g = ImageEnhance.Sharpness(g).enhance(2.0)
    return g.convert("RGB")


def is_vertical_dimension(img: Image.Image) -> bool:
    w, h = img.size
    return h > w * 1.35


def orientations_for_ocr(img: Image.Image) -> list[tuple[str, Image.Image]]:
    """Try both rotations for vertical CAD text (primary: -90° / CW)."""
    if not is_vertical_dimension(img):
        return [("h", img)]

    fill = (255, 255, 255)
    return [
        ("cw", img.rotate(-90, expand=True, fillcolor=fill)),
        ("ccw", img.rotate(90, expand=True, fillcolor=fill)),
    ]


def primary_oriented(img: Image.Image) -> Image.Image:
    """Vertical CAD text → rotate -90° (CW) for left-to-right OCR."""
    if not is_vertical_dimension(img):
        return img
    return img.rotate(-90, expand=True, fillcolor=(255, 255, 255))


def bbox_from_oriented_to_original(
    bbox: dict[str, float],
    orig_w: int,
    orig_h: int,
) -> dict[str, float]:
    """
    Map an axis-aligned bbox from ``primary_oriented`` (-90° CW) space back
    into the original received-image coordinates.
    """
    x, y, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

    def inv(xr: float, yr: float) -> tuple[float, float]:
        return yr, orig_h - xr

    corners = [inv(x, y), inv(x + bw, y), inv(x, y + bh), inv(x + bw, y + bh)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(float(orig_w), max(xs))
    y1 = min(float(orig_h), max(ys))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return bbox
    return {
        "x": round(x0, 1),
        "y": round(y0, 1),
        "width": round(x1 - x0, 1),
        "height": round(y1 - y0, 1),
    }


def prepare_ocr_variants(img: Image.Image) -> list[tuple[str, Image.Image]]:
    base = upscale_min_edge(pad_image(img))
    variants: list[tuple[str, Image.Image]] = []

    for oname, oriented in orientations_for_ocr(base):
        variants.extend(
            [
                (f"{oname}_clahe", clahe_rgb(oriented)),
                (f"{oname}_sharp", sharpen_rgb(oriented)),
                (f"{oname}_cad", cad_ink_to_gray(oriented)),
            ]
        )

    # Deskewed pass for values aligned to a slanted leader. Prefixed "dsk_" so
    # the voter knows not to trust its word boxes for balloon placement (they
    # live in rotated space). Only added when a real slant is detected, so
    # upright text pays nothing.
    deskewed = deskew_to_horizontal(base)
    if deskewed is not None:
        variants.append(("dsk_clahe", clahe_rgb(deskewed)))
        variants.append(("dsk_cad", cad_ink_to_gray(deskewed)))

    return variants


def prepare_primary_ocr_variant(img: Image.Image) -> tuple[str, Image.Image]:
    """Build only the first variant used by the fast single-pass profile."""

    base = upscale_min_edge(pad_image(img))
    orientation_name, oriented = orientations_for_ocr(base)[0]
    return f"{orientation_name}_clahe", clahe_rgb(oriented)
