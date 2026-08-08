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

    return variants


def prepare_primary_ocr_variant(img: Image.Image) -> tuple[str, Image.Image]:
    """Build only the first variant used by the fast single-pass profile."""

    base = upscale_min_edge(pad_image(img))
    orientation_name, oriented = orientations_for_ocr(base)[0]
    return f"{orientation_name}_clahe", clahe_rgb(oriented)
