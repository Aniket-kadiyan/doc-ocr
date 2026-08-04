"""Crop leading / symbol zones for dedicated detection."""

from __future__ import annotations

from PIL import Image

from image_preprocess import is_vertical_dimension, upscale_min_edge

# Narrow strip = symbol glyph only (reduces digit bleed into prefix OCR)
PREFIX_FRAC_H = 0.28
PREFIX_FRAC_V = 0.32


def split_symbol_zones(
    img: Image.Image,
    *,
    from_vertical_rotated: bool = False,
) -> dict[str, Image.Image]:
    """
    Split crop into prefix (symbol) and body (digits) zones.

    ``from_vertical_rotated=True`` when *img* is the -90° rotated view of a
    vertical crop — reading start (Ø) is on the **right**, not the left.
    """
    img = upscale_min_edge(img)
    w, h = img.size
    zones: dict[str, Image.Image] = {}

    if is_vertical_dimension(img):
        strip = max(14, int(h * PREFIX_FRAC_V))
        zones["prefix"] = img.crop((0, h - strip, w, h))
        zones["body"] = img.crop((0, 0, w, h - strip))
    else:
        strip = max(14, int(w * PREFIX_FRAC_H))
        if from_vertical_rotated:
            zones["prefix"] = img.crop((w - strip, 0, w, h))
            zones["body"] = img.crop((0, 0, w - strip, h))
        else:
            zones["prefix"] = img.crop((0, 0, strip, h))
            zones["body"] = img.crop((strip, 0, w, h))

    zones["full"] = img
    return zones


def enlarge_zone(zone: Image.Image, factor: float = 2.5) -> Image.Image:
    nw = max(1, int(zone.width * factor))
    nh = max(1, int(zone.height * factor))
    return zone.resize((nw, nh), Image.Resampling.LANCZOS)
