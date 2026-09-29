"""
Tests for image_preprocess deskew helpers.

Run: PYTHONPATH=. python test_deskew.py
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from image_preprocess import deskew_to_horizontal, estimate_skew_angle

_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render(text: str, rotate_deg: float = 0.0) -> Image.Image:
    font = _font(44)
    tmp = Image.new("RGB", (10, 10))
    box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 40, box[3] - box[1] + 40
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((20 - box[0], 20 - box[1]), text,
                             fill=(20, 60, 200), font=font)
    if rotate_deg:
        img = img.rotate(rotate_deg, expand=True, fillcolor=(255, 255, 255))
    return img


def _check(name: str, cond: bool) -> bool:
    print(("OK   " if cond else "FAIL ") + name)
    return cond


def main() -> int:
    fails = 0

    # Upright text: no meaningful skew, no deskew variant produced.
    up = estimate_skew_angle(_render("27.43"))
    fails += not _check(f"upright angle ~0 (got {up})",
                        up is None or abs(up) < 4.0)
    fails += not _check("upright -> no deskew",
                        deskew_to_horizontal(_render("27.43")) is None)

    # A crop rotated by ~15° is detected and corrected back toward horizontal.
    slant = _render("27.43", rotate_deg=15.0)
    est = estimate_skew_angle(slant)
    fails += not _check(f"15deg detected (got {est})",
                        est is not None and 8.0 <= abs(est) <= 25.0)
    fixed = deskew_to_horizontal(slant)
    fails += not _check("15deg -> deskew produced", fixed is not None)
    if fixed is not None:
        residual = estimate_skew_angle(fixed)
        fails += not _check(f"residual skew small (got {residual})",
                            residual is None or abs(residual) < 6.0)

    print("\nPASS" if not fails else f"\n{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
