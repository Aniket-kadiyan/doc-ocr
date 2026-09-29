"""
Tests for stroke_filter.is_stray_line.

Run: PYTHONPATH=. python test_stroke_filter.py
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from stroke_filter import is_stray_line

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


def _render(text: str) -> Image.Image:
    font = _font(48)
    tmp = Image.new("RGB", (10, 10))
    box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 24, box[3] - box[1] + 24
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((12 - box[0], 12 - box[1]), text,
                             fill=(20, 60, 200), font=font)
    return img


def _vline(w: int = 14, h: int = 150) -> Image.Image:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.line([(w // 2, 4), (w // 2, h - 4)], fill=(20, 60, 200), width=3)
    return img


def _diag_line(w: int = 140, h: int = 120) -> Image.Image:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).line([(6, h - 6), (w - 6, 6)], fill=(20, 60, 200), width=3)
    return img


def _check(name: str, cond: bool) -> bool:
    print(("OK   " if cond else "FAIL ") + name)
    return cond


def main() -> int:
    fails = 0

    # Straight lines read as "1" -> rejected as stray.
    fails += not _check("vertical line + '1' -> stray",
                        is_stray_line(_vline(), "1"))
    fails += not _check("diagonal line + 'l' -> stray",
                        is_stray_line(_diag_line(), "l"))
    fails += not _check("vertical line + '11' -> stray",
                        is_stray_line(_vline(), "11"))

    # A printed digit is NOT a stray, even when it reads one-like.
    fails += not _check("printed '1' -> kept",
                        not is_stray_line(_render("1"), "1"))
    fails += not _check("printed '111' -> kept",
                        not is_stray_line(_render("111"), "111"))

    # A real multi-digit value is never eligible (not one-like) -> kept.
    fails += not _check("value '1.08' -> kept",
                        not is_stray_line(_render("1.08"), "1.08"))
    fails += not _check("line but value text '27.43' -> kept",
                        not is_stray_line(_vline(), "27.43"))

    print("\nPASS" if not fails else f"\n{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
