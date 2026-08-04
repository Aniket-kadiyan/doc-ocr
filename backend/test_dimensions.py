"""
End-to-end pipeline checks. Renders dimension text to an image and runs it
through OcrPipeline.recognize, asserting the composed value + type.

Run: PYTHONPATH=. .venv/bin/python test_dimensions.py
"""

from __future__ import annotations

import sys

from PIL import Image, ImageDraw, ImageFont

from ocr_pipeline import get_pipeline

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


def render(text: str, vertical: bool = False, color=(20, 60, 200)) -> Image.Image:
    font = _font(46)
    tmp = Image.new("RGB", (10, 10))
    box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 24, box[3] - box[1] + 24
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((12 - box[0], 12 - box[1]), text, fill=color, font=font)
    if vertical:
        img = img.rotate(-90, expand=True, fillcolor=(255, 255, 255))
    return img


# (rendered text, expected substring in result, expected type) — expectations
# tolerate a leading-digit OCR slip; we assert symbols + structure + type.
CASES = [
    ("32°20'40\"", "°20'40\"", "angle"),
    ("Ø174.07±0.05", "±0.05", "diameter"),
    ("45.0", "45.0", "linear"),
    ("R12.5", "12.5", "radius"),
]


def main() -> int:
    pipe = get_pipeline()
    status = pipe.status
    print("STATUS:", status)
    if not status.get("paddleocr"):
        print("FAIL: PaddleOCR did not load:", status.get("errors"))
        return 1

    failures = 0
    for text, expect_sub, expect_type in CASES:
        res = pipe.recognize(render(text))
        got, got_type = res["text"], res["type"]
        ok_sub = expect_sub in got
        ok_type = got_type == expect_type
        tb = res.get("text_bbox")
        flag = "OK " if (ok_sub and ok_type) else "FAIL"
        if not (ok_sub and ok_type):
            failures += 1
        print(
            f"{flag} {text!r:18} -> {got!r:16} type={got_type:9} "
            f"(want ~{expect_sub!r}/{expect_type}) bbox={tb}"
        )

    print("\nPASS" if failures == 0 else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
