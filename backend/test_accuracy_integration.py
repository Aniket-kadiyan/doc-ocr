"""
End-to-end checks for the accuracy mechanisms through the real pipeline.

Renders CAD-style values and runs OcrPipeline.recognize / .segment, asserting:
  - a dual-unit pair reads and cross-checks as consistent
  - a value rendered at an angle still reads correctly (deskew variant)
  - a stack of two dual-unit callouts segments into two regions

Run: PYTHONPATH=. .venv/bin/python test_accuracy_integration.py
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from ocr_pipeline import get_pipeline

_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render(text: str, size: int = 40, rotate: float = 0.0) -> Image.Image:
    font = _font(size)
    tmp = Image.new("RGB", (10, 10))
    box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 30, box[3] - box[1] + 30
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((15 - box[0], 15 - box[1]), text,
                             fill=(20, 60, 200), font=font)
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=(255, 255, 255))
    return img


def _stack(lines: list[str], size: int = 38) -> Image.Image:
    font = _font(size)
    tmp = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(tmp)
    widths, heights = [], []
    for ln in lines:
        b = draw.textbbox((0, 0), ln, font=font)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    line_h = max(heights) + 18
    text_w = max(widths)
    # Pad the canvas so the text block is a realistic fraction of a selection
    # (a real user crop is much larger than the digits), not ~100% of a tight
    # box — otherwise image-fraction thresholds in segmentation misbehave.
    w = text_w * 3
    h = line_h * len(lines) + 120
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    x0 = (w - text_w) // 2
    for i, ln in enumerate(lines):
        d.text((x0, 60 + i * line_h), ln, fill=(20, 60, 200), font=font)
    return img


def _check(name: str, cond: bool, detail: str = "") -> bool:
    print(("OK   " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    return cond


def main() -> int:
    pipe = get_pipeline()
    if not pipe.status.get("paddleocr"):
        print("FAIL: PaddleOCR did not load:", pipe.status.get("errors"))
        return 1

    fails = 0

    # 1. Dual-unit pair reads and cross-checks consistent.
    res = pipe.recognize(_render("1.08 [27.43]"))
    dual = res.get("dual_unit", {})
    got = res["text"]
    fails += not _check(
        "dual pair reads + consistent",
        "27.43" in got and "1.08" in got and dual.get("status") == "consistent",
        f"text={got!r} dual={dual.get('status')}",
    )

    # 2. Angled value still reads (deskew helps voting).
    res = pipe.recognize(_render("27.43", rotate=14.0))
    got = res["text"]
    fails += not _check("angled value reads 27.43", "27.43" in got,
                        f"text={got!r}")

    # 3. Stacked dual-unit callouts segment into two regions.
    seg = pipe.segment(_stack(["1.28 [32.51]", "1.10 [27.94]"]))
    texts = [r["text"] for r in seg["regions"]]
    fails += not _check(
        "stack -> 2 regions",
        seg["count"] >= 2,
        f"count={seg['count']} texts={texts}",
    )

    print("\nPASS" if not fails else f"\n{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
