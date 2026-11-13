"""Report a sheet's sub-drawing sections and save an overlay.

    python scripts/report_sections.py drawing.pdf [--page 1] [--out overlay.png] [--crops]

An ``--out`` ending in .pdf keeps full resolution and the source page size.
With ``--crops`` (PDF only), each section follows on its own page, cropped
to its region as ``section_crop`` gives it for processing.
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drawing_sections import detect_drawing_sections, section_crop  # noqa: E402
from page_layout import analyze_page_layout  # noqa: E402

def section_colours(count: int) -> list[tuple[int, int, int]]:
    """One distinct, print-dark colour per section: hues a golden angle apart."""

    colours = []
    for index in range(count):
        hue = (index * 0.618034) % 1.0
        value = 0.85 if index % 2 == 0 else 0.6
        red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, value)
        # Yellows and cyans are pale on white paper; darken them to match.
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        if luminance > 0.45:
            red, green, blue = (channel * 0.45 / luminance for channel in (red, green, blue))
        colours.append((int(red * 255), int(green * 255), int(blue * 255)))
    return colours


def label_point(section, font_size: int) -> tuple[int, int]:
    """Where the label sits: high in the region, but well inside it. A box
    corner, or the region's topmost point, can sit in a neighbour's region."""

    box = section.bbox
    step = max(1, max(box.width, box.height) // 400)
    mask = Image.new("L", (box.width // step + 2, box.height // step + 2), 0)
    draw = ImageDraw.Draw(mask)
    for outline in section.outlines:
        draw.polygon([((x - box.x) // step, (y - box.y) // step) for x, y in outline], fill=255)
    inside = np.asarray(mask) > 0
    depth = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3) * step
    # The highest row with room for the label, else the deepest point.
    room = np.argwhere(depth >= font_size * 1.2)
    if room.size:
        row, column = room[np.lexsort((room[:, 1], room[:, 0]))[0]]
    else:
        row, column = np.unravel_index(np.argmax(depth), depth.shape)
    return box.x + int(column) * step - font_size // 2, box.y + int(row) * step - font_size // 2


def load_page(path: Path, page: int, short_side: int) -> tuple[Image.Image, float]:
    """The page image and the dpi that keeps the source page size."""

    if path.suffix.lower() != ".pdf":
        image = Image.open(path).convert("RGB")
        return image, float(image.info.get("dpi", (72, 72))[0])
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    pdf_page = document[page - 1]
    scale = short_side / min(pdf_page.get_size())
    return pdf_page.render(scale=scale).to_pil().convert("RGB"), 72 * scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--short-side", type=int, default=4120)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--crops", action="store_true", help="add a page per section (PDF)")
    args = parser.parse_args()
    if args.crops and (args.out is None or args.out.suffix.lower() != ".pdf"):
        parser.error("--crops needs an --out ending in .pdf")

    image, dpi = load_page(args.source, args.page, args.short_side)
    started = time.perf_counter()
    layout = analyze_page_layout(image)
    sections = detect_drawing_sections(image, table_masks=layout.table_masks)
    elapsed = time.perf_counter() - started
    print(f"{args.source.name} page {args.page}: {image.width}x{image.height}")
    print(f"{len(sections)} sections in {elapsed:.1f}s (incl. layout)")
    for section in sections:
        box = section.bbox
        print(
            f"  {section.section_id:>4} {section.kind:<4} "
            f"x={box.x:>5} y={box.y:>5} w={box.width:>5} h={box.height:>5} "
            f"glyphs={section.glyph_count}"
        )
    if args.out:
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        stroke = max(3, image.width // 450)
        font_size = max(24, image.width // 55)
        for section, colour in zip(sections, section_colours(len(sections))):
            box = section.bbox
            width = stroke if section.kind == "view" else max(2, stroke // 2)
            # The region outline: regions never overlap, boxes can.
            for outline in section.outlines or (
                ((box.x, box.y), (box.x1, box.y), (box.x1, box.y1), (box.x, box.y1)),
            ):
                draw.line([*outline, outline[0]], fill=colour, width=width, joint="curve")
            origin = (box.x + stroke * 2, box.y + stroke)
            if section.outlines:
                origin = label_point(section, font_size)
            # A white backing keeps the label readable over the drawing.
            label = draw.textbbox(origin, section.section_id, font_size=font_size)
            draw.rectangle(label, fill="white")
            draw.text(origin, section.section_id, fill=colour, font_size=font_size)
        if args.out.suffix.lower() == ".pdf":
            pages = []
            if args.crops:
                colours = section_colours(len(sections))
                for section, colour in zip(sections, colours):
                    crop, _ = section_crop(image, section, padding=40)
                    header = font_size * 2
                    page = Image.new("RGB", (crop.width, crop.height + header), "white")
                    page.paste(crop, (0, header))
                    box = section.bbox
                    ImageDraw.Draw(page).text(
                        (font_size // 2, font_size // 2),
                        f"{section.section_id} ({section.kind})  "
                        f"x={box.x} y={box.y} {box.width}x{box.height}",
                        fill=colour,
                        font_size=font_size,
                    )
                    pages.append(page)
            overlay.save(args.out, "PDF", resolution=dpi, save_all=True, append_images=pages)
        else:
            overlay.thumbnail((2400, 2400))
            overlay.save(args.out)
        print(f"overlay: {args.out}")

if __name__ == "__main__":
    main()
