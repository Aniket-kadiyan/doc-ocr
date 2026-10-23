"""
Feature control frames drawn the way a vector PDF renders them: hairlines.

The profile frame "⌓ | 0.2 Ⓤ" on 3015-100-2232F came out of the pipeline as a
bare "0.2" labelled a linear dimension. Each test here pins one of the causes,
on synthetic frames drawn with one-pixel rules:

  * the binarization's median filter erased one-pixel rules, so the frame's top
    rule never existed and the frame was never proposed;
  * the side rule sat one pixel past the horizontal rules' measured ends;
  * a hairline Ⓤ scored 0.07 against the bold circled-letter template, so the
    modifier could not be identified, and OCR read it as a trailing "0";
  * the frame's text dropped every modifier other than a material condition.

    python -m pytest -q test_fcf_hairline.py
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

import fcf_parse
import gdt_pipeline
from gdt_detect import binarize, to_gray


#: Frames are drawn this many times larger and downsampled, so their strokes
#: arrive as antialiased one-pixel hairlines, the way pdfium renders a vector
#: drawing's lines, rather than as hard-edged pixels nothing ever produces.
SUPERSAMPLE = 4


def hairline_frame(modifier: str | None = "U", value: str = "0.2", scale: int = 1) -> Image.Image:
    """A two-compartment frame: a ⌓ glyph, then the value and a circled letter."""
    k = SUPERSAMPLE * scale
    image = Image.new("RGB", (260 * k, 100 * k), "white")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = 40 * k, 30 * k, 191 * k, 70 * k
    # The horizontal rules stop one pixel short of the left side, as a
    # rendered corner does.
    draw.line([(x0 + k, y0), (x1, y0)], fill="black", width=SUPERSAMPLE)
    draw.line([(x0 + k, y1), (x1, y1)], fill="black", width=SUPERSAMPLE)
    draw.line([(x0, y0), (x0, y1)], fill="black", width=SUPERSAMPLE)
    draw.line([(x1, y0), (x1, y1)], fill="black", width=SUPERSAMPLE)
    divider = x0 + 39 * k
    draw.line([(divider, y0), (divider, y1)], fill="black", width=SUPERSAMPLE)
    # Profile of a surface: a closed half-disc outline in uniform hairlines.
    left, top, right, bottom = x0 + 6 * k, y0 + 10 * k, x0 + 33 * k, y0 + 36 * k
    middle = (top + bottom) // 2
    draw.arc([left, top, right, bottom], 180, 360, fill="black", width=SUPERSAMPLE)
    draw.line([(left, middle), (right, middle)], fill="black", width=SUPERSAMPLE)
    draw.text((divider + 8 * k, y0 + 8 * k), value, fill="black", font=ImageFont.load_default(size=22 * k))
    if modifier:
        cx, cy, r = x1 - 18 * k, y0 + 20 * k, 14 * k
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=SUPERSAMPLE)
        draw.text((cx, cy), modifier, fill="black", font=ImageFont.load_default(size=18 * k), anchor="mm")
    return image.resize((260 * scale, 100 * scale), Image.LANCZOS)


def test_one_pixel_strokes_survive_binarization_unless_they_are_stray():
    image = Image.new("RGB", (200, 60), "white")
    ImageDraw.Draw(image).line([(10, 30), (190, 30)], fill="black", width=1)
    # A hairline rule survives when reading structure.
    assert (binarize(to_gray(image), clean=False)[30, 10:190] > 0).mean() > 0.9
    # A glyph read keeps a hairline mark that is part of the glyph — a ring
    # beside bold digits — but still drops a hairline running across the
    # crop, which there is a leader or a border, as it always did.
    glyph = Image.new("RGB", (80, 60), "white")
    draw = ImageDraw.Draw(glyph)
    draw.rectangle([6, 15, 26, 45], fill="black")  # a bold mark
    draw.ellipse([40, 15, 70, 45], outline="black", width=1)  # a hairline ring
    draw.line([(0, 4), (79, 4)], fill="black", width=1)  # a leader across
    binary = binarize(to_gray(glyph))
    assert (binary[15:46, 40:71] > 0).sum() > 60
    assert (binary[4, :] > 0).mean() < 0.1


def test_a_hairline_frame_is_found_with_its_side_rule_past_the_rule_ends():
    frames = fcf_parse.find_frames(hairline_frame())
    assert len(frames) == 1
    x, y, w, h = frames[0]
    # The box reaches the side rule, one pixel outside the horizontal rules.
    assert abs(x - 40) <= 1 and abs(w - 152) <= 2 and abs(h - 41) <= 2


def ocr_reading(cell_text: str):
    """OCR stub: the whole tolerance cell reads ``cell_text``; a crop without
    the modifier reads the value; a ring's interior reads its letter."""

    def ocr(crop: Image.Image) -> tuple[str, float]:
        if crop.width < crop.height * 1.6:
            return "U", 0.97  # a circled letter's interior
        if crop.width > crop.height * 3.4:
            return cell_text, 0.9  # the whole compartment, ring and all
        return "0.2", 0.99

    return ocr


def test_the_circled_letter_is_read_from_inside_its_ring():
    frame_image = hairline_frame()
    x, y, w, h = fcf_parse.find_frames(frame_image)[0]
    crop = frame_image.crop((int(x), int(y), int(x + w), int(y + h)))
    frame = fcf_parse.parse_frame(crop, bbox=(x, y, w, h), ocr_fn=ocr_reading("0.20"))
    assert frame.symbol is not None and frame.symbol.symbol_id == "profile_of_a_surface"
    assert frame.modifiers == ["unequally_disposed_profile"]
    assert frame.tolerance.value == 0.2
    # And the frame's text, which the drawing view and every export show,
    # carries it.
    assert gdt_pipeline.frame_text(frame) == "⌓ 0.2 Ⓤ"


def test_a_value_misread_through_the_modifier_is_read_again_without_it():
    # "1Ⓤ" read whole as "10": the ring taken for a zero. The value is read
    # again from the compartment up to the located ring.
    frame_image = hairline_frame(value="1")
    x, y, w, h = fcf_parse.find_frames(frame_image)[0]
    crop = frame_image.crop((int(x), int(y), int(x + w), int(y + h)))

    wide_reads: list[int] = []

    def ocr(cell: Image.Image) -> tuple[str, float]:
        if cell.width < cell.height * 1.6:
            return "U", 0.97  # inside the ring
        # The first wide read is the whole compartment; the next is the
        # re-read of the part before the ring.
        wide_reads.append(1)
        return ("10", 0.9) if len(wide_reads) == 1 else ("1", 0.99)

    frame = fcf_parse.parse_frame(crop, bbox=(x, y, w, h), ocr_fn=ocr)
    assert frame.tolerance.value == 1
    assert any("re-read without the modifier" in warning for warning in frame.warnings)


def test_a_trailing_zero_is_not_a_modifier():
    # A tall oval is a digit 0, not a ring: no modifier, and nothing unreadable.
    frame_image = hairline_frame(modifier=None, value="0.20")
    x, y, w, h = fcf_parse.find_frames(frame_image)[0]
    crop = frame_image.crop((int(x), int(y), int(x + w), int(y + h)))
    frame = fcf_parse.parse_frame(
        crop, bbox=(x, y, w, h), ocr_fn=lambda c: ("0.20", 0.99) if c.width > c.height else ("", 0.0)
    )
    assert frame.modifiers == []
    assert frame.material_condition.state != fcf_parse.UNREADABLE
    assert "?" not in gdt_pipeline.frame_text(frame)


def test_the_symbol_is_read_wherever_the_box_edges_fall():
    # The finder's box may or may not include a rule row; the symbol must
    # read either way.
    frame_image = hairline_frame()
    x, y, w, h = (int(v) for v in fcf_parse.find_frames(frame_image)[0])
    for dx, dy, dw, dh in ((0, 0, 0, 0), (1, 1, -2, -2), (-1, -1, 2, 2)):
        box = (x + dx, y + dy, w + dw, h + dh)
        crop = frame_image.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
        frame = fcf_parse.parse_frame(crop, bbox=box)
        assert frame.symbol is not None and frame.symbol.symbol_id == "profile_of_a_surface", box


def test_words_against_a_frame_are_kept_with_it_and_values_are_not():
    region = {"bbox": {"x": 100, "y": 100, "width": 150, "height": 40}, "label": "GD&T Profile of a Surface"}
    gdt_pipeline.attach_adjacent_text(
        [region],
        [
            ("OUTWARDS", {"x": 110, "y": 145, "width": 90, "height": 18}),
            ("25.4", {"x": 110, "y": 145, "width": 40, "height": 18}),
            ("SECTION A-A", {"x": 900, "y": 900, "width": 120, "height": 18}),
        ],
    )
    assert region["associated_text"] == "OUTWARDS"
    assert region["label"] == "GD&T Profile of a Surface · OUTWARDS"


def test_frame_text_shows_an_unreadable_modifier_instead_of_dropping_it():
    frame = fcf_parse.FeatureControlFrame(id="f", bbox=(0, 0, 1, 1), page=1)
    frame.tolerance = fcf_parse.Field.present(0.3, 0.9)
    frame.material_condition = fcf_parse.Field.unreadable(raw="0.3")
    assert gdt_pipeline.frame_text(frame) == "0.3 ?"


def test_the_same_frame_drawn_larger_is_found_too():
    # Twice the size, the same hairline strokes: a higher-resolution render,
    # placed on a sheet so the frame is small beside it, as on a page.
    sheet = Image.new("RGB", (1200, 800), "white")
    sheet.paste(hairline_frame(scale=2), (300, 300))
    assert len(fcf_parse.find_frames(sheet)) == 1
