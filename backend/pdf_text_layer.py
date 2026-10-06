"""
Read a drawing's callouts from the PDF's own text layer.

A vector PDF already carries every value as text, with exact digits and exact
glyph boxes, so reading it is both perfectly accurate and about three orders
of magnitude faster than OCR (5 ms against 30 s on the one sample drawing that
has a text layer). It is an accelerator, not a replacement: five of the six
sample drawings are raster scans with no fonts and no text at all, and those
still need the OCR route.

Two things make a CAD text layer different from ordinary document text:

* **Symbol fonts.** A diameter is not "Ø" in the file. AutoCAD writes it with
  its ``AIGDT`` symbol font, where the glyph code is the letter ``P`` (and
  ``B`` is ±). The same letters appear in the ordinary fonts on the same page,
  so the substitution has to be keyed on the font or the part name ``PUNCH``
  becomes ``ØUNCH``.
* **Rotation, and callouts split across text objects.** A sheet carries text
  at several angles at once, and one callout arrives as several runs: ``70
  ±0.2`` comes through as ``7``, ``0`` and ``±0.2``. Runs are therefore
  grouped per angle, in a frame where that angle's text reads horizontally.
"""

from __future__ import annotations

import ctypes
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

# CAD symbol fonts: their glyph codes are letters, so they are only remapped
# for text actually drawn in one of them.
SYMBOL_FONTS = frozenset({"AIGDT", "GDT", "AMGDT", "ISOGDT"})
SYMBOL_GLYPHS = {"P": "Ø", "B": "±", "n": "Ø"}

# A text font whose 0x80-0x9f codes carry the CAD encoding rather than
# Windows-1252. Those code points are C1 control characters in Unicode, so
# text containing them is already meaningless; mapping the ones we have seen
# is safe and leaves anything else untouched.
# ``0x91`` is the diameter: AutoCAD's "%%c" written in a text font rather
# than in AIGDT, which left "Ø37.5 +0.2" published as a value opening with a
# control character.
CONTROL_GLYPHS = {"\x83": "°", "\x91": "Ø"}

# Fonts used by "searchable scan" tools to hide OCR text behind the image.
# That text IS OCR output, and worse than our own, so a page carrying it is
# treated as having no usable text layer.
OCR_LAYER_FONTS = frozenset({"GlyphLessFont"})


def _base_font_name(font: str) -> str:
    """Remove the six-letter subset prefix used by embedded PDF fonts."""

    return font.split("+", 1)[-1]


@dataclass(frozen=True)
class TextRun:
    """
    One run of text from the PDF, in page-render pixels.

    ``lx``/``ly``/``lw``/``lh`` are the same box in the frame where this run's
    own angle reads horizontally. Grouping happens there, because a sheet
    carries several angles at once and the axis-aligned boxes of vertical text
    run straight through the horizontal callouts.
    """

    text: str
    x: float
    y: float
    width: float
    height: float
    angle: float
    font: str
    lx: float
    ly: float
    lw: float
    lh: float
    #: The font's own size in page-render pixels. The levelled height above
    #: is derived from an axis-aligned box, which is inflated for text at a
    #: diagonal; the font size is exact at any angle, so it is what decides
    #: which runs share a line and which run is the value a stack qualifies.
    size: float


def _decode_symbols(char: str, font: str) -> str:
    if _base_font_name(font) in SYMBOL_FONTS:
        return SYMBOL_GLYPHS.get(char, char)
    return CONTROL_GLYPHS.get(char, char)


def page_text_runs(
    pdf: Any,
    page_index: int = 0,
    *,
    dpi: float = 250.0,
    target_size: tuple[int, int] | None = None,
) -> tuple[list[TextRun], tuple[int, int], set[str]]:
    """
    Every text run on one page, in the pixels of a ``dpi`` render.

    ``pdf`` is a path, bytes, or an open :class:`pypdfium2.PdfDocument`. A run
    ends at a line break in the text layer or wherever the text changes angle.
    Returns ``(runs, (width, height), fonts)``.
    """
    document = pdf if isinstance(pdf, pdfium.PdfDocument) else pdfium.PdfDocument(pdf)
    page = document[page_index]
    text_page = page.get_textpage()
    count = text_page.count_chars()
    if target_size is None:
        scale_x = scale_y = dpi / 72.0
        size = (int(page.get_width() * scale_x), int(page.get_height() * scale_y))
    else:
        size = (int(target_size[0]), int(target_size[1]))
        if size[0] < 1 or size[1] < 1:
            raise ValueError("target_size must contain positive dimensions")
        scale_x = size[0] / max(float(page.get_width()), 1.0)
        scale_y = size[1] / max(float(page.get_height()), 1.0)
    if count == 0:
        return [], size, set()

    page_height = page.get_height()
    name_buffer = ctypes.create_string_buffer(256)
    fonts: set[str] = set()
    runs: list[TextRun] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        if current["text"].strip():
            theta = math.radians(current["angle"])
            x, y = current["x0"], current["y0"]
            width = current["x1"] - x
            height = current["y1"] - y
            lx, ly, lw, lh = _level_box(x, y, width, height, theta)
            runs.append(
                TextRun(
                    text=current["text"].strip(),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    angle=current["angle"],
                    font=current["font"],
                    lx=lx,
                    ly=ly,
                    lw=lw,
                    lh=lh,
                    size=current["size"],
                )
            )
        current = None

    for index in range(count):
        char = text_page.get_text_range(index, 1)
        if char in "\r\n":
            flush()
            continue
        flags = ctypes.c_int()
        written = pdfium_raw.FPDFText_GetFontInfo(
            text_page.raw, index, name_buffer, 256, ctypes.byref(flags)
        )
        font = (
            name_buffer.raw[: max(written - 1, 0)].decode("utf-8", "replace")
            if written
            else ""
        )
        fonts.add(font)
        char = _decode_symbols(char, font)
        left, bottom, right, top = text_page.get_charbox(index)
        angle = round(
            math.degrees(pdfium_raw.FPDFText_GetCharAngle(text_page.raw, index)), 1
        )
        font_size = (
            pdfium_raw.FPDFText_GetFontSize(text_page.raw, index)
            * (scale_x + scale_y)
            / 2.0
        )
        x0, y0 = left * scale_x, (page_height - top) * scale_y
        x1, y1 = right * scale_x, (page_height - bottom) * scale_y
        lx, ly, lw, lh = _level_box(x0, y0, x1 - x0, y1 - y0, math.radians(angle))

        if current is not None:
            broken = abs(angle - current["angle"]) > 1.0
            if not broken:
                # One text object can hold two separate labels with a wide
                # space between them ("SHARP" ... "po"), and joining those
                # gives a run that reaches across a callout it has nothing to
                # do with. A gap wider than a character ends the run.
                line = max(current["lh"], lh, 1.0)
                if lx - current["lx1"] > 1.2 * line:
                    broken = True
                else:
                    # Same line of text, tested by overlap rather than by the
                    # distance between top edges: a decimal point is a tiny
                    # box sitting at the baseline, and comparing edges tears
                    # it off its own number.
                    overlap = min(current["ly1"], ly + lh) - max(current["ly"], ly)
                    if overlap <= 0:
                        broken = True
            if broken:
                flush()

        if current is None:
            current = {
                "text": char,
                "angle": angle,
                "font": font,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "lx": lx,
                "ly": ly,
                "lx1": lx + lw,
                "ly1": ly + lh,
                "lh": lh,
                "size": font_size,
            }
        else:
            current["text"] += char
            current["x0"] = min(current["x0"], x0)
            current["y0"] = min(current["y0"], y0)
            current["x1"] = max(current["x1"], x1)
            current["y1"] = max(current["y1"], y1)
            current["lx1"] = max(current["lx1"], lx + lw)
            current["ly"] = min(current["ly"], ly)
            current["ly1"] = max(current["ly1"], ly + lh)
            current["lh"] = max(current["lh"], lh)
            current["size"] = max(current["size"], font_size)
    flush()

    return runs, size, fonts


# A drawing whose picture is one big raster cannot have its dimensions in the
# text layer, whatever text sits on top of it. Measured over the sample
# drawings: a scan's image covers 0.91 to 1.00 of the page, while a vector
# drawing's largest image is a logo at 0.01 to 0.02.
RASTER_PAGE_COVERAGE = 0.6

# A slashed ring with both its holes, scored by phi_detector. Its own
# documentation puts a confirmed diameter at 0.8 or above.
PHI_TOPOLOGY_MIN = 0.8

# Breathing room around a callout's box, as a fraction of its line height. The
# glyph boxes are exact, so without this a balloon sits hard against the ink.
BOX_PADDING = 0.3

# How far two side-by-side runs' font sizes may differ and still be one callout.
#
# Runs are banded into lines by vertical overlap, measured against the SMALLER
# of the two heights on purpose: a decimal point is a tiny box at the baseline
# and has to stay with its number. The side effect is that a small run falling
# anywhere inside a tall run's vertical span lands in its band, so the 21pt
# view label "View I" took in the 12pt "+0.2" of the callout beside it and the
# whole group was then thrown away as prose. Size tells them apart: CAD sets a
# deviation at about three quarters of its value, so ratios up to about 1.25
# are still one callout and anything past this is two kinds of text.
#
# The test is between neighbours rather than over the band, because a band is
# a page-wide strip: its own size range is set by whatever else happens to sit
# at that height, which is usually nothing to do with the callout.
SAME_LINE_SIZE_RATIO = 1.4

# A deviation is a signed number and nothing else. Requiring the shape keeps
# the rule below from sweeping up a title-block label that happens to be set
# in smaller type beside a value ("23-11-2018" next to "Dat.:").
_DEVIATION_RE = re.compile(r"^[+\-\u2212\u00b1]?\s*\d*\.?\d+$")

# Marks that make a number a measurement on their own.
_VALUE_MARKS = "\u00d8\u2300\u00f8\u03c6\u03a6\u00b0\u00b1\u2032\u2033'\""
_RADIUS_RE = re.compile(r"[Rr]\s*\d")
_BRACKET_RE = re.compile(r"[\[\](){}]")
_QUALIFIER_RE = re.compile(r"(MIN|MAX|REF|TYP|THK|DEEP)", re.I)
# A word, as opposed to the single letters of a datum or a feature control
# frame ("Ø0.5 A B C"), which belong to the callout they sit in.
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
# A signed value standing entirely on its own: "+0.5", "-0.8".
_SIGNED_ALONE_RE = re.compile(r"^[+\-\u2212\u00b1]\s*\d+(?:[.,]\d+)?$")


def _is_prose(members: Iterable[TextRun]) -> bool:
    """Whether a group reads as a label rather than as part of a measurement."""
    return any(
        _WORD_RE.search(member.text) and not _QUALIFIER_RE.search(member.text)
        for member in members
    )


def _is_value_shaped(members: Iterable[TextRun]) -> bool:
    """
    Whether a group carries enough to be the value that a stack qualifies.

    A deviation stacks under something it measures. Two lone digits stacking
    is the numbering of a note list, where "2." sits above "3." in a column
    of its own, and the callout gate reads the pair as the dimension "2 3".
    """
    text = "".join(member.text for member in members)
    if any(mark in text for mark in _VALUE_MARKS) or _RADIUS_RE.search(text):
        return True
    return sum(character.isdigit() for character in text) >= 2


def is_callout(text: str) -> bool:
    """
    Whether one piece of exact text is a callout worth a balloon.

    The OCR routes use ``is_segment_worthy``, which throws away anything
    under three characters because a short OCR read is usually a fragment of
    a misread. Nothing here is misread, so that rule silently dropped real
    values: ``7.3``, ``3 MIN``, ``(25)``, ``0.2``, the ``21`` of an angle
    whose degree sign is drawn as geometry. What is left to exclude is prose
    and the lone digits a numbered note leaves behind.

    ``has_dimension_value`` carries one more rule of the same kind, and it is
    skipped here for the same reason: it drops a signed value standing alone
    as "a tolerance with nothing to apply it to". For an OCR read that is a
    sign left behind when the segmenter tore a value in half. In exact text
    it is the whole callout, and this drawing really does carry "+0.5" and
    "-0.5" by themselves against a step. They are published for review,
    because a deviation with nothing beside it is worth a second look.
    """
    from segment_quality import has_dimension_value

    stripped = (text or "").strip()
    if not has_dimension_value(stripped) and not _SIGNED_ALONE_RE.match(stripped):
        return False
    digits = sum(character.isdigit() for character in stripped)
    if digits == 0:
        return False
    if any(mark in stripped for mark in _VALUE_MARKS):
        return True
    if _RADIUS_RE.search(stripped) or _BRACKET_RE.search(stripped):
        return True
    if _QUALIFIER_RE.search(stripped) or "." in stripped:
        return True
    # A lone digit is a note's number or a datum, never a dimension.
    return digits >= 2


def is_bare_number(text: str) -> bool:
    """
    Whether a callout carries no mark saying what it measures.

    On a drawing that writes Ø, ° and x as geometry rather than text, such a
    value has probably lost one, so it is published for the inspector to
    confirm rather than dropped or presented as certain.
    """
    stripped = (text or "").strip()
    # A lone deviation says how much, never of what. Whatever it qualifies is
    # drawn rather than written, so it always goes to the inspector.
    if _SIGNED_ALONE_RE.match(stripped):
        return True
    if any(mark in stripped for mark in _VALUE_MARKS):
        return False
    if _RADIUS_RE.search(stripped) or _QUALIFIER_RE.search(stripped):
        return False
    if _BRACKET_RE.search(stripped):
        return False
    return "." not in stripped


def page_raster_coverage(pdf: Any, page_index: int = 0) -> float:
    """The page fraction covered by its largest embedded image."""
    document = pdf if isinstance(pdf, pdfium.PdfDocument) else pdfium.PdfDocument(pdf)
    page = document[page_index]
    area = page.get_width() * page.get_height()
    if area <= 0:
        return 0.0
    largest = 0.0
    for obj in page.get_objects(max_depth=15):
        if obj.type != pdfium_raw.FPDF_PAGEOBJ_IMAGE:
            continue
        try:
            left, bottom, right, top = obj.get_bounds()
        except Exception:
            continue
        largest = max(largest, max(0.0, right - left) * max(0.0, top - bottom))
    return largest / area


def has_usable_text_layer(
    runs: Iterable[TextRun],
    fonts: Iterable[str],
    raster_coverage: float = 0.0,
) -> bool:
    """
    Whether this page's text can be trusted instead of reading the pixels.

    A page with no text at all is a scan. So are the two cases that would
    otherwise be worse than a scan, because they look like a text layer and
    are not: text hidden behind the image by a "make searchable" tool, which
    is somebody else's OCR, and a raster drawing carrying a few vector
    labels, whose dimensions are all in the picture. Both fall back to our
    own pipeline rather than publish a handful of values and call the sheet
    done.
    """
    if any(_base_font_name(font) in OCR_LAYER_FONTS for font in fonts):
        return False
    if raster_coverage >= RASTER_PAGE_COVERAGE:
        return False
    return any(run.text.strip() for run in runs)


def _level_box(
    x: float, y: float, width: float, height: float, theta: float
) -> tuple[float, float, float, float]:
    """One axis-aligned box mapped into the frame that levels ``theta``."""
    centre_x, centre_y = x + width / 2.0, y + height / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    levelled_width = abs(cos_t) * width + abs(sin_t) * height
    levelled_height = abs(sin_t) * width + abs(cos_t) * height
    return (
        (centre_x * cos_t + centre_y * sin_t) - levelled_width / 2.0,
        (-centre_x * sin_t + centre_y * cos_t) - levelled_height / 2.0,
        levelled_width,
        levelled_height,
    )


def _unlevel(x: float, y: float, theta: float) -> tuple[float, float]:
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def _comparable_sizes(left: TextRun, right: TextRun) -> bool:
    """Whether two neighbouring runs are set close enough in size to be one callout."""
    smaller = max(min(left.size, right.size), 0.1)
    return max(left.size, right.size) <= SAME_LINE_SIZE_RATIO * smaller


def _reading_order(members: list[TextRun]) -> list[TextRun]:
    """
    A callout's runs in the order a person reads them.

    Rows are decided by whether the boxes actually overlap, not by dividing
    the distance from the top by a line height: a bracket is half the height
    of the number it encloses, so quantising ``( 67.1 )`` by line put the
    brackets on a row of their own and read it as ``67.1()``.
    """
    if len(members) < 2:
        return list(members)
    # Compared against the SMALLEST text in the group, not the median: a
    # value with two deviations beside it is a group of three where the
    # median is already the value's own size, and it would never stand out.
    body = min(run.size for run in members)
    # A run printed larger than the rest is the value, and the smaller runs
    # are the deviations stacked beside it. It is read first however its box
    # happens to overlap theirs. CAD sets a deviation at about three quarters
    # of its value's size, so the real ratios are around 1.25; anything under
    # this is one line of text in one size.
    lead = [run for run in members if run.size > 1.15 * body]
    rest = [run for run in members if run not in lead]

    rows: list[list[TextRun]] = []
    for run in sorted(rest, key=lambda r: r.ly):
        for row in rows:
            top = min(r.ly for r in row)
            bottom = max(r.ly + r.lh for r in row)
            overlap = min(bottom, run.ly + run.lh) - max(top, run.ly)
            if overlap >= 0.4 * min(bottom - top, run.lh):
                row.append(run)
                break
        else:
            rows.append([run])
    rows.sort(key=lambda row: min(r.ly for r in row))
    return sorted(lead, key=lambda r: r.lx) + [
        run for row in rows for run in sorted(row, key=lambda r: r.lx)
    ]


def _join_runs(ordered: list[TextRun], line: float) -> str:
    """
    Join a callout's runs, inserting a space only where the drawing has one.

    CAD can write one number as several runs — "70" arrives as "7" then "0" —
    so joining unconditionally with a space would rewrite the value.
    """
    value_size = max(run.size for run in ordered)
    text = ordered[0].text
    for previous, run in zip(ordered, ordered[1:]):
        same_line = abs(run.ly - previous.ly) < 0.5 * line
        gap = run.lx - (previous.lx + previous.lw)
        if same_line and gap < 0.25 * line:
            separator = ""
        elif (
            not same_line
            # Both are deviations rather than the value they qualify, which
            # is set larger: without this the value itself took the slash
            # and "21 +1 0" came out as "21/+1/0".
            and previous.size < value_size
            and run.size < value_size
            and _DEVIATION_RE.match(previous.text.strip())
            and _DEVIATION_RE.match(run.text.strip())
        ):
            # The upper and lower limits of one deviation, stacked. Written
            # as "+0.2/0", the same form the OCR route publishes.
            separator = "/"
        else:
            separator = " "
        text += separator + run.text
    return text


def group_runs(
    runs: list[TextRun],
    page_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """
    Group runs into callouts, one angle at a time.

    Text at one angle never joins text at another: on a real sheet the
    vertical dimensions run right through the horizontal ones, and grouping
    their axis-aligned boxes together fuses callouts that never touch.

    Within an angle the boxes are exact, so the rule is plain geometry rather
    than the adaptive thresholds the OCR path needs: runs sharing a line join
    when the gap between them is under a character, and a line joins the line
    above when it sits directly under it, which is how a value and its
    stacked deviations arrive.
    """
    del page_size  # the levelled frame is self-describing
    out: list[dict[str, Any]] = []
    for angle in sorted({run.angle for run in runs}):
        here = [run for run in runs if run.angle == angle]
        heights = sorted(run.lh for run in here)
        line = max(heights[len(heights) // 2], 1.0)

        lines: list[list[TextRun]] = []
        for run in sorted(here, key=lambda r: (r.ly, r.lx)):
            placed = False
            for existing in lines:
                top = min(r.ly for r in existing)
                bottom = max(r.ly + r.lh for r in existing)
                overlap = min(bottom, run.ly + run.lh) - max(top, run.ly)
                if overlap >= 0.4 * min(bottom - top, run.lh):
                    existing.append(run)
                    placed = True
                    break
            if not placed:
                lines.append([run])

        groups: list[list[TextRun]] = []
        for members in lines:
            members.sort(key=lambda r: r.lx)
            run_group = [members[0]]
            for run in members[1:]:
                gap = run.lx - max(r.lx + r.lw for r in run_group)
                # A bracket is punctuation around a value, so it joins it
                # from further away than ordinary text would.
                bracket = _BRACKET_RE.fullmatch(run.text.strip()) or _BRACKET_RE.fullmatch(
                    run_group[-1].text.strip()
                )
                if gap <= (3.0 if bracket else 1.2) * line and _comparable_sizes(
                    run_group[-1], run
                ):
                    run_group.append(run)
                else:
                    groups.append(run_group)
                    run_group = [run]
            groups.append(run_group)

        # A value and the deviations stacked BESIDE it are one callout too:
        # "Ø8 +0.2/0" and "21° +1°/0°" put a two-row column of smaller text
        # immediately to the right of the value, and neither row shares
        # enough of the value's own row to be banded with it.
        merged = True
        while merged:
            merged = False
            for i, value in enumerate(groups):
                v_size = max(r.size for r in value)
                v_x1 = max(r.lx + r.lw for r in value)
                v_top = min(r.ly for r in value)
                v_bottom = max(r.ly + r.lh for r in value)
                v_mid = (v_top + v_bottom) / 2.0
                for j, stack in enumerate(groups):
                    if i == j:
                        continue
                    if max(r.size for r in stack) >= 0.9 * v_size:
                        continue
                    if not all(_DEVIATION_RE.match(r.text.strip()) for r in stack):
                        continue
                    gap = min(r.lx for r in stack) - v_x1
                    if not -0.2 * line <= gap <= 2.0 * line:
                        continue
                    s_top = min(r.ly for r in stack)
                    s_bottom = max(r.ly + r.lh for r in stack)
                    if abs((s_top + s_bottom) / 2.0 - v_mid) > 1.2 * line:
                        continue
                    groups[i] = value + stack
                    groups.pop(j)
                    merged = True
                    break
                if merged:
                    break

        # A value and the deviations stacked under it are one callout: the
        # lower group sits within the upper one's width, a fraction of a line
        # below it.
        merged = True
        while merged:
            merged = False
            for i, upper in enumerate(groups):
                for j, lower in enumerate(groups):
                    if i == j:
                        continue
                    ux0 = min(r.lx for r in upper)
                    ux1 = max(r.lx + r.lw for r in upper)
                    uy1 = max(r.ly + r.lh for r in upper)
                    lx0 = min(r.lx for r in lower)
                    lx1 = max(r.lx + r.lw for r in lower)
                    ly0 = min(r.ly for r in lower)
                    # Unlike the merge above, this one has only geometry to go
                    # on, and a label printed under a callout stacks exactly
                    # like a deviation does. Absorbing one turned an exact
                    # value into prose that the callout gate then threw away
                    # whole: "2.8" over "Spline Run-off" and "(15.9)" over
                    # "Effective Spline Length" were both lost that way.
                    if _is_prose(lower) or _is_prose(upper):
                        continue
                    if not _is_value_shaped(upper):
                        continue
                    # Rows of a deviation column can overlap slightly, and the
                    # line height here is a median over text of two sizes.
                    if not -0.3 * line <= ly0 - uy1 <= 0.7 * line:
                        continue
                    shared = min(ux1, lx1) - max(ux0, lx0)
                    if shared < 0.6 * min(ux1 - ux0, lx1 - lx0):
                        continue
                    groups[i] = upper + lower
                    groups.pop(j)
                    merged = True
                    break
                if merged:
                    break

        theta = math.radians(angle)
        for members in groups:
            ordered = _reading_order(members)
            pad = BOX_PADDING * max(r.size for r in ordered)
            x0 = min(r.lx for r in ordered) - pad
            y0 = min(r.ly for r in ordered) - pad
            x1 = max(r.lx + r.lw for r in ordered) + pad
            y1 = max(r.ly + r.lh for r in ordered) + pad
            corners = [
                _unlevel(x, y, theta)
                for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            ]
            out.append(
                {
                    "text": _join_runs(ordered, line),
                    "angle": angle,
                    "runs": ordered,
                    "corners": corners,
                    # Kept so a caller that finds a symbol drawn beside the
                    # text can grow the box over it.
                    "bounds": (x0, y0, x1, y1),
                }
            )
    return out


def _boxes_from_corners(
    corners: list[tuple[float, float]]
) -> tuple[dict[str, float], dict[str, float]]:
    """The axis-aligned hull and the tight rotated rectangle for one callout."""
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    bbox = {
        "x": round(min(xs), 1),
        "y": round(min(ys), 1),
        "width": round(max(xs) - min(xs), 1),
        "height": round(max(ys) - min(ys), 1),
    }
    (tlx, tly), (trx, try_), _, (blx, bly) = corners
    oriented = {
        "x": round(tlx, 1),
        "y": round(tly, 1),
        "width": round(math.hypot(trx - tlx, try_ - tly), 1),
        "height": round(math.hypot(blx - tlx, bly - tly), 1),
        "rotation": round(math.degrees(math.atan2(try_ - tly, trx - tlx)), 1),
    }
    return bbox, oriented


def _prefix_zone(
    page_image: Any, runs: list[TextRun], angle: float
) -> Any | None:
    """
    The strip of drawing immediately before a callout, read upright.

    Some CAD exports draw Ø and ° as geometry rather than text, so the text
    layer holds every digit exactly and no symbol at all. The digits are
    still worth far more than a scan; the symbol is recovered from the
    picture, in the one place it can be.
    """
    from PIL import Image

    if not runs:
        return None
    lead = min(runs, key=lambda r: r.lx)
    size = max(lead.size, 1.0)
    # A zone one and a half characters wide, ending where the value starts.
    x0, x1 = lead.lx - 1.6 * size, lead.lx - 0.05 * size
    y0, y1 = lead.ly - 0.25 * size, lead.ly + lead.lh + 0.25 * size
    theta = math.radians(angle)
    corners = [
        _unlevel(x, y, theta) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    ]
    left = int(min(x for x, _ in corners))
    top = int(min(y for _, y in corners))
    right = int(max(x for x, _ in corners))
    bottom = int(max(y for _, y in corners))
    if right - left < 8 or bottom - top < 8:
        return None
    crop = page_image.crop(
        (
            max(0, left),
            max(0, top),
            min(page_image.width, right),
            min(page_image.height, bottom),
        )
    )
    if crop.width < 8 or crop.height < 8:
        return None
    if abs(angle) < 1.0:
        return crop
    return crop.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")


def _render_page(
    pdf: Any,
    page_index: int,
    dpi: float,
    target_size: tuple[int, int] | None = None,
) -> Any:
    document = pdf if isinstance(pdf, pdfium.PdfDocument) else pdfium.PdfDocument(pdf)
    page = document[page_index]
    scale = (
        target_size[0] / max(float(page.get_width()), 1.0)
        if target_size is not None
        else dpi / 72.0
    )
    rendered = page.render(scale=scale).to_pil().convert("RGB")
    if target_size is not None and rendered.size != target_size:
        from PIL import Image

        rendered = rendered.resize(target_size, resample=Image.BICUBIC)
    return rendered


def text_layer_regions(
    pdf: Any,
    page_index: int = 0,
    *,
    dpi: float = 250.0,
    recover_symbols: bool = True,
    target_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """
    Read one page's callouts from its text layer.

    Returns the same region shape the OCR routes publish, so the client maps
    and balloons them without knowing which route produced them. ``usable``
    is False when the page carries no text worth trusting, which is the
    signal to fall back to a scan.
    """
    from dimension_compose import compose_engineering_dimension
    from feature_classifier import classify_feature
    from symbol_vision import DetectedSymbols

    runs, size, fonts = page_text_runs(
        pdf,
        page_index,
        dpi=dpi,
        target_size=target_size,
    )
    coverage = page_raster_coverage(pdf, page_index)
    if not has_usable_text_layer(runs, fonts, coverage):
        if any(_base_font_name(font) in OCR_LAYER_FONTS for font in fonts):
            fallback_reason = "searchable_ocr_text_layer"
        elif coverage >= RASTER_PAGE_COVERAGE:
            fallback_reason = "raster_dominant_page"
        else:
            fallback_reason = "no_native_text"
        return {
            "usable": False,
            "page_size": size,
            "regions": [],
            "excluded": [],
            "fonts": sorted(fonts),
            "raster_coverage": round(coverage, 3),
            "fallback_reason": fallback_reason,
        }

    regions: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    page_image: Any | None = None
    for group in group_runs(runs, size):
        raw = group["text"]
        composed = compose_engineering_dimension(raw, None, DetectedSymbols())
        text = (composed.text or "").strip() or raw.strip()
        bbox, oriented = _boxes_from_corners(group["corners"])
        if not any(character.isdigit() for character in text):
            excluded.append({"text": text, "bbox": bbox})
            continue
        # Recovery runs BEFORE the gate: a position tolerance reads "2 A B C"
        # until its Ø is found, and a lone digit does not earn a balloon.
        if recover_symbols and not any(mark in text for mark in "Ø⌀φ"):
            if page_image is None:
                page_image = _render_page(pdf, page_index, dpi, target_size)
            zone = _prefix_zone(page_image, group["runs"], group["angle"])
            if zone is not None:
                # The topology detector, which wants a slashed ring with its
                # two holes, not merely round ink. The lighter prefix scorer
                # was tried here and fires on arrowheads and on any "O" in
                # the title block: it put a Ø on "Form.: A4" and on "2x R29.9".
                from phi_detector import detect_phi_topology

                score, _details = detect_phi_topology(zone)
                if score >= PHI_TOPOLOGY_MIN:
                    text = "Ø" + text
                    composed = compose_engineering_dimension(
                        text, None, DetectedSymbols(diameter=True)
                    )
                    text = (composed.text or "").strip() or text
                    # The balloon covers the symbol it just read, not only
                    # the digits that were in the text layer.
                    bx0, by0, bx1, by1 = group["bounds"]
                    lead_size = max(run.size for run in group["runs"])
                    theta = math.radians(group["angle"])
                    bbox, oriented = _boxes_from_corners(
                        [
                            _unlevel(x, y, theta)
                            for x, y in (
                                (bx0 - 1.6 * lead_size, by0),
                                (bx1, by0),
                                (bx1, by1),
                                (bx0 - 1.6 * lead_size, by1),
                            )
                        ]
                    )
        if not is_callout(text):
            excluded.append({"text": text, "bbox": bbox})
            continue
        review = is_bare_number(text)
        if review and recover_symbols:
            # A number with no mark has probably lost one the drawing drew
            # instead of writing: "1 × 45°" reaches the text layer as "1" and
            # "45" with nothing between them. Reading that ink recovers the
            # mark, and the exact digits still decide whether to believe it.
            if page_image is None:
                page_image = _render_page(pdf, page_index, dpi, target_size)
            lead_size = max(run.size for run in group["runs"])
            crop = _levelled_crop(
                page_image,
                group["bounds"],
                group["angle"],
                pad_left=4.0 * lead_size,
                pad_right=2.0 * lead_size,
                pad_cross=0.4 * lead_size,
            )
            if crop is not None:
                from ocr_pipeline import get_pipeline

                result = get_pipeline().recognize(crop, compute_text_bbox=False)
                rescued = rescue_geometry_marks(
                    text, str(result.get("text") or "").strip()
                )
                if rescued and float(result.get("confidence") or 0.0) >= 0.9:
                    text = rescued
                    review = False
                    bbox, oriented = _boxes_from_corners(
                        [
                            _unlevel(x, y, math.radians(group["angle"]))
                            # The recovered marks sit on both sides: the "×"
                            # before the value, the "°" after it.
                            for x, y in (
                                (group["bounds"][0] - 4.0 * lead_size, group["bounds"][1]),
                                (group["bounds"][2] + 0.9 * lead_size, group["bounds"][1]),
                                (group["bounds"][2] + 0.9 * lead_size, group["bounds"][3]),
                                (group["bounds"][0] - 4.0 * lead_size, group["bounds"][3]),
                            )
                        ]
                    )
        feature = classify_feature(text)
        upright = abs(group["angle"]) < 1.0 or abs(abs(group["angle"]) - 180.0) < 1.0
        region = {
            "bbox": bbox,
            "text": text,
            "confidence": 0.995,
            "type": composed.kind,
            "category": feature.category,
            "subtype": feature.subtype,
            "label": feature.label,
            "orientation": "horizontal" if upright else "vertical",
            "rotation": 0 if upright else oriented["rotation"],
            "needs_review": review,
            "agreement": 1.0,
            "engine": "pdf-text-layer",
            "symbols_detected": None,
            "recognized": True,
            "recognition_source": "native_pdf",
            "recognition_evidence": {
                "selected_source": "native_pdf",
                "sources": ["native_pdf"],
                "native_text": text,
                "ocr_text": "",
                "agreement": 1.0,
                "conflict": False,
                "native_span_ids": [],
                "native_bbox": dict(bbox),
            },
        }
        if not upright:
            region["oriented_box"] = oriented
        regions.append(region)

    return {
        "usable": True,
        "page_size": size,
        "regions": regions,
        "excluded": excluded,
        "fonts": sorted(fonts),
        "raster_coverage": round(coverage, 3),
        "fallback_reason": None,
    }


# What a rescued read must gain over the text layer's own: a mark the drawing
# wrote as geometry rather than as text.
_GEOMETRY_MARKS = "×x°Ø⌀±R"


def _levelled_crop(
    page_image: Any,
    bounds: tuple[float, float, float, float],
    angle: float,
    pad_left: float,
    pad_right: float,
    pad_cross: float,
) -> Any | None:
    """A callout's neighbourhood, rotated so its own text reads upright."""
    from PIL import Image

    x0, y0, x1, y1 = bounds
    corners = [
        _unlevel(x, y, math.radians(angle))
        for x, y in (
            (x0 - pad_left, y0 - pad_cross),
            (x1 + pad_right, y0 - pad_cross),
            (x1 + pad_right, y1 + pad_cross),
            (x0 - pad_left, y1 + pad_cross),
        )
    ]
    left = max(0, int(min(x for x, _ in corners)))
    top = max(0, int(min(y for _, y in corners)))
    right = min(page_image.width, int(max(x for x, _ in corners)))
    bottom = min(page_image.height, int(max(y for _, y in corners)))
    if right - left < 10 or bottom - top < 10:
        return None
    crop = page_image.crop((left, top, right, bottom))
    if abs(angle) < 1.0:
        return crop
    return crop.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")


def rescue_geometry_marks(text: str, read: str) -> str | None:
    """
    Whether an OCR read of the same ink may replace the text layer's version.

    Only when it says the same number — the digits of the exact text appear
    in it, in order, unbroken — and adds a mark the text layer could not
    carry because the drawing drew it rather than wrote it. Anything else is
    the recogniser disagreeing with a value that is already certain, and the
    certain one wins.
    """
    exact_digits = "".join(c for c in text if c.isdigit())
    read_digits = "".join(c for c in read if c.isdigit())
    if not exact_digits or exact_digits not in read_digits:
        return None
    if not any(mark in read for mark in _GEOMETRY_MARKS):
        return None
    if any(mark in text for mark in _GEOMETRY_MARKS):
        return None
    # Only a plain number may be replaced wholesale. A value that already
    # carries a sign or a deviation pair has structure the text layer got
    # right and the recogniser would flatten: "21+1/0" came back "21°+1°0".
    if not re.fullmatch(r"\d+(?:\.\d+)?", text.strip()):
        return None
    return read
