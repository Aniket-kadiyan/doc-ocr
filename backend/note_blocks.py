"""
Rebuild a drawing's notes from loose pieces of text.

Both readers hand over text in pieces that are smaller than a note: the PDF
text layer as runs (one per text object, often one per word or per glyph
cluster), the OCR detector as one box per printed line segment. Neither
says which lines make up one note, and a note that is published line by line
reads as three unrelated annotations — or, worse, its "0.2 MAX" comes back
as a dimension balloon with the sentence around it thrown away.

This module is deliberately source-agnostic: it reasons only about geometry
and the text itself, so a vector PDF and a scan of the same sheet group the
same way. Two stages:

* :func:`build_lines` joins fragments that share a baseline into lines,
  per text angle, and splits a line wherever the gap is wide enough to be a
  second column.
* :func:`build_blocks` stacks lines into paragraphs when their size, pitch
  and alignment agree, and :func:`classify_block` decides which paragraphs
  are notes. A block that is not a note is still returned, labelled, so a
  caller can keep it for audit instead of losing it.

Coordinates are page pixels with a top-left origin, the convention every
route in this backend uses. Angles are degrees CLOCKWISE on the page, as
pdfium reports them (text running up the sheet is 270); text at a non-zero
angle is grouped in a frame where it reads horizontally and mapped back
afterwards.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# A gap this many characters wide between two fragments on one line is a
# column break, not a word space. Notes set in two columns, and a note
# sharing its baseline with an unrelated label, are separated here.
COLUMN_GAP_EM = 2.5

# Native runs closer than this (in characters) are one word that the PDF
# happened to write as two objects ("7" + "0"); anything wider is a space.
GLUED_GAP_EM = 0.2

# Consecutive lines of one paragraph sit at most this many characters apart,
# baseline to baseline. CAD leading is 1.2-1.7; 2.1 tolerates a detector
# that boxes each line a little loosely without reaching the next note.
MAX_LINE_PITCH_EM = 2.1

# Lines of one paragraph are set in one size. Headings may be larger.
SAME_SIZE_RATIO = 1.3
OCR_SAME_SIZE_RATIO = 1.6
HEADING_SIZE_RATIO = 1.8

# Alignment tolerance, in characters, for left/centre/right aligned lines.
ALIGN_EM = 1.0

# A wrapped line under a numbered point starts after the marker, so it is
# indented from the block's left edge by up to this many characters.
HANGING_INDENT_EM = 5.0

# Prose evidence for an unnumbered paragraph to count as a note: enough real
# words, mostly letters. "SECTION A-A" and "Designed by" stay labels; "ALL
# DRAWINGS OF CAST / MACHINED COMPONENTS…" is a note.
NOTE_MIN_WORDS = 5
NOTE_MIN_WORDS_PER_LINE = 2.0
NOTE_MAX_DIGIT_FRACTION = 0.3

# A cell neighbour this close on the same baseline marks a table row.
TABLE_NEIGHBOUR_EM = 12.0

NOTES_HEADING_RE = re.compile(
    r"^\s*(?:GENERAL\s+)?N[OU0]TES?\s*[:.\-]?\s*$|^\s*注記\s*$", re.I
)
# "1." / "2)" / "3 WORD WORD": the shapes ocr_pipeline accepts as markers.
NOTE_MARKER_RE = re.compile(
    r"^\s*(\d{1,2})\s*[.)]\s*\S|^\s*(\d{1,2})\s+[^\W\d_]{3,}\s+\S"
)
_WORD_RE = re.compile(r"[^\W\d_][^\W\d_'’\-]+", re.UNICODE)
# Sizing words that sit inside a callout ("3 MIN", "R2 TYP") and are not prose.
_QUALIFIERS = frozenset({"MIN", "MAX", "REF", "TYP", "THK", "DEEP", "THRU", "PLACES", "PL"})
# A proprietary or copyright notice: legally a note, never an inspection item.
_LEGAL_RE = re.compile(
    r"\b(PROPRIETARY|CONFIDENTIAL|COPYRIGHT|REPRODUCTION|SOLE\s+PROPERTY|PROPERTY\s+OF"
    r"|(?:BUSINESS|TRADE)\s+SECRET|ILLEGAL|ALL\s+RIGHTS\s+RESERVED"
    r"|WITHOUT\s+(?:THE\s+)?(?:PRIOR|WRITTEN)\s+(?:APPROVAL|PERMISSION|CONSENT))\b",
    re.I,
)
# Title-block field labels: "Drawn by", "Part Name:", "Drawing nr.:".
_FIELD_LABEL_RE = re.compile(r"(?:\b(?:by|nr|no|name|date)\b\.?\s*:?\s*$)|:\s*$", re.I)
# A field label anywhere in the text: "Drawing Name : ICE…", "DWG NO.: 47630".
_TITLE_FIELD_RE = re.compile(
    r"\b(?:DRAWING|DWG|PART|DRAWN|CHECKED|APPROVED|DESIGNED|SHEET|SCALE|TITLE|REV(?:ISION)?|PROJ(?:ECT)?"
    r"|MAT|TREAT|SURF|DIM|DAT|FORM|CODE)"
    r"\.?\s*(?:NAME|NO|NR|NUMBER|BY|DATE)?\.?\s*:",
    re.I,
)
# A revision-table entry: an engineering change number, or the verbs a
# revision description is written in ("HOOK HEIGHT TOLERANCE WAS ±.030").
_ECN_RE = re.compile(r"\b(?:ECN|ECR|ECO)\s*-?\s*\d", re.I)
_REVISION_VERB_RE = re.compile(
    r"\b(?:WAS|WERE|ADDED|CHANGED|MODIFIED|RELEASED|REVISED|DELETED|UPDATED|REDRAWN)\b", re.I
)
# Words that make a sentence of a phrase. A note is written as one ("ALL
# DIMENSIONS ARE IN M.M."); a table's row label is a noun phrase ("UTILIZABLE
# LENGTH OF PATH OF CONTACT"). Only consulted for Latin text in a table.
_FUNCTION_WORDS = frozenset(
    "A AN THE TO BE IS ARE AND OR OF IN ON AT BY FOR FROM WITH WITHOUT AS PER ALL ANY "
    "NOT NO SHALL MUST SHOULD WILL MAY UNLESS OTHERWISE THAN AFTER BEFORE".split()
)
# A company name in the title block.
_COMPANY_RE = re.compile(r"\b(?:LTD|LIMITED|INC|GMBH|CORP|PVT|B\.V|S\.A|LLC|CO\.)\b\.?", re.I)

# Glyphs a text layer emits when it cannot map a character to Unicode.
_UNMAPPED_RE = re.compile("[�\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f-]")


@dataclass
class Fragment:
    """One piece of text with an axis-aligned box in page pixels."""

    text: str
    x: float
    y: float
    w: float
    h: float
    angle: float = 0.0
    #: Glyph size in pixels. The font size for native text (exact at any
    #: angle); the box height for OCR, where nothing better is known.
    size: float = 0.0
    source: str = "ocr"
    confidence: float = 1.0
    #: Set by the caller for its own bookkeeping (an index into its list).
    ref: Any = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            self.size = self._levelled()[3]

    def _levelled(self) -> tuple[float, float, float, float]:
        return level_box(self.x, self.y, self.w, self.h, math.radians(self.angle))

    @property
    def suspect(self) -> bool:
        """Whether the text layer failed to decode part of this text."""
        return bool(_UNMAPPED_RE.search(self.text))


@dataclass
class TextLine:
    fragments: list[Fragment]
    text: str
    angle: float
    lx: float
    ly: float
    lw: float
    lh: float
    size: float
    #: Other segments sharing this baseline, as (left, right, text) in the
    #: levelled frame. Cells of a table row see each other here.
    neighbours: list[tuple[float, float, str]] = field(default_factory=list)

    @property
    def right(self) -> float:
        return self.lx + self.lw

    @property
    def bottom(self) -> float:
        return self.ly + self.lh

    @property
    def centre(self) -> float:
        return self.lx + self.lw / 2.0

    def page_box(self) -> dict[str, float]:
        return levelled_to_page_box(self.lx, self.ly, self.lw, self.lh, self.angle)


@dataclass
class TextBlock:
    lines: list[TextLine]
    heading: TextLine | None = None
    category: str = "unclassified"
    reason: str = ""
    #: Value-like lines beside the block within its own height: the value
    #: cell of a table row whose label wrapped onto two lines.
    side_cells: int = 0

    @property
    def angle(self) -> float:
        return self.lines[0].angle

    @property
    def all_lines(self) -> list[TextLine]:
        return ([self.heading] if self.heading else []) + self.lines

    @property
    def text(self) -> str:
        """The note's text, one printed line per text line, heading excluded."""
        return "\n".join(line.text for line in self.lines)

    @property
    def fragments(self) -> list[Fragment]:
        return [f for line in self.all_lines for f in line.fragments]

    def levelled_bounds(self, include_heading: bool = False) -> tuple[float, float, float, float]:
        lines = self.all_lines if include_heading else self.lines
        return (
            min(line.lx for line in lines),
            min(line.ly for line in lines),
            max(line.right for line in lines),
            max(line.bottom for line in lines),
        )

    def page_box(self, include_heading: bool = False) -> dict[str, float]:
        x0, y0, x1, y1 = self.levelled_bounds(include_heading)
        return levelled_to_page_box(x0, y0, x1 - x0, y1 - y0, self.angle)

    @property
    def confidence(self) -> float:
        frags = self.fragments
        return sum(f.confidence for f in frags) / len(frags) if frags else 0.0


# ── ruling ──────────────────────────────────────────────────────────────────

# A rule is ink along at least this share of the span it is tested over.
# A glyph stroke ("I", "l", "[") stays well under it: the span reaches past
# the glyph box, which a rule crosses and a letter does not.
RULE_COVERAGE = 0.9


class InkMap:
    """
    The page's dark pixels, for asking where the drawing is ruled.

    Geometry alone cannot tell a table from a paragraph: a column of row
    labels stacks and aligns exactly like the lines of a note. The rules
    can. A sentence never has a vertical rule between two of its words, and
    the lines of a paragraph never have a cell border between them.
    """

    def __init__(self, image: Any, threshold: int = 140) -> None:
        import numpy as np

        self._np = np
        self.dark = np.asarray(image.convert("L")) < threshold
        self.height, self.width = self.dark.shape

    def _clip(self, x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
        return (
            max(0, int(math.floor(x0))),
            max(0, int(math.floor(y0))),
            min(self.width, int(math.ceil(x1))),
            min(self.height, int(math.ceil(y1))),
        )

    def vertical_rule(self, x0: float, x1: float, y0: float, y1: float) -> bool:
        """Whether some column in [x0, x1] is inked down the whole of [y0, y1]."""
        cx0, cy0, cx1, cy1 = self._clip(x0 - 1, y0, x1 + 1, y1)
        if cx1 - cx0 < 1 or cy1 - cy0 < 4:
            return False
        band = self.dark[cy0:cy1, cx0:cx1]
        # A scanned rule wanders a pixel; any of three adjacent columns counts.
        wide = band.copy()
        wide[:, 1:] |= band[:, :-1]
        wide[:, :-1] |= band[:, 1:]
        return bool(wide.mean(axis=0).max() >= RULE_COVERAGE)

    def horizontal_rule(
        self, x0: float, x1: float, y0: float, y1: float, margin: float = 0.0
    ) -> bool:
        """
        Whether some row in [y0, y1] is inked across [x0, x1] and on past it.

        With ``margin``, the same row must also be inked for ``margin``
        beyond BOTH ends, each side judged on its own: a rule running past
        the text either way, not an underline that starts where it starts.
        """
        cx0, cy0, cx1, cy1 = self._clip(x0 - margin, y0 - 1, x1 + margin, y1 + 1)
        if cy1 - cy0 < 1 or cx1 - cx0 < 4:
            return False
        band = self.dark[cy0:cy1, cx0:cx1]
        wide = band.copy()
        wide[1:, :] |= band[:-1, :]
        wide[:-1, :] |= band[1:, :]
        inner0 = max(0, int(round(x0 - cx0)))
        inner1 = max(inner0 + 1, int(round(x1 - cx0)))
        covered = wide[:, inner0:inner1].mean(axis=1) >= RULE_COVERAGE
        if margin > 0:
            if inner0 < 1 or inner1 >= wide.shape[1]:
                return False  # the page edge cuts the margin off
            covered &= wide[:, :inner0].mean(axis=1) >= RULE_COVERAGE
            covered &= wide[:, inner1:].mean(axis=1) >= RULE_COVERAGE
        return bool(covered.any())

    def _run_end(self, row: int, start: int, step: int) -> int:
        """Walk a dark row from ``start`` until it ends (gaps of 2px bridged)."""
        x, gap = start, 0
        while 0 <= x + step < self.width and gap <= 2:
            x += step
            band = self.dark[max(0, row - 1):row + 2, x]
            gap = 0 if band.any() else gap + 1
        return x - step * gap

    def _meets_vertical(self, row: int, x: int, reach: int) -> bool:
        """Whether a vertical rule leaves (x, row) upward or downward."""
        x0, x1 = max(0, x - 3), min(self.width, x + 4)
        for y0, y1 in ((row - reach, row - 1), (row + 2, row + reach + 1)):
            y0, y1 = max(0, y0), min(self.height, y1)
            if y1 - y0 < 3:
                continue
            column = self.dark[y0:y1, x0:x1].any(axis=1)
            if column.mean() >= 0.8:
                return True
        return False

    def table_rule(self, x0: float, x1: float, y0: float, y1: float, reach: float) -> bool:
        """
        Whether a row in [y0, y1] is a table rule across [x0, x1].

        A rule inked across the span whose two ends each meet a vertical
        rule: a cell border, ending at the table's corners and T-junctions.
        An underline is inked the same way but its ends are free.
        """
        cx0, cy0, cx1, cy1 = self._clip(x0, y0, x1, y1)
        if cy1 - cy0 < 1 or cx1 - cx0 < 4:
            return False
        band = self.dark[cy0:cy1, cx0:cx1]
        rows = band.mean(axis=1) >= RULE_COVERAGE
        depth = max(4, int(reach))
        for offset in self._np.nonzero(rows)[0]:
            row = cy0 + int(offset)
            left = self._run_end(row, cx0, -1)
            right = self._run_end(row, cx1 - 1, 1)
            if self._meets_vertical(row, left, depth) and self._meets_vertical(row, right, depth):
                return True
        return False


# ── geometry ────────────────────────────────────────────────────────────────


def level_box(
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


def unlevel(x: float, y: float, theta: float) -> tuple[float, float]:
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def levelled_to_page_box(
    x: float, y: float, w: float, h: float, angle: float
) -> dict[str, float]:
    theta = math.radians(angle)
    corners = [unlevel(px, py, theta) for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
    xs = [cx for cx, _ in corners]
    ys = [cy for _, cy in corners]
    return {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)}


def _angle_key(angle: float) -> float:
    return round(angle % 360.0) % 360


# ── lines ───────────────────────────────────────────────────────────────────


def _join(frags: list[Fragment], levelled: list[tuple[float, float, float, float]]) -> str:
    text = frags[0].text.strip()
    for (prev, pbox), (frag, box) in zip(
        zip(frags, levelled), zip(frags[1:], levelled[1:])
    ):
        piece = frag.text.strip()
        if not piece:
            continue
        gap = box[0] - (pbox[0] + pbox[2])
        size = max(min(prev.size, frag.size), 1.0)
        glued = (
            prev.source == frag.source == "pdf-text"
            and gap < GLUED_GAP_EM * size
        )
        text += ("" if glued else " ") + piece
    return text


def build_lines(fragments: Iterable[Fragment], ink: InkMap | None = None) -> list[TextLine]:
    """
    Join fragments that share a baseline into lines, per text angle.

    Banding is by vertical overlap against the SMALLER height — a decimal
    point is a tiny box on the baseline and has to stay with its number — and
    by distance from the band's own centre line, never its running extent:
    measured against the extent, table rows a box-height apart chained into
    one "line" ("MODULE NO.OF SHIFT FACTOR TEETH"). A gap wider than
    :data:`COLUMN_GAP_EM` characters starts a new segment, and so does a
    vertical rule between two fragments when ``ink`` is given: that is a
    second column, a table cell or an unrelated label, not the next word.
    """
    by_angle: dict[float, list[Fragment]] = {}
    for frag in fragments:
        if frag.text.strip():
            by_angle.setdefault(_angle_key(frag.angle), []).append(frag)

    out: list[TextLine] = []
    for angle, frags in sorted(by_angle.items()):
        theta = math.radians(angle)
        boxes = {id(f): level_box(f.x, f.y, f.w, f.h, theta) for f in frags}
        bands: list[list[Fragment]] = []
        for frag in sorted(frags, key=lambda f: (boxes[id(f)][1], boxes[id(f)][0])):
            _, fy, _, fh = boxes[id(frag)]
            best, best_offset = None, math.inf
            for band in bands:
                centres = [boxes[id(b)][1] + boxes[id(b)][3] / 2.0 for b in band]
                heights = sorted(boxes[id(b)][3] for b in band)
                band_mid = sum(centres) / len(centres)
                band_h = heights[len(heights) // 2]
                overlap = min(band_mid + band_h / 2.0, fy + fh) - max(band_mid - band_h / 2.0, fy)
                if overlap < 0.5 * min(band_h, fh):
                    continue
                offset = abs((fy + fh / 2.0) - band_mid)
                tolerance = 0.45 * max(min(band_h, fh), 0.5 * max(band_h, fh))
                if offset > tolerance:
                    continue
                if offset < best_offset:
                    best, best_offset = band, offset
            if best is None:
                bands.append([frag])
            else:
                best.append(frag)

        for band in bands:
            band.sort(key=lambda f: boxes[id(f)][0])
            segments: list[list[Fragment]] = [[band[0]]]
            for frag in band[1:]:
                prev = segments[-1]
                reach = max(boxes[id(p)][0] + boxes[id(p)][2] for p in prev)
                size = max(min(prev[-1].size, frag.size), 1.0)
                if boxes[id(frag)][0] - reach > COLUMN_GAP_EM * size or (
                    ink is not None and angle == 0 and _ruled_apart(ink, prev[-1], frag)
                ):
                    segments.append([frag])
                else:
                    prev.append(frag)
            spans = [
                (
                    min(boxes[id(f)][0] for f in seg),
                    max(boxes[id(f)][0] + boxes[id(f)][2] for f in seg),
                    " ".join(f.text.strip() for f in seg),
                )
                for seg in segments
            ]
            for index, seg in enumerate(segments):
                levelled = [boxes[id(f)] for f in seg]
                x0 = min(b[0] for b in levelled)
                y0 = min(b[1] for b in levelled)
                x1 = max(b[0] + b[2] for b in levelled)
                y1 = max(b[1] + b[3] for b in levelled)
                sizes = sorted(f.size for f in seg)
                out.append(
                    TextLine(
                        fragments=seg,
                        text=_join(seg, levelled),
                        angle=angle,
                        lx=x0,
                        ly=y0,
                        lw=x1 - x0,
                        lh=y1 - y0,
                        size=sizes[len(sizes) // 2],
                        neighbours=[s for i, s in enumerate(spans) if i != index],
                    )
                )
    return out


def _ruled_apart(ink: InkMap, left: Fragment, right: Fragment) -> bool:
    """A vertical rule between two fragments on one line: they are table cells."""
    # Detector boxes are padded, so two cells' boxes can overlap the rule
    # between them; the rule is looked for across the overlap as well.
    # The search also reaches a little past both box edges: a box can stop
    # two pixels short of the rule it sits against (TS2's bottom strip).
    gap0, gap1 = sorted((left.x + left.w, right.x))
    top = min(left.y, right.y)
    bottom = max(left.y + left.h, right.y + right.h)
    pad = 0.1 * (bottom - top)
    reach = max(3.0, 0.15 * (bottom - top))
    return ink.vertical_rule(gap0 - reach, gap1 + reach, top - pad, bottom + pad)


# ── blocks ──────────────────────────────────────────────────────────────────


def _letters(text: str) -> int:
    return sum(c.isalpha() for c in text)


def _prose_words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text) if w.upper() not in _QUALIFIERS]


def is_value_line(text: str) -> bool:
    """A line that is a measurement, not words: "R0.2 +0.3", "0.2 MAX"."""
    return any(c.isdigit() for c in text) and not _prose_words(text)


def is_heading(line: TextLine) -> bool:
    return bool(NOTES_HEADING_RE.match(line.text))


def _aligned(prev: TextLine, line: TextLine, block: list[TextLine], size: float) -> bool:
    tol = ALIGN_EM * size
    left = min(l.lx for l in block)
    if abs(line.lx - prev.lx) <= tol or abs(line.lx - left) <= tol:
        return True
    if abs(line.centre - prev.centre) <= tol or abs(line.right - prev.right) <= tol:
        return True
    # A wrapped line under a numbered point: indented past the marker, never
    # further left than the block itself.
    numbered = any(NOTE_MARKER_RE.match(l.text) for l in block)
    if numbered and left - tol <= line.lx <= left + HANGING_INDENT_EM * size:
        return True
    return False


def _ruled_between(ink: InkMap, prev: TextLine, line: TextLine, size: float) -> bool:
    """
    A cell border between two stacked lines: they are rows, not one paragraph.

    The rule must end at vertical rules at both ends. An underline (a
    Japanese notes block underlines every line) is inked just the same,
    but its ends are free; a table rule runs into the cell's borders.
    """
    x0 = max(prev.lx, line.lx)
    x1 = min(prev.right, line.right)
    if x1 - x0 < 4:
        return False
    y0 = prev.bottom - 0.15 * size
    y1 = line.ly + 0.15 * size
    return ink.table_rule(x0, x1, y0, y1, reach=0.6 * size)


def _continues(
    prev: TextLine, line: TextLine, block: list[TextLine], ink: InkMap | None = None
) -> float | None:
    """The pitch at which ``line`` continues the paragraph ending in ``prev``, or None."""
    small, large = sorted((prev.size, line.size))
    # OCR "size" is a box height, which moves with ascenders and descenders
    # ("Remove burrs" against "Roughness"); a font size does not.
    ocr_only = all(f.source == "ocr" for f in prev.fragments + line.fragments)
    ratio = OCR_SAME_SIZE_RATIO if ocr_only else SAME_SIZE_RATIO
    if large > ratio * max(small, 0.1):
        return None
    size = max(small, 1.0)
    pitch = line.ly - prev.ly
    gap = line.ly - prev.bottom
    if pitch <= 0.5 * size or pitch > MAX_LINE_PITCH_EM * size or gap < -0.35 * size:
        return None
    # Lines that never share a column are not one paragraph, however close.
    if min(prev.right, line.right) - max(prev.lx, line.lx) <= 0:
        return None
    if not _aligned(prev, line, block, size):
        return None
    # A value printed under a note is not its next line. A continuation is
    # words, unless the sentence above has not finished yet.
    if len(_prose_words(line.text)) < 2 and prev.text.rstrip().endswith("."):
        return None
    if _letters(line.text) < 2:
        return None
    # Nor does a paragraph open with a value: a callout printed above a note
    # ("R0.2 +0.3" over "All around the teeth profile") would be swallowed
    # by it, and the balloon lost with it.
    if len(block) == 1 and is_value_line(prev.text):
        return None
    if ink is not None and line.angle == 0 and _ruled_between(ink, prev, line, size):
        return None
    return pitch


def build_blocks(lines: Iterable[TextLine], ink: InkMap | None = None) -> list[TextBlock]:
    """
    Stack lines into paragraphs.

    Lines join when they are set in the same size, sit within one leading of
    each other, overlap horizontally and share an alignment (left, centre,
    right, or the hanging indent of a numbered list). Each line joins the
    paragraph it continues most closely, so two notes side by side stay two
    blocks. A "NOTES" heading attaches to the paragraph directly under it.
    """
    by_angle: dict[float, list[TextLine]] = {}
    for line in lines:
        by_angle.setdefault(line.angle, []).append(line)

    out: list[TextBlock] = []
    for _angle, here in sorted(by_angle.items()):
        blocks: list[TextBlock] = []
        for line in sorted(here, key=lambda l: (l.ly, l.lx)):
            if is_heading(line):
                blocks.append(TextBlock(lines=[], heading=line))
                continue
            best: TextBlock | None = None
            best_pitch = math.inf
            for block in blocks:
                if not block.lines:
                    continue
                prev = block.lines[-1]
                pitch = _continues(prev, line, block.lines, ink)
                if pitch is not None and pitch < best_pitch:
                    best, best_pitch = block, pitch
            if best is not None:
                best.lines.append(line)
                continue
            # A paragraph directly under a heading belongs to it.
            heading_block = next(
                (
                    b
                    for b in blocks
                    if b.heading is not None
                    and not b.lines
                    and _heading_owns(b.heading, line)
                ),
                None,
            )
            if heading_block is not None:
                heading_block.lines.append(line)
            else:
                blocks.append(TextBlock(lines=[line]))
        for block in blocks:
            if block.lines:
                out.append(block)
            elif block.heading is not None:
                # A heading with nothing under it is still text on the page.
                out.append(TextBlock(lines=[block.heading], category="label"))
    return out


def _heading_owns(heading: TextLine, line: TextLine) -> bool:
    size = max(min(heading.size, line.size), 1.0)
    if max(heading.size, line.size) > HEADING_SIZE_RATIO * size:
        return False
    gap = line.ly - heading.bottom
    if not -0.35 * size <= gap <= 1.6 * size:
        return False
    return heading.lx - HANGING_INDENT_EM * size <= line.lx <= heading.right


# ── classification ──────────────────────────────────────────────────────────


def _sentence_line(text: str) -> bool:
    """A line written as a sentence: four or more words, two of them function words."""
    tokens = [t.strip(".,;:()").upper() for t in text.split()]
    function = {t for t in tokens if t in _FUNCTION_WORDS}
    return len(_prose_words(text)) >= 4 and len(function) >= 2


def _reads_as_sentence(text: str) -> bool:
    """
    Whether unnumbered text is written as a sentence rather than as labels.

    A column of row labels ("PRESSURE ANGLE / PITCH CIRCLE DIA (d) / …") or
    an address stacks exactly like a paragraph; what it lacks is the small
    words a sentence is built from. Text that is not Latin script (a
    Japanese note) is not judged this way.
    """
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if ord(c) >= 0x2E80) >= 0.3 * len(letters):
        return True
    tokens = {t.strip(".,;:()*\"'").upper() for t in text.split()}
    return len(tokens & _FUNCTION_WORDS) >= 2


def _revision_like(block: TextBlock) -> bool:
    text = " ".join(line.text for line in block.lines)
    if _ECN_RE.search(text):
        return True
    return bool(_REVISION_VERB_RE.search(text)) and any(line.neighbours for line in block.lines)


def _table_like(block: TextBlock) -> bool:
    """Most lines have a cell neighbour on their own baseline: a table, not prose."""
    # A wrapped row label ("UTILIZABLE LENGTH OF / PATH OF CONTACT") has its
    # value centred beside it, on neither line's own baseline.
    if block.side_cells and len(block.lines) <= 3:
        return True
    if len(block.lines) < 2:
        return False
    with_cells = 0
    for line in block.lines:
        reach = TABLE_NEIGHBOUR_EM * max(line.size, 1.0)
        # A table cell beside a row label holds a value or a word or two; a
        # sentence beside a line is a second note in the next column.
        if any(
            (0 <= left - line.right <= reach or 0 <= line.lx - right <= reach)
            and (is_value_line(text) or len(text.split()) <= 2)
            for left, right, text in line.neighbours
        ):
            with_cells += 1
    return with_cells >= 0.5 * len(block.lines)


def _scrappy(text: str) -> bool:
    """A misread: most tokens are single glyphs ("drain ha esi t 7 C a n S m")."""
    tokens = text.split()
    single = sum(1 for t in tokens if len(t) == 1 and not t.isdigit())
    return len(tokens) >= 4 and single > 0.3 * len(tokens)


def classify_block(block: TextBlock) -> TextBlock:
    """
    Label a paragraph ``note``, ``title_block``, ``table`` or ``label``.

    Nothing is discarded here: the label decides whether a caller publishes
    the block as a note, and the others are kept for audit. ``reason`` says
    which evidence decided it.
    """
    text = " ".join(line.text for line in block.lines)
    words = _prose_words(text)
    compact = text.replace(" ", "")
    digits = sum(c.isdigit() for c in compact)
    letters = _letters(compact)
    numbered = [line for line in block.lines if NOTE_MARKER_RE.match(line.text)]

    if _LEGAL_RE.search(text):
        block.category, block.reason = "title_block", "proprietary notice"
    elif not numbered and block.heading is None and (
        _TITLE_FIELD_RE.search(text) or _COMPANY_RE.search(text)
    ):
        block.category, block.reason = "title_block", "title-block field or company name"
    elif _revision_like(block):
        block.category, block.reason = "revision", "revision-table entry"
    elif _table_like(block):
        # A note printed in a table cell ("ADDENDUM SHALL NOT RISE SHARPLY",
        # "ALL DIMENSIONS ARE IN M.M.") still reads as a sentence; the row
        # labels around it do not.
        if any(_sentence_line(line.text) for line in block.lines):
            block.category, block.reason = "note", "sentence in a ruled cell"
        else:
            block.category, block.reason = "table", "lines have cell neighbours"
    elif block.heading is not None and letters >= 4:
        block.category, block.reason = "note", "under a NOTES heading"
    elif numbered and len(words) >= 3:
        block.category, block.reason = "note", "numbered note"
    elif _scrappy(text):
        block.category, block.reason = "label", "mostly single-glyph scraps"
    elif (
        len(words) >= NOTE_MIN_WORDS
        # A wrapped paragraph carries several words a line; a column of
        # one-word labels ("Root Diameter", "Tip Diameter") does not.
        and len(words) >= NOTE_MIN_WORDS_PER_LINE * len(block.lines)
        and letters >= 0.6 * max(len(compact), 1)
        and digits <= NOTE_MAX_DIGIT_FRACTION * max(len(compact), 1)
        and not all(_FIELD_LABEL_RE.search(line.text) for line in block.lines)
        and _reads_as_sentence(text)
    ):
        block.category, block.reason = "note", "prose paragraph"
    else:
        block.category, block.reason = "label", "too little prose for a note"
    return block


def _count_side_cells(block: TextBlock, lines: list[TextLine]) -> int:
    """Short value-like lines level with the block and just beside it."""
    x0, y0, x1, y1 = block.levelled_bounds()
    size = max(line.size for line in block.lines)
    reach = TABLE_NEIGHBOUR_EM * size
    own = {id(line) for line in block.all_lines}
    count = 0
    for line in lines:
        if id(line) in own or line.angle != block.angle:
            continue
        centre = line.ly + line.lh / 2
        if not y0 <= centre <= y1:
            continue
        if not (0 <= line.lx - x1 <= reach or 0 <= x0 - line.right <= reach):
            continue
        if is_value_line(line.text) or len(line.text.split()) <= 2:
            count += 1
    return count


def note_blocks(fragments: Iterable[Fragment], ink: InkMap | None = None) -> list[TextBlock]:
    """Every paragraph on the page, classified, in reading order."""
    lines = build_lines(fragments, ink)
    built = build_blocks(lines, ink)
    for block in built:
        if len(block.lines) >= 2:
            block.side_cells = _count_side_cells(block, lines)
    blocks = [classify_block(b) for b in built]
    return sorted(blocks, key=lambda b: (round(b.page_box()["y"]), b.page_box()["x"]))


def block_lines_payload(block: TextBlock, digits: int = 1) -> list[dict[str, Any]]:
    """The constituent lines of a block, so its grouping stays auditable."""
    return [
        {
            "text": line.text,
            "bbox": {k: round(v, digits) for k, v in line.page_box().items()},
            "heading": line is block.heading,
        }
        for line in block.all_lines
    ]


# ── normalisation and reading quality ───────────────────────────────────────

_COMMA_MARKER_RE = re.compile(r"^(\s*\d{1,2})\s*,\s*(?=[^\W\d_])")
_BARE_MARKER_RE = re.compile(r"^(\s*\d{1,2})\s+(?=[^\W\d_]{2,})")
_PUNCTUATED_MARKER_RE = re.compile(r"^\s*\d{1,2}\s*[.)]")


def normalized_note_text(block: TextBlock) -> str:
    """
    The note's text with its point markers written one way.

    The client splits a notes block into points on "N." at a line start.
    A list that writes one marker as "3," (a scanned dot) or drops the dot
    altogether ("2 PART TO BE DRY") would have that point glued onto the one
    above it, so inside a list that already numbers with "N." those markers
    are written the same way. The raw text is kept beside this unchanged.
    """
    lines = [line.text for line in block.lines]
    lines = [_COMMA_MARKER_RE.sub(r"\1. ", line) for line in lines]
    if any(_PUNCTUATED_MARKER_RE.match(line) for line in lines):
        lines = [_BARE_MARKER_RE.sub(r"\1. ", line) for line in lines]
    return "\n".join(re.sub(r"\s+", " ", line).strip() for line in lines)


def reading_quality(lines: Iterable[TextLine]) -> float:
    """
    How much a reading of a paragraph can be believed.

    Letters weighted by the recogniser's confidence, squared so a confident
    read of a line outweighs a longer, garbled one ("OFTTEAD" at 0.86 against
    "OF THE HEAD TO THE LAST THREAD" at 0.98).
    """
    total = 0.0
    for line in lines:
        conf = sum(f.confidence for f in line.fragments) / max(len(line.fragments), 1)
        total += _letters(line.text) * conf * conf
    return total
