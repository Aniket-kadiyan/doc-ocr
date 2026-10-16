"""Find where a reference table's point names are printed in the views.

Once the coordinate table has been read, every one of its point names — ``a``,
``b``, ``c`` … — appears somewhere in the drawing as a bare glyph on its own
leader line, often several times because the same point is called out in more
than one view.  Those glyph positions are what gets ballooned.

Recognition is the wrong tool here and was measured to be: run over tiles of
the sheet, PaddleOCR found roughly half the markers on the benchmark drawing
and invented a handful that were not there.  A lone 10-pixel letter carries
almost no context, which is exactly what a text recogniser relies on.

Template matching does have the context, because the drawing supplies it.  The
table prints the very same names in the very same font, so its name cells are
cut out and used as the templates, and the digits in its coordinate cells are
cut out as *counter*-templates.  A candidate glyph in the views is accepted
only when it looks more like one of the point names than like any digit, by a
clear margin.  Everything compared comes from this one sheet, so the matcher
never has to generalise across fonts, scanners or line weights.

Scale is handled by normalising both sides into the same box: the table prints
its names larger than the views do, typically two to three times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from page_layout import LayoutBox


# Every glyph is compared inside this box. Large enough to keep the strokes
# that separate b/h and c/e apart, small enough that matching stays cheap.
GLYPH_NORM = 28

# Standard deviation, in normalised-patch pixels, of the blur applied to both
# sides before they are correlated.
#
# CAD lettering is a hairline: at the resolution this runs at, a point name is
# one or two pixels of stroke with nothing either side of it. A mean-centred
# cosine over two such patches measures stroke *registration*, not letter
# shape — offset the same glyph by one pixel and the two vectors are close to
# orthogonal. On the benchmark tube drawing the correct identity scored 0.20
# for 'k', 0.22 for 'a' and 0.42 for 'g', while 'd' beat its own template's
# rival by 0.02, so eight of the nine printed names failed the acceptance bar
# and three were found. Softening both sides first restores the comparison the
# scores were meant to express: the same eight glyphs score 0.56 to 0.94 and
# every one of them picks its own identity.
#
# Softening this far and no further. The useful band on that sheet runs from
# about 0.8 to 1.5; at 1.6 a false positive appears, and by 1.8 the blur has
# taken the shape out of the glyphs altogether — every thin upright on the
# page matches 'f' or 'k' and the sheet comes back with forty-one markers,
# none of them real.
GLYPH_BLUR = 1.2

# A marker is a smaller print of the same name: the views label points at the
# sheet's note size while the table prints them at its own. Outside this band
# relative to the table's glyphs the candidate is something else entirely.
MARKER_MIN_SCALE = 0.25
MARKER_MAX_SCALE = 1.35

# A printed letter fills a modest part of its bounding box. A solid arrowhead
# or dot fills nearly all of it, a rule or leader line almost none.
MIN_FILL_RATIO = 0.10
MAX_FILL_RATIO = 0.78

# Letters are roughly as tall as they are wide, give or take a descender.
MIN_ASPECT = 0.22
MAX_ASPECT = 2.0

# A marker printed inside dense artwork — over hatching, between the walls of
# a sectioned part — cannot be read reliably whatever it scores. Foreign ink
# in a margin this wide around the glyph, measured against the glyph's own
# ink, is the test. Generous on purpose: a marker normally has its own leader
# line a few pixels away, and that is not a reason to drop it.
ISOLATION_MARGIN_RATIO = 0.5
ISOLATION_MAX_FOREIGN = 1.5

# A marker is a lone glyph; a word or a dimension is a row of them. Whitespace
# alone does not separate the two, because drawing offices letter-space their
# headings — "G u a r a n t e e d  M e c h a n i c a l  P r o p e r t i e s"
# passes any isolation test and then offers one convincing 'a' and 'e' after
# another.
#
# What does separate them is how much company there is along the line. A line
# of type puts a run of same-size glyphs on one foot; a marker has at most a
# leader line and a stray tick beside it. The count is taken over a window
# rather than from the immediate neighbours, which is what makes the test hold
# at a word boundary: the last letter of a letter-spaced word has nothing
# within a letter's width of it on one side.
#
# Alignment is measured at the foot of the glyph, not its centre, because that
# is what a line of type actually shares. Comparing centres missed an 'e' set
# between an ascender and a descender, which is most of a line of lower case.
#
# The test stays blind to shape on purpose: a drawing is full of short strokes
# that would each look like an 'l' or an 'i', and excluding them individually
# was measured to take the thin letters out of "Material" with them, leaving
# its 'a' looking like a lone callout.
TEXT_LINE_SPAN_RATIO = 9.0
TEXT_LINE_MIN_COMPANY = 2
TEXT_RUN_FOOT_ABOVE = 0.3
TEXT_RUN_FOOT_BELOW = 0.45
TEXT_RUN_HEIGHT_MIN = 0.45
TEXT_RUN_HEIGHT_MAX = 2.2

# Correlation against the best-matching template, and how far that must beat
# the best template of any other identity (including the digits). Both were
# set from the benchmark sheet; the margin is what rejects digits that happen
# to resemble a letter at this size.
#
# The bar moved up with GLYPH_BLUR, which lifts every score: on the benchmark
# tube drawing a correct marker now reads 0.74 to 0.97 while the best
# impostor — the "0" of a scale bar's "10" — reads 0.62.
MIN_MATCH_SCORE = 0.70
MIN_MATCH_MARGIN = 0.06

# What the first pass must reach before a marker is trusted enough to become a
# template for the second. Well above the acceptance bar, because one wrong
# exemplar here teaches the second pass to repeat the mistake.
CONFIDENT_SCORE = 0.75
CONFIDENT_MARGIN = 0.18

# How far, in normalised-patch pixels, a template may be slid over a
# candidate before they are scored. See :func:`_correlate`.
SHIFT_SEARCH = 2

# How far a match may stand from the height the sheet's confident matches
# agree on, and how many of those it takes before that agreement is trusted.
# See :func:`_size_agreeing`.
SIZE_CONSENSUS_MIN_RATIO = 0.65
SIZE_CONSENSUS_MAX_RATIO = 1.45
SIZE_CONSENSUS_MIN_MARKERS = 3

# Guard against a pathological page: a drawing covered in small marks would
# otherwise spend minutes in the comparison loop.
MAX_CANDIDATES = 6000

# Label used for every counter-template. Not a point name, so a candidate that
# matches one is rejected.
NOT_A_NAME = "#"


@dataclass
class GlyphTemplate:
    """One exemplar cut from the sheet, kept at the size it was printed.

    The pixels are held rather than a finished patch because the comparison
    is made at the *candidate's* size: see :func:`_normalise`.
    """

    label: str
    gray: np.ndarray
    mask: np.ndarray
    _patches: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def is_name(self) -> bool:
        return self.label != NOT_A_NAME

    def patch(self, height: int) -> np.ndarray:
        """This glyph as it would look printed `height` pixels tall."""

        cached = self._patches.get(height)
        if cached is None:
            cached = _normalise(self.gray, self.mask, height)
            self._patches[height] = cached
        return cached


@dataclass(frozen=True)
class SymbolMarker:
    """One place a point name is printed in the views."""

    symbol: str
    bbox: LayoutBox
    score: float
    margin: float


@dataclass(frozen=True)
class _Candidate:
    """A lone glyph in the views, normalised and waiting to be identified."""

    box: LayoutBox
    gray: np.ndarray
    mask: np.ndarray
    patch: np.ndarray


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binarise dark-on-white artwork, keeping ink that is merely grey.

    Otsu alone is the wrong test for a drawing, and was measured to be: it
    assumes two comparable populations and splits the difference between them,
    so on a sheet that is 99% paper and 1% black linework it parks the
    threshold near mid-grey. Anything printed lighter than that — and point
    names are routinely set in a grey half the weight of the artwork they sit
    on — falls on the paper side and never becomes a candidate at all. On the
    synthetic sheet, markers at grey 130 lost two of three names and markers
    at grey 170 lost all three, before a single correlation was computed.

    Triangle is the test for this histogram shape: one dominant peak with a
    tail, with the threshold set out at the end of the tail. It keeps the grey
    markers, and because it follows the paper distribution rather than a fixed
    level it tightens by itself on a noisy scan instead of flooding — ink came
    back at 0.8–1.0% of the page across the noise range, against Otsu's 0.7%.

    The two are combined rather than swapped. Taking whichever threshold
    admits more ink makes this strictly more permissive than Otsu on its own,
    so a marker that is found today cannot be lost to an unusual histogram
    where Triangle lands lower — an inverted or very dark scan, say.
    """

    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    triangle, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE
    )
    _, mask = cv2.threshold(
        gray,
        max(otsu, triangle),
        255,
        cv2.THRESH_BINARY_INV,
    )
    return mask


def _normalise(
    gray: np.ndarray,
    mask: np.ndarray,
    target_height: int | None = None,
) -> np.ndarray:
    """Ink-weighted square patch, aspect preserved and centred.

    Ink *intensity* rather than the bare binary mask: at marker size a stroke
    is one or two pixels wide and partly anti-aliased, so thresholding alone
    throws away most of what distinguishes one letter from another. The mask
    is still applied, to keep a neighbouring stroke clipped by the bounding
    box out of the template.

    ``target_height`` reprints the glyph at that many pixels before anything
    else happens, and the table's templates are always compared at the height
    of the candidate in hand. Scaling a crisp 60-pixel 'b' straight into the
    28-pixel comparison box and a blurred 18-pixel marker into the same box
    compares two different things: the template keeps a gap between bowl and
    stem that the marker physically cannot have. Reprinting the template small
    first costs that same detail, and it is what separates b from h and c from
    e at the size the views actually use.
    """

    if target_height is not None and target_height != mask.shape[0]:
        scale = target_height / max(mask.shape[0], 1)
        size = (
            max(1, int(round(mask.shape[1] * scale))),
            max(1, target_height),
        )
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)

    height, width = mask.shape
    side = max(height, width, 1)
    canvas = np.zeros((side, side), np.float32)
    top = (side - height) // 2
    left = (side - width) // 2
    ink = (255.0 - gray.astype(np.float32)) * (mask > 0)
    canvas[top : top + height, left : left + width] = ink
    patch = cv2.resize(
        canvas,
        (GLYPH_NORM, GLYPH_NORM),
        interpolation=cv2.INTER_AREA,
    )
    peak = float(patch.max())
    if peak > 0:
        patch /= peak
    # See GLYPH_BLUR: a hairline stroke has to be given some width before two
    # prints of it can be compared by correlation at all.
    if GLYPH_BLUR > 0:
        patch = cv2.GaussianBlur(patch, (0, 0), GLYPH_BLUR)
    centred = patch - patch.mean()
    norm = float(np.linalg.norm(centred))
    scaled = centred / norm if norm > 0 else centred
    return np.ascontiguousarray(scaled, dtype=np.float32)


def _correlate(candidate: np.ndarray, template: np.ndarray) -> float:
    """Best correlation between a glyph and a template over small shifts.

    Two prints of the same letter never land on the same pixel grid. The
    candidate's bounding box comes from a thresholded component and the
    template's from a table cell, and each one's idea of where the glyph
    starts is off by up to a pixel; normalising both into the same box turns
    that into a sub-pixel offset that no amount of care removes. A fixed
    comparison then charges the glyph for it, which at this stroke width is
    most of the score.

    So the template is slid over the candidate and the best position wins.
    What that buys is not recall for its own sake but *stability*: before
    this, re-rendering the benchmark sheet at 10.0 instead of 9.95 — a half
    percent — moved the sheet from twelve correct markers to ten and swapped
    two of the identities. With the shift allowed, 9.5, 9.95 and 10.0 all
    return the same reading of the drawing.

    ``SHIFT_SEARCH`` of two normalised pixels is where the gain levels off;
    at three the same sheets start admitting false positives instead.
    """

    padded = cv2.copyMakeBorder(
        candidate,
        SHIFT_SEARCH,
        SHIFT_SEARCH,
        SHIFT_SEARCH,
        SHIFT_SEARCH,
        cv2.BORDER_CONSTANT,
        value=0.0,
    )
    scores = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
    return float(scores.max())


def _classify(
    patch: np.ndarray,
    height: int,
    templates: Sequence[GlyphTemplate],
    *,
    itself: np.ndarray | None = None,
) -> tuple[str, float, float]:
    """Best identity for a glyph, its score, and its lead over the next one.

    ``itself`` is the candidate's own pixels. A glyph the first pass was sure
    of becomes a template for the second, and is then in this list: scoring it
    against itself returns 1.0 and says nothing about anything.
    """

    # Best score per identity, so the runner-up is a genuinely different
    # reading and not the same letter's second exemplar.
    per_label: dict[str, float] = {}
    for template in templates:
        if itself is not None and template.gray is itself:
            continue
        score = _correlate(patch, template.patch(height))
        if score > per_label.get(template.label, -1.0):
            per_label[template.label] = score
    if not per_label:
        return "", 0.0, 0.0
    best_label = max(per_label, key=lambda label: per_label[label])
    best_score = per_label[best_label]
    rival = max(
        (score for label, score in per_label.items() if label != best_label),
        default=0.0,
    )
    return best_label, best_score, best_score - max(rival, 0.0)


# A stroke this small a fraction of the name cell's largest one is a speck of
# scanner dirt or the stub of a rule the inset failed to cut away, not part of
# the glyph. Low on purpose: the arms of a 'k' and the dot of an 'i' are a
# modest share of the stem they belong to.
INTERIOR_STROKE_MIN_AREA = 0.05


def _interior_glyph(
    gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Cut the printed glyph out of a table cell.

    The cell crop still carries slivers of the rules that bound it, and those
    are wider than anything printed inside, so the largest component is not
    safe to take. Components touching the crop edge are dropped first.

    Every surviving stroke is then kept, not merely the biggest one. A letter
    is not always one connected component: in the font on the benchmark tube
    drawing a 'k' is a stem with its arms drawn separately, so taking only the
    largest left a template that was the bare "<" — and 'k' was then unfindable
    whatever the views printed. The same would hollow out an 'i', and a
    two-character name like "P1" is two components by construction.
    """

    mask = _ink_mask(gray)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    interior = [
        index
        for index in range(1, count)
        for x, y, w, h, _ in (stats[index],)
        if x > 0 and y > 0 and x + w < width and y + h < height
    ]
    if not interior:
        return None
    largest = max(stats[index][4] for index in interior)
    strokes = [
        index
        for index in interior
        if stats[index][4] >= largest * INTERIOR_STROKE_MIN_AREA
    ]
    x = min(stats[index][0] for index in strokes)
    y = min(stats[index][1] for index in strokes)
    x1 = max(stats[index][0] + stats[index][2] for index in strokes)
    y1 = max(stats[index][1] + stats[index][3] for index in strokes)
    component = np.isin(labels[y:y1, x:x1], strokes).astype(np.uint8)
    return gray[y:y1, x:x1], component


def build_templates(
    image: Image.Image,
    name_cells: Mapping[str, LayoutBox],
    counter_cells: Sequence[LayoutBox],
) -> list[GlyphTemplate]:
    """Cut point-name and counter templates out of the reference table.

    ``counter_cells`` are the table's coordinate cells. Each glyph in them is
    a digit, a sign or a decimal point — the characters a marker must *not* be
    confused with, printed in the same font at the same size as the names they
    sit beside, which makes them the fairest possible negative evidence.
    """

    page = np.asarray(image.convert("L"))
    templates: list[GlyphTemplate] = []

    for symbol, box in name_cells.items():
        inset = _inset(box)
        cut = _interior_glyph(page[inset.y : inset.y1, inset.x : inset.x1])
        if cut is None:
            continue
        gray, component = cut
        templates.append(
            GlyphTemplate(label=symbol, gray=gray, mask=component)
        )

    name_height = _median_height(templates)
    for box in counter_cells:
        inset = _inset(box)
        cell = page[inset.y : inset.y1, inset.x : inset.x1]
        mask = _ink_mask(cell)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cell_height, cell_width = mask.shape
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if x <= 0 or y <= 0 or x + w >= cell_width or y + h >= cell_height:
                continue
            # A digit stands about as tall as the names beside it; anything
            # much shorter is the decimal point, which is not a glyph worth
            # comparing against and matches almost everything round.
            if h < name_height * 0.55 or area < 8:
                continue
            templates.append(
                GlyphTemplate(
                    label=NOT_A_NAME,
                    gray=cell[y : y + h, x : x + w],
                    mask=(labels[y : y + h, x : x + w] == index).astype(np.uint8),
                )
            )

    return templates


def _inset(box: LayoutBox) -> LayoutBox:
    """Pull a cell in off its own rules before anything is cut out of it."""

    margin = max(1, int(round(min(box.width, box.height) * 0.12)))
    return LayoutBox(
        box.x + margin,
        box.y + margin,
        max(1, box.width - margin * 2),
        max(1, box.height - margin * 2),
    )


def _median_height(templates: Sequence[GlyphTemplate]) -> float:
    heights = [template.height for template in templates if template.is_name]
    return float(np.median(heights)) if heights else 0.0


def _page_glyphs(
    stats: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Every component on the page that reads as a printed character.

    The baseline test needs the glyphs a candidate sits *between*, and those
    are usually characters the point names do not include — the digits of a
    dimension, the letters of a heading.  Leader lines and hatching have to
    stay out of it, or a marker with a leader beside it looks like a word.
    """

    rows: list[np.ndarray] = []
    for index in range(1, stats.shape[0]):
        _, _, w, h, _ = stats[index]
        if not minimum * TEXT_RUN_HEIGHT_MIN <= h <= maximum * TEXT_RUN_HEIGHT_MAX:
            continue
        # Width is the only other condition, so that the narrow letters a
        # marker hides between still count. Requiring a letter-like aspect
        # here was measured to drop the 'i' and 'l' out of "Material" and
        # leave its 'a' looking like a lone callout.
        if w > h * MAX_ASPECT:
            continue
        rows.append(stats[index, :4])
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def locate_markers(
    image: Image.Image,
    templates: Sequence[GlyphTemplate],
    *,
    excluded: Sequence[LayoutBox] = (),
) -> list[SymbolMarker]:
    """Find every printing of a point name outside the sheet's ruled tables.

    ``excluded`` is every table found on the page, not only the reference
    table: a name in the revision block or the parts list is that table's
    text, not a callout on the part.

    Matching runs twice.  The table's templates are two to three times the
    size the views print at, and that gap is what makes ``c`` and ``e``, or
    ``b`` and ``h``, hard to separate at marker size.  So the first pass keeps
    only the markers it is sure of, and the second pass re-reads everything
    with those markers added as templates — exemplars now at exactly the size,
    weight and resolution of the thing being matched.  Glyphs rejected as part
    of a word or a dimension are added as counter-templates in the same step,
    for the same reason.
    """

    if not any(template.is_name for template in templates):
        return []
    name_height = _median_height(templates)
    if name_height <= 0:
        return []

    page = np.asarray(image.convert("L"))
    page_height, page_width = page.shape
    mask = _ink_mask(page)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    minimum = max(3.0, name_height * MARKER_MIN_SCALE)
    maximum = name_height * MARKER_MAX_SCALE
    glyphs = _page_glyphs(stats, minimum=minimum, maximum=maximum)

    candidates: list[_Candidate] = []

    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not minimum <= h <= maximum:
            continue
        if not MIN_ASPECT <= w / max(h, 1) <= MAX_ASPECT:
            continue
        if not MIN_FILL_RATIO <= area / float(max(w * h, 1)) <= MAX_FILL_RATIO:
            continue
        if any(_inside(x, y, w, h, box) for box in excluded):
            continue

        margin = max(2, int(round(h * ISOLATION_MARGIN_RATIO)))
        window = labels[
            max(0, y - margin) : min(page_height, y + h + margin),
            max(0, x - margin) : min(page_width, x + w + margin),
        ]
        foreign = int(np.count_nonzero((window != 0) & (window != index)))
        glyph = page[y : y + h, x : x + w]
        component = (labels[y : y + h, x : x + w] == index).astype(np.uint8)
        patch = _normalise(glyph, component)

        # Flanked on both sides, so it is a character of a word or of a
        # dimension rather than a callout. Tempting as it is to keep these as
        # counter-templates at marker size, they are the wrong evidence: the
        # 'd' in "DETAIL" really is a d, and teaching the matcher otherwise
        # was measured to reject most of the genuine markers on this sheet.
        if _inside_a_word(stats, index, neighbours=glyphs):
            continue
        if foreign > area * ISOLATION_MAX_FOREIGN:
            continue
        candidates.append(
            _Candidate(
                LayoutBox(int(x), int(y), int(w), int(h)),
                glyph,
                component,
                patch,
            )
        )
        if len(candidates) >= MAX_CANDIDATES:
            break

    confident: list[tuple[str, _Candidate]] = []
    for candidate in candidates:
        label, score, gap = _classify(
            candidate.patch, candidate.box.height, templates
        )
        if (
            label != NOT_A_NAME
            and score >= CONFIDENT_SCORE
            and gap >= CONFIDENT_MARGIN
        ):
            confident.append((label, candidate))

    # Nothing is promoted to a template until it agrees with the size the
    # other confident matches were printed at. The second pass compounds
    # whatever it is given: on one render of the benchmark sheet two
    # six-pixel ticks scraped over the confidence bar as 'f', and every other
    # tick on the page then matched those exemplars at 0.9 and above — the
    # sheet came back with fifteen 'f' markers and lost d, e, g and h. The
    # same render passes once the two ticks are kept out of the template set.
    typical = _consensus_height([item.box.height for _, item in confident])
    learned = [
        GlyphTemplate(label=label, gray=item.gray, mask=item.mask)
        for label, item in confident
        if _size_agrees(item.box.height, typical)
    ]

    enlarged = [*templates, *learned]
    markers: list[SymbolMarker] = []
    for candidate in candidates:
        label, score, gap = _classify(
            candidate.patch,
            candidate.box.height,
            enlarged,
            itself=candidate.gray,
        )
        if not label or label == NOT_A_NAME:
            continue
        if score < MIN_MATCH_SCORE or gap < MIN_MATCH_MARGIN:
            continue
        markers.append(
            SymbolMarker(
                symbol=label,
                bbox=candidate.box,
                score=round(score, 3),
                margin=round(gap, 3),
            )
        )

    agreed = _size_agreeing(markers)
    return sorted(agreed, key=lambda marker: (marker.symbol, marker.bbox.y))


def _consensus_height(heights: Sequence[int]) -> float | None:
    """The height the sheet letters its callouts at, if it is knowable here.

    A drawing office sets every point name in one view at one size, so the
    confident matches agree on a height and that height is the sheet's own
    answer to "how big is a marker here". Nothing else in this module knows
    it: :data:`MARKER_MIN_SCALE` has to stay loose enough for the sheets that
    print their callouts at half the table's size, which on a dense drawing
    also admits ten-pixel tick marks and the stubs of extension lines.

    ``None`` when too few matches exist to form a consensus. A sheet with two
    markers has no distribution, and guessing at one would throw away the
    second. The median is deliberate: it survives a minority of the confident
    matches being junk, which is exactly the case this exists to contain.
    """

    if len(heights) < SIZE_CONSENSUS_MIN_MARKERS:
        return None
    typical = float(np.median(heights))
    return typical if typical > 0 else None


def _size_agrees(height: int, typical: float | None) -> bool:
    """Whether a glyph is the size this sheet prints its callouts at."""

    if typical is None:
        return True
    return (
        SIZE_CONSENSUS_MIN_RATIO * typical
        <= height
        <= SIZE_CONSENSUS_MAX_RATIO * typical
    )


def _size_agreeing(markers: Sequence[SymbolMarker]) -> list[SymbolMarker]:
    """Drop matches printed at a size the sheet does not letter callouts at.

    The last word on what :func:`_consensus_height` already decided for the
    template set: the softened correlation lets the odd tick mark over the
    acceptance bar too, on the benchmark tube drawing twenty of them scoring
    between 0.62 and 0.70 beside true markers at 0.89 and above.
    """

    typical = _consensus_height(
        [
            marker.bbox.height
            for marker in markers
            if marker.score >= CONFIDENT_SCORE
        ]
    )
    return [
        marker for marker in markers if _size_agrees(marker.bbox.height, typical)
    ]


def _inside_a_word(
    stats: np.ndarray,
    index: int,
    *,
    neighbours: np.ndarray,
) -> bool:
    """Whether enough same-size glyphs share this one's foot to be a line."""

    x, y, w, h, _ = stats[index]
    if h <= 0:
        return True
    foot = y + h
    span = h * TEXT_LINE_SPAN_RATIO

    other_x = neighbours[:, 0]
    other_y = neighbours[:, 1]
    other_w = neighbours[:, 2]
    other_h = neighbours[:, 3]
    other_foot = other_y + other_h
    ratio = other_h / float(h)
    company = (
        (ratio >= TEXT_RUN_HEIGHT_MIN)
        & (ratio <= TEXT_RUN_HEIGHT_MAX)
        & (other_foot - foot >= -h * TEXT_RUN_FOOT_ABOVE)
        & (other_foot - foot <= h * TEXT_RUN_FOOT_BELOW)
        # Edge-to-edge horizontal distance, negative when the boxes overlap.
        & (np.maximum(x - (other_x + other_w), other_x - (x + w)) <= span)
    )
    # The candidate matches itself, so its own entry is discounted.
    return int(np.count_nonzero(company)) - 1 >= TEXT_LINE_MIN_COMPANY


def _inside(x: int, y: int, w: int, h: int, box: LayoutBox) -> bool:
    centre_x = x + w / 2
    centre_y = y + h / 2
    return box.x <= centre_x <= box.x1 and box.y <= centre_y <= box.y1
