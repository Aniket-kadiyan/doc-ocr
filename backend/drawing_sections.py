"""Split a drawing sheet into its sub-drawings (views, sections, details).

A sheet usually carries several views, each drawn as one web of connected
lines: outlines, hatching, centre lines, and the extension and dimension lines
hung off them. The values and labels written beside those lines are separate
glyphs that touch nothing. So the sheet is sectioned in two passes:

1. Lines. Every ink component that is not glyph-shaped is a line. The line
   ink is padded so that a dashed centre line, or the small gap a drafter
   leaves between an outline and its extension line, still connects; the
   padded regions are the connected graph of the drawing. One stray leader
   or long dimension line can still join two views, so the regions are
   opened with a disc a little wider than the padding: an isolated line
   (a padded strip) vanishes while the dense body of a view survives as a
   *core*. Each line pixel then goes to the nearest core in its own padded
   region, which cuts a bridging line at the midpoint between the views it
   joins instead of fusing them.

   Cores need three corrections. A view in clean thin lines (a gear face)
   is mostly paper and cores only at its dense spots, so cores ringing one
   compact enclosed area of paper are merged, and grow over it. Two views
   drawn close can share a core through a neck of dimensions, so a core
   that falls into view-sized bodies under a wider opening is split. And a
   dimension stack or frame off a view's body cores on its own, so a small
   section owning little ink rejoins the view beside it.
2. Glyphs. Values, labels and section titles attach to the nearest section
   ink within reach. What is left over, such as notes and title-block text,
   is grouped into text blocks.

All geometry is computed at a working scale (short side ~2000 px), so the
ratios below do not depend on the render resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

from page_layout import LayoutBox, detect_table_masks

SectionKind = Literal["view", "text"]

SECTION_WORK_SHORT_SIDE = 2000
SECTION_INK_THRESHOLD = 160
# All sizes below are fractions of the page's short side.
# A component whose longer side is under this is a glyph, not a line.
SECTION_GLYPH_RATIO = 0.012
# Padding that joins line ink into one region: bridges dash gaps and the
# outline-to-extension-line offset, but not the white space between views.
SECTION_LINE_PAD_RATIO = 0.006
# Opening radius, as a multiple of the padding, that erases isolated lines
# (leaders, long dimension lines) but keeps the dense body of a view.
SECTION_CORE_RADIUS_FACTOR = 1.6
SECTION_MIN_CORE_RATIO = 0.05
# Enclosed paper larger than this share of the sheet is not one view.
SECTION_MAX_ENCLOSED_SHARE = 0.15
# A view encloses compact paper (a bore, the inside of an outline); paper
# fenced in by leaders between two views is ragged. Small holes always fill.
SECTION_HOLE_SOLIDITY = 0.8
SECTION_MIN_SOLID_HOLE = 0.0005
# Line ink within this many paddings of a core sets the section's extent.
SECTION_BODY_REACH_FACTOR = 3.0
# Extra opening radii (multiples of the core radius) tried per core to find
# a neck between two views that share it.
SECTION_NECK_FACTORS = (1.5, 2.0, 2.5)
SECTION_NECK_BODY_SHARE = 0.15
# How far a glyph may sit from its section's ink and still belong to it.
SECTION_GLYPH_REACH_RATIO = 0.03
# Sections smaller than this (longer side) are fragments of a neighbour:
# a dimension stack or a frame that only touches its view through a leader.
SECTION_MIN_VIEW_RATIO = 0.12
# A fragment's body is under this multiple of the smallest view, and it owns
# under this share of the ink of the section it rejoins.
SECTION_FRAGMENT_EXTENT = 1.25
SECTION_FRAGMENT_INK_SHARE = 0.5
SECTION_FRAGMENT_GAP_FACTOR = 3.0
# Two touching sections are one view when the seam between them carries
# line ink at this share of the sparser one's density: a wire-frame view
# cores only at its dense spots. Measured: pieces of one view 0.78-2.1
# (129E01, BS1801025); separate views 0.65 or less (113, 503645 View I).
SECTION_SEAM_DENSITY_SHARE = 0.72
# And the seam runs this far (of the short side): a leader's is 0.008, the
# shortest real cut (129E01) 0.033.
SECTION_SEAM_MIN_LENGTH = 0.02
# Sheet rules: long anywhere, or moderately long inside the edge band.
SECTION_RULE_RATIO = 0.25
SECTION_EDGE_RULE_RATIO = 0.06
SECTION_EDGE_BAND_RATIO = 0.06
SECTION_TEXT_BLOCK_GAP_RATIO = 0.015
# Drawing frame: a side is a rule within this band of the sheet edge that
# covers this share of the sheet, found in segments of this length and with
# this much skew allowed across it.
SECTION_FRAME_BAND_RATIO = 0.1
SECTION_FRAME_MIN_COVERAGE = 0.5
SECTION_FRAME_SEGMENT_RATIO = 0.08
SECTION_FRAME_SKEW_RATIO = 0.005
# Glyph ink per line ink above which a cored section is text, not a view.
SECTION_TEXT_GLYPH_RATIO = 1.5
# A leftover text block this small, next to a view, is that view's title.
SECTION_TITLE_MAX_RATIO = 0.15


@dataclass(frozen=True)
class DrawingSection:
    """One sub-drawing on the sheet, in page coordinates."""

    section_id: str
    kind: SectionKind
    bbox: LayoutBox
    # The lines alone, without the values and labels attached around them.
    line_bbox: LayoutBox
    glyph_count: int
    # The section's region as closed polygons in page coordinates. Regions
    # never overlap, unlike bounding boxes around L-shaped views.
    outlines: tuple[tuple[tuple[int, int], ...], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.section_id,
            "kind": self.kind,
            **self.bbox.to_dict(),
            "line_bbox": self.line_bbox.to_dict(),
            "glyph_count": self.glyph_count,
            "outlines": [[list(point) for point in outline] for outline in self.outlines],
        }


def _disc(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _sheet_frame(ink: np.ndarray) -> tuple[int, int, int, int] | None:
    """The drawing frame as (left, top, right, bottom), or None.

    ``page_layout.detect_drawing_border`` wants four straight rules meeting
    at the corners, which a scanned sheet's skewed frame does not give. Here
    each side is the innermost rule in its edge band that runs at least
    half the sheet, measured over a few rows so a skewed rule still counts.
    """

    height, width = ink.shape
    sides: list[int] = []
    for horizontal in (True, False):
        extent = width if horizontal else height
        length = max(3, int(extent * SECTION_FRAME_SEGMENT_RATIO))
        kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
        rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        skew = max(3, int(min(width, height) * SECTION_FRAME_SKEW_RATIO))
        rules = cv2.dilate(rules, np.ones((skew, 1) if horizontal else (1, skew), np.uint8))
        # A side is one rule, broken at most by gaps shorter than a segment
        # (a scanned rule's dropouts). Summing the whole row would let a box
        # edge and a dimension line beside it pass as the frame together.
        rules = cv2.morphologyEx(
            rules, cv2.MORPH_CLOSE, np.ones((1, length) if horizontal else (length, 1), np.uint8)
        )
        coverage = _longest_runs(rules if horizontal else rules.T) / extent
        band = int(coverage.size * SECTION_FRAME_BAND_RATIO)
        near = np.flatnonzero(coverage[:band] >= SECTION_FRAME_MIN_COVERAGE)
        far = np.flatnonzero(coverage[-band:] >= SECTION_FRAME_MIN_COVERAGE)
        if near.size == 0 or far.size == 0:
            return None
        # Innermost, and past the dilation, so the rule itself is outside.
        sides.extend((int(near.max()) + 1, coverage.size - band + int(far.min()) - 1))
    top, bottom, left, right = sides
    if right - left < width / 2 or bottom - top < height / 2:
        return None
    return left, top, right, bottom


def _longest_runs(mask: np.ndarray) -> np.ndarray:
    """Per row, the length of the longest run of set pixels."""

    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), np.int8)
    padded[:, 1:-1] = mask > 0
    steps = np.diff(padded, axis=1)
    longest = np.zeros(mask.shape[0], np.int64)
    rows, starts = np.nonzero(steps == 1)
    _rows, ends = np.nonzero(steps == -1)
    # Starts and ends pair up in order within each row.
    np.maximum.at(longest, rows, ends - starts)
    return longest


def _erase_sheet_rules(ink: np.ndarray) -> None:
    """Remove the frame, zone and title-block rules that join every view.

    ``page_layout`` only knows a border with all four sides intact; a scanned
    sheet's frame is often skewed or broken, so rules are removed by length.
    """

    height, width = ink.shape
    band = int(min(width, height) * SECTION_EDGE_BAND_RATIO)
    outer = np.ones_like(ink)
    outer[band : height - band, band : width - band] = 0
    for horizontal in (True, False):
        extent = width if horizontal else height
        for ratio, source in (
            (SECTION_RULE_RATIO, ink),
            (SECTION_EDGE_RULE_RATIO, ink & outer),
        ):
            length = max(3, int(extent * ratio))
            kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
            rules = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
            ink[cv2.dilate(rules, np.ones((3, 3), np.uint8)) > 0] = 0


def _line_components(stats: np.ndarray, unit: int) -> np.ndarray:
    """True per component label for line ink, False for glyphs."""

    widths, heights, areas = stats[:, 2], stats[:, 3], stats[:, 4]
    longer = np.maximum(widths, heights)
    shorter = np.minimum(widths, heights)
    fill = areas / np.maximum(widths * heights, 1)
    glyph = unit * SECTION_GLYPH_RATIO
    # A word is several touching glyphs: text-high and dense. A dimension
    # line with its arrowheads is as high and as full of its box, but its
    # mean stroke across the run is a line's, not a row of letters'.
    word = (
        (shorter <= glyph * 1.3)
        & (shorter >= glyph * 0.35)
        & (fill >= 0.12)
        & (areas / np.maximum(longer, 1) >= glyph * 0.2)
    )
    # A dash of a centre or hidden line is short but thin and straight; it
    # must still outrun a glyph box, or every "1" and "l" would be a line.
    dash = (longer >= glyph) & (shorter <= max(2, glyph * 0.15)) & (
        longer >= 3 * shorter
    )
    lines = ~((longer < glyph) | word) | dash
    lines[0] = False
    return lines


def _nearest_label_map(seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel, the distance to and the label of the nearest seed."""

    distance, pixel = cv2.distanceTransformWithLabels(
        (seeds == 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    ys, xs = np.nonzero(seeds)
    lookup = np.zeros(int(pixel.max()) + 1, np.int32)
    lookup[pixel[ys, xs]] = seeds[ys, xs]
    return distance, lookup[pixel]


def _enclosed_paper(mask: np.ndarray) -> list[np.ndarray]:
    """Compact paper areas enclosed by line ink, as (label-window) masks.

    A view drawn in clean thin lines (a gear, a section's walls) is mostly
    empty paper; opening its padded ink leaves only the dense corners, so it
    yields several cores. Its outline encloses compact paper, though, so the
    cores ringing one such area are one view. Paper fenced in between two
    views by leaders is ragged, and too large an area is a surviving frame.
    """

    height, width = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask == 0).astype(np.uint8),
        connectivity=4,
    )
    sheet = height * width
    found: list[tuple[int, int, int, int, np.ndarray]] = []
    for hole in range(1, count):
        x, y, w, h, area = stats[hole]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        if not sheet * SECTION_MIN_SOLID_HOLE <= area <= sheet * SECTION_MAX_ENCLOSED_SHARE:
            continue
        inside = (labels[y : y + h, x : x + w] == hole).astype(np.uint8)
        contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Measured on the outer contour: the detail drawn inside an outline
        # makes islands in its paper, which do not make the paper ragged.
        outline = max(contours, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(outline))
        if cv2.contourArea(outline) >= hull * SECTION_HOLE_SOLIDITY:
            found.append((x, y, w, h, inside))
    return found


def _merge_ringing_cores(
    core_labels: np.ndarray,
    keep: np.ndarray,
    holes: list,
    reach: int,
) -> None:
    """Merge the cores that ring one enclosed paper area, in place."""

    parent = list(range(len(keep)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    height, width = core_labels.shape
    for x, y, w, h, inside in holes:
        x0, y0 = max(0, x - reach), max(0, y - reach)
        x1, y1 = min(width, x + w + reach), min(height, y + h + reach)
        ring = np.zeros((y1 - y0, x1 - x0), np.uint8)
        ring[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = inside
        ring = cv2.dilate(ring, _disc(reach))
        touching = np.unique(core_labels[y0:y1, x0:x1][ring > 0])
        touching = [int(t) for t in touching if t and keep[t]]
        for other in touching[1:]:
            parent[find(other)] = find(touching[0])
    roots = np.array([find(i) for i in range(len(keep))], np.int32)
    if np.any(roots != np.arange(len(keep))):
        core_labels[:] = roots[core_labels]
        keep &= roots == np.arange(len(keep))


def _grow_into_enclosed(
    core_labels: np.ndarray,
    keep: np.ndarray,
    holes: list,
    reach: int,
) -> None:
    """Grow each core over the enclosed paper it touches, in place.

    Then an outline drawn well clear of a view's dense interior is still
    near its core, and counts toward the view's extent.
    """

    height, width = core_labels.shape
    for x, y, w, h, inside in holes:
        x0, y0 = max(0, x - reach), max(0, y - reach)
        x1, y1 = min(width, x + w + reach), min(height, y + h + reach)
        ring = np.zeros((y1 - y0, x1 - x0), np.uint8)
        ring[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = inside
        ring = cv2.dilate(ring, _disc(reach))
        touching = core_labels[y0:y1, x0:x1][ring > 0]
        touching = touching[(touching > 0)]
        touching = touching[keep[touching]]
        if touching.size == 0:
            continue
        owner = int(np.bincount(touching).argmax())
        window = core_labels[y : y + h, x : x + w]
        window[(inside > 0) & (window == 0)] = owner


def _split_necked_cores(
    core_labels: np.ndarray,
    keep: np.ndarray,
    core_region: np.ndarray,
    *,
    radius: int,
    min_view: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a core that is two views joined by a neck.

    Two views drawn close together can share a core through a stack of
    dimensions between them. Each core is opened further on its own; if
    it falls apart into two or more view-sized bodies, those become the
    seeds instead and the neck goes to whichever body is nearer. Also
    returns each core's origin: the core it was split from, else itself.
    """

    # Labels index ``keep``; dropped cores leave gaps below its length.
    next_label = len(keep)
    new_keep = list(keep)
    new_region = list(core_region)
    origin = list(range(len(keep)))
    for core in np.flatnonzero(keep):
        ys, xs = np.nonzero(core_labels == core)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if max(y1 - y0, x1 - x0) < min_view * 2:
            continue
        mask = (core_labels[y0:y1, x0:x1] == core).astype(np.uint8)
        for factor in SECTION_NECK_FACTORS:
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _disc(int(radius * factor)))
            count, parts, stats, _ = cv2.connectedComponentsWithStats(opened)
            # Each side of a real neck is a whole view, not a lobe of one.
            bodies = [
                part
                for part in range(1, count)
                if max(stats[part, 2], stats[part, 3]) >= min_view
                and stats[part, 4] >= mask.sum() * SECTION_NECK_BODY_SHARE
            ]
            groups = _ring_groups(mask, parts, bodies)
            if len(groups) < 2:
                continue
            # The lobes and the neck go to the nearest body, so the core's
            # lines stay with its own views.
            seeds = np.zeros(mask.shape, np.int32)
            for group in groups:
                seeds[np.isin(parts, group)] = next_label
                new_keep.append(True)
                new_region.append(core_region[core])
                origin.append(int(core))
                next_label += 1
            _, nearest = _nearest_label_map(seeds)
            window = core_labels[y0:y1, x0:x1]
            window[mask > 0] = nearest[mask > 0]
            new_keep[core] = False
            break
    return (
        core_labels,
        np.array(new_keep, bool),
        np.array(new_region, np.int32),
        np.array(origin, np.int32),
    )


def _ring_groups(mask: np.ndarray, parts: np.ndarray, bodies: list[int]) -> list[list[int]]:
    """Bodies of one opened core, grouped where they are one view.

    Two views joined by a neck meet at one place. A view drawn round, such
    as a flange's concentric outline, opens into arcs that meet at both
    ends: bodies linked by two or more separate necks are one ring.
    """

    parent = {body: body for body in bodies}

    def root(body: int) -> int:
        while parent[body] != body:
            body = parent[body]
        return body

    rest = ((mask > 0) & ~np.isin(parts, bodies)).astype(np.uint8)
    count, necks = cv2.connectedComponents(rest, connectivity=8)
    grown = cv2.dilate(np.where(np.isin(parts, bodies), parts, 0).astype(np.uint16), np.ones((3, 3), np.uint8))
    links: dict[tuple[int, int], int] = {}
    for neck in range(1, count):
        touched = sorted(set(np.unique(grown[necks == neck]).tolist()) & set(bodies))
        for index, first in enumerate(touched):
            for second in touched[index + 1 :]:
                links[(first, second)] = links.get((first, second), 0) + 1
    for (first, second), joins in links.items():
        if joins >= 2:
            parent[root(second)] = root(first)
    groups: dict[int, list[int]] = {}
    for body in bodies:
        groups.setdefault(root(body), []).append(body)
    return list(groups.values())


def _seam_merges(
    section_ink: np.ndarray,
    lines: np.ndarray,
    region_cores: dict[int, list[int]],
    origin: np.ndarray,
    pad: int,
    min_seam: float,
) -> dict[int, int]:
    """Cores whose owned ink is cut through drawing, not across paper.

    Two views on one leader touch at a line or two, with empty paper
    between them. A sparse view split at its dense spots is cut through its
    own lines, so the seam is as inked as the pieces. Halves of a necked
    core were split on purpose and stay apart. Returns core -> core it
    joins (a root that maps to itself is left out).
    """

    parent: dict[int, int] = {}

    def root(core: int) -> int:
        while parent.get(core, core) != core:
            core = parent[core]
        return core

    touch_kernel = np.ones((3, 3), np.uint8)
    for members in region_cores.values():
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                if origin[first] == origin[second]:
                    continue
                a = section_ink == first
                b = section_ink == second
                if not (a & (cv2.dilate(b.astype(np.uint8), touch_kernel) > 0)).any():
                    continue
                ys, xs = np.nonzero(a | b)
                window = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                a, b, ink = a[window], b[window], lines[window] > 0
                hulls = []
                for piece in (a, b):
                    hull = np.zeros(piece.shape, np.uint8)
                    cv2.fillPoly(hull, [cv2.convexHull(cv2.findNonZero(piece.astype(np.uint8)))], 1)
                    hulls.append(hull > 0)
                seeds = np.zeros(a.shape, np.int32)
                seeds[a], seeds[b] = 1, 2
                _, nearest = _nearest_label_map(seeds)
                seam = cv2.morphologyEx(
                    (nearest == 1).astype(np.uint8), cv2.MORPH_GRADIENT, touch_kernel
                )
                band = (cv2.dilate(seam, _disc(pad * 2)) > 0) & (hulls[0] | hulls[1])
                # A leader between two views meets the pieces' hulls over a
                # sliver; the band is 4 pads wide, so this is its length.
                if band.sum() < min_seam * pad * 4:
                    continue
                sparser = min(ink[hull].mean() for hull in hulls)
                joined = root(first), root(second)
                if joined[0] != joined[1] and ink[band].mean() >= sparser * SECTION_SEAM_DENSITY_SHARE:
                    parent[joined[1]] = joined[0]
    return {core: root(core) for core in parent if root(core) != core}


def _box_gap(a: list[int], b: list[int]) -> int:
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def _union(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def detect_drawing_sections(
    image: Image.Image,
    *,
    table_masks: Sequence[LayoutBox] | None = None,
) -> tuple[DrawingSection, ...]:
    """Detect the sheet's sub-drawings, in reading order.

    ``table_masks`` defaults to ``detect_table_masks(image)``; pass a
    ``PageLayout.table_masks`` to reuse an analysed page.
    """

    if cv2 is None:
        return ()
    width, height = image.size
    if width < 64 or height < 64:
        return ()
    if table_masks is None:
        table_masks = detect_table_masks(image)

    page_ink = (np.asarray(image.convert("L")) < SECTION_INK_THRESHOLD).astype(np.uint8)
    scale = min(1.0, SECTION_WORK_SHORT_SIDE / min(width, height))
    w, h = max(1, int(width * scale)), max(1, int(height * scale))

    def work_ink() -> np.ndarray:
        if scale == 1.0:
            return page_ink.copy()
        return (
            cv2.resize(page_ink * 255, (w, h), interpolation=cv2.INTER_AREA) > 40
        ).astype(np.uint8)

    # Everything outside the drawing frame (margin notes, zone ruler, the
    # stamp strip under the title block) is not part of the drawing. Found
    # before the table masks: a header table along the frame masks its rule.
    frame = _sheet_frame(work_ink())
    for mask in table_masks:
        page_ink[mask.y : mask.y1, mask.x : mask.x1] = 0
    ink = work_ink()
    unit = min(w, h)
    if frame is not None:
        left, top, right, bottom = frame
        inside = np.zeros_like(ink)
        inside[top:bottom, left:right] = 1
        ink &= inside
    _erase_sheet_rules(ink)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    is_line = _line_components(stats, unit)
    lines = is_line[labels].astype(np.uint8)

    pad = max(2, int(unit * SECTION_LINE_PAD_RATIO))
    padded = cv2.dilate(lines, _disc(pad))
    # An outline and the detail drawn inside it are one region even when the
    # paper between them is wider than the padding.
    enclosed = _enclosed_paper(padded)
    joined = padded.copy()
    for x, y, hw, hh, inside in enclosed:
        joined[y : y + hh, x : x + hw] |= inside
    region_count, regions, region_stats, _ = cv2.connectedComponentsWithStats(
        joined,
        connectivity=8,
    )
    cores = cv2.morphologyEx(
        padded,
        cv2.MORPH_OPEN,
        _disc(int(pad * SECTION_CORE_RADIUS_FACTOR)),
    )
    core_count, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(
        cores,
        connectivity=8,
    )
    keep = np.zeros(core_count, bool)
    core_region = np.zeros(core_count, np.int32)
    edge = int(unit * SECTION_EDGE_BAND_RATIO)
    for core in range(1, core_count):
        x, y, cw, ch, _area = core_stats[core]
        if max(cw, ch) < unit * SECTION_MIN_CORE_RATIO:
            continue
        # A zone ruler down the sheet edge: thin, and wholly in the band.
        if min(cw, ch) < unit * 0.03 and (
            x + cw < edge or y + ch < edge or x > w - edge or y > h - edge
        ):
            continue
        keep[core] = True
        ys, xs = np.nonzero(core_labels[y : y + ch, x : x + cw] == core)
        core_region[core] = regions[y + ys[0], x + xs[0]]
    core_labels[~keep[core_labels]] = 0
    _merge_ringing_cores(core_labels, keep, enclosed, pad * 2)
    core_labels, keep, core_region, origin = _split_necked_cores(
        core_labels,
        keep,
        core_region,
        radius=int(pad * SECTION_CORE_RADIUS_FACTOR),
        min_view=unit * SECTION_MIN_VIEW_RATIO,
    )

    _grow_into_enclosed(core_labels, keep, enclosed, pad * 2)

    # A view drawn only in thin lines (a bare outline) leaves no core at all;
    # a view-sized region without one is its own core.
    cored = set(int(region) for region in core_region[keep])
    extra_keep: list[bool] = []
    extra_region: list[int] = []
    for region in range(1, region_count):
        x, y, rw, rh, _area = region_stats[region]
        if region in cored or max(rw, rh) < unit * SECTION_MIN_VIEW_RATIO:
            continue
        # Two-dimensional, unlike a lone long line or a note's underlines.
        if min(rw, rh) < unit * SECTION_MIN_VIEW_RATIO * 0.25:
            continue
        window = core_labels[y : y + rh, x : x + rw]
        window[(regions[y : y + rh, x : x + rw] == region) & (window == 0)] = (
            len(keep) + len(extra_keep)
        )
        extra_keep.append(True)
        extra_region.append(region)
    if extra_keep:
        keep = np.concatenate([keep, np.array(extra_keep, bool)])
        core_region = np.concatenate([core_region, np.array(extra_region, np.int32)])
        origin = np.concatenate([origin, np.arange(len(origin), len(keep), dtype=np.int32)])

    # Each line pixel goes to the nearest core of its own padded region.
    # The nearest core overall may sit across the gap in another region, so
    # each region is resolved among its own cores only.
    section_ink = np.zeros(lines.shape, np.int32)
    region_cores: dict[int, list[int]] = {}
    for core in np.flatnonzero(keep):
        region_cores.setdefault(int(core_region[core]), []).append(int(core))
    for region, members in region_cores.items():
        x, y, rw, rh, _area = region_stats[region]
        window = (slice(y, y + rh), slice(x, x + rw))
        inside = (regions[window] == region) & (lines[window] > 0)
        if len(members) == 1:
            section_ink[window][inside] = members[0]
            continue
        seeds = np.where(np.isin(core_labels[window], members), core_labels[window], 0)
        _, nearest = _nearest_label_map(seeds.astype(np.int32))
        section_ink[window][inside] = nearest[inside]
    for core, target in _seam_merges(
        section_ink, lines, region_cores, origin, pad, unit * SECTION_SEAM_MIN_LENGTH
    ).items():
        section_ink[section_ink == core] = target
        core_labels[core_labels == core] = target
        keep[core] = False
    owned = section_ink > 0

    # A section's extent is its line ink near its core. A leader or an
    # extension line reaching far across the sheet stays the section's, but
    # does not stretch its box over the neighbouring views.
    core_distance, nearest_core = _nearest_label_map(core_labels)
    body = np.where(
        (core_distance <= pad * SECTION_BODY_REACH_FACTOR) & (nearest_core == section_ink),
        section_ink,
        0,
    )

    boxes: dict[int, list[int]] = {}
    line_boxes: dict[int, list[int]] = {}
    glyphs: dict[int, int] = {}
    glyph_px: dict[int, int] = {}
    line_px: dict[int, int] = {}
    # Everything a section owns, leaders included: how much of a drawing it
    # is, when deciding whether it is a whole view or a fragment of one.
    owned_px: dict[int, int] = {}
    for core in np.flatnonzero(keep):
        ys, xs = np.nonzero(section_ink == core)
        if ys.size:
            owned_px[int(core)] = int(ys.size)
        ys, xs = np.nonzero(body == core)
        if ys.size:
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            boxes[int(core)] = list(box)
            line_boxes[int(core)] = box
            line_px[int(core)] = int(ys.size)
            glyphs[int(core)] = 0
            glyph_px[int(core)] = 0

    # A glyph goes to the nearest body within reach, or failing that to the
    # line it sits on: a value written on a dimension line drawn well out
    # from the view, past the body.
    distance, nearest_section = _nearest_label_map(body)
    line_distance, nearest_line = _nearest_label_map(section_ink)
    reach = unit * SECTION_GLYPH_REACH_RATIO
    leftover = np.zeros_like(ink)
    glyph_owner = np.zeros(count, np.int32)
    for component in range(1, count):
        x, y, cw, ch, area = stats[component]
        if is_line[component]:
            if not owned[y : y + ch, x : x + cw].any():
                leftover[y : y + ch, x : x + cw] |= (
                    labels[y : y + ch, x : x + cw] == component
                ).astype(np.uint8)
            continue
        cy, cx = y + ch // 2, x + cw // 2
        section = int(nearest_section[cy, cx])
        if distance[cy, cx] > reach or section not in boxes:
            section = int(nearest_line[cy, cx])
            if line_distance[cy, cx] > reach / 2:
                section = 0
        if section in boxes:
            glyph_owner[component] = section
            boxes[section] = _union(boxes[section], [x, y, x + cw, y + ch])
            glyphs[section] += 1
            glyph_px[section] += int(area)
        else:
            leftover[y : y + ch, x : x + cw] = 1

    # Fragments (a dimension stack, a feature control frame on its leader)
    # rejoin the view their lines connect to: the nearest section, by ink,
    # in the same padded region. A small view standing on its own, such as
    # a detail or a narrow section, is left alone however small it is.
    small = unit * SECTION_MIN_VIEW_RATIO

    # A stack of dimension lines spans a view's width with little ink.
    sparse = float(np.median(list(owned_px.values()))) * 0.15 if owned_px else 0.0
    alias: dict[int, int] = {}
    for key in sorted(boxes, key=lambda k: owned_px[k]):
        box = line_boxes[key]
        compact = max(box[2] - box[0], box[3] - box[1]) < small * SECTION_FRAGMENT_EXTENT
        if not compact and owned_px[key] >= sparse:
            continue
        # A fragment hangs off a much larger view. Two small views of like
        # size on one leader are two views.
        partners = [
            other
            for other in boxes
            if other != key
            and core_region[other] == core_region[key]
            and owned_px[key] < owned_px[other] * SECTION_FRAGMENT_INK_SHARE
        ]
        if not partners:
            continue
        away = cv2.distanceTransform((body != key).astype(np.uint8), cv2.DIST_L2, 3)
        gaps = {other: float(away[body == other].min()) for other in partners}
        target = min(partners, key=gaps.__getitem__)
        # Pieces of one view lie close (a sparse view's dimension stack can
        # sit a little off its body); a detail at the end of a long leader
        # is a view of its own.
        if gaps[target] > reach * SECTION_FRAGMENT_GAP_FACTOR:
            continue
        body[body == key] = target
        alias[key] = target
        boxes[target] = _union(boxes[target], boxes.pop(key))
        line_boxes[target] = _union(line_boxes[target], line_boxes.pop(key))
        line_px[target] += line_px.pop(key)
        owned_px[target] += owned_px.pop(key)
        glyphs[target] += glyphs.pop(key)
        glyph_px[target] += glyph_px.pop(key)

    # Bold note text (kanji, underlined rows) can pack into a core of its
    # own. A view carries at most ~0.4 px of glyph ink per px of line ink;
    # measured notes blocks carry 3.5 or more.
    views: dict[int, list[int]] = {}
    texts: dict[int, list[int]] = {}
    for key in boxes:
        if glyph_px[key] > owned_px[key] * SECTION_TEXT_GLYPH_RATIO:
            texts[key] = [*boxes[key], glyphs[key]]
        else:
            views[key] = [*boxes[key], glyphs[key]]

    # Leftover glyphs and lines form text blocks. A small one beside a view
    # is its title ("SECTION AA" sits below the view, past the glyph reach)
    # or the title's underline; the rest are notes and title-block text.
    gap = max(2, int(unit * SECTION_TEXT_BLOCK_GAP_RATIO))
    blocks = cv2.dilate(leftover, np.ones((gap, gap * 2), np.uint8))
    block_count, block_labels, block_stats, _ = cv2.connectedComponentsWithStats(blocks)
    block_glyphs = np.bincount(
        block_labels[stats[1:, 1] + stats[1:, 3] // 2, stats[1:, 0] + stats[1:, 2] // 2][
            ~is_line[1:]
        ],
        minlength=block_count,
    )
    block_owner = np.zeros(block_count, np.int32)
    view_ink = np.where(np.isin(body, list(views)), body, 0).astype(np.int32)
    view_distance, nearest_view = _nearest_label_map(view_ink)
    next_key = len(keep)
    for block in range(1, block_count):
        x, y, bw, bh, _area = block_stats[block]
        box = [x + gap, y + gap // 2, x + bw - gap, y + bh - gap // 2]
        if box[2] <= box[0] or box[3] <= box[1]:
            box = [x, y, x + bw, y + bh]
        # Titles run along the view; rotated values stand beside it. A column
        # down the sheet edge is margin boilerplate, not either.
        title_like = (
            max(bw, bh) < unit * SECTION_TITLE_MAX_RATIO
            and min(bw, bh) <= unit * SECTION_GLYPH_RATIO * 4
            and edge <= x
            and edge <= y
            and x + bw <= w - edge
            and y + bh <= h - edge
        )
        # Text inside a view's own lines (a bore's callout, a hatch note)
        # belongs to the smallest view around it.
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        around = [
            key
            for key in views
            if line_boxes[key][0] <= cx <= line_boxes[key][2]
            and line_boxes[key][1] <= cy <= line_boxes[key][3]
        ]
        if around:
            owner = min(
                around,
                key=lambda key: (line_boxes[key][2] - line_boxes[key][0])
                * (line_boxes[key][3] - line_boxes[key][1]),
            )
            views[owner][:4] = _union(views[owner], box)
            views[owner][4] += int(block_glyphs[block])
            block_owner[block] = owner
            continue
        # A title belongs to the view whose ink is nearest, not whose box
        # is: "DETAIL AREA L" sits under its detail, inside a big neighbour's
        # bounding box.
        if title_like and views:
            here = block_labels[y : y + bh, x : x + bw] == block
            distances = view_distance[y : y + bh, x : x + bw][here]
            nearest = int(nearest_view[y : y + bh, x : x + bw][here][np.argmin(distances)])
            if distances.min() <= reach * 2 and nearest in views:
                views[nearest][:4] = _union(views[nearest], box)
                views[nearest][4] += int(block_glyphs[block])
                block_owner[block] = nearest
                continue
        # Measured without the dilation, which alone is wider than this. A
        # block thinner than a glyph is a stray rule, such as a title's
        # underline, not text.
        if max(box[2] - box[0], box[3] - box[1]) < unit * 0.04:
            continue
        if min(box[2] - box[0], box[3] - box[1]) < unit * SECTION_GLYPH_RATIO * 0.5:
            continue
        texts[next_key] = [*box, int(block_glyphs[block])]
        block_owner[block] = next_key
        next_key += 1

    # Paragraphs of one notes column become one text section.
    merged = True
    while merged:
        merged = False
        for first, second in [(a, b) for a in texts for b in texts if a < b]:
            if _box_gap(texts[first], texts[second]) <= gap * 2:
                texts[first][:4] = _union(texts[first], texts[second])
                texts[first][4] += texts.pop(second)[4]
                alias[second] = first
                merged = True
                break

    # One owner per ink pixel: lines, the glyphs attached to them, and the
    # titles and text blocks around them. Extension and dimension lines past
    # the body are the section's too; only far leaders are left out.
    owner = body.copy()
    resolved = section_ink.copy()
    for key in alias:
        target = alias[key]
        while target in alias:
            target = alias[target]
        resolved[resolved == key] = target
    body_distance, nearest_body = _nearest_label_map(body)
    near_lines = (
        (owner == 0)
        & (resolved > 0)
        & (nearest_body == resolved)
        & (body_distance <= reach * 2)
    )
    owner[near_lines] = resolved[near_lines]
    # Past that, a line is still the section's when it carries one of the
    # section's own values: a dimension line drawn well out from the view
    # with its number on it. A bare leader carries none and stays out.
    far_lines = (
        (owner == 0) & (resolved > 0) & (nearest_body == resolved) & (lines > 0)
    ).astype(np.uint8)
    if far_lines.any():
        _far_count, far_labels = cv2.connectedComponents(far_lines, connectivity=8)
        far_distance, nearest_far = _nearest_label_map(far_labels.astype(np.int32))
        carrying: set[int] = set()
        for component in np.flatnonzero(glyph_owner):
            x, y, cw, ch, _area = stats[component]
            cy, cx = y + ch // 2, x + cw // 2
            line = int(nearest_far[cy, cx])
            if (
                line
                and far_distance[cy, cx] <= reach / 2
                and resolved[far_labels == line][0] == glyph_owner[component]
            ):
                carrying.add(line)
        if carrying:
            carried = np.isin(far_labels, list(carrying))
            owner[carried] = resolved[carried]
    attached = glyph_owner[labels]
    owner = np.where((owner == 0) & (attached > 0), attached, owner)
    grouped = block_owner[block_labels]
    owner = np.where((owner == 0) & (leftover > 0) & (grouped > 0), grouped, owner)
    for key in sorted(alias, reverse=True):
        target = alias[key]
        while target in alias:
            target = alias[target]
        owner[owner == key] = target

    sections: list[tuple[SectionKind, int, list[int], list[int], int]] = [
        ("view", key, views[key][:4], line_boxes[key], views[key][4]) for key in views
    ] + [("text", key, text[:4], text[:4], text[4]) for key, text in texts.items()]

    if frame is not None:
        left, top, right, bottom = frame
        clipped = []
        for kind, key, box, line_box, glyph_count in sections:
            box = [max(box[0], left), max(box[1], top), min(box[2], right), min(box[3], bottom)]
            if min(box[2] - box[0], box[3] - box[1]) >= unit * SECTION_GLYPH_RATIO * 0.5:
                clipped.append((kind, key, box, line_box, glyph_count))
        sections = clipped
    kept = {key for _kind, key, *_rest in sections}
    owner[~np.isin(owner, list(kept))] = 0
    regions_by_key = _section_regions(owner, int(reach))

    def to_page(box: list[int]) -> LayoutBox:
        x0 = max(0, int(box[0] / scale))
        y0 = max(0, int(box[1] / scale))
        x1 = min(width, int(np.ceil(box[2] / scale)))
        y1 = min(height, int(np.ceil(box[3] / scale)))
        return LayoutBox(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    left, top, right, bottom = frame if frame is not None else (0, 0, w, h)

    def outline_to_page(polygon: np.ndarray) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                min(width, int(round(min(max(px, left), right) / scale))),
                min(height, int(round(min(max(py, top), bottom) / scale))),
            )
            for px, py in polygon
        )

    band = unit * 0.2
    sections.sort(key=lambda item: (item[0] != "view", int(item[2][1] // band), item[2][0]))
    numbers = {"view": 0, "text": 0}
    result: list[DrawingSection] = []
    for kind, key, box, line_box, glyph_count in sections:
        numbers[kind] += 1
        outlines = regions_by_key.get(key, ())
        if outlines:
            points = np.clip(np.vstack(outlines), (left, top), (right, bottom))
            box = [
                int(points[:, 0].min()),
                int(points[:, 1].min()),
                int(points[:, 0].max()) + 1,
                int(points[:, 1].max()) + 1,
            ]
        result.append(
            DrawingSection(
                section_id=f"{'V' if kind == 'view' else 'T'}{numbers[kind]}",
                kind=kind,
                bbox=to_page(box),
                line_bbox=to_page(line_box),
                glyph_count=glyph_count,
                outlines=tuple(outline_to_page(polygon) for polygon in outlines),
            )
        )
    return tuple(result)


def _section_regions(owner: np.ndarray, reach: int) -> dict[int, list[np.ndarray]]:
    """Each section's region, as outline polygons in work coordinates.

    Every pixel within ``reach`` of owned ink goes to the nearest owner, so
    regions never overlap; each is then closed into a solid shape, still
    without crossing into a neighbour's territory.
    """

    distance, nearest = _nearest_label_map(owner)
    territory = np.where(distance <= reach, nearest, 0).astype(np.int32)
    height, width = owner.shape
    regions: dict[int, list[np.ndarray]] = {}
    for key in np.unique(territory):
        if key == 0:
            continue
        ys, xs = np.nonzero(territory == key)
        y0, y1 = max(0, ys.min() - reach), min(height, ys.max() + reach + 1)
        x0, x1 = max(0, xs.min() - reach), min(width, xs.max() + reach + 1)
        window = territory[y0:y1, x0:x1]
        mine = (window == key).astype(np.uint8)
        others = (window > 0) & (window != key)
        solid = cv2.morphologyEx(mine, cv2.MORPH_CLOSE, _disc(max(1, reach // 2)))
        solid[others] = 0
        contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Drop crumbs: the territory of a lone leader end far from the view.
        minimum = reach * reach
        polygons = [
            cv2.approxPolyDP(contour, max(1.5, reach * 0.1), True).reshape(-1, 2) + (x0, y0)
            for contour in contours
            if cv2.contourArea(contour) >= minimum
        ]
        if polygons:
            regions[int(key)] = polygons
    return regions


def section_crop(
    image: Image.Image,
    section: DrawingSection,
    *,
    padding: int = 20,
) -> tuple[Image.Image, tuple[int, int]]:
    """The section's pixels alone: its bounding box, with everything outside
    its outlines whitened, so a neighbour's values are never read twice.

    Returns the crop and its page-coordinate origin.
    """

    from PIL import ImageDraw

    box = section.bbox
    x0, y0 = max(0, box.x - padding), max(0, box.y - padding)
    x1, y1 = min(image.width, box.x1 + padding), min(image.height, box.y1 + padding)
    crop = image.convert("RGB").crop((x0, y0, x1, y1))
    if not section.outlines:
        return crop, (x0, y0)
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    for outline in section.outlines:
        draw.polygon([(px - x0, py - y0) for px, py in outline], fill=255)
    white = Image.new("RGB", crop.size, (255, 255, 255))
    return Image.composite(crop, white, mask), (x0, y0)


def tag_regions_with_sections(
    regions: Sequence[dict],
    sections: Sequence[DrawingSection],
) -> None:
    """Tag each scan region with the section its centre falls in.

    Sets ``section_id`` and ``section_rank`` (the section's position in
    reading order), so balloons can be numbered one section at a time. A
    centre inside no outline (a value on a frame rule, a dropped margin)
    goes to the section whose box is nearest. Regions keep their order.
    """

    if not sections:
        return
    contours = [
        [np.asarray(outline, dtype=np.int32).reshape(-1, 1, 2) for outline in section.outlines]
        for section in sections
    ]
    for region in regions:
        box = region.get("bbox") or {}
        try:
            cx = float(box["x"]) + float(box["width"]) / 2
            cy = float(box["y"]) + float(box["height"]) / 2
        except (KeyError, TypeError, ValueError):
            continue
        rank = next(
            (
                index
                for index, polygons in enumerate(contours)
                if any(cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0 for polygon in polygons)
            ),
            None,
        )
        if rank is None:
            def distance(section: DrawingSection) -> float:
                b = section.bbox
                dx = max(b.x - cx, 0.0, cx - b.x1)
                dy = max(b.y - cy, 0.0, cy - b.y1)
                return math.hypot(dx, dy)

            rank = min(range(len(sections)), key=lambda index: distance(sections[index]))
        region["section_id"] = sections[rank].section_id
        region["section_rank"] = rank
