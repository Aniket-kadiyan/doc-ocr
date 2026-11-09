"""Page-level layout analysis for whole-page and section auto-balloon scans.

The layout stage is deliberately independent from OCR.  It detects tabular
regions from repeated horizontal/vertical line intersections, creates
temporary overlapping processing panels from whitespace and ink density, and
provides page-coordinate geometry for the live debug overlay.

Nothing in this module decides whether a recognized value is technically
meaningful. Whole-page and selected-section scans consume the same table masks
and adaptive panels so a value cannot change merely because the user selected
the area around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, Literal, Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - the populated-rows test then passes
    cv2 = None  # type: ignore


SCAN_DEBUG_OVERLAY_ENABLED = True

LAYOUT_MIN_PANELS = 4
LAYOUT_MAX_PANELS = 8
LAYOUT_PANEL_OVERLAP_RATIO = 0.04
LAYOUT_PANEL_OVERLAP_MIN = 24
LAYOUT_PANEL_OVERLAP_MAX = 80

TABLE_MASK_PADDING_RATIO = 0.002
# Width of the left and right margin strips that get a table pass of their own,
# as a fraction of the sheet. A rule only counts as a rule if it is long
# relative to the page, which is the right test for a title block spanning the
# sheet and the wrong one for the boilerplate column down a margin: this
# drawing's geometric-tolerance legend rules its rows over 190px of a 5875px
# page, just under the threshold, so not one row was found and every garbled
# label in it -- "3D-CAD", "A3", "CONCENTRICITY TOLERANCE" -- was ballooned.
# Measuring against the strip instead finds them. Lowering the page-wide
# threshold to reach them does not work: at 0.025 it starts calling the pipe
# outlines and their centre lines a grid, and masks real dimensions.
TABLE_MASK_MARGIN_BAND_RATIO = 0.08


@dataclass(frozen=True)
class LayoutBox:
    """Axis-aligned page-coordinate rectangle."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x1, self.y1

    def to_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }


@dataclass(frozen=True)
class LayoutPanel:
    """One temporary detector panel in page coordinates."""

    panel_id: str
    bbox: LayoutBox

    def to_dict(self, *, state: str = "pending") -> dict[str, object]:
        return {
            "id": self.panel_id,
            "label": self.panel_id,
            "state": state,
            **self.bbox.to_dict(),
        }


@dataclass(frozen=True)
class PageLayout:
    """Reusable page geometry cached by full-image fingerprint."""

    width: int
    height: int
    table_masks: tuple[LayoutBox, ...]
    panels: tuple[LayoutPanel, ...]
    overlaps: tuple[LayoutBox, ...]
    # The drawing border, when the sheet has a full one. Its outside is
    # already among ``table_masks``; this is for passes that never see the
    # masked page, such as the notes and feature-control-frame reads.
    border: LayoutBox | None = None

    def overlay(
        self,
        *,
        scope_kind: Literal["page", "section"],
        panel_states: dict[str, str] | None = None,
        candidates: Sequence[dict[str, object]] = (),
    ) -> dict[str, object] | None:
        """Return the exact geometry consumed by processing for the UI."""

        if not SCAN_DEBUG_OVERLAY_ENABLED:
            return None
        states = panel_states or {}
        return {
            "enabled": True,
            "page_width": self.width,
            "page_height": self.height,
            "scope_kind": scope_kind,
            # Both scan scopes now consume the same page masks. Section mode
            # processes only panels whose state is not ``skipped``.
            "table_masks": [box.to_dict() for box in self.table_masks],
            "panels": [
                panel.to_dict(
                    state=states.get(panel.panel_id, "pending")
                )
                for panel in self.panels
                if scope_kind == "page"
                or (
                    panel_states is not None
                    and states.get(panel.panel_id, "pending") != "skipped"
                )
            ],
            "overlaps": (
                [box.to_dict() for box in self.overlaps]
                if scope_kind == "page"
                else []
            ),
            "candidates": list(candidates),
        }


@dataclass(frozen=True)
class _LineSegment:
    """Horizontal or vertical line segment used only during grid analysis."""

    coordinate: int
    start: int
    end: int


def _ink_map(image: Image.Image) -> np.ndarray:
    """Return True for non-paper pixels, including blue and light-grey lines."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.min(rgb, axis=2) < 245


def _runs_with_small_gaps(
    values: np.ndarray,
    *,
    max_gap: int,
    min_length: int,
) -> list[tuple[int, int]]:
    positions = np.flatnonzero(values)
    if positions.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = previous = int(positions[0])
    for raw_position in positions[1:]:
        position = int(raw_position)
        if position - previous > max_gap + 1:
            if previous - start + 1 >= min_length:
                runs.append((start, previous + 1))
            start = position
        previous = position
    if previous - start + 1 >= min_length:
        runs.append((start, previous + 1))
    return runs


def _merge_nearby_segments(
    segments: Iterable[_LineSegment],
    *,
    coordinate_tolerance: int,
) -> list[_LineSegment]:
    """Collapse the 1–3 pixel thickness of a printed grid line."""

    merged: list[dict[str, int]] = []
    for segment in sorted(
        segments,
        key=lambda item: (item.coordinate, item.start, item.end),
    ):
        match: dict[str, int] | None = None
        for existing in reversed(merged):
            if segment.coordinate - existing["coordinate"] > coordinate_tolerance:
                break
            overlap = min(segment.end, existing["end"]) - max(
                segment.start,
                existing["start"],
            )
            smaller = max(
                1,
                min(
                    segment.end - segment.start,
                    existing["end"] - existing["start"],
                ),
            )
            if overlap / smaller >= 0.65:
                match = existing
                break
        if match is None:
            merged.append(
                {
                    "coordinate": segment.coordinate,
                    "start": segment.start,
                    "end": segment.end,
                    "count": 1,
                }
            )
            continue

        count = match["count"] + 1
        match["coordinate"] = int(
            round(
                (
                    match["coordinate"] * match["count"]
                    + segment.coordinate
                )
                / count
            )
        )
        match["start"] = min(match["start"], segment.start)
        match["end"] = max(match["end"], segment.end)
        match["count"] = count

    return [
        _LineSegment(
            coordinate=item["coordinate"],
            start=item["start"],
            end=item["end"],
        )
        for item in merged
    ]


def _extract_axis_lines(
    ink: np.ndarray,
    *,
    horizontal: bool,
) -> list[_LineSegment]:
    height, width = ink.shape
    primary = height if horizontal else width
    secondary = width if horizontal else height
    min_length = max(32, int(round(secondary * 0.035)))
    max_gap = max(1, int(round(min(width, height) / 900)))
    # Page-border lines may also be legitimate outer boundaries of title and
    # tolerance tables.  Keep them; isolated page-frame rectangles are removed
    # later because they do not form repeated adjacent cells.
    edge_margin = 1

    segments: list[_LineSegment] = []
    for coordinate in range(edge_margin, primary - edge_margin):
        values = ink[coordinate, :] if horizontal else ink[:, coordinate]
        for start, end in _runs_with_small_gaps(
            values,
            max_gap=max_gap,
            min_length=min_length,
        ):
            segments.append(
                _LineSegment(
                    coordinate=coordinate,
                    start=start,
                    end=end,
                )
            )

    return _merge_nearby_segments(
        segments,
        coordinate_tolerance=max(2, int(round(min(width, height) / 600))),
    )


def _boxes_touch_or_align(
    left: LayoutBox,
    right: LayoutBox,
    *,
    gap: int,
) -> bool:
    overlap_x = min(left.x1, right.x1) - max(left.x, right.x)
    overlap_y = min(left.y1, right.y1) - max(left.y, right.y)
    gap_x = max(0, max(left.x, right.x) - min(left.x1, right.x1))
    gap_y = max(0, max(left.y, right.y) - min(left.y1, right.y1))
    aligned_x = overlap_x >= 0.45 * min(left.width, right.width)
    aligned_y = overlap_y >= 0.45 * min(left.height, right.height)
    return (gap_x <= gap and aligned_y) or (gap_y <= gap and aligned_x)


def _merge_layout_boxes(
    boxes: Sequence[LayoutBox],
    *,
    gap: int,
) -> list[LayoutBox]:
    pending = list(boxes)
    merged: list[LayoutBox] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remaining: list[LayoutBox] = []
            for other in pending:
                if not _boxes_touch_or_align(current, other, gap=gap):
                    remaining.append(other)
                    continue
                x0 = min(current.x, other.x)
                y0 = min(current.y, other.y)
                x1 = max(current.x1, other.x1)
                y1 = max(current.y1, other.y1)
                current = LayoutBox(x0, y0, x1 - x0, y1 - y0)
                changed = True
            pending = remaining
        merged.append(current)
    return merged


def _line_covers(
    line: _LineSegment,
    start: int,
    end: int,
    *,
    tolerance: int,
) -> bool:
    return line.start - tolerance <= start and line.end + tolerance >= end


# How close to the sheet's edge a row's rules have to run before the edge is
# read as the missing side of a cell, and how far in a real rule may be and
# still count as that side instead.
_CLIPPED_EDGE_REACH = 0.004
_CLIPPED_EDGE_RULE_GAP = 0.02


def _clipped_edges(
    top: _LineSegment,
    bottom: _LineSegment,
    rules: set[int],
    *,
    page_width: int,
    tolerance: int,
) -> set[int]:
    """Page edges that close a cell the sheet itself does not close.

    A PDF whose media box cuts through the drawing leaves its right-hand
    column with three sides: the table's rules run off the paper and there is
    no fourth line to pair them with, so no cell forms there and nothing in
    that column is ever masked as table content. On the benchmark sheet that
    published the tensile-strength figure as a dimension and queued the
    column's heading for review, while the yield column beside it — closed on
    both sides — was masked correctly.

    The edge is only taken when the row's own rules run to it and no real
    rule is already standing near it, which is what distinguishes a clipped
    table from a sheet whose border frame happens to sit close to the paper.
    """

    reach = max(tolerance, int(round(page_width * _CLIPPED_EDGE_REACH)))
    gap = max(tolerance, int(round(page_width * _CLIPPED_EDGE_RULE_GAP)))
    edges: set[int] = set()
    for edge, runs_to_edge in (
        (0, top.start <= reach and bottom.start <= reach),
        (
            page_width - 1,
            top.end >= page_width - 1 - reach
            and bottom.end >= page_width - 1 - reach,
        ),
    ):
        if not runs_to_edge:
            continue
        if any(abs(rule - edge) <= gap for rule in rules):
            continue
        edges.add(edge)
    return edges


def _detect_grid_cells(
    horizontal: Sequence[_LineSegment],
    vertical: Sequence[_LineSegment],
    *,
    page_width: int,
    page_height: int,
    tolerance: int,
) -> list[LayoutBox]:
    """Find individual cells bounded by two horizontal and two vertical lines.

    Returning cells first is important for engineering drawings: title blocks
    and tolerance tables often form an L-shaped connected grid.  A single
    connected-component bounding box would cover the useful drawing between
    those two tables.
    """

    h_lines = sorted(horizontal, key=lambda line: line.coordinate)
    cells: list[LayoutBox] = []
    maximum_row_height = max(40, int(page_height * 0.18))

    for top_index, top in enumerate(h_lines):
        best_row_cells: list[LayoutBox] = []
        best_row_score = 0
        for bottom in h_lines[top_index + 1 :]:
            row_height = bottom.coordinate - top.coordinate
            if row_height < max(4, tolerance * 2):
                continue
            if row_height > maximum_row_height:
                break

            rules = {
                line.coordinate
                for line in vertical
                if _line_covers(
                    line,
                    top.coordinate,
                    bottom.coordinate,
                    tolerance=tolerance,
                )
                and _line_covers(
                    top,
                    line.coordinate,
                    line.coordinate,
                    tolerance=tolerance,
                )
                and _line_covers(
                    bottom,
                    line.coordinate,
                    line.coordinate,
                    tolerance=tolerance,
                )
            }
            boundaries = sorted(
                rules
                | _clipped_edges(
                    top,
                    bottom,
                    rules,
                    page_width=page_width,
                    tolerance=tolerance,
                )
            )
            row_cells: list[LayoutBox] = []
            for left, right in zip(boundaries, boundaries[1:]):
                cell_width = right - left
                if cell_width < max(8, tolerance * 3):
                    continue
                if cell_width > page_width * 0.75:
                    continue
                if not (
                    _line_covers(top, left, right, tolerance=tolerance)
                    and _line_covers(bottom, left, right, tolerance=tolerance)
                ):
                    continue
                row_cells.append(
                    LayoutBox(
                        left,
                        top.coordinate,
                        cell_width,
                        row_height,
                    )
                )

            row_score = sum(cell.width for cell in row_cells)
            if row_score > best_row_score:
                best_row_cells = row_cells
                best_row_score = row_score

        cells.extend(best_row_cells)

    return cells


def _cells_are_adjacent(
    left: LayoutBox,
    right: LayoutBox,
    *,
    tolerance: int,
) -> bool:
    overlap_x = min(left.x1, right.x1) - max(left.x, right.x)
    overlap_y = min(left.y1, right.y1) - max(left.y, right.y)
    vertical_edge = (
        min(abs(left.x1 - right.x), abs(right.x1 - left.x)) <= tolerance
        and overlap_y >= 0.65 * min(left.height, right.height)
    )
    horizontal_edge = (
        min(abs(left.y1 - right.y), abs(right.y1 - left.y)) <= tolerance
        and overlap_x >= 0.65 * min(left.width, right.width)
    )
    return vertical_edge or horizontal_edge


#: What makes a group of adjacent cells a table rather than a part view.
#:
#: A machined block in orthographic projection IS a grid of rectangles — the
#: outline, its pockets and its steps are all cells bounded by rules — and
#: the grid detector cannot tell it from a table by geometry alone. On one
#: drawing that produced 27 masks on a sheet with no tables, and because a
#: mask is applied by whitening the page BEFORE detection, every dimension
#: inside those views (16 of the 17 the author pointed at) was never seen.
#:
#: What separates them was measured over six sheets: a table's rows are text
#: height and nearly every cell holds text; a part view's cells are large
#: and mostly empty. Tables measured at a median cell height of 0.010–0.060
#: of the page, with 57–100% of cells populated; part views at 0.068–0.150
#: with 0–44% populated (one at 67%, caught by its height). The thresholds sit
#: between the two populations.
TABLE_CELL_MAX_HEIGHT_RATIO = 0.065
TABLE_CELL_MIN_OCCUPANCY = 0.5
TABLE_CELL_MIN_TEXT_PER_CELL = 3.0
#: Size band for a text-like ink component inside a cell, as a fraction of
#: the page height: a glyph on a 250dpi render is 30–40px of ~2900.
TEXT_COMPONENT_MIN_HEIGHT_RATIO = 0.0025
TEXT_COMPONENT_MAX_HEIGHT_RATIO = 0.025

#: How many adjacent cells a group needs before it is called a table.
#:
#: Two was not enough. A drawing that frames its dimension text — a common
#: CAD style, and what a basic dimension always looks like — puts two framed
#: numbers next to each other all over the views, and every such pair was
#: being masked. The mask is applied by whitening the page BEFORE detection,
#: so those values were not merely rejected: the detector never saw them, and
#: they reached neither a balloon nor the review queue. On one sheet that is
#: what removed "59.8" from the scan entirely.
#:
#: Three is the smallest group a ruled table can have and still be one, and
#: measured over five real sheets it removes the pairs sitting on the views
#: while leaving every title block and revision grid masked.
TABLE_MASK_MIN_CELLS = 3

#: Dimension lines end in arrowheads; table rules never do.
#:
#: A densely dimensioned view passes every test above. Its extension and
#: dimension lines cross into text-height "cells", and each one holds the
#: value written on its dimension line, so the view looks like a populated
#: table. On 113-M02-292A that masked three whole views (41.7±0.2, 27, 48.5,
#: R1000, 8-R2, 30°, 45° and more were never detected).
#:
#: A cell edge counts as a dimension line when one end, past the rule it
#: starts on, widens into a wedge on BOTH sides of the line. The two-sided
#: test is what keeps table text out: a glyph or signature touching a rule
#: sits on one side of it. Measured over 69 grids on 40 raster sheets, the
#: three dimensioned views had 17–31% of their cell edges ending in arrows;
#: the highest real table had 10%, from one signature stroke in a 10-cell
#: block. The decision is per group only: a single arrowed cell is noise
#: (it unmasked half a gear-specification table when tried), so a real
#: table is masked whole or not at all.
TABLE_ARROW_LENGTH_RATIO = 0.008
TABLE_VIEW_MIN_ARROW_EDGE_RATIO = 0.13
TABLE_VIEW_MIN_ARROW_EDGES = 3


def _group_repeated_cells(
    cells: Sequence[LayoutBox],
    *,
    tolerance: int,
    minimum_cells: int = TABLE_MASK_MIN_CELLS,
) -> list[list[LayoutBox]]:
    """Adjacent cells grouped, keeping only groups large enough to be a grid."""

    parent = list(range(len(cells)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(cells):
        for right_index in range(left_index + 1, len(cells)):
            if _cells_are_adjacent(left, cells[right_index], tolerance=tolerance):
                union(left_index, right_index)

    counts: dict[int, int] = {}
    for index in range(len(cells)):
        root = find(index)
        counts[root] = counts.get(root, 0) + 1
    groups: dict[int, list[LayoutBox]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault(find(index), []).append(cell)
    return [group for group in groups.values() if len(group) >= minimum_cells]


def _retain_repeated_cell_groups(
    cells: Sequence[LayoutBox],
    *,
    tolerance: int,
    minimum_cells: int = TABLE_MASK_MIN_CELLS,
) -> list[LayoutBox]:
    """Discard isolated drawing rectangles while retaining repeated grids."""

    return [
        cell
        for group in _group_repeated_cells(
            cells,
            tolerance=tolerance,
            minimum_cells=minimum_cells,
        )
        for cell in group
    ]


def _text_component_count(
    ink: np.ndarray,
    *,
    page_height: int,
) -> int:
    """How many glyph-sized ink components a cell's interior holds."""

    if cv2 is None or ink.size == 0 or not ink.any():
        return 0
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8),
        connectivity=8,
    )
    if stats.shape[0] <= 1:
        return 0
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    min_height = max(5, int(page_height * TEXT_COMPONENT_MIN_HEIGHT_RATIO))
    max_height = max(min_height + 1, int(page_height * TEXT_COMPONENT_MAX_HEIGHT_RATIO))
    glyph_like = (
        (heights >= min_height)
        & (heights <= max_height)
        & (widths >= 2)
        & (widths <= 6 * heights)
        & (heights <= 8 * widths)
    )
    return int(np.count_nonzero(glyph_like))


def _group_looks_like_a_table(
    group: Sequence[LayoutBox],
    ink: np.ndarray,
    *,
    page_height: int,
    tolerance: int,
) -> bool:
    """Text-height rows, nearly all of them holding text. See the constants."""

    heights = np.array([cell.height for cell in group], dtype=float)
    if float(np.median(heights)) > page_height * TABLE_CELL_MAX_HEIGHT_RATIO:
        return False
    if cv2 is None:
        return True
    inset = max(2, tolerance)
    populated = 0
    components = 0
    for cell in group:
        interior = ink[
            cell.y + inset : max(cell.y + inset, cell.y1 - inset),
            cell.x + inset : max(cell.x + inset, cell.x1 - inset),
        ]
        count = _text_component_count(interior, page_height=page_height)
        components += count
        populated += count >= 2
    occupancy = populated / len(group)
    return (
        occupancy > TABLE_CELL_MIN_OCCUPANCY
        or components / len(group) >= TABLE_CELL_MIN_TEXT_PER_CELL
    )


def _ink_run_across(
    ink: np.ndarray,
    *,
    along: int,
    across: int,
    horizontal: bool,
    reach: int,
) -> tuple[int, int] | None:
    """
    The run of ink crossing a line, in one slice across it.

    Returned as the run's first and last coordinate across the line, or
    ``None`` when the slice falls off the page. An empty slice is a zero
    length run at ``across``.
    """

    if horizontal:
        if not 0 <= along < ink.shape[1]:
            return None
        start = max(0, across - reach)
        line = ink[start : across + reach + 1, along]
    else:
        if not 0 <= along < ink.shape[0]:
            return None
        start = max(0, across - reach)
        line = ink[along, start : across + reach + 1]
    centre = across - start
    touching = [
        index
        for index in range(max(0, centre - 3), min(len(line), centre + 4))
        if line[index]
    ]
    if not touching:
        return across, across - 1
    first = last = min(touching, key=lambda index: abs(index - centre))
    while first > 0 and line[first - 1] and centre - first < reach:
        first -= 1
    while last < len(line) - 1 and line[last + 1] and last - centre < reach:
        last += 1
    return start + first, start + last


def _edge_ends_in_arrowhead(
    ink: np.ndarray,
    *,
    across: int,
    tip: int,
    step: int,
    horizontal: bool,
    length: int,
) -> bool:
    """Whether a cell edge widens into a two-sided wedge just inside ``tip``."""

    runs: list[tuple[int, int]] = []
    for offset in range(2, 3 * length):
        run = _ink_run_across(
            ink,
            along=tip + step * offset,
            across=across,
            horizontal=horizontal,
            reach=2 * length,
        )
        if run is None:
            return False
        runs.append(run)
    first = np.array([run[0] for run in runs], dtype=float)
    last = np.array([run[1] for run in runs], dtype=float)
    width = last - first + 1
    # Past the head the edge is a plain line. Its middle there is the line's
    # real centre; a scanned rule rarely sits exactly on the cell coordinate,
    # and the wedge's two sides are measured from it.
    line_width = float(np.median(width[-length:]))
    line_centre = float(np.median((first[-length:] + last[-length:]) / 2))
    # The edge starts on the perpendicular line that closes the cell; that
    # line is far wider than any arrowhead, so step past it first.
    start = 0
    while start < length and width[start] > 2 * length:
        start += 1
    head = width[start : start + 2 * length]
    widest = start + int(head.argmax())
    peak = float(width[widest])
    if peak < max(3 * line_width, line_width + 4) or peak > 2 * length:
        return False
    # A glyph or signature touching a rule sits on one side of it. A scanned
    # arrowhead is rarely centred on its line, so up to 1:3 still counts.
    above = line_centre - first[widest]
    below = last[widest] - line_centre
    return min(above, below) >= 2 and min(above, below) >= max(above, below) / 3


def _arrowed_edge_count(
    ink: np.ndarray,
    cell: LayoutBox,
    *,
    page_height: int,
) -> int:
    """How many of a cell's four edges end in an arrowhead at either end."""

    length = max(6, int(page_height * TABLE_ARROW_LENGTH_RATIO))
    count = 0
    for across, start, end, horizontal in (
        (cell.y, cell.x, cell.x1, True),
        (cell.y1, cell.x, cell.x1, True),
        (cell.x, cell.y, cell.y1, False),
        (cell.x1, cell.y, cell.y1, False),
    ):
        count += any(
            _edge_ends_in_arrowhead(
                ink,
                across=across,
                tip=tip,
                step=step,
                horizontal=horizontal,
                length=length,
            )
            for tip, step in ((start, 1), (end, -1))
        )
    return count


def _group_is_a_dimensioned_view(
    group: Sequence[LayoutBox],
    ink: np.ndarray,
    *,
    page_height: int,
) -> bool:
    """Too many of the group's edges are dimension lines. See the constants."""

    arrowed = sum(
        _arrowed_edge_count(ink, cell, page_height=page_height)
        for cell in group
    )
    return (
        arrowed >= TABLE_VIEW_MIN_ARROW_EDGES
        and arrowed / (4 * len(group)) >= TABLE_VIEW_MIN_ARROW_EDGE_RATIO
    )


def _merge_rectangular_cells(
    cells: Sequence[LayoutBox],
    *,
    tolerance: int,
) -> list[LayoutBox]:
    """Merge equal rows/columns without filling an L-shaped empty corner."""

    boxes = list(cells)
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(boxes):
            merged_index: int | None = None
            merged_box: LayoutBox | None = None
            for right_index in range(left_index + 1, len(boxes)):
                right = boxes[right_index]
                same_row = (
                    abs(left.y - right.y) <= tolerance
                    and abs(left.height - right.height) <= tolerance
                    and min(abs(left.x1 - right.x), abs(right.x1 - left.x))
                    <= tolerance
                )
                same_column = (
                    abs(left.x - right.x) <= tolerance
                    and abs(left.width - right.width) <= tolerance
                    and min(abs(left.y1 - right.y), abs(right.y1 - left.y))
                    <= tolerance
                )
                if not (same_row or same_column):
                    continue
                x0 = min(left.x, right.x)
                y0 = min(left.y, right.y)
                x1 = max(left.x1, right.x1)
                y1 = max(left.y1, right.y1)
                merged_index = right_index
                merged_box = LayoutBox(x0, y0, x1 - x0, y1 - y0)
                break
            if merged_index is None or merged_box is None:
                continue
            boxes[left_index] = merged_box
            boxes.pop(merged_index)
            changed = True
            break
    return boxes


def _discard_mostly_contained_boxes(
    boxes: Sequence[LayoutBox],
) -> list[LayoutBox]:
    """Remove nested duplicates produced by thick or decorative grid lines."""

    accepted: list[LayoutBox] = []
    for candidate in sorted(
        boxes,
        key=lambda box: box.width * box.height,
        reverse=True,
    ):
        candidate_area = max(candidate.width * candidate.height, 1)
        mostly_contained = False
        for existing in accepted:
            x0 = max(candidate.x, existing.x)
            y0 = max(candidate.y, existing.y)
            x1 = min(candidate.x1, existing.x1)
            y1 = min(candidate.y1, existing.y1)
            intersection = max(0, x1 - x0) * max(0, y1 - y0)
            if intersection / candidate_area >= 0.80:
                mostly_contained = True
                break
        if not mostly_contained:
            accepted.append(candidate)
    return accepted


def _grid_masks(
    ink: np.ndarray,
    *,
    page_width: int,
    page_height: int,
) -> list[LayoutBox]:
    """Repeated-cell grids in one ink map, in that map's own coordinates."""

    tolerance = max(2, int(round(min(page_width, page_height) / 500)))
    cells = _detect_grid_cells(
        _extract_axis_lines(ink, horizontal=True),
        _extract_axis_lines(ink, horizontal=False),
        page_width=page_width,
        page_height=page_height,
        tolerance=tolerance,
    )
    repeated_cells = [
        cell
        for group in _group_repeated_cells(cells, tolerance=tolerance * 2)
        if _group_looks_like_a_table(
            group,
            ink,
            page_height=page_height,
            tolerance=tolerance,
        )
        and not _group_is_a_dimensioned_view(
            group,
            ink,
            page_height=page_height,
        )
        for cell in group
    ]
    return _discard_mostly_contained_boxes(
        _merge_rectangular_cells(repeated_cells, tolerance=tolerance * 2)
    )


def _margin_band_masks(
    ink: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[LayoutBox]:
    """Grids in the sheet's four margin strips, measured against the strip.

    A drawing keeps its boilerplate down the margins, and a narrow table's
    rules are short in page terms however obviously they rule a table. Running
    the same detection over the strip alone restores the proportion the
    page-wide pass assumes. Spurious grids here cost nothing: the strips hold
    the border and the boilerplate, not dimensions.
    """

    masks: list[LayoutBox] = []
    band = int(round(width * TABLE_MASK_MARGIN_BAND_RATIO))
    if band >= 64:
        for x0 in (0, max(0, width - band)):
            x1 = min(width, x0 + band)
            if x1 - x0 < 64:
                continue
            for box in _grid_masks(
                ink[:, x0:x1],
                page_width=x1 - x0,
                page_height=height,
            ):
                masks.append(LayoutBox(box.x + x0, box.y, box.width, box.height))
    # The same along the top and bottom: a revision strip or a parts list
    # across the head or foot of the sheet has rows too short in page terms.
    band = int(round(height * TABLE_MASK_MARGIN_BAND_RATIO))
    if band >= 64:
        for y0 in (0, max(0, height - band)):
            y1 = min(height, y0 + band)
            if y1 - y0 < 64:
                continue
            # Rules are measured against the strip, which is what lets its
            # short column rules count; cell sizes are still judged against
            # the page, or every row would look too tall to be a table's.
            for box in _grid_masks(
                ink[y0:y1, :],
                page_width=width,
                page_height=height,
            ):
                masks.append(LayoutBox(box.x, box.y + y0, box.width, box.height))
    return masks


# The drawing border: four long rules, each within this fraction of its own
# edge of the sheet, at least this fraction of the sheet long, and meeting the
# other three at the corners. Everything between the border and the paper edge
# -- a boilerplate column down the margin, a print stamp under the frame -- is
# outside the drawing and never a dimension. All four sides are required: one
# long line near an edge is just as likely a shaft's outline, and masking
# beyond it would delete the dimensions over it.
BORDER_EDGE_BAND_RATIO = 0.15
BORDER_MIN_LENGTH_RATIO = 0.6
BORDER_CORNER_TOLERANCE_RATIO = 0.02
# Rules this close to the paper edge are the scan's own edge, not a border.
BORDER_PAPER_EDGE_RATIO = 0.003


def _border_sides(ink: np.ndarray) -> tuple[int, int, int, int] | None:
    """The drawing border's left, top, right and bottom rules, if it has one."""

    height, width = ink.shape
    horizontal = _extract_axis_lines(ink, horizontal=True)
    vertical = _extract_axis_lines(ink, horizontal=False)

    def outermost(
        segments: Sequence[_LineSegment],
        extent: int,
        length: int,
        *,
        far: bool,
    ) -> list[_LineSegment]:
        band = extent * BORDER_EDGE_BAND_RATIO
        paper = extent * BORDER_PAPER_EDGE_RATIO
        found = [
            segment
            for segment in segments
            if segment.end - segment.start >= length * BORDER_MIN_LENGTH_RATIO
            and (
                paper <= extent - segment.coordinate <= band
                if far
                else paper <= segment.coordinate <= band
            )
        ]
        return sorted(found, key=lambda s: -s.coordinate if far else s.coordinate)

    tolerance = max(width, height) * BORDER_CORNER_TOLERANCE_RATIO
    tops = outermost(horizontal, height, width, far=False)
    bottoms = outermost(horizontal, height, width, far=True)
    lefts = outermost(vertical, width, height, far=False)
    rights = outermost(vertical, width, height, far=True)
    # The outermost rule on each side that closes the rectangle with the
    # others; an inner rule (a header table's row line) is tried only if the
    # outer one does not.
    for top in tops:
        for bottom in bottoms:
            for left in lefts:
                for right in rights:
                    if all(
                        segment.start <= low + tolerance
                        and segment.end >= high - tolerance
                        for segment, low, high in (
                            (top, left.coordinate, right.coordinate),
                            (bottom, left.coordinate, right.coordinate),
                            (left, top.coordinate, bottom.coordinate),
                            (right, top.coordinate, bottom.coordinate),
                        )
                    ):
                        return (
                            left.coordinate,
                            top.coordinate,
                            right.coordinate,
                            bottom.coordinate,
                        )
    return None


def _outside_border_masks(
    border: LayoutBox | None,
    *,
    width: int,
    height: int,
) -> list[LayoutBox]:
    """The strips between the drawing border and the paper edge."""

    if border is None:
        return []
    left, top, right, bottom = border.x, border.y, border.x1 - 1, border.y1 - 1
    # Stop short of the rule itself, which a title block or a tolerance table
    # inside the border may share as its own outer edge.
    gap = max(2, int(round(min(width, height) / 600)))
    strips = (
        LayoutBox(0, 0, left - gap, height),
        LayoutBox(right + gap + 1, 0, width - right - gap - 1, height),
        LayoutBox(0, 0, width, top - gap),
        LayoutBox(0, bottom + gap + 1, width, height - bottom - gap - 1),
    )
    return [strip for strip in strips if strip.width > 0 and strip.height > 0]


def detect_drawing_border(image: Image.Image) -> LayoutBox | None:
    """The drawing border as a box, or None when the sheet has no full frame."""

    width, height = image.size
    if width < 64 or height < 64:
        return None
    return _border_box(_ink_map(image))


def _border_box(ink: np.ndarray) -> LayoutBox | None:
    sides = _border_sides(ink)
    if sides is None:
        return None
    left, top, right, bottom = sides
    return LayoutBox(left, top, right - left + 1, bottom - top + 1)


def detect_table_masks(
    image: Image.Image,
    *,
    include_border: bool = True,
) -> tuple[LayoutBox, ...]:
    """Detect every repeated-cell grid that should be masked in page scans,
    plus the strips outside the drawing border unless told otherwise."""

    width, height = image.size
    if width < 64 or height < 64:
        return ()

    ink = _ink_map(image)
    intersection_tolerance = max(2, int(round(min(width, height) / 500)))
    padding = max(
        2,
        int(round(min(width, height) * TABLE_MASK_PADDING_RATIO)),
    )

    boxes = _discard_mostly_contained_boxes(
        _grid_masks(ink, page_width=width, page_height=height)
        + _margin_band_masks(ink, width=width, height=height)
    )
    padded: list[LayoutBox] = []
    for box in boxes:
        x0 = max(0, box.x - padding)
        y0 = max(0, box.y - padding)
        x1 = min(width, box.x1 + padding)
        y1 = min(height, box.y1 + padding)
        padded.append(LayoutBox(x0, y0, x1 - x0, y1 - y0))
    # Unpadded: the strips already stop just short of the border rule, and
    # padding would carry them over it onto values drawn against the frame.
    if include_border:
        padded.extend(
            _outside_border_masks(_border_box(ink), width=width, height=height)
        )
    return tuple(sorted(padded, key=lambda box: (box.y, box.x)))


def _mask_ink(ink: np.ndarray, masks: Sequence[LayoutBox]) -> np.ndarray:
    masked = ink.copy()
    for mask in masks:
        masked[mask.y : mask.y1, mask.x : mask.x1] = False
    return masked


def mask_table_regions(
    image: Image.Image,
    masks: Sequence[LayoutBox],
) -> Image.Image:
    """Whiten table regions before OCR detection to avoid wasted work."""

    if not masks:
        return image.copy()
    output = image.convert("RGB").copy()
    white = Image.new("RGB", output.size, (255, 255, 255))
    for mask in masks:
        output.paste(white.crop(mask.box), (mask.x, mask.y))
    return output


def _smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values.astype(float), kernel, mode="same")


def _best_split(
    region: LayoutBox,
    ink: np.ndarray,
    *,
    page_width: int,
    page_height: int,
) -> tuple[Literal["x", "y"], int] | None:
    """Choose a balanced low-ink split; fall back to the longest axis."""

    minimum_width = max(96, int(page_width * 0.18))
    minimum_height = max(72, int(page_height * 0.18))
    options: list[tuple[float, Literal["x", "y"], int]] = []

    for axis in ("x", "y"):
        length = region.width if axis == "x" else region.height
        minimum = minimum_width if axis == "x" else minimum_height
        if length < minimum * 2:
            continue

        view = ink[region.y : region.y1, region.x : region.x1]
        projection = view.sum(axis=0 if axis == "x" else 1)
        window = max(3, int(round(length * 0.025)))
        smoothed = _smooth_projection(projection, window)
        low = max(minimum, int(length * 0.28))
        high = min(length - minimum, int(length * 0.72))
        if high <= low:
            continue
        local = smoothed[low:high]
        relative = int(np.argmin(local)) + low
        peak = max(float(np.percentile(smoothed, 80)), 1.0)
        whitespace = 1.0 - min(1.0, float(smoothed[relative]) / peak)
        balance = 1.0 - abs(relative / length - 0.5) * 2.0
        axis_preference = (
            region.width / max(region.height, 1)
            if axis == "x"
            else region.height / max(region.width, 1)
        )
        score = whitespace * 2.0 + balance * 0.35 + axis_preference * 0.15
        absolute = region.x + relative if axis == "x" else region.y + relative
        options.append((score, axis, absolute))

    if options:
        _score, axis, position = max(options, key=lambda item: item[0])
        return axis, position

    if region.width >= region.height and region.width >= minimum_width * 2:
        return "x", region.x + region.width // 2
    if region.height >= minimum_height * 2:
        return "y", region.y + region.height // 2
    return None


def _target_panel_count(width: int, height: int) -> int:
    megapixels = width * height / 1_000_000.0
    if megapixels >= 4.0:
        return 8
    if megapixels >= 2.0:
        return 6
    return LAYOUT_MIN_PANELS


def _expand_panel(
    region: LayoutBox,
    *,
    page_width: int,
    page_height: int,
    overlap: int,
) -> LayoutBox:
    x0 = region.x if region.x == 0 else max(0, region.x - overlap)
    y0 = region.y if region.y == 0 else max(0, region.y - overlap)
    x1 = region.x1 if region.x1 == page_width else min(
        page_width,
        region.x1 + overlap,
    )
    y1 = region.y1 if region.y1 == page_height else min(
        page_height,
        region.y1 + overlap,
    )
    return LayoutBox(x0, y0, x1 - x0, y1 - y0)


def _panel_overlaps(panels: Sequence[LayoutPanel]) -> tuple[LayoutBox, ...]:
    overlaps: list[LayoutBox] = []
    for left_index, left in enumerate(panels):
        for right in panels[left_index + 1 :]:
            x0 = max(left.bbox.x, right.bbox.x)
            y0 = max(left.bbox.y, right.bbox.y)
            x1 = min(left.bbox.x1, right.bbox.x1)
            y1 = min(left.bbox.y1, right.bbox.y1)
            if x1 > x0 and y1 > y0:
                overlaps.append(LayoutBox(x0, y0, x1 - x0, y1 - y0))
    return tuple(_merge_layout_boxes(overlaps, gap=0))


def build_adaptive_panels(
    image: Image.Image,
    table_masks: Sequence[LayoutBox] = (),
) -> tuple[tuple[LayoutPanel, ...], tuple[LayoutBox, ...]]:
    """Build 4–8 overlapping panels using layout cues, with grid fallback."""

    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("Page dimensions must be positive")

    ink = _mask_ink(_ink_map(image), table_masks)
    target = min(
        LAYOUT_MAX_PANELS,
        max(LAYOUT_MIN_PANELS, _target_panel_count(width, height)),
    )
    regions = [LayoutBox(0, 0, width, height)]
    while len(regions) < target:
        candidates = sorted(
            enumerate(regions),
            key=lambda item: (
                item[1].width * item[1].height,
                int(
                    ink[
                        item[1].y : item[1].y1,
                        item[1].x : item[1].x1,
                    ].sum()
                ),
            ),
            reverse=True,
        )
        split_applied = False
        for region_index, region in candidates:
            split = _best_split(
                region,
                ink,
                page_width=width,
                page_height=height,
            )
            if split is None:
                continue
            axis, position = split
            if axis == "x":
                first = LayoutBox(
                    region.x,
                    region.y,
                    position - region.x,
                    region.height,
                )
                second = LayoutBox(
                    position,
                    region.y,
                    region.x1 - position,
                    region.height,
                )
            else:
                first = LayoutBox(
                    region.x,
                    region.y,
                    region.width,
                    position - region.y,
                )
                second = LayoutBox(
                    region.x,
                    position,
                    region.width,
                    region.y1 - position,
                )
            regions[region_index : region_index + 1] = [first, second]
            split_applied = True
            break
        if not split_applied:
            break

    # Very small pages may not satisfy the preferred minimum panel dimensions.
    # A guaranteed 2x2 fallback still prevents full-page one-panel processing.
    if len(regions) == 1 and width >= 2 and height >= 2:
        mid_x = width // 2
        mid_y = height // 2
        regions = [
            LayoutBox(0, 0, mid_x, mid_y),
            LayoutBox(mid_x, 0, width - mid_x, mid_y),
            LayoutBox(0, mid_y, mid_x, height - mid_y),
            LayoutBox(mid_x, mid_y, width - mid_x, height - mid_y),
        ]

    overlap = max(
        LAYOUT_PANEL_OVERLAP_MIN,
        min(
            LAYOUT_PANEL_OVERLAP_MAX,
            int(round(min(width, height) * LAYOUT_PANEL_OVERLAP_RATIO)),
        ),
    )
    expanded = [
        _expand_panel(
            region,
            page_width=width,
            page_height=height,
            overlap=overlap,
        )
        for region in regions[:LAYOUT_MAX_PANELS]
    ]
    expanded.sort(key=lambda box: (box.y, box.x))
    panels = tuple(
        LayoutPanel(panel_id=f"P{index}", bbox=box)
        for index, box in enumerate(expanded, start=1)
    )
    return panels, _panel_overlaps(panels)


def analyze_page_layout(image: Image.Image) -> PageLayout:
    """Analyze one complete rendered drawing in source-page coordinates."""

    rgb = image.convert("RGB")
    tables = detect_table_masks(rgb, include_border=False)
    border = detect_drawing_border(rgb)
    # Panels are split around the tables alone. The border strips hold
    # almost no ink, but feeding them in still moves the split lines, and
    # every value near a moved line reads a little differently.
    panels, overlaps = build_adaptive_panels(rgb, tables)
    masks = tuple(
        sorted(
            tables
            + tuple(
                _outside_border_masks(border, width=rgb.width, height=rgb.height)
            ),
            key=lambda box: (box.y, box.x),
        )
    )
    return PageLayout(
        width=rgb.width,
        height=rgb.height,
        table_masks=masks,
        panels=panels,
        overlaps=overlaps,
        border=border,
    )


def crop_section(
    image: Image.Image,
    scope_bbox: dict[str, float],
    *,
    padding: int = 20,
) -> tuple[Image.Image, tuple[int, int], LayoutBox]:
    """Return a padded raw section crop in the page coordinate system.

    The returned origin maps crop-local coordinates—including padding—back to
    the full page. Pixels are returned unchanged; callers that need scan parity
    must apply the page layout separately.
    """

    x0 = max(0, floor(float(scope_bbox["x"])))
    y0 = max(0, floor(float(scope_bbox["y"])))
    x1 = min(
        image.width,
        ceil(float(scope_bbox["x"]) + float(scope_bbox["width"])),
    )
    y1 = min(
        image.height,
        ceil(float(scope_bbox["y"]) + float(scope_bbox["height"])),
    )
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The selected scan section is outside the page")

    section = image.convert("RGB").crop((x0, y0, x1, y1))
    padded = Image.new(
        "RGB",
        (section.width + padding * 2, section.height + padding * 2),
        (255, 255, 255),
    )
    padded.paste(section, (padding, padding))
    return padded, (x0 - padding, y0 - padding), LayoutBox(
        x0,
        y0,
        x1 - x0,
        y1 - y0,
    )
