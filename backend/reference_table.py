"""Find the coordinate reference table on a drawing and read its rows.

Some sheets carry a small table of named points — ``a``, ``b``, ``c`` … — each
with its X/Y/Z coordinates, and print the bare point name beside the matching
feature in the views.  The table is what gives those letters meaning, so it has
to be located before anything in the drawing can be ballooned against it.

The title above such a table cannot be relied on.  "(REFERENCE)", "(参考)",
"BEND POINTS" and no caption at all have all been seen on real sheets, so the
caption is never consulted.  What is stable is the *header row*: a column of
point names followed by columns headed by the axis letters.  Identification
therefore reads the grid, not the prose:

    POINT │   X   │   Y   │   Z        <- axis letters in adjacent columns
    ──────┼───────┼───────┼──────
      a   │  0.0  │  0.0  │  0.0       <- a short name, then numbers
      b   │  0.0  │ 35.6  │  6.9

The grid itself comes from the same horizontal/vertical line analysis the page
scanner already uses for its table masks, but kept as a *lattice* of rows and
columns rather than collapsed to one rectangle: the cells are what the caller
crops to read a point name and its coordinates.

This module is pure geometry and text logic.  It never runs a model; callers
hand in recognized boxes through :func:`assign_text_to_cells`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from PIL import Image

from page_layout import (
    LayoutBox,
    _detect_grid_cells,
    _extract_axis_lines,
    _ink_map,
)


# Axis columns recognised as coordinate axes, in their conventional order.
AXIS_LETTERS = ("X", "Y", "Z")

# Readings that are the axis letter with OCR noise around it. The header cell
# is tiny and isolated, so a stray tick or the cell rule is occasionally fused
# into the box.
_AXIS_NOISE = re.compile(r"[^A-Z]")

# A lone capital X in a ruled cell comes back as the multiplication sign about
# as often as it comes back as the letter, and stripping non-letters would
# then leave nothing at all. Applied before that strip, never after.
_AXIS_SHAPES = {"×": "X", "✕": "X", "╳": "X", "Χ": "X", "Υ": "Y", "Ζ": "Z"}

# A point name: one or two characters, starting with a letter. Real sheets use
# a, b, c … and occasionally P1/A1. A pure number is never a point name — that
# is a coordinate that landed in the wrong column.
_POINT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]?$")

# A coordinate: optionally signed decimal. "0.0", "-71.9", "126.5".
_COORDINATE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

# Characters OCR commonly substitutes in this font when reading a lone glyph
# out of a ruled cell. Applied only to the point-name column, where the
# alternatives are known to be a short name rather than arbitrary text.
_POINT_NAME_FIXUPS = {"O": "0", "l": "1", "|": "1"}

# Grid shapes worth reading. A reference table has one name column plus at
# least two axis columns, and one row per point.
MIN_TABLE_COLUMNS = 3
MAX_TABLE_COLUMNS = 16
MIN_TABLE_ROWS = 3
MAX_TABLE_ROWS = 60

# At least two of X/Y/Z must be present. Sheets that dimension a flat part in
# two axes are still worth ballooning, and demanding all three would drop them.
MIN_AXIS_COLUMNS = 2


@dataclass(frozen=True)
class GridCell:
    """One lattice cell of a ruled table, in page coordinates."""

    box: LayoutBox
    row: int
    column: int


@dataclass(frozen=True)
class GridTable:
    """A ruled table recovered as a row/column lattice."""

    bbox: LayoutBox
    cells: tuple[GridCell, ...]
    row_count: int
    column_count: int

    def cell(self, row: int, column: int) -> GridCell | None:
        for item in self.cells:
            if item.row == row and item.column == column:
                return item
        return None


@dataclass(frozen=True)
class ReferencePoint:
    """One point name and the coordinates printed against it."""

    symbol: str
    """The name as it is printed in the drawing, e.g. "a"."""

    coordinates: tuple[tuple[str, str], ...]
    """Axis letter and its value, in column order: (("X", "0.0"), …)."""

    name_box: LayoutBox
    """The point-name cell, which also supplies the glyph template."""

    row_box: LayoutBox
    """The whole row, so the UI can highlight it on the sheet."""


@dataclass(frozen=True)
class ReferenceTable:
    """A located coordinate table and the points it defines."""

    bbox: LayoutBox
    axes: tuple[str, ...]
    name_header: str
    header_box: LayoutBox
    points: tuple[ReferencePoint, ...]

    value_boxes: tuple[LayoutBox, ...] = ()
    """Every coordinate cell. These hold the digits, signs and decimal points
    that a point name must not be confused with, which makes them the
    counter-templates for the marker search."""


def _cluster_coordinates(values: Iterable[int], tolerance: int) -> list[int]:
    """Collapse near-identical line coordinates to one representative each."""

    ordered = sorted(values)
    clusters: list[int] = []
    for value in ordered:
        if clusters and value - clusters[-1] <= tolerance:
            continue
        clusters.append(value)
    return clusters


def _lattice_from_cells(
    cells: Sequence[LayoutBox],
    *,
    tolerance: int,
) -> GridTable | None:
    """Index a connected cell group by its distinct row and column bands."""

    if not cells:
        return None

    tops = _cluster_coordinates((cell.y for cell in cells), tolerance)
    lefts = _cluster_coordinates((cell.x for cell in cells), tolerance)
    if len(tops) < MIN_TABLE_ROWS or len(lefts) < MIN_TABLE_COLUMNS:
        return None
    if len(tops) > MAX_TABLE_ROWS or len(lefts) > MAX_TABLE_COLUMNS:
        return None

    def band(value: int, bands: Sequence[int]) -> int:
        best = 0
        for index, start in enumerate(bands):
            if value >= start - tolerance:
                best = index
        return best

    indexed: dict[tuple[int, int], GridCell] = {}
    for cell in cells:
        row = band(cell.y, tops)
        column = band(cell.x, lefts)
        # A merged cell spanning two bands is indexed once, at its top-left.
        indexed.setdefault((row, column), GridCell(cell, row, column))

    x0 = min(cell.x for cell in cells)
    y0 = min(cell.y for cell in cells)
    x1 = max(cell.x1 for cell in cells)
    y1 = max(cell.y1 for cell in cells)
    return GridTable(
        bbox=LayoutBox(x0, y0, x1 - x0, y1 - y0),
        cells=tuple(
            indexed[key] for key in sorted(indexed, key=lambda k: (k[0], k[1]))
        ),
        row_count=len(tops),
        column_count=len(lefts),
    )


def _group_connected_cells(
    cells: Sequence[LayoutBox],
    *,
    tolerance: int,
) -> list[list[LayoutBox]]:
    """Split loose cells into separate tables by shared edges."""

    parent = list(range(len(cells)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def touching(left: LayoutBox, right: LayoutBox) -> bool:
        overlap_x = min(left.x1, right.x1) - max(left.x, right.x)
        overlap_y = min(left.y1, right.y1) - max(left.y, right.y)
        side_by_side = (
            min(abs(left.x1 - right.x), abs(right.x1 - left.x)) <= tolerance
            and overlap_y >= 0.5 * min(left.height, right.height)
        )
        stacked = (
            min(abs(left.y1 - right.y), abs(right.y1 - left.y)) <= tolerance
            and overlap_x >= 0.5 * min(left.width, right.width)
        )
        return side_by_side or stacked

    for left_index, left in enumerate(cells):
        for right_index in range(left_index + 1, len(cells)):
            if touching(left, cells[right_index]):
                root_left = find(left_index)
                root_right = find(right_index)
                if root_left != root_right:
                    parent[root_right] = root_left

    groups: dict[int, list[LayoutBox]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault(find(index), []).append(cell)
    return list(groups.values())


def extract_grid_tables(image: Image.Image) -> list[GridTable]:
    """Recover every ruled table on the page as a row/column lattice.

    Deliberately separate from :func:`page_layout.detect_table_masks`, which
    merges its cells into one rectangle per table.  Reading a reference table
    needs the opposite: the individual cells, because each one is cropped and
    recognised on its own.
    """

    width, height = image.size
    if width < 64 or height < 64:
        return []

    ink = _ink_map(image)
    horizontal = _extract_axis_lines(ink, horizontal=True)
    vertical = _extract_axis_lines(ink, horizontal=False)
    tolerance = max(2, int(round(min(width, height) / 500)))

    cells = _detect_grid_cells(
        horizontal,
        vertical,
        page_width=width,
        page_height=height,
        tolerance=tolerance,
    )
    tables: list[GridTable] = []
    for group in _group_connected_cells(cells, tolerance=tolerance * 2):
        lattice = _lattice_from_cells(group, tolerance=tolerance * 2)
        if lattice is not None:
            tables.append(lattice)
    return sorted(tables, key=lambda table: (table.bbox.y, table.bbox.x))


def assign_text_to_cells(
    table: GridTable,
    boxes: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int], str]:
    """Place recognized boxes into the cells that contain their centres.

    Boxes are keyed to a cell by centre rather than overlap: the recogniser
    routinely returns a box a pixel or two wider than the printed glyphs, and
    an overlap rule would let a value bleed into the neighbouring column.
    """

    contents: dict[tuple[int, int], list[tuple[float, float, str]]] = {}
    for box in boxes:
        text = str(box.get("text", "")).strip()
        if not text:
            continue
        centre_x = float(box["x"]) + float(box["w"]) / 2
        centre_y = float(box["y"]) + float(box["h"]) / 2
        for cell in table.cells:
            if (
                cell.box.x <= centre_x <= cell.box.x1
                and cell.box.y <= centre_y <= cell.box.y1
            ):
                contents.setdefault((cell.row, cell.column), []).append(
                    (centre_y, centre_x, text)
                )
                break

    return {
        key: " ".join(part for _, _, part in sorted(items))
        for key, items in contents.items()
    }


def _union(boxes: Sequence[LayoutBox]) -> LayoutBox:
    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    return LayoutBox(x0, y0, x1 - x0, y1 - y0)


def _unify_name_case(points: Sequence[ReferencePoint]) -> tuple[ReferencePoint, ...]:
    """Fold the odd misread capital into the case the table is printed in.

    Lone ``c`` and ``k`` in this drafting font read back as ``C`` and ``K``
    often enough to matter, and a name shown as "C" beside a drawing that
    prints "c" looks like a different point.  The table's own majority decides:
    the glyph templates come from the cell images, so recognition case never
    affects which markers are found — only what the row is labelled.
    """

    letters = [point.symbol for point in points if point.symbol.isalpha()]
    if not letters:
        return tuple(points)
    lower = sum(1 for name in letters if name.islower())
    if lower * 2 <= len(letters):
        return tuple(points)
    return tuple(
        point
        if not point.symbol.isalpha()
        else ReferencePoint(
            symbol=point.symbol.lower(),
            coordinates=point.coordinates,
            name_box=point.name_box,
            row_box=point.row_box,
        )
        for point in points
    )


def _axis_letter(text: str) -> str | None:
    """Return the axis a header cell names, or None if it names something else."""

    shaped = "".join(_AXIS_SHAPES.get(ch, ch) for ch in (text or "").upper())
    stripped = _AXIS_NOISE.sub("", shaped)
    return stripped if stripped in AXIS_LETTERS else None


def _clean_point_name(text: str) -> str | None:
    """Return a usable point name from a name cell, or None."""

    candidate = (text or "").strip().strip(".,:;")
    if not candidate:
        return None
    if len(candidate) > 2:
        return None
    candidate = "".join(_POINT_NAME_FIXUPS.get(ch, ch) for ch in candidate)
    return candidate if _POINT_NAME.match(candidate) else None


def _is_coordinate(text: str) -> bool:
    # Readings arrive with the odd stray space inside a number ("12 6.5") when
    # the recogniser splits a cell, so spaces are closed before matching.
    return bool(_COORDINATE.match((text or "").replace(" ", "")))


def _normalize_coordinate(text: str) -> str:
    return (text or "").replace(" ", "")


def _find_axis_header(
    table: GridTable,
    cell_text: Mapping[tuple[int, int], str],
) -> tuple[int, dict[str, int]] | None:
    """Locate the header row and which column carries each axis.

    Scans every row rather than assuming the first: sheets print a merged
    caption band above the header often enough that row 0 is frequently the
    caption, not the axis letters.
    """

    for row in range(table.row_count - 1):
        axes: dict[str, int] = {}
        for column in range(table.column_count):
            axis = _axis_letter(cell_text.get((row, column), ""))
            # The first column to read as an axis letter wins it; a repeat
            # further right is a different table's column bleeding in.
            if axis and axis not in axes:
                axes[axis] = column
        if len(axes) < MIN_AXIS_COLUMNS:
            continue
        # The point-name column sits to the left of every axis column.
        if min(axes.values()) < 1:
            continue
        return row, axes
    return None


def identify_reference_table(
    tables: Sequence[GridTable],
    cell_text_by_table: Sequence[Mapping[tuple[int, int], str]],
) -> ReferenceTable | None:
    """Pick the coordinate table out of every ruled table on the sheet.

    A table qualifies on its header alone — two or more axis letters in
    separate columns with a name column to their left — and then on its body
    actually reading as named points with numeric coordinates.  When a sheet
    has more than one qualifying table the one defining the most points wins.
    """

    best: ReferenceTable | None = None
    for table, cell_text in zip(tables, cell_text_by_table):
        header = _find_axis_header(table, cell_text)
        if header is None:
            continue
        header_row, axes = header
        name_column = min(axes.values()) - 1

        own_columns = {name_column, *axes.values()}
        points: list[ReferencePoint] = []
        value_boxes: list[LayoutBox] = []
        for row in range(header_row + 1, table.row_count):
            name_cell = table.cell(row, name_column)
            if name_cell is None:
                continue
            symbol = _clean_point_name(cell_text.get((row, name_column), ""))
            if symbol is None:
                continue
            coordinates: list[tuple[str, str]] = []
            for axis in AXIS_LETTERS:
                column = axes.get(axis)
                if column is None:
                    continue
                raw = cell_text.get((row, column), "")
                if not _is_coordinate(raw):
                    continue
                coordinates.append((axis, _normalize_coordinate(raw)))
            # A name with no coordinate beside it is a stray read, not a point.
            if len(coordinates) < MIN_AXIS_COLUMNS:
                continue
            row_cells = [
                cell for cell in table.cells
                if cell.row == row and cell.column in own_columns
            ]
            value_boxes.extend(
                cell.box for cell in row_cells if cell.column != name_column
            )
            points.append(
                ReferencePoint(
                    symbol=symbol,
                    coordinates=tuple(coordinates),
                    name_box=name_cell.box,
                    row_box=_union([cell.box for cell in row_cells]),
                )
            )

        if len(points) < MIN_TABLE_ROWS - 1:
            continue
        named = _unify_name_case(points)

        header_box = _union(
            [
                cell.box
                for cell in table.cells
                if cell.row == header_row and cell.column in own_columns
            ]
        )
        candidate = ReferenceTable(
            # Only this table's own cells: a reference table printed against
            # the title block shares its rules with it, so the two come back
            # as one lattice and the lattice bbox would cover both.
            bbox=_union([header_box, *(point.row_box for point in named)]),
            axes=tuple(axis for axis in AXIS_LETTERS if axis in axes),
            name_header=cell_text.get((header_row, name_column), "").strip(),
            header_box=header_box,
            points=named,
            value_boxes=tuple(value_boxes),
        )
        if best is None or len(candidate.points) > len(best.points):
            best = candidate
    return best
