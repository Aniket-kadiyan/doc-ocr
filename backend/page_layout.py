"""Page-level layout analysis for whole-page and section auto-balloon scans.

The layout stage is deliberately independent from OCR.  It detects tabular
regions from repeated horizontal/vertical line intersections, creates
temporary overlapping processing panels from whitespace and ink density, and
provides page-coordinate geometry for the live debug overlay.

Nothing in this module decides whether a recognized value is technically
meaningful.  Table masks are a whole-page hard exclusion; selected-section
pixels remain raw and use only the light value gate in :mod:`page_value_filters`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, Literal, Sequence

import numpy as np
from PIL import Image


SCAN_DEBUG_OVERLAY_ENABLED = True

LAYOUT_MIN_PANELS = 4
LAYOUT_MAX_PANELS = 8
LAYOUT_PANEL_OVERLAP_RATIO = 0.04
LAYOUT_PANEL_OVERLAP_MIN = 24
LAYOUT_PANEL_OVERLAP_MAX = 80

TABLE_MASK_PADDING_RATIO = 0.002


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
            "table_masks": (
                [box.to_dict() for box in self.table_masks]
                if scope_kind == "page"
                else []
            ),
            "panels": (
                [
                    panel.to_dict(
                        state=states.get(panel.panel_id, "pending")
                    )
                    for panel in self.panels
                ]
                if scope_kind == "page"
                else []
            ),
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

            boundaries = sorted(
                {
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


def _retain_repeated_cell_groups(
    cells: Sequence[LayoutBox],
    *,
    tolerance: int,
) -> list[LayoutBox]:
    """Discard isolated drawing rectangles while retaining repeated grids."""

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
    return [
        cell
        for index, cell in enumerate(cells)
        if counts.get(find(index), 0) >= 2
    ]


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


def detect_table_masks(image: Image.Image) -> tuple[LayoutBox, ...]:
    """Detect every repeated-cell grid that should be masked in page scans."""

    width, height = image.size
    if width < 64 or height < 64:
        return ()

    ink = _ink_map(image)
    horizontal = _extract_axis_lines(ink, horizontal=True)
    vertical = _extract_axis_lines(ink, horizontal=False)
    intersection_tolerance = max(2, int(round(min(width, height) / 500)))
    padding = max(
        2,
        int(round(min(width, height) * TABLE_MASK_PADDING_RATIO)),
    )

    cells = _detect_grid_cells(
        horizontal,
        vertical,
        page_width=width,
        page_height=height,
        tolerance=intersection_tolerance,
    )
    repeated_cells = _retain_repeated_cell_groups(
        cells,
        tolerance=intersection_tolerance * 2,
    )
    boxes = _discard_mostly_contained_boxes(
        _merge_rectangular_cells(
            repeated_cells,
            tolerance=intersection_tolerance * 2,
        )
    )
    padded: list[LayoutBox] = []
    for box in boxes:
        x0 = max(0, box.x - padding)
        y0 = max(0, box.y - padding)
        x1 = min(width, box.x1 + padding)
        y1 = min(height, box.y1 + padding)
        padded.append(LayoutBox(x0, y0, x1 - x0, y1 - y0))
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
    masks = detect_table_masks(rgb)
    panels, overlaps = build_adaptive_panels(rgb, masks)
    return PageLayout(
        width=rgb.width,
        height=rgb.height,
        table_masks=masks,
        panels=panels,
        overlaps=overlaps,
    )


def crop_section(
    image: Image.Image,
    scope_bbox: dict[str, float],
    *,
    padding: int = 20,
) -> tuple[Image.Image, tuple[int, int], LayoutBox]:
    """Return a padded raw section crop in the page coordinate system.

    The returned origin maps crop-local coordinates—including padding—back to
    the full page.  Section mode is deliberately user-bounded and layout-mask
    free, so table-like pixels are retained exactly as selected. A later light
    text gate may still discard pure letters, symbols, and prose.
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
