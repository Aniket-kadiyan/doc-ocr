"""Whole-page tiling and strict cross-tile candidate deduplication.

Whole-page auto-ballooning deliberately treats a drawing as a set of
overlapping selection-sized scans.  This module owns only page orchestration
geometry; it does not perform detection or OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf
from typing import Any

from detection_passes import DetectionTile


PAGE_SCAN_TILE_MIN_WIDTH = 700
PAGE_SCAN_TILE_MIN_HEIGHT = 500
PAGE_SCAN_TILE_MAX_WIDTH = 2000
PAGE_SCAN_TILE_MAX_HEIGHT = 1400
PAGE_SCAN_TILE_OVERLAP_X = 160
PAGE_SCAN_TILE_OVERLAP_Y = 120


@dataclass(frozen=True)
class PageCandidate:
    """One tile-local object restored to whole-page coordinates."""

    bbox: dict[str, float]
    tile_index: int
    tile: DetectionTile
    boundary_clearance: float
    boundary_review: bool
    detection_confidence: float = 0.0


def _axis_starts(
    length: int,
    *,
    max_edge: int,
    overlap: int,
) -> list[int]:
    """Return balanced starts with at least the requested overlap."""

    if length <= max_edge:
        return [0]

    stride = max_edge - overlap
    tile_count = ceil((length - max_edge) / stride) + 1
    travel = length - max_edge
    return [
        int(round(index * travel / max(tile_count - 1, 1)))
        for index in range(tile_count)
    ]


def build_page_tiles(
    image_or_size: Any,
    *,
    max_edge: int | None = None,
    overlap: int | None = None,
) -> list[DetectionTile]:
    """Cover a page with balanced, overlapping selection-sized tiles.

    Production defaults split an ordinary drawing into approximately four
    selections.  Tile dimensions grow with high-resolution sources until the
    bounded maximum is reached. ``max_edge``/``overlap`` remain explicit test
    and tuning overrides for a square grid.
    """

    width, height = (
        image_or_size.size
        if hasattr(image_or_size, "size")
        else image_or_size
    )
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("Page dimensions must be positive")

    if max_edge is not None:
        if max_edge < 64:
            raise ValueError("Page tile edge must be at least 64 pixels")
        overlap_x = overlap_y = (
            PAGE_SCAN_TILE_OVERLAP_X if overlap is None else overlap
        )
        tile_width = tile_height = max_edge
    else:
        if overlap is not None:
            raise ValueError("overlap requires an explicit max_edge")
        overlap_x = PAGE_SCAN_TILE_OVERLAP_X
        overlap_y = PAGE_SCAN_TILE_OVERLAP_Y
        tile_width = min(
            PAGE_SCAN_TILE_MAX_WIDTH,
            max(
                PAGE_SCAN_TILE_MIN_WIDTH,
                ceil((width + overlap_x) / 2),
            ),
        )
        tile_height = min(
            PAGE_SCAN_TILE_MAX_HEIGHT,
            max(
                PAGE_SCAN_TILE_MIN_HEIGHT,
                ceil((height + overlap_y) / 2),
            ),
        )

    if overlap_x < 0 or overlap_x >= tile_width:
        raise ValueError("Horizontal page overlap must be within the tile width")
    if overlap_y < 0 or overlap_y >= tile_height:
        raise ValueError("Vertical page overlap must be within the tile height")

    x_starts = _axis_starts(
        width,
        max_edge=tile_width,
        overlap=overlap_x,
    )
    y_starts = _axis_starts(
        height,
        max_edge=tile_height,
        overlap=overlap_y,
    )
    return [
        DetectionTile(
            x=x,
            y=y,
            width=min(tile_width, width - x),
            height=min(tile_height, height - y),
        )
        for y in y_starts
        for x in x_starts
    ]


def _candidate_boundary_clearance(
    bbox: dict[str, float],
    tile: DetectionTile,
    *,
    page_size: tuple[int, int],
) -> float:
    """Distance to the nearest internal tile edge.

    Page exterior edges are ignored because no neighbouring tile can provide a
    more complete view there.
    """

    page_width, page_height = page_size
    local_x = float(bbox["x"])
    local_y = float(bbox["y"])
    width = float(bbox["width"])
    height = float(bbox["height"])
    clearances: list[float] = []

    if tile.x > 0:
        clearances.append(local_x)
    if tile.y > 0:
        clearances.append(local_y)
    if tile.x + tile.width < page_width:
        clearances.append(float(tile.width) - (local_x + width))
    if tile.y + tile.height < page_height:
        clearances.append(float(tile.height) - (local_y + height))

    return max(0.0, min(clearances)) if clearances else inf


def map_tile_candidate(
    bbox: dict[str, float],
    tile: DetectionTile,
    *,
    tile_index: int,
    page_size: tuple[int, int],
    detection_confidence: float = 0.0,
) -> PageCandidate | None:
    """Clip and restore one tile-local object to whole-page coordinates."""

    page_width, page_height = page_size
    local_x = float(bbox["x"])
    local_y = float(bbox["y"])
    local_width = float(bbox["width"])
    local_height = float(bbox["height"])
    if local_width <= 0 or local_height <= 0:
        return None

    page_x0 = max(0.0, min(float(page_width), tile.x + local_x))
    page_y0 = max(0.0, min(float(page_height), tile.y + local_y))
    page_x1 = max(
        0.0,
        min(float(page_width), tile.x + local_x + local_width),
    )
    page_y1 = max(
        0.0,
        min(float(page_height), tile.y + local_y + local_height),
    )
    if page_x1 - page_x0 < 1.0 or page_y1 - page_y0 < 1.0:
        return None

    clearance = _candidate_boundary_clearance(
        bbox,
        tile,
        page_size=page_size,
    )
    boundary_margin = max(8.0, min(local_width, local_height) * 0.5)
    return PageCandidate(
        bbox={
            "x": round(page_x0, 1),
            "y": round(page_y0, 1),
            "width": round(page_x1 - page_x0, 1),
            "height": round(page_y1 - page_y0, 1),
        },
        tile_index=tile_index,
        tile=tile,
        boundary_clearance=clearance,
        boundary_review=clearance < boundary_margin,
        detection_confidence=float(detection_confidence),
    )


def _intersection_over_union(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    x0 = max(float(left["x"]), float(right["x"]))
    y0 = max(float(left["y"]), float(right["y"]))
    x1 = min(
        float(left["x"] + left["width"]),
        float(right["x"] + right["width"]),
    )
    y1 = min(
        float(left["y"] + left["height"]),
        float(right["y"] + right["height"]),
    )
    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    left_area = max(float(left["width"] * left["height"]), 1.0)
    right_area = max(float(right["width"] * right["height"]), 1.0)
    return intersection / max(left_area + right_area - intersection, 1.0)


def strict_same_object(
    left: dict[str, float],
    right: dict[str, float],
    *,
    min_size_ratio: float = 0.65,
) -> bool:
    """Match only similar-size boxes at the same position.

    Containment alone is intentionally insufficient: a large box containing
    several dimensions must never suppress its individual children.
    """

    left_width = max(float(left["width"]), 1.0)
    left_height = max(float(left["height"]), 1.0)
    right_width = max(float(right["width"]), 1.0)
    right_height = max(float(right["height"]), 1.0)
    width_ratio = min(left_width, right_width) / max(left_width, right_width)
    height_ratio = min(left_height, right_height) / max(left_height, right_height)
    if width_ratio < min_size_ratio or height_ratio < min_size_ratio:
        return False

    left_center = (
        float(left["x"]) + left_width / 2,
        float(left["y"]) + left_height / 2,
    )
    right_center = (
        float(right["x"]) + right_width / 2,
        float(right["y"]) + right_height / 2,
    )
    centre_is_close = (
        abs(left_center[0] - right_center[0])
        <= 0.35 * max(left_width, right_width)
        and abs(left_center[1] - right_center[1])
        <= 0.35 * max(left_height, right_height)
    )
    return (
        _intersection_over_union(left, right) >= 0.45
        or centre_is_close
    )


def _candidate_preference(candidate: PageCandidate) -> tuple[float, ...]:
    area = float(candidate.bbox["width"] * candidate.bbox["height"])
    return (
        0.0 if candidate.boundary_review else 1.0,
        candidate.boundary_clearance,
        candidate.detection_confidence,
        area,
        -float(candidate.tile_index),
    )


def deduplicate_page_candidates(
    candidates: list[PageCandidate],
) -> list[PageCandidate]:
    """Remove only repeated cross-tile views of the same object."""

    preferred_first = sorted(
        candidates,
        key=_candidate_preference,
        reverse=True,
    )
    accepted: list[PageCandidate] = []
    for candidate in preferred_first:
        duplicate = any(
            existing.tile_index != candidate.tile_index
            and strict_same_object(existing.bbox, candidate.bbox)
            for existing in accepted
        )
        if not duplicate:
            accepted.append(candidate)

    return sorted(
        accepted,
        key=lambda candidate: (
            candidate.bbox["y"],
            candidate.bbox["x"],
            candidate.tile_index,
        ),
    )
