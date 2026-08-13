"""Bounded whole-page detector planning and strict candidate deduplication.

Whole-page auto-ballooning has a dedicated, detector-only route.  Page layout
analysis now supplies four to eight adaptive overlapping panels; each panel is
inspected at 0 and 90 degrees before candidates are deduplicated globally.

This module owns only page orchestration geometry.  It does not perform model
inference, recognition, or value filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf
from typing import Mapping, Sequence

from detection_passes import DetectionTile


PAGE_SCAN_ROTATIONS_CW = (0, 90)
PAGE_SCAN_MAX_TILES = 8
PAGE_SCAN_MAX_DETECTOR_CALLS = 16
# Preserve approximately the text scale that worked in selection scans without
# restoring the selection cascade. The shared detector still caps at 2400px.
PAGE_SCAN_DETECTOR_MIN_LONG_EDGE = 2000
PAGE_SCAN_RECOGNITION_BATCH_SIZE = 16
EXISTING_VALUE_OVERLAP_THRESHOLD = 0.55


def bbox_overlap_fraction(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    """Intersection as a fraction of the smaller box."""

    x0 = max(float(left["x"]), float(right["x"]))
    y0 = max(float(left["y"]), float(right["y"]))
    x1 = min(
        float(left["x"]) + float(left["width"]),
        float(right["x"]) + float(right["width"]),
    )
    y1 = min(
        float(left["y"]) + float(left["height"]),
        float(right["y"]) + float(right["height"]),
    )
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = max(float(left["width"]) * float(left["height"]), 1.0)
    right_area = max(float(right["width"]) * float(right["height"]), 1.0)
    return intersection / min(left_area, right_area)


def overlaps_existing_value(
    bbox: Mapping[str, float],
    existing_value_boxes: Sequence[Mapping[str, float]],
    *,
    threshold: float = EXISTING_VALUE_OVERLAP_THRESHOLD,
) -> bool:
    """Return whether a detected object is already represented by a balloon."""

    return any(
        bbox_overlap_fraction(bbox, existing) >= threshold
        for existing in existing_value_boxes
    )


@dataclass(frozen=True)
class PageCandidate:
    """One detector-pass object restored to whole-page coordinates."""

    bbox: dict[str, float]
    tile_index: int
    tile: DetectionTile
    boundary_clearance: float
    boundary_review: bool
    detection_confidence: float = 0.0
    pass_index: int = 0
    rotation_cw: int = 0
    polygon: tuple[tuple[float, float], ...] = ()
    candidate_id: str = ""


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
    pass_index: int = 0,
    rotation_cw: int = 0,
    polygon: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
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
    local_polygon = polygon or (
        (local_x, local_y),
        (local_x + local_width, local_y),
        (local_x + local_width, local_y + local_height),
        (local_x, local_y + local_height),
    )
    page_polygon: list[tuple[float, float]] = []
    for point in local_polygon:
        if len(point) < 2:
            continue
        point_x = max(
            0.0,
            min(float(page_width), tile.x + float(point[0])),
        )
        point_y = max(
            0.0,
            min(float(page_height), tile.y + float(point[1])),
        )
        page_polygon.append((round(point_x, 1), round(point_y, 1)))
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
        pass_index=pass_index,
        rotation_cw=rotation_cw,
        polygon=tuple(page_polygon),
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
    """Remove only repeated same-size views of the same atomic object."""

    preferred_first = sorted(
        candidates,
        key=_candidate_preference,
        reverse=True,
    )
    accepted: list[PageCandidate] = []
    for candidate in preferred_first:
        duplicate = any(
            strict_same_object(existing.bbox, candidate.bbox)
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


def assign_candidate_ids(
    candidates: list[PageCandidate],
) -> list[PageCandidate]:
    """Give deduplicated page objects stable, page-order identifiers."""

    return [
        replace(candidate, candidate_id=f"C{index:04d}")
        for index, candidate in enumerate(candidates, start=1)
    ]
