"""
Group detected text boxes into individual dimensions.

PaddleOCR detection returns one box per text region. A single CAD dimension
(e.g. a vertical ``Ø215,37 ±0.05``) often detects as 2–3 stacked fragments,
while separate dimensions sit far apart (different columns / rows). We merge
fragments that belong together with a simple proximity rule: inflate every box
by a margin proportional to its own size, then union-find any boxes whose
inflated rectangles overlap.

This stays orientation-agnostic — vertical Ø dimensions and a horizontal angle
in the same selection cluster correctly without special-casing either.
"""

from __future__ import annotations

from collections import defaultdict
from math import floor
from statistics import median
from typing import Any, Iterator

# Each box is a dict with at least: x, y, w, h. Extra keys (text, conf) ride along.
Box = dict[str, Any]
Rect = tuple[float, float, float, float]

_MIN_SPATIAL_CELL = 32.0
_MAX_SPATIAL_CELL = 256.0
_MAX_GRID_CELLS_PER_RECT = 256


def _normalise_rect(rect: Rect) -> Rect:
    """Return a left-to-right, top-to-bottom rectangle."""

    x0, y0, x1, y1 = (float(value) for value in rect)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _spatial_cell_size(rects: list[Rect]) -> float:
    """Choose a stable grid size from the typical candidate extent."""

    extents = [
        max(abs(rect[2] - rect[0]), abs(rect[3] - rect[1]), 1.0)
        for rect in rects
    ]
    if not extents:
        return _MIN_SPATIAL_CELL
    return max(
        _MIN_SPATIAL_CELL,
        min(_MAX_SPATIAL_CELL, float(median(extents)) * 2.0),
    )


class _SpatialRectIndex:
    """Exact rectangle-neighbour lookup backed by a bounded uniform grid.

    Very large drawing-geometry boxes are kept in a small fallback set instead
    of being copied into thousands of cells. Queries still test them exactly,
    so this changes comparison cost without changing grouping results.
    """

    def __init__(self, cell_size: float) -> None:
        self._cell_size = max(float(cell_size), 1.0)
        self._buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
        self._large: set[int] = set()
        self._rects: dict[int, Rect] = {}

    def _cell_bounds(self, rect: Rect) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = _normalise_rect(rect)
        return (
            floor(x0 / self._cell_size),
            floor(y0 / self._cell_size),
            floor(x1 / self._cell_size),
            floor(y1 / self._cell_size),
        )

    @staticmethod
    def _cell_count(bounds: tuple[int, int, int, int]) -> int:
        gx0, gy0, gx1, gy1 = bounds
        return (gx1 - gx0 + 1) * (gy1 - gy0 + 1)

    def insert(self, key: int, rect: Rect) -> None:
        """Insert or expand one indexed rectangle.

        Updated keys may remain referenced by an old bucket. Query performs an
        exact final overlap check, so a stale bucket can add only a harmless
        false candidate and avoids costly grid deletion during fragment merges.
        """

        normalised = _normalise_rect(rect)
        self._rects[key] = normalised
        bounds = self._cell_bounds(normalised)
        if self._cell_count(bounds) > _MAX_GRID_CELLS_PER_RECT:
            self._large.add(key)
            return

        gx0, gy0, gx1, gy1 = bounds
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                self._buckets[(gx, gy)].add(key)

    def query(self, rect: Rect) -> list[int]:
        """Return indexed keys whose current rectangles overlap ``rect``."""

        normalised = _normalise_rect(rect)
        bounds = self._cell_bounds(normalised)
        if self._cell_count(bounds) > _MAX_GRID_CELLS_PER_RECT:
            candidates = set(self._rects)
        else:
            candidates = set(self._large)
            gx0, gy0, gx1, gy1 = bounds
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    candidates.update(self._buckets.get((gx, gy), ()))

        return sorted(
            key
            for key in candidates
            if _overlap(normalised, self._rects[key])
        )


def _spatial_overlap_pairs(rects: list[Rect]) -> Iterator[tuple[int, int]]:
    """Yield every exactly-overlapping pair without an all-pairs scan."""

    index = _SpatialRectIndex(_spatial_cell_size(rects))
    for right, rect in enumerate(rects):
        for left in index.query(rect):
            yield left, right
        index.insert(right, rect)


def _inflate(box: Box, margin_ratio: float) -> tuple[float, float, float, float]:
    """
    Grow a box anisotropically: more along its own reading axis, less across it.

    A dimension's fragments stack along its primary axis (vertical text grows in
    y, horizontal text in x), so we inflate generously there to join them. The
    perpendicular axis is inflated only slightly so neighbouring *parallel*
    dimensions (separate columns / rows) stay in distinct clusters.
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    # Base the margin on the SHORT side (~character size), not the long side, so
    # a wide blob (e.g. a horizontal angle) doesn't reach sideways into adjacent
    # columns. Inflate more along the reading axis to join split fragments.
    base = min(w, h)
    primary = base * margin_ratio  # along the reading axis
    perp = base * margin_ratio * 0.4  # across it
    floor = 3.0
    if h >= w:  # vertical / tall text — reading axis is y
        my = max(primary, floor)
        mx = max(perp, floor)
    else:  # wide text — reading axis is x
        mx = max(primary, floor)
        my = max(perp, floor)
    return (x - mx, y - my, x + w + mx, y + h + my)


def _overlap(a: Rect, b: Rect) -> bool:
    """Axis-aligned rectangle overlap test (touching edges count as overlap)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _reading_axis(box: Box) -> str:
    """Primary reading axis from aspect ratio: ``vertical``, ``horizontal``, or ``square``."""
    w, h = box["w"], box["h"]
    if h >= w * 1.35:
        return "vertical"
    if w >= h * 1.35:
        return "horizontal"
    return "square"


def _raw_rect(box: Box) -> tuple[float, float, float, float]:
    return (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])


def _axis_gap(a: Box, b: Box, axis: str) -> float:
    """Gap between two boxes along their shared reading axis (0 = overlap)."""
    if axis == "vertical":
        a0, a1 = a["y"], a["y"] + a["h"]
        b0, b1 = b["y"], b["y"] + b["h"]
    else:
        a0, a1 = a["x"], a["x"] + a["w"]
        b0, b1 = b["x"], b["x"] + b["w"]
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def _should_merge(a: Box, b: Box, margin_ratio: float) -> bool:
    """
    Merge stacked fragments of one dimension, but keep perpendicular neighbours
    separate unless their *tight* boxes already overlap (same ink region).
    """
    oa, ob = _reading_axis(a), _reading_axis(b)
    if oa != ob and "square" not in (oa, ob):
        return _overlap(_raw_rect(a), _raw_rect(b))

    if oa == ob and oa in ("vertical", "horizontal"):
        char = min(min(a["w"], a["h"]), min(b["w"], b["h"]))
        gap = _axis_gap(a, b, oa)
        # Separate dimensions in one column have a larger gap than split
        # fragments of a single value (Ø / digits / tolerance).
        if gap > char * 1.65:
            return False

    return _overlap(_inflate(a, margin_ratio), _inflate(b, margin_ratio))


def _union_size(a: Box, b: Box) -> tuple[float, float]:
    x0 = min(a["x"], b["x"])
    y0 = min(a["y"], b["y"])
    x1 = max(a["x"] + a["w"], b["x"] + b["w"])
    y1 = max(a["y"] + a["h"], b["y"] + b["h"])
    return x1 - x0, y1 - y0


def drop_bridge_boxes(boxes: list[Box]) -> list[Box]:
    """
    Remove detection boxes that bridge two separate perpendicular columns.

    OCR/morphology sometimes proposes a wide horizontal box spanning the gap
    between two vertical dimension columns (or a tall box across two rows). Such
    a bridge overlaps ≥2 perpendicular boxes that do *not* overlap each other,
    and it fuses otherwise-distinct dimensions into one cluster — e.g. ``Ø174,07``
    and ``Ø175,32`` collapsing together. Distinct columns never share a real
    box, so any such bridge is spurious and is dropped.
    """
    if len(boxes) <= 2:
        return boxes

    raw_rects = [_raw_rect(box) for box in boxes]
    axes = [_reading_axis(box) for box in boxes]
    crossed: list[list[int]] = [[] for _ in boxes]
    for left, right in _spatial_overlap_pairs(raw_rects):
        left_axis = axes[left]
        right_axis = axes[right]
        if (
            left_axis == "square"
            or right_axis == "square"
            or left_axis == right_axis
        ):
            continue
        crossed[left].append(right)
        crossed[right].append(left)

    keep: list[Box] = []
    for i, b in enumerate(boxes):
        ax = axes[i]
        if ax == "square":
            keep.append(b)
            continue

        # Axis-aligned rectangles are pairwise-overlapping exactly when their
        # x intervals and y intervals each share a common point. This replaces
        # the former nested all-pairs test inside every candidate.
        neighbours = [raw_rects[index] for index in crossed[i]]
        bridge = len(neighbours) >= 2 and (
            max(rect[0] for rect in neighbours)
            > min(rect[2] for rect in neighbours)
            or max(rect[1] for rect in neighbours)
            > min(rect[3] for rect in neighbours)
        )
        if not bridge:
            keep.append(b)
    return keep


def cluster_boxes(
    boxes: list[Box],
    margin_ratio: float = 0.6,
    *,
    img_w: float = 0.0,
    img_h: float = 0.0,
) -> list[list[Box]]:
    """
    Group ``boxes`` into clusters via union-find on inflated rectangles.

    ``margin_ratio`` controls how aggressively neighbouring fragments merge:
    larger = fewer, bigger clusters. 0.6 reliably joins the stacked pieces of
    one vertical dimension while keeping adjacent columns separate.

    Returns a list of clusters (each a list of the original box dicts).
    """
    n = len(boxes)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    max_w = img_w * 0.34 if img_w else 0.0
    max_h = img_h * 0.42 if img_h else 0.0

    inflated = [_inflate(box, margin_ratio) for box in boxes]
    for i, j in _spatial_overlap_pairs(inflated):
        if not _should_merge(boxes[i], boxes[j], margin_ratio):
            continue
        if max_w and max_h:
            uw, uh = _union_size(boxes[i], boxes[j])
            if uw > max_w or uh > max_h:
                continue
        union(i, j)

    groups: dict[int, list[Box]] = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)

    return list(groups.values())


def union_bbox(cluster: list[Box]) -> dict[str, float]:
    """Tight axis-aligned bounding box covering every box in ``cluster``."""
    x0 = min(b["x"] for b in cluster)
    y0 = min(b["y"] for b in cluster)
    x1 = max(b["x"] + b["w"] for b in cluster)
    y1 = max(b["y"] + b["h"] for b in cluster)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _box_w(box: Box) -> float:
    return float(box.get("w", box.get("width", 0)))


def _box_h(box: Box) -> float:
    return float(box.get("h", box.get("height", 0)))


def _perp_overlap(a: Box, b: Box, axis: str) -> float:
    """Fraction of the smaller box overlapped on the perpendicular axis."""
    if axis == "vertical":
        a0, a1 = a["x"], a["x"] + _box_w(a)
        b0, b1 = b["x"], b["x"] + _box_w(b)
    else:
        a0, a1 = a["y"], a["y"] + _box_h(a)
        b0, b1 = b["y"], b["y"] + _box_h(b)
    ix0, ix1 = max(a0, b0), min(a1, b1)
    if ix1 <= ix0:
        return 0.0
    inter = ix1 - ix0
    short = min(a1 - a0, b1 - b0)
    return inter / max(short, 1.0)


def merge_fragment_clusters(
    clusters: list[list[Box]],
    *,
    max_fragment_area: float = 650.0,
    max_absorb_gap: float = 20.0,
) -> list[list[Box]]:
    """
    Glue tiny proposal boxes onto the nearest full dimension in the same column.

    Morphology / projection can leave Ø, digits, and tolerance as separate
    slivers; this merges them back before OCR runs once per value.
    """
    if len(clusters) <= 1:
        return clusters

    pools = [list(c) for c in clusters]
    areas = [union_bbox(p)["width"] * union_bbox(p)["height"] for p in pools]
    anchors = [i for i, a in enumerate(areas) if a >= max_fragment_area]
    if not anchors:
        return clusters

    anchor_set = set(anchors)
    anchor_rects: list[Rect] = []
    for index in anchors:
        anchor = union_bbox(pools[index])
        anchor_rects.append(
            (
                anchor["x"] - max_absorb_gap,
                anchor["y"] - max_absorb_gap,
                anchor["x"] + anchor["width"] + max_absorb_gap,
                anchor["y"] + anchor["height"] + max_absorb_gap,
            )
        )
    anchor_index = _SpatialRectIndex(_spatial_cell_size(anchor_rects))
    for index, rect in zip(anchors, anchor_rects):
        anchor_index.insert(index, rect)

    for i, area in enumerate(areas):
        if i in anchor_set or not pools[i]:
            continue
        frag_ub = union_bbox(pools[i])
        frag_axis = _reading_axis(pools[i][0])
        best: int | None = None
        best_gap = float("inf")
        frag_rect = (
            frag_ub["x"],
            frag_ub["y"],
            frag_ub["x"] + frag_ub["width"],
            frag_ub["y"] + frag_ub["height"],
        )
        for j in anchor_index.query(frag_rect):
            if not pools[j]:
                continue
            anchor_ub = union_bbox(pools[j])
            anchor_axis = _reading_axis(pools[j][0])
            if frag_axis != anchor_axis and frag_axis != "square":
                continue
            axis = anchor_axis if anchor_axis != "square" else frag_axis
            if axis == "square":
                continue
            if _perp_overlap(frag_ub, anchor_ub, axis) < 0.35:
                continue
            gap = _axis_gap(
                {
                    "x": frag_ub["x"],
                    "y": frag_ub["y"],
                    "w": frag_ub["width"],
                    "h": frag_ub["height"],
                },
                {
                    "x": anchor_ub["x"],
                    "y": anchor_ub["y"],
                    "w": anchor_ub["width"],
                    "h": anchor_ub["height"],
                },
                axis,
            )
            if gap <= max_absorb_gap and gap < best_gap:
                best_gap = gap
                best = j
        if best is not None:
            pools[best].extend(pools[i])
            pools[i] = []
            updated = union_bbox(pools[best])
            anchor_index.insert(
                best,
                (
                    updated["x"] - max_absorb_gap,
                    updated["y"] - max_absorb_gap,
                    updated["x"] + updated["width"] + max_absorb_gap,
                    updated["y"] + updated["height"] + max_absorb_gap,
                ),
            )

    return [p for p in pools if p]


def split_mixed_clusters(clusters: list[list[Box]]) -> list[list[Box]]:
    """
    Break clusters that still hold both vertical and horizontal dimensions.

    This is a safety net when morphology bridged perpendicular text before
    ``_refine_blob`` could split them.
    """
    out: list[list[Box]] = []
    for cluster in clusters:
        axes = {_reading_axis(b) for b in cluster}
        perpendicular = "vertical" in axes and "horizontal" in axes
        if not perpendicular or len(cluster) <= 1:
            out.append(cluster)
            continue
        for axis in ("vertical", "horizontal", "square"):
            group = [b for b in cluster if _reading_axis(b) == axis]
            if group:
                out.append(group)
    return out


def _ubbox_overlap_frac(a: dict[str, float], b: dict[str, float]) -> float:
    """Intersection of two union-bboxes as a fraction of the smaller area."""
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = ax0 + a["width"], ay0 + a["height"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = bx0 + b["width"], by0 + b["height"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1.0)
    area_b = max((bx1 - bx0) * (by1 - by0), 1.0)
    return inter / min(area_a, area_b)


def merge_overlapping_clusters(
    clusters: list[list[Box]], *, min_overlap: float = 0.1
) -> list[list[Box]]:
    """
    Union clusters whose bounding boxes overlap — fragments of one dimension.

    Distinct dimensions occupy separate columns/rows and never overlap, so any
    meaningful bbox overlap means the pieces belong together. This re-joins a
    value that ``split_mixed_clusters`` broke onto a perpendicular axis (e.g. a
    vertical ``Ø175,32`` and its wider ``REF.`` tag) so it is read as one box.
    """
    n = len(clusters)
    if n <= 1:
        return clusters
    boxes = [union_bbox(c) for c in clusters]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rects = [
        (
            box["x"],
            box["y"],
            box["x"] + box["width"],
            box["y"] + box["height"],
        )
        for box in boxes
    ]
    for i, j in _spatial_overlap_pairs(rects):
        if _ubbox_overlap_frac(boxes[i], boxes[j]) >= min_overlap:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    groups: dict[int, list[Box]] = {}
    for i, cluster in enumerate(clusters):
        groups.setdefault(find(i), []).extend(cluster)
    return list(groups.values())


def order_clusters(clusters: list[list[Box]]) -> list[list[Box]]:
    """
    Sort clusters into a stable reading order: top-to-bottom, then left-to-right.

    Uses each cluster's union-box top-left corner with a coarse row band so that
    roughly-aligned dimensions read left-to-right before dropping to the next row.
    """
    boxed = [(union_bbox(c), c) for c in clusters]
    if not boxed:
        return []
    avg_h = sum(b["height"] for b, _ in boxed) / len(boxed)
    band = max(avg_h, 1.0)
    return [
        c
        for _, c in sorted(
            boxed, key=lambda bc: (round(bc[0]["y"] / band), bc[0]["x"])
        )
    ]
