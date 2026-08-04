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

from typing import Any

# Each box is a dict with at least: x, y, w, h. Extra keys (text, conf) ride along.
Box = dict[str, Any]


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


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
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
    keep: list[Box] = []
    for i, b in enumerate(boxes):
        ax = _reading_axis(b)
        if ax == "square":
            keep.append(b)
            continue
        crossed = [
            o
            for j, o in enumerate(boxes)
            if j != i
            and _reading_axis(o) not in (ax, "square")
            and _overlap(_raw_rect(b), _raw_rect(o))
        ]
        bridge = any(
            not _overlap(_raw_rect(crossed[m]), _raw_rect(crossed[n]))
            for m in range(len(crossed))
            for n in range(m + 1, len(crossed))
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

    for i in range(n):
        for j in range(i + 1, n):
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

    for i, area in enumerate(areas):
        if i in anchors or not pools[i]:
            continue
        frag_ub = union_bbox(pools[i])
        frag_axis = _reading_axis(pools[i][0])
        best: int | None = None
        best_gap = float("inf")
        for j in anchors:
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

    for i in range(n):
        for j in range(i + 1, n):
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
