"""
Oriented-rectangle geometry for text that runs along a leader line.

Axis-aligned boxes describe upright CAD text well, but a callout drawn along a
45° leader has an axis-aligned hull far larger than its ink — two callouts on
*parallel* leaders then overlap heavily even though they never touch. Every
comparison between regions (dedupe, redundancy, containment) therefore works on
the tight **oriented** rectangle when a region carries one, and falls back to
its axis-aligned box otherwise.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

Point = tuple[float, float]
Polygon = list[Point]

_EPS = 1e-9


def rect_polygon(box: dict[str, float]) -> Polygon:
    """Corners of an axis-aligned ``{x, y, width, height}`` box."""
    x, y = float(box["x"]), float(box["y"])
    w, h = float(box["width"]), float(box["height"])
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def oriented_polygon(box: dict[str, float]) -> Polygon:
    """
    Corners of an oriented ``{x, y, width, height, rotation}`` rectangle.

    ``rotation`` is clockwise degrees (Konva's convention, y axis pointing
    down) about the rectangle's own top-left corner — the same convention
    ``OcrPipeline._oriented_box_from_rot`` emits.
    """
    x, y = float(box["x"]), float(box["y"])
    w, h = float(box["width"]), float(box["height"])
    a = math.radians(float(box.get("rotation", 0.0)))
    cos, sin = math.cos(a), math.sin(a)
    return [
        (x + lx * cos - ly * sin, y + lx * sin + ly * cos)
        for lx, ly in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))
    ]


def region_polygon(region: dict[str, Any]) -> Polygon:
    """Tightest polygon available for a region: oriented box, else its bbox."""
    oriented = region.get("oriented_box")
    if oriented:
        return oriented_polygon(oriented)
    return rect_polygon(region["bbox"])


def _signed_area(poly: Sequence[Point]) -> float:
    total = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def polygon_area(poly: Sequence[Point]) -> float:
    return abs(_signed_area(poly))


def _counter_clockwise(poly: Sequence[Point]) -> Polygon:
    return list(poly) if _signed_area(poly) >= 0 else list(poly)[::-1]


def _left_of(pt: Point, a: Point, b: Point) -> bool:
    return (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) >= -_EPS


def _edge_crossing(p: Point, q: Point, a: Point, b: Point) -> Point:
    x1, y1 = p
    x2, y2 = q
    x3, y3 = a
    x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return q
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def intersection_area(a: Sequence[Point], b: Sequence[Point]) -> float:
    """
    Area shared by two convex polygons (Sutherland–Hodgman clipping).

    Both rectangles here are convex by construction, so clipping the first
    against every edge of the second leaves exactly their intersection.
    """
    subject = _counter_clockwise(a)
    clip = _counter_clockwise(b)
    for i in range(len(clip)):
        if not subject:
            return 0.0
        edge_a = clip[i]
        edge_b = clip[(i + 1) % len(clip)]
        source, subject = subject, []
        for j, current in enumerate(source):
            previous = source[j - 1]
            cur_in = _left_of(current, edge_a, edge_b)
            prev_in = _left_of(previous, edge_a, edge_b)
            if cur_in:
                if not prev_in:
                    subject.append(_edge_crossing(previous, current, edge_a, edge_b))
                subject.append(current)
            elif prev_in:
                subject.append(_edge_crossing(previous, current, edge_a, edge_b))
    return polygon_area(subject) if len(subject) >= 3 else 0.0


def overlap_frac(a: dict[str, Any], b: dict[str, Any]) -> float:
    """
    Shared area of two regions as a fraction of the smaller one.

    Uses each region's oriented rectangle when it has one, so two callouts on
    parallel leaders score ~0 even though their axis-aligned hulls overlap.
    """
    pa, pb = region_polygon(a), region_polygon(b)
    area_a, area_b = polygon_area(pa), polygon_area(pb)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return intersection_area(pa, pb) / min(area_a, area_b)


def contains_point(region: dict[str, Any], pt: Point) -> bool:
    """True when ``pt`` lies inside the region's tight polygon."""
    poly = _counter_clockwise(region_polygon(region))
    return all(
        _left_of(pt, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))
    )


def region_center(region: dict[str, Any]) -> Point:
    """Centre of the region's tight polygon."""
    poly = region_polygon(region)
    return (
        sum(p[0] for p in poly) / len(poly),
        sum(p[1] for p in poly) / len(poly),
    )


def polygons_of(regions: Iterable[dict[str, Any]]) -> list[Polygon]:
    return [region_polygon(r) for r in regions]
