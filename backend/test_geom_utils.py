"""
Tests for oriented-rectangle geometry (backend/geom_utils.py).

Run: PYTHONPATH=. .venv/bin/python -m pytest test_geom_utils.py
"""

from __future__ import annotations

import math

from geom_utils import (
    contains_point,
    intersection_area,
    overlap_frac,
    polygon_area,
    rect_polygon,
    region_center,
    region_polygon,
)


def _axis(x, y, w, h, **extra):
    return {"bbox": {"x": x, "y": y, "width": w, "height": h}, **extra}


def _oriented(x, y, w, h, rot):
    return {
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "oriented_box": {"x": x, "y": y, "width": w, "height": h, "rotation": rot},
    }


def test_axis_aligned_overlap_matches_hand_calculation():
    a = _axis(0, 0, 100, 100)
    b = _axis(50, 0, 100, 100)
    assert abs(overlap_frac(a, b) - 0.5) < 1e-6


def test_disjoint_boxes_do_not_overlap():
    assert overlap_frac(_axis(0, 0, 10, 10), _axis(100, 100, 10, 10)) == 0.0


def test_containment_is_full_overlap_of_smaller():
    outer = _axis(0, 0, 100, 100)
    inner = _axis(25, 25, 10, 10)
    assert abs(overlap_frac(outer, inner) - 1.0) < 1e-6


def test_rotation_preserves_area():
    poly = region_polygon(_oriented(10, 10, 80, 20, 37))
    assert abs(polygon_area(poly) - 80 * 20) < 1e-6


def test_parallel_leaders_separate_though_hulls_overlap():
    """
    The case this module exists for: the two GPD 18T fit callouts.

    Boxes are the ones the pipeline actually produced for ``Ø20H10 +0.084/0``
    and ``Ø18H10 +0.070/0``. Their axis-aligned hulls overlap by more than half
    — enough for the old dedupe to delete one — while the leader-aligned
    rectangles share no ink at all.
    """
    a = {
        "bbox": {"x": 175, "y": 77, "width": 179, "height": 198},
        "oriented_box": {"x": 180, "y": 250, "width": 210, "height": 40, "rotation": -35},
    }
    b = {
        "bbox": {"x": 246, "y": 96, "width": 267, "height": 225},
        "oriented_box": {"x": 250, "y": 320, "width": 210, "height": 40, "rotation": -35},
    }
    assert overlap_frac({"bbox": a["bbox"]}, {"bbox": b["bbox"]}) > 0.5
    assert overlap_frac(a, b) < 0.05


def test_identical_rotated_rects_fully_overlap():
    a = _oriented(10, 10, 100, 30, 45)
    b = _oriented(10, 10, 100, 30, 45)
    assert abs(overlap_frac(a, b) - 1.0) < 1e-6


def test_center_and_contains_point():
    region = _oriented(0, 0, 100, 40, 30)
    center = region_center(region)
    assert contains_point(region, center)
    assert not contains_point(region, (center[0] + 500, center[1]))


def test_center_of_axis_box():
    assert region_center(_axis(10, 20, 100, 40)) == (60.0, 40.0)


def test_intersection_area_of_rotated_square():
    """A square rotated 45° about a shared centre overlaps a known area."""
    side = 100.0
    upright = rect_polygon({"x": 0, "y": 0, "width": side, "height": side})
    c = side / 2
    a = math.radians(45)
    turned = [
        (
            c + (px - c) * math.cos(a) - (py - c) * math.sin(a),
            c + (px - c) * math.sin(a) + (py - c) * math.cos(a),
        )
        for px, py in upright
    ]
    # The intersection is a regular octagon of area 2*(sqrt(2)-1)*side^2.
    expected = 2 * (math.sqrt(2) - 1) * side * side
    assert abs(intersection_area(upright, turned) - expected) < 1.0
