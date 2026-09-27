"""
Tests for reading text orientation from the detector's quadrilaterals.

The detector returns four corners per text line, so a callout drawn along a
leader reports its own angle. These cover the angle read-out, the grouping into
slant families, and the neighbourhood regions the angled pass uses.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_quad_angle.py
"""

from __future__ import annotations

import math

import pytest
from PIL import Image

from ocr_pipeline import OcrPipeline


def _quad(cx, cy, length, height, degrees):
    """Corners of a text line of the given size centred at (cx, cy)."""
    a = math.radians(degrees)
    cos, sin = math.cos(a), math.sin(a)
    corners = []
    for lx, ly in (
        (-length / 2, -height / 2),
        (length / 2, -height / 2),
        (length / 2, height / 2),
        (-length / 2, height / 2),
    ):
        corners.append((cx + lx * cos - ly * sin, cy + lx * sin + ly * cos))
    return corners


def _box(cx, cy, length, height, degrees, text="18H10", conf=0.95):
    quad = _quad(cx, cy, length, height, degrees)
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return {
        "x": min(xs), "y": min(ys),
        "w": max(xs) - min(xs), "h": max(ys) - min(ys),
        "text": text, "conf": conf, "quad": quad,
    }


@pytest.mark.parametrize("angle", [-60.0, -36.6, -12.0, 0.0, 15.0, 45.0, 72.0])
def test_quad_angle_reads_back(angle):
    measured = OcrPipeline._quad_angle(_quad(100, 100, 200, 40, angle))
    assert measured == pytest.approx(angle, abs=0.5)


def test_quad_angle_uses_the_long_edge():
    """A tall, narrow quad reads along its long side, not its top edge."""
    assert OcrPipeline._quad_angle(_quad(50, 50, 30, 150, 0.0)) == pytest.approx(90.0, abs=0.5)


def test_quad_angle_missing_or_degenerate():
    assert OcrPipeline._quad_angle(None) is None
    assert OcrPipeline._quad_angle([]) is None
    assert OcrPipeline._quad_angle([(0, 0), (0, 0), (0, 0), (0, 0)]) is None


def test_real_detector_quad_angle():
    """The corners PaddleOCR returned for the GPD 18T ``Ø18H10 +0.070``."""
    quad = [(475.0, 530.0), (930.0, 192.0), (1029.0, 322.0), (574.0, 661.0)]
    assert OcrPipeline._quad_angle(quad) == pytest.approx(-36.6, abs=0.3)


def test_slant_groups_ignore_upright_and_vertical_text():
    """Only diagonal text votes: the other two are read by their own passes."""
    boxes = [
        _box(100, 100, 200, 40, 0.0),      # horizontal
        _box(300, 300, 200, 40, 90.0),     # vertical
        _box(500, 500, 200, 40, 2.0),      # near-horizontal
        _box(700, 700, 200, 40, 88.0),     # near-vertical
    ]
    assert OcrPipeline._slant_angle_groups(boxes) == []


def test_slant_groups_merge_within_tolerance():
    boxes = [
        _box(100, 100, 200, 40, -36.0),
        _box(300, 300, 190, 40, -40.0),
        _box(500, 500, 195, 40, -38.0),
    ]
    groups = OcrPipeline._slant_angle_groups(boxes)
    assert len(groups) == 1
    assert groups[0] == pytest.approx(-38.0, abs=2.0)


def test_slant_groups_separate_opposite_diagonals():
    """A chamfer on one diagonal and a fit callout on the other are two families."""
    boxes = [
        _box(100, 100, 300, 40, 45.0),
        _box(600, 100, 300, 40, 44.0),
        _box(100, 600, 200, 40, -38.0),
    ]
    groups = OcrPipeline._slant_angle_groups(boxes)
    assert len(groups) == 2
    assert groups[0] == pytest.approx(44.5, abs=1.5)   # heaviest family first
    assert groups[1] == pytest.approx(-38.0, abs=1.0)


def test_slant_groups_respect_max_groups():
    boxes = [
        _box(100, 100, 300, 40, 45.0),
        _box(600, 100, 200, 40, -38.0),
        _box(100, 600, 100, 40, 20.0),
    ]
    assert len(OcrPipeline._slant_angle_groups(boxes, max_groups=2)) == 2


def test_neighbourhood_covers_diagonal_text_only():
    image = Image.new("RGB", (2000, 1500), "white")
    pipeline = OcrPipeline()
    boxes = [
        _box(1600, 400, 200, 40, -38.0),
        _box(1700, 500, 160, 40, -38.0),
        _box(300, 1200, 200, 40, 0.0),      # upright text, must not pull the region
    ]
    rois = pipeline._slant_neighbourhoods(image, boxes)
    assert len(rois) == 1
    x0, y0, x1, y1 = rois[0]
    assert x0 > 1000 and y1 < 1000          # nowhere near the upright text
    assert x1 - x0 > 200 and y1 - y0 > 100  # padded beyond the boxes themselves


def test_neighbourhoods_split_distant_groups():
    image = Image.new("RGB", (3000, 2000), "white")
    pipeline = OcrPipeline()
    boxes = [
        _box(300, 300, 150, 30, -38.0),
        _box(2600, 1700, 150, 30, 45.0),
    ]
    assert len(pipeline._slant_neighbourhoods(image, boxes)) == 2


def test_neighbourhood_clamped_to_image():
    image = Image.new("RGB", (400, 300), "white")
    pipeline = OcrPipeline()
    rois = pipeline._slant_neighbourhoods(image, [_box(30, 30, 100, 30, -40.0)])
    x0, y0, x1, y1 = rois[0]
    assert x0 >= 0 and y0 >= 0 and x1 <= 400 and y1 <= 300


def test_neighbourhood_sign():
    pipeline = OcrPipeline()
    boxes = [_box(500, 500, 300, 40, -38.0), _box(520, 520, 100, 20, 30.0)]
    # The heavier box decides.
    assert pipeline._neighbourhood_sign(boxes, (0, 0, 1000, 1000)) == -1.0
    # Boxes outside the region do not vote.
    assert pipeline._neighbourhood_sign(boxes, (0, 0, 100, 100)) is None


def test_no_quads_means_no_groups():
    """Boxes from the morphology proposer carry no corners, so they cannot vote."""
    plain = [{"x": 0, "y": 0, "w": 100, "h": 20, "text": "12.5", "conf": 0.9}]
    assert OcrPipeline._slant_angle_groups(plain) == []
