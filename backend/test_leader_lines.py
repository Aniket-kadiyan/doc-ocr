"""
Tests for leader-line detection (backend/region_detect.detect_leader_lines).

A callout lettered along a leader shares that leader's direction, so the line
gives the angle to level by. These draw known lines and check they come back.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_leader_lines.py
"""

from __future__ import annotations

import math

import pytest
from PIL import Image, ImageDraw

from ocr_pipeline import OcrPipeline
from region_detect import detect_leader_lines, opencv_available

pytestmark = pytest.mark.skipif(not opencv_available(), reason="needs OpenCV")


def _sheet(lines, size=(900, 700)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for (x1, y1, x2, y2) in lines:
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=2)
    return image


def _segment(cx, cy, length, degrees):
    a = math.radians(degrees)
    dx, dy = math.cos(a) * length / 2, math.sin(a) * length / 2
    return (cx - dx, cy - dy, cx + dx, cy + dy)


@pytest.mark.parametrize("angle", [-57.0, -34.5, -45.0, 30.0, 45.0, 60.0])
def test_leader_angle_is_measured(angle):
    image = _sheet([_segment(450, 350, 400, angle)])
    lines = detect_leader_lines(image)
    assert lines, angle
    assert lines[0]["angle"] == pytest.approx(angle, abs=2.0)
    assert lines[0]["length"] == pytest.approx(400, abs=40)


def test_axis_aligned_lines_are_skipped():
    """Dimension and extension lines are read by the upright and 90° passes."""
    image = _sheet([
        _segment(450, 200, 500, 0.0),
        _segment(450, 400, 500, 90.0),
        _segment(450, 500, 500, 4.0),
        _segment(450, 600, 500, 87.0),
    ])
    assert detect_leader_lines(image) == []


def test_two_leaders_are_separate_and_longest_first():
    image = _sheet([
        _segment(300, 300, 240, -35.0),
        _segment(650, 400, 420, -57.0),
    ])
    lines = detect_leader_lines(image)
    assert len(lines) == 2
    assert lines[0]["length"] >= lines[1]["length"]
    assert {round(line["angle"]) for line in lines} == {-57, -35}


def test_collinear_segments_merge_into_one_leader():
    """A leader broken by the text it carries is still one line."""
    a = _segment(300, 300, 200, -40.0)
    b = _segment(460, 165, 200, -40.0)
    lines = detect_leader_lines(_sheet([a, b]))
    assert len(lines) == 1
    assert lines[0]["angle"] == pytest.approx(-40.0, abs=2.0)
    assert lines[0]["length"] > 300


def test_short_strokes_are_not_leaders():
    image = _sheet([_segment(450, 350, 30, -45.0)])
    assert detect_leader_lines(image) == []


def test_blank_and_tiny_images():
    assert detect_leader_lines(Image.new("RGB", (900, 700), "white")) == []
    assert detect_leader_lines(Image.new("RGB", (8, 8), "white")) == []


def test_leader_angles_for_roi_selects_by_position():
    lines = [
        {"angle": -57.0, "x1": 1441, "y1": 649, "x2": 1666, "y2": 302, "length": 414.0},
        {"angle": -34.6, "x1": 1387, "y1": 684, "x2": 1779, "y2": 414, "length": 476.0},
        {"angle": 45.0, "x1": 100, "y1": 100, "x2": 300, "y2": 300, "length": 283.0},
    ]
    # The fit-callout region: both fit leaders run through it, the chamfer's does not.
    angles = OcrPipeline._leader_angles_for_roi(lines, (1300, 250, 1900, 750))
    assert angles == [-57.0, -34.6]
    # A region holding only the chamfer leader.
    assert OcrPipeline._leader_angles_for_roi(lines, (50, 50, 350, 350)) == [45.0]
    # A region with no leader at all.
    assert OcrPipeline._leader_angles_for_roi(lines, (0, 900, 200, 1000)) == []


def test_leader_angles_for_roi_dedupes_near_equal_angles():
    lines = [
        {"angle": -45.0, "x1": 100, "y1": 400, "x2": 300, "y2": 200, "length": 283.0},
        {"angle": -47.0, "x1": 120, "y1": 420, "x2": 320, "y2": 220, "length": 283.0},
    ]
    assert OcrPipeline._leader_angles_for_roi(lines, (0, 0, 500, 500)) == [-45.0]
