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


def test_full_page_slant_jobs_share_one_bounded_budget(monkeypatch):
    pipeline = object.__new__(OcrPipeline)
    requested_max_lines = []
    leaders = [
        {
            "angle": -35.0,
            "x1": float(index * 120),
            "y1": 200.0,
            "x2": float(index * 120 + 100),
            "y2": 130.0,
            "length": 122.0,
        }
        for index in range(10)
    ]

    def detect_leaders(_image, *, max_lines=6, **_kwargs):
        requested_max_lines.append(max_lines)
        return leaders[:max_lines]

    monkeypatch.setattr("region_detect.detect_leader_lines", detect_leaders)
    pipeline._leader_text_side = lambda *_args, **_kwargs: 1.0

    def corridor(_image, line, _line_h, **_kwargs):
        x0 = int(line["x1"])
        box = (x0, 100, x0 + 110, 230)
        return box, Image.new("RGB", (110, 130), "white"), [
            (x0, 100),
            (x0 + 110, 100),
            (x0 + 110, 230),
            (x0, 230),
        ]

    pipeline._leader_corridor = corridor
    pipeline._callout_neighbourhoods = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("four leader jobs exhaust the shared budget")
    )
    pipeline._worker_pool = lambda: None
    pipeline._angled_in_roi = lambda _image, **_kwargs: [
        {
            "text": "R4.5",
            "bbox": {"x": 1.0, "y": 2.0, "width": 30.0, "height": 14.0},
        }
    ]
    pipeline._supersede_fused_base = lambda *_args: set()
    pipeline._dedupe_angled = lambda angled, _base: angled
    events = []
    diagnostics = {}

    recovered = pipeline._detect_angled_regions(
        Image.new("RGB", (2400, 1600), "white"),
        [],
        det_boxes=[],
        max_neighbourhoods=4,
        progress_callback=lambda **event: events.append(event),
        diagnostics=diagnostics,
    )

    assert requested_max_lines == [4]
    assert len(recovered) == 4
    assert diagnostics["leader_job_count"] == 4
    assert diagnostics["quad_job_count"] == 0
    assert diagnostics["job_count"] == 4
    assert diagnostics["completed_job_count"] == 4
    assert diagnostics["recovered_count"] == 4
    assert events[-1]["completed"] == events[-1]["total"] == 4


def test_remaining_slant_budget_is_offered_to_quad_neighbourhoods(monkeypatch):
    pipeline = object.__new__(OcrPipeline)
    requested_quad_budget = []

    monkeypatch.setattr(
        "region_detect.detect_leader_lines",
        lambda _image, **_kwargs: [],
    )

    def callouts(_image, _boxes, *, max_rois, with_quads):
        requested_quad_budget.append((max_rois, with_quads))
        return []

    pipeline._callout_neighbourhoods = callouts
    pipeline._worker_pool = lambda: None
    pipeline._supersede_fused_base = lambda *_args: set()
    pipeline._dedupe_angled = lambda angled, _base: angled

    recovered = pipeline._detect_angled_regions(
        Image.new("RGB", (2400, 1600), "white"),
        [],
        det_boxes=[],
        max_neighbourhoods=24,
    )

    assert recovered == []
    assert requested_quad_budget == [(24, True)]


def test_slant_worker_pool_reports_each_completed_neighbourhood(monkeypatch):
    pipeline = object.__new__(OcrPipeline)
    leaders = [
        {
            "angle": -35.0,
            "x1": float(index * 200),
            "y1": 300.0,
            "x2": float(index * 200 + 120),
            "y2": 220.0,
            "length": 144.0,
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        "region_detect.detect_leader_lines",
        lambda _image, **_kwargs: leaders,
    )
    pipeline._leader_text_side = lambda *_args, **_kwargs: 1.0
    pipeline._leader_corridor = lambda _image, line, _line_h, **_kwargs: (
        (int(line["x1"]), 180, int(line["x1"]) + 150, 340),
        Image.new("RGB", (150, 160), "white"),
        [
            (line["x1"], 180.0),
            (line["x1"] + 150.0, 180.0),
            (line["x1"] + 150.0, 340.0),
            (line["x1"], 340.0),
        ],
    )
    pipeline._callout_neighbourhoods = lambda *_args, **_kwargs: []
    pipeline._angled_in_roi = lambda _image, **_kwargs: [
        {
            "text": "17",
            "bbox": {"x": 1.0, "y": 2.0, "width": 20.0, "height": 12.0},
        }
    ]
    pipeline._supersede_fused_base = lambda *_args: set()
    pipeline._dedupe_angled = lambda angled, _base: angled
    map_calls = []

    class Pool:
        def map(self, task, payloads, *, local, on_item_done):
            map_calls.append((task, len(payloads)))
            results = []
            for index, payload in enumerate(payloads):
                result = local(payload)
                results.append(result)
                on_item_done(index, result)
            return results

    pipeline._worker_pool = lambda: Pool()
    events = []
    diagnostics = {}

    recovered = pipeline._detect_angled_regions(
        Image.new("RGB", (2400, 1600), "white"),
        [],
        det_boxes=[],
        max_neighbourhoods=2,
        progress_callback=lambda **event: events.append(event),
        diagnostics=diagnostics,
    )

    assert map_calls == [("angled_roi", 2)]
    assert len(recovered) == 2
    assert diagnostics["worker_pool_used"] is True
    assert diagnostics["completed_job_count"] == 2
    assert events[-1]["completed"] == events[-1]["total"] == 2
