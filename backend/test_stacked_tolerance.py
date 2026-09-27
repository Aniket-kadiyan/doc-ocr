"""
Tests for the stacked deviation reader (backend/stacked_tolerance.py).

Box layouts are taken from real PaddleOCR detections on the GPD 18T fit
callouts, levelled along their leader lines.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_stacked_tolerance.py
"""

from __future__ import annotations

from stacked_tolerance import stacked_deviation_text


def _b(x, y, w, h, text, conf=1.0):
    return {"x": x, "y": y, "w": w, "h": h, "text": text, "conf": conf}


def test_reads_real_fit_callout():
    """Ø18H10 with +0.070 over 0 — parts as detected in the levelled frame."""
    boxes = [
        _b(10, 11, 160, 56, "18H10"),
        _b(162, 9, 107, 37, "+0.070"),
        _b(204, 40, 26, 16, "0", 0.86),
    ]
    assert stacked_deviation_text(boxes) == "18H10 +0.070/0"


def test_ignores_speck_and_neighbouring_callout():
    """A sliver of the next callout below, and a stray speck, are excluded."""
    boxes = [
        _b(20, 81, 174, 102, "20H10", 0.97),
        _b(162, 27, 114, 76, "+0.084"),
        _b(208, 71, 36, 32, "0", 0.94),
        _b(224, 187, 83, 22, "0.070", 0.94),  # the callout on the next leader
        _b(266, 79, 11, 9, "2", 0.45),  # speck, low confidence and tiny
    ]
    assert stacked_deviation_text(boxes) == "20H10 +0.084/0"


def test_plain_tolerance_is_not_stacked():
    boxes = [_b(0, 0, 100, 40, "215.37"), _b(110, 0, 60, 40, "±0.05")]
    assert stacked_deviation_text(boxes) is None


def test_side_by_side_parts_rejected():
    """Two deviations on the same row are not a stack."""
    boxes = [
        _b(0, 0, 100, 40, "20H10"),
        _b(110, 0, 50, 20, "+0.08"),
        _b(170, 0, 50, 20, "0"),
    ]
    assert stacked_deviation_text(boxes) is None


def test_low_confidence_part_rejected():
    boxes = [
        _b(10, 11, 160, 56, "18H10"),
        _b(162, 9, 107, 37, "+0.070", 0.4),
        _b(204, 40, 26, 16, "0"),
    ]
    assert stacked_deviation_text(boxes) is None


def test_non_numeric_deviation_rejected():
    boxes = [
        _b(10, 11, 160, 56, "18H10"),
        _b(162, 9, 107, 37, "+0.070"),
        _b(204, 40, 26, 16, "A"),
    ]
    assert stacked_deviation_text(boxes) is None


def test_part_left_of_value_rejected():
    """A value with something before it is a fragment, not a callout start."""
    boxes = [
        _b(200, 11, 160, 56, "18H10"),
        _b(360, 9, 107, 37, "+0.070"),
        _b(400, 40, 26, 16, "0"),
        _b(10, 20, 80, 40, "12.5"),
    ]
    assert stacked_deviation_text(boxes) is None


def test_too_few_parts():
    assert stacked_deviation_text([]) is None
    assert stacked_deviation_text([_b(0, 0, 50, 20, "20")]) is None
