"""
Regression tests for ``OcrPipeline._dedupe_det_boxes``.

Overlap in that dedupe is measured against the smaller of two boxes, so a
fragment inside a text line is a positional duplicate of the line. Which one
survives is decided by the visiting order, and on a real sheet the wrong order
threw away every small-text callout: a morphology fragment over "48" displaced
the detector's line around ``Ø0.148 [3.76] REF.``. These tests pin the order.
"""

from __future__ import annotations

from ocr_pipeline import OcrPipeline


def _box(x, y, w, h, *, conf=0.0, text="", **extra):
    return {"x": x, "y": y, "w": w, "h": h, "conf": conf, "text": text, **extra}


def test_fragment_inside_a_detector_line_does_not_displace_the_line() -> None:
    line = _box(1383, 1321, 402, 62, conf=0.89, _detection_pass="detector")
    fragment = _box(1483, 1334, 55, 38, conf=0.0, _detection_pass="morphology")

    # Morphology candidates are listed first in the caller, which is exactly
    # the order that used to lose the line.
    kept = OcrPipeline._dedupe_det_boxes([fragment, line], overlap_thresh=0.62)

    assert kept == [line]


def test_speck_from_the_same_detector_pass_does_not_displace_the_line() -> None:
    speck = _box(1046, 322, 12, 12, conf=0.5)
    line = _box(1017, 300, 327, 61, conf=0.79)

    kept = OcrPipeline._dedupe_det_boxes([speck, line], overlap_thresh=0.62)

    assert kept == [line]


def test_same_location_from_two_passes_is_proposed_once() -> None:
    upright = _box(1000, 203, 311, 63, conf=0.86)
    turned = _box(999, 206, 307, 61, conf=0.83)

    kept = OcrPipeline._dedupe_det_boxes([turned, upright], overlap_thresh=0.62)

    assert len(kept) == 1
    assert kept[0]["w"] * kept[0]["h"] == 311 * 63


def test_distinct_locations_are_all_kept() -> None:
    a = _box(0, 0, 100, 40, conf=0.9)
    b = _box(0, 100, 100, 40, conf=0.9)
    c = _box(300, 0, 20, 20)  # a morphology proposal with nothing near it

    kept = OcrPipeline._dedupe_det_boxes([c, a, b], overlap_thresh=0.62)

    assert sorted((k["x"], k["y"]) for k in kept) == [(0, 0), (0, 100), (300, 0)]


def test_ink_fill_still_ranks_first_for_the_quick_path() -> None:
    # The quick proposer scores each box by ink fill; a well-filled phantom-free
    # box must keep winning over a poorly filled larger one at the same spot.
    filled = _box(10, 10, 80, 30, conf=0.9, _fill=0.4)
    sparse = _box(0, 0, 120, 50, conf=0.9, _fill=0.03)

    kept = OcrPipeline._dedupe_det_boxes([sparse, filled], overlap_thresh=0.62)

    assert kept == [filled]
