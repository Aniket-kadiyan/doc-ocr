"""Tests for segment_quality.is_segment_worthy."""

from __future__ import annotations

from segment_quality import dedupe_regions, is_segment_worthy


def test_accepts_full_dimensions():
    assert is_segment_worthy("Ø215.37±0.05")
    assert is_segment_worthy("32°20'40\"")
    assert is_segment_worthy("Ø174.07±0.05")


def test_rejects_fragments():
    assert not is_segment_worthy(".05")
    assert not is_segment_worthy("05")
    assert not is_segment_worthy("215.")
    assert not is_segment_worthy("174.")


def _r(text, x, y, w, h, conf=0.9):
    return {"text": text, "confidence": conf, "bbox": {"x": x, "y": y, "width": w, "height": h}}


def test_dedupe_suppresses_overlapping_duplicate():
    # Ø175,32 column detected as a proper box and an overlapping sliver.
    regions = [
        _r("Ø175.32REF.", 355, 215, 60, 130, 0.92),
        _r("175REF", 350, 250, 111, 32, 0.71),
    ]
    out = dedupe_regions(regions)
    assert [r["text"] for r in out] == ["Ø175.32REF."]


def test_dedupe_keeps_distinct_columns():
    regions = [
        _r("Ø215,37±0.05", 90, 250, 35, 190),
        _r("Ø174.07±0.05", 205, 245, 34, 195),
        _r("203.20", 690, 330, 35, 120),
    ]
    assert len(dedupe_regions(regions)) == 3


def test_dedupe_preserves_reading_order():
    regions = [
        _r("Ø215,37±0.05", 90, 250, 35, 190),
        _r("Ø175.32REF.", 355, 215, 60, 130, 0.92),
        _r("175REF", 350, 250, 111, 32, 0.71),  # dup of the previous
        _r("203.20", 690, 330, 35, 120),
    ]
    out = [r["text"] for r in dedupe_regions(regions)]
    assert out == ["Ø215,37±0.05", "Ø175.32REF.", "203.20"]


if __name__ == "__main__":
    test_accepts_full_dimensions()
    test_rejects_fragments()
    test_dedupe_suppresses_overlapping_duplicate()
    test_dedupe_keeps_distinct_columns()
    test_dedupe_preserves_reading_order()
    print("OK")
