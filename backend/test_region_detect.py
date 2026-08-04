"""
Unit tests for ink-gap blob refinement (backend/region_detect.py).

Run: PYTHONPATH=. .venv/bin/python test_region_detect.py
"""

from __future__ import annotations

import numpy as np

from region_detect import _bands_from_projection, _refine_blob


def _mask_with_bands(bands: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Build a text mask with rectangular ink bands (y0, y1, x0, x1)."""
    h = max(b[1] for b in bands) + 4
    w = max(b[3] for b in bands) + 4
    roi = np.zeros((h, w), dtype=np.uint8)
    for y0, y1, x0, x1 in bands:
        roi[y0:y1, x0:x1] = 255
    return roi


def test_row_bands_split_stacked_text():
    roi = _mask_with_bands(
        [
            (4, 18, 4, 16),
            (34, 46, 6, 52),
        ]
    )
    bands = _bands_from_projection(roi, along="rows", char=6, min_area=20)
    assert len(bands) == 2


def test_refine_blob_splits_tall_merged_region():
    roi = _mask_with_bands(
        [
            (5, 20, 5, 15),
            (40, 52, 8, 20),
            (78, 95, 5, 15),
        ]
    )
    h, w = roi.shape
    assert h >= w * 2.6
    refined = _refine_blob(0, 0, w, h, roi, min_area=15)
    assert len(refined) >= 2


if __name__ == "__main__":
    test_row_bands_split_stacked_text()
    test_refine_blob_splits_tall_merged_region()
    print("OK")
