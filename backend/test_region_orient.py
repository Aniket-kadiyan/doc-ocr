"""Rotation-aware region detection: coordinate mapping + slant detection."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from region_detect import dominant_slant_angles

cv2 = pytest.importorskip("cv2")

from ocr_pipeline import OcrPipeline  # noqa: E402


def _mark(image, cx, cy, color=(255, 0, 0), r=4):
    d = ImageDraw.Draw(image)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


@pytest.mark.parametrize("angle", [30.0, -30.0, 45.0, -45.0, 60.0])
def test_bbox_rot_to_source_roundtrip(angle):
    """A point placed in the source must map back onto itself through rotate+inverse."""
    iw, ih = 400, 300
    img = Image.new("RGB", (iw, ih), (255, 255, 255))
    # Known source point.
    sx, sy = 260.0, 90.0
    rot, inv = OcrPipeline._rotate_expand(img, angle)

    # Forward map source -> rotated frame using the same rotation matrix.
    fwd = cv2.getRotationMatrix2D(((iw - 1) / 2.0, (ih - 1) / 2.0), angle, 1.0)
    cos, sin = abs(fwd[0, 0]), abs(fwd[0, 1])
    nw = int(ih * sin + iw * cos)
    nh = int(ih * cos + iw * sin)
    fwd[0, 2] += (nw - iw) / 2.0
    fwd[1, 2] += (nh - ih) / 2.0
    rx = fwd[0, 0] * sx + fwd[0, 1] * sy + fwd[0, 2]
    ry = fwd[1, 0] * sx + fwd[1, 1] * sy + fwd[1, 2]

    # A 1px box at (rx, ry) in the rotated frame must map back to (sx, sy).
    bbox = OcrPipeline._bbox_rot_to_source(rx - 0.5, ry - 0.5, 1.0, 1.0, inv, iw, ih)
    assert bbox is not None
    mcx = bbox["x"] + bbox["width"] / 2.0
    mcy = bbox["y"] + bbox["height"] / 2.0
    assert abs(mcx - sx) < 1.5
    assert abs(mcy - sy) < 1.5


@pytest.mark.parametrize("angle", [30.0, -45.0, 60.0])
def test_oriented_box_matches_konva_reconstruction(angle):
    """The oriented box (x,y,w,h,rotation) must reconstruct — via Konva's
    rotate-about-(x,y) rule — to exactly the rotated-frame rectangle's corners
    mapped back to source. Proves diagonal balloons are drawn tight, not loose."""
    import math

    iw, ih = 500, 400
    img = Image.new("RGB", (iw, ih), (255, 255, 255))
    _, inv = OcrPipeline._rotate_expand(img, angle)
    # A rotated-frame box (horizontal text region there).
    bx, by, bw, bh = 120.0, 60.0, 90.0, 24.0
    ob = OcrPipeline._oriented_box_from_rot(bx, by, bw, bh, inv)

    r = math.radians(ob["rotation"])
    cos, sin = math.cos(r), math.sin(r)
    for dx, dy in [(0, 0), (bw, 0), (0, bh), (bw, bh)]:
        # Konva: local (dx,dy) about (x,y) -> screen.
        kx = ob["x"] + dx * cos - dy * sin
        ky = ob["y"] + dx * sin + dy * cos
        # Direct: map the same rotated-frame corner through inv.
        mx = inv[0, 0] * (bx + dx) + inv[0, 1] * (by + dy) + inv[0, 2]
        my = inv[1, 0] * (bx + dx) + inv[1, 1] * (by + dy) + inv[1, 2]
        assert abs(kx - mx) < 0.5 and abs(ky - my) < 0.5, (dx, dy, kx, mx, ky, my)


def test_dominant_slant_detects_diagonal_text():
    """A crop dominated by ~45° strokes reports a ~45° magnitude; a plain
    horizontal/vertical crop reports nothing."""
    diag = Image.new("RGB", (300, 300), (255, 255, 255))
    d = ImageDraw.Draw(diag)
    for off in range(-60, 61, 12):
        d.line([(40 + off, 250), (200 + off, 90)], fill=(0, 0, 0), width=3)
    mags = dominant_slant_angles(diag)
    assert mags, "expected a dominant slant on diagonal strokes"
    assert any(30 <= m <= 60 for m in mags), mags

    axis = Image.new("RGB", (300, 300), (255, 255, 255))
    d = ImageDraw.Draw(axis)
    for y in range(60, 240, 18):
        d.line([(40, y), (260, y)], fill=(0, 0, 0), width=3)
    assert dominant_slant_angles(axis) == []
