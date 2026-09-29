"""
Tests for the slashed-ring (Ø) topology detector in phi_detector.

Renders glyph runs with a system TrueType font when one is available and
checks that a leading Ø is found while digit runs (including the two-hole 8
and touching zeros) are not. Skips cleanly when no font can be loaded.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_phi_topology.py
"""

from __future__ import annotations

import glob

import pytest
from PIL import Image, ImageDraw, ImageFont

from phi_detector import LEGACY_CAP, TOPO_ACCEPT, detect_phi_multi_strip, detect_phi_topology

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _font(size: int):
    for path in _FONT_CANDIDATES + glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)[:5]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


def _render(text: str, size: int = 28, vertical: bool = False) -> Image.Image:
    font = _font(size)
    if font is None:
        pytest.skip("no TrueType font available for glyph rendering")
    im = Image.new("RGB", (size * len(text) + 30, size + 24), "white")
    ImageDraw.Draw(im).text((12, 8), text, font=font, fill=(0, 0, 0))
    return im.rotate(90, expand=True, fillcolor="white") if vertical else im


@pytest.mark.parametrize("text", ["Ø215.37", "Ø20.5", "ø8.2", "Ø0.03 A"])
@pytest.mark.parametrize("vertical", [False, True])
def test_leading_phi_detected(text, vertical):
    score, info = detect_phi_topology(_render(text, vertical=vertical))
    assert score >= TOPO_ACCEPT, info


@pytest.mark.parametrize(
    "text", ["8.25", "80.5", "00.5", "B 0.03", "R5.00", "18H10", "38.8", "100", "0.084"]
)
@pytest.mark.parametrize("vertical", [False, True])
def test_digit_runs_not_phi(text, vertical):
    score, info = detect_phi_topology(_render(text, vertical=vertical))
    assert score == 0.0, info


def test_multi_strip_caps_legacy_without_topology():
    """Without a slashed ring the (noisy) strip detectors can't clear 0.5."""
    im = _render("80.5")
    _, score, details = detect_phi_multi_strip(im, vertical=False)
    assert score <= LEGACY_CAP
    assert any(d["strip"] == "topology" for d in details)


def test_multi_strip_uses_topology_when_found():
    im = _render("Ø20.5")
    has, score, _ = detect_phi_multi_strip(im, vertical=False)
    assert has and score >= TOPO_ACCEPT


def test_robust_to_scale():
    for scale in (0.6, 1.6):
        im = _render("Ø174.07")
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        score, info = detect_phi_topology(im)
        assert score >= TOPO_ACCEPT, (scale, info)
