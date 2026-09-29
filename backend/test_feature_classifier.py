"""Rule-engine tests driven by the GDTOCR spec's own examples."""

from __future__ import annotations

import pytest

import feature_dictionary as fd
from feature_classifier import classify_feature


@pytest.mark.parametrize(
    "text,expected",
    [
        # Diameter
        ("Ø20", fd.CAT_DIAMETER),
        ("Ø10 ±0.02", fd.CAT_DIAMETER),
        ("Ø174.07±0.05", fd.CAT_DIAMETER),
        # Radius
        ("R10", fd.CAT_RADIUS),
        ("R0.5", fd.CAT_RADIUS),
        # Chamfer (before angle)
        ("2X45°", fd.CAT_CHAMFER),
        ("0.5×45°", fd.CAT_CHAMFER),
        # Angle
        ("45°", fd.CAT_ANGLE),
        ("30°±3°", fd.CAT_ANGLE),
        # Thread
        ("M6", fd.CAT_THREAD),
        ("M10×1.25", fd.CAT_THREAD),
        ("1/4-20 UNC", fd.CAT_THREAD),
        ("G1/4", fd.CAT_THREAD),
        ("NPT", fd.CAT_THREAD),
        # Hole features
        ("Ø8 THRU", fd.CAT_HOLE),
        ("4X Ø10 THRU", fd.CAT_HOLE),
        # Counterbore (12), Depth (15) and GD&T (16) are switched off in
        # feature_rules, so these fall through to the diameter they carry.
        # test_feature_rules.py covers that decision; this file records what
        # the classifier does with the sections that are on.
        ("Ø12 CBORE Ø20", fd.CAT_DIAMETER),
        ("Ø10 × 15 DEEP", fd.CAT_DIAMETER),
        ("POSITION Ø0.1 A B", fd.CAT_DIAMETER),
        ("FLATNESS 0.05", fd.CAT_LINEAR),
        ("ROUNDNESS(2PT./180)", fd.CAT_LINEAR),
        # Surface finish
        ("Ra 1.6", fd.CAT_SURFACE),
        ("Rz 3.2", fd.CAT_SURFACE),
        ("N7", fd.CAT_SURFACE),
        # Material / heat / coating
        ("MATERIAL SAE52100", fd.CAT_MATERIAL),
        ("HARDNESS Rc58~63", fd.CAT_HEAT),
        ("BLACK OXIDE", fd.CAT_COATING),
        # Notes
        ("REMOVE ALL BURRS", fd.CAT_NOTE),
        ("UNLESS OTHERWISE SPECIFIED", fd.CAT_NOTE),
        # Reference Dimension (25) is switched off, so these read as linear.
        ("(25)", fd.CAT_LINEAR),
        ("25 REF", fd.CAT_LINEAR),
        # Linear
        ("25", fd.CAT_LINEAR),
        ("40 ±0.1", fd.CAT_TOLERANCE),
    ],
)
def test_category(text, expected):
    assert classify_feature(text).category == expected


def test_quantity_prefix_is_out_of_scope():
    """Section 24 is switched off: the hole is still read, the 4X is not."""
    feat = classify_feature("4X Ø10 THRU")
    assert feat.category == fd.CAT_HOLE
    assert feat.quantity is None
    assert feat.subtype == "Through Hole"
    assert feat.label == "Through Hole"


def test_diameter_requires_text_glyph():
    # A Ø glyph in the (composed) text is authoritative.
    assert classify_feature("Ø50.22±0.05").category == fd.CAT_DIAMETER


def test_vision_only_diameter_is_not_trusted():
    # Vision claims a diameter but the text has no glyph and compose did not
    # inject one -> we must NOT invent a Diameter (the false-positive fix).
    feat = classify_feature("50.22±0.05", symbols={"diameter": True})
    assert feat.category == fd.CAT_TOLERANCE
    assert "diameter" not in feat.symbols
    assert "vision:diameter" in feat.symbols


def test_vision_degree_does_not_override_text():
    # "45°" with a spurious vision-diameter flag stays an Angle, not Diameter.
    feat = classify_feature("45°", symbols={"diameter": True})
    assert feat.category == fd.CAT_ANGLE


def test_keyword_boundary_no_false_positives():
    # "TAIL" ⊂ "DETAIL" and "RA" ⊂ "RADIUS"/"DRAWN" must not misfire.
    assert classify_feature("DETAIL A SCALE 2:1").category != fd.CAT_WELD
    assert classify_feature("RADIUS ±0.2").category != fd.CAT_SURFACE
    assert classify_feature("REVISION DRAWN").category != fd.CAT_SURFACE


def test_overall_dimension_by_span():
    feat = classify_feature("250", geometry={"span_ratio": 0.9})
    assert feat.category == fd.CAT_LINEAR
    assert feat.subtype == "Overall"


def test_chamfer_beats_angle():
    assert classify_feature("2X45°").category == fd.CAT_CHAMFER
    assert classify_feature("45°").category == fd.CAT_ANGLE


def test_labels_are_populated():
    for text in ("Ø20", "R10", "M6", "Ra 1.6", "45°", "MATERIAL EN8"):
        assert classify_feature(text).label
