"""
Tests for GDTOCR spec-section coverage (backend/feature_rules.py).

The spec numbers 26 detection rules. Several are marked out of scope and must
not fire; the rest must keep classifying. These pin both halves down, because
switching a section off is easy to do and easy to do too widely.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_feature_rules.py
"""

from __future__ import annotations

import pytest

import feature_dictionary as fd
import feature_rules as rules
from feature_classifier import classify_feature

OUT_OF_SCOPE_SECTIONS = {
    rules.FILLET,
    rules.COUNTERBORE,
    rules.DEPTH,
    rules.QUANTITY_PREFIX,
    rules.REFERENCE,
}


# --- the registry itself ------------------------------------------------------

def test_every_spec_section_is_listed_once():
    numbers = [r.section for r in rules.SPEC_RULES]
    assert numbers == list(range(1, 27)), numbers


def test_sections_marked_out_of_scope_are_off():
    off = {r.section for r in rules.disabled_rules()}
    assert OUT_OF_SCOPE_SECTIONS <= off


def test_every_disabled_section_says_why():
    for r in rules.disabled_rules():
        assert r.reason, f"section {r.section} ({r.name}) is off with no reason"


def test_overall_dimensions_is_off_for_a_different_reason():
    """Not out of scope — it needs geometry the pipeline does not measure."""
    overall = rules.rule(rules.OVERALL_DIMENSIONS)
    assert not overall.enabled
    assert overall.reason == rules.NEEDS_GEOMETRY


def test_the_working_sections_are_on():
    for section in (
        rules.LINEAR, rules.DIAMETER, rules.RADIUS, rules.CHAMFER, rules.ANGLE,
        rules.TAPER, rules.THREAD, rules.THRU, rules.COUNTERSINK, rules.SPOTFACE,
        rules.GDT, rules.DATUM, rules.SURFACE_FINISH, rules.WELD, rules.MATERIAL,
        rules.HEAT_TREATMENT, rules.COATING, rules.DEBURRING, rules.BASIC,
        rules.GENERAL_NOTES,
    ):
        assert rules.is_enabled(section), rules.rule(section).name


def test_describe_lists_every_section():
    text = rules.describe()
    for r in rules.SPEC_RULES:
        assert r.name in text


# --- hole keywords follow their sections -------------------------------------

def test_disabled_hole_sections_drop_their_keywords():
    active = rules.active_hole_modifiers()
    for word in ("CBORE", "COUNTERBORE", "C'BORE", "DEEP", "DP", "DEPTH"):
        assert word not in active, word


def test_enabled_hole_sections_keep_theirs():
    active = rules.active_hole_modifiers()
    for word in ("THRU", "THROUGH", "CSK", "COUNTERSINK", "SF", "SPOTFACE"):
        assert word in active, word


def test_subtypes_follow_the_same_rule():
    names = [name for _, name in rules.active_hole_subtypes()]
    assert "Counterbore" not in names
    assert "Blind Hole" not in names
    assert "Through Hole" in names and "Countersink" in names


# --- what the classifier does with them --------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ø12 CBORE Ø20", fd.CAT_DIAMETER),   # 12 Counterbore, off
        ("Ø10 × 15 DEEP", fd.CAT_DIAMETER),   # 15 Depth, off
        ("25 REF", fd.CAT_LINEAR),            # 25 Reference, off
        ("(25)", fd.CAT_LINEAR),
    ],
)
def test_out_of_scope_sections_do_not_claim_text(text, expected):
    assert classify_feature(text).category == expected


def test_fillet_does_not_become_a_subtype():
    feature = classify_feature("R3 FILLET")
    assert feature.category == fd.CAT_RADIUS
    assert feature.subtype != "Fillet"


def test_quantity_prefix_is_not_read():
    feature = classify_feature("4X Ø10 THRU")
    assert feature.quantity is None
    assert feature.category == fd.CAT_HOLE
    assert "4X" not in feature.label


@pytest.mark.parametrize(
    "text,category,label",
    [
        ("Ø20", fd.CAT_DIAMETER, "Diameter"),
        ("R10", fd.CAT_RADIUS, "Radius"),
        ("0.5×45°", fd.CAT_CHAMFER, "Chamfer"),
        ("45°", fd.CAT_ANGLE, "Angle"),
        ("TAPER 1:20", fd.CAT_TAPER, "Taper"),
        ("M6×1", fd.CAT_THREAD, "Thread M6×1"),
        ("Ø8 THRU", fd.CAT_HOLE, "Through Hole"),
        ("Ø10 CSK", fd.CAT_HOLE, "Countersink"),
        ("Ø12 SF", fd.CAT_HOLE, "Spotface"),
        ("Ra 1.6", fd.CAT_SURFACE, "Surface Finish"),
        ("⏥ 0.05 A", fd.CAT_GDT, "GD&T Flatness"),
        ("ZINC PLATE", fd.CAT_COATING, "Coating"),
        ("HRC 58", fd.CAT_HEAT, "Heat Treatment"),
        ("ALUMINIUM 6061", fd.CAT_MATERIAL, "Material"),
        ("REMOVE ALL BURRS", fd.CAT_NOTE, "General Note"),
        ("40±0.1", fd.CAT_TOLERANCE, "Toleranced Dimension"),
        ("25", fd.CAT_LINEAR, "Linear Dimension"),
        ("□25□", fd.CAT_BASIC, "Basic Dimension"),
    ],
)
def test_working_sections_still_classify_and_label(text, category, label):
    feature = classify_feature(text)
    assert feature.category == category, feature
    assert feature.label == label, feature


def test_every_classified_feature_gets_a_label():
    """A balloon always needs something to show, whatever the category."""
    for text in ("Ø20", "R10", "45°", "M6", "Ra 1.6", "25", "", "????"):
        feature = classify_feature(text)
        assert isinstance(feature.label, str)
