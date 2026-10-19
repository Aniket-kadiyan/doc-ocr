"""
Tests for context-aware disambiguation (backend/gdt_context.py).

The rule under test throughout: context resolves what the pixels left open,
and never overrules what they settled. Several of these exist because the
opposite happened — an early version promoted an alternative over a confirmed
reading and turned a confident profile-of-a-line into circularity.

Pure logic: no images, no cv2 required.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_gdt_context.py
"""

from __future__ import annotations

import pytest

import gdt_context as ctx
from gdt_detect import (
    STATUS_AMBIGUOUS,
    STATUS_CONFIRMED,
    Candidate,
    Detection,
)


def make(
    symbol_id: str,
    *,
    status: str = STATUS_AMBIGUOUS,
    alternatives: list[str] | None = None,
    confidence: float = 0.7,
    bbox: tuple[float, float, float, float] = (10, 10, 20, 20),
) -> Detection:
    import gdt_registry as reg

    return Detection(
        symbol_id=symbol_id,
        name=reg.get(symbol_id).name,
        status=status,
        confidence=confidence,
        bbox=bbox,
        alternatives=[
            Candidate(a, confidence - 0.02, "template_match")
            for a in (alternatives or [])
        ],
    )


# --- the governing rule ------------------------------------------------------


def test_context_does_not_overrule_a_confirmed_detection():
    """The regression this module exists to prevent."""
    det = make(
        "profile_of_a_line",
        status=STATUS_CONFIRMED,
        alternatives=["profile_of_a_surface", "circularity"],
        confidence=0.87,
    )
    ctx.resolve_by_position(det, in_frame=True, frame_compartment=0)
    assert det.symbol_id == "profile_of_a_line"
    assert det.status == STATUS_CONFIRMED


def test_context_does_resolve_an_ambiguous_detection():
    det = make("arc_length", alternatives=["profile_of_a_line"])
    ctx.resolve_by_position(det, in_frame=True, frame_compartment=0)
    assert det.symbol_id == "profile_of_a_line"
    assert det.status == STATUS_CONFIRMED
    assert "context" in det.evidence


def test_a_promotion_keeps_the_demoted_reading_as_an_alternative():
    """A context swap must stay auditable."""
    det = make("arc_length", alternatives=["profile_of_a_line"])
    ctx.resolve_by_position(det, in_frame=True, frame_compartment=0)
    assert "arc_length" in {a.symbol_id for a in det.alternatives}


# --- the specific confusions the brief names ---------------------------------


def test_the_arc_glyph_outside_a_frame_is_an_arc_length():
    det = make("profile_of_a_line", alternatives=["arc_length"])
    ctx.resolve_by_position(det, in_frame=False)
    assert det.symbol_id == "arc_length"


def test_a_counterbore_glyph_with_sf_is_a_spotface():
    det = make("counterbore", alternatives=["spotface"])
    ctx.resolve_by_position(det, in_frame=False, nearby_text="⌴ Ø12 SF")
    assert det.symbol_id == "spotface"


def test_a_counterbore_glyph_without_sf_stays_a_counterbore():
    det = make("counterbore", alternatives=["spotface"])
    ctx.resolve_by_position(det, in_frame=False, nearby_text="Ø12 ↧5")
    assert det.symbol_id == "counterbore"


def test_a_circle_in_a_frames_first_compartment_is_circularity():
    det = make("diameter", alternatives=["circularity"])
    ctx.resolve_by_position(det, in_frame=True, frame_compartment=0)
    assert det.symbol_id == "circularity"


def test_a_circle_before_a_number_is_a_diameter():
    det = make("circularity", alternatives=["diameter"])
    ctx.resolve_by_position(det, in_frame=False, nearby_text="25.4")
    assert det.symbol_id == "diameter"


def test_a_circle_on_a_leader_kink_is_all_around():
    det = make("circularity", alternatives=["all_around"])
    ctx.resolve_by_position(det, in_frame=False, on_leader=True)
    assert det.symbol_id == "all_around"


def test_an_s_prefix_makes_a_diameter_spherical():
    det = make("diameter", alternatives=["spherical_diameter"])
    ctx.resolve_by_position(det, in_frame=False, nearby_text="SØ30")
    assert det.symbol_id == "spherical_diameter"


def test_a_triangle_with_a_letter_is_a_datum_feature():
    det = make("datum_triangle", alternatives=["datum_feature_symbol"])
    ctx.resolve_by_position(det, in_frame=False, nearby_text="A")
    assert det.symbol_id == "datum_feature_symbol"


def test_a_triangle_with_no_letter_stays_an_unlabelled_marker():
    det = make("datum_feature_symbol", alternatives=["datum_triangle"])
    ctx.resolve_by_position(det, in_frame=False, nearby_text="")
    assert det.symbol_id == "datum_triangle"


# --- radius: a letter, not a glyph -------------------------------------------


@pytest.mark.parametrize(
    "text,symbol_id,value",
    [
        ("R12", "radius", 12.0),
        ("R 3.5", "radius", 3.5),
        ("CR5", "controlled_radius", 5.0),
        ("SR20", "spherical_radius", 20.0),
    ],
)
def test_radius_callouts_are_read(text, symbol_id, value):
    parsed = ctx.radius_from_text(text)
    assert parsed is not None
    assert parsed["symbol_id"] == symbol_id
    assert parsed["value"] == value
    assert parsed["unit"] == "drawing_unit"


@pytest.mark.parametrize(
    "text", ["Ra 0.8", "RA1.6", "Rc 58", "REF", "REV B", "ROUGHNESS", "R"]
)
def test_the_letter_r_in_other_words_is_not_a_radius(text):
    assert ctx.radius_from_text(text) is None


# --- values ------------------------------------------------------------------


def test_a_ratio_is_read_as_a_ratio_not_a_length():
    value = ctx.parse_value("conical_taper", "1:5")
    assert value["kind"] == "ratio"
    assert value["numerator"] == 1 and value["denominator"] == 5


def test_a_count_prefix_is_read():
    value = ctx.parse_value("multiple_identical_features", "4X Ø6")
    assert value["kind"] == "count" and value["value"] == 4


def test_no_value_invents_a_unit():
    """The brief: never assume millimetres."""
    for symbol_id, text in (
        ("diameter", "Ø25"),
        ("depth", "12"),
        ("basic_dimension", "50"),
    ):
        value = ctx.parse_value(symbol_id, text)
        assert value is not None
        assert value.get("unit") == "drawing_unit"


def test_a_symbol_that_takes_no_value_returns_none():
    assert ctx.parse_value("all_around", "25") is None
    assert ctx.parse_value("mmc", "0.5") is None


# --- attaching text ----------------------------------------------------------


def test_text_to_the_right_on_the_same_line_is_preferred():
    det = make("diameter", bbox=(100, 100, 20, 20))
    texts = [
        ctx.TextBox("99", (100, 160, 20, 20)),     # below
        ctx.TextBox("25.4", (124, 100, 40, 20)),   # right, same line
    ]
    ctx.attach_text([det], texts)
    assert det.associated_text == "25.4"
    assert det.value["value"] == 25.4


def test_a_symbol_with_nothing_beside_it_gets_no_invented_value():
    det = make("diameter", bbox=(100, 100, 20, 20))
    far_away = [ctx.TextBox("25.4", (900, 900, 40, 20))]
    ctx.attach_text([det], far_away)
    assert det.associated_text is None
    assert det.value is None


# --- loose characteristics ---------------------------------------------------


def test_a_characteristic_outside_a_frame_is_demoted_and_explained():
    """A drawing is full of right angles; most are not perpendicularity."""
    det = make("perpendicularity", status=STATUS_CONFIRMED, confidence=0.9)
    ctx.demote_loose_characteristics([det])
    assert det.status == STATUS_AMBIGUOUS
    assert det.confidence < 0.9
    assert any("outside any feature control frame" in w for w in det.warnings)


def test_a_characteristic_inside_a_frame_is_left_alone():
    det = make("perpendicularity", status=STATUS_CONFIRMED, confidence=0.9)
    det.frame_id = "frame-1"
    ctx.demote_loose_characteristics([det])
    assert det.status == STATUS_CONFIRMED
    assert det.confidence == 0.9


# --- datum association -------------------------------------------------------


class _Ref:
    def __init__(self, label, precedence, material=None):
        self.label = label
        self.precedence = precedence
        self.material_condition = material


class _Frame:
    def __init__(self, datums):
        self.id = "frame-1"
        self.datums = datums


def test_a_datum_reference_resolves_to_its_feature_symbol():
    feature = make("datum_feature_symbol", bbox=(5, 5, 20, 20))
    feature.value = {"kind": "letter", "value": "A"}
    records = ctx.associate_datums([_Frame([_Ref("A", 1)])], [feature])
    assert len(records) == 1
    assert records[0]["resolved"] is True
    assert records[0]["datum_feature_bbox"] is not None


def test_a_reference_with_no_matching_feature_is_reported_unresolved():
    """A frame citing a datum the drawing lacks is a finding, not a gap."""
    records = ctx.associate_datums([_Frame([_Ref("B", 1)])], [])
    assert records[0]["resolved"] is False
    assert "no datum feature symbol labelled" in records[0]["note"]


def test_precedence_is_carried_through():
    records = ctx.associate_datums(
        [_Frame([_Ref("A", 1), _Ref("B", 2), _Ref("C", 3)])], []
    )
    assert [r["precedence"] for r in records] == [1, 2, 3]


# --- standards ---------------------------------------------------------------


def test_an_asme_only_symbol_on_an_iso_drawing_is_reported():
    det = make("independency", status=STATUS_CONFIRMED)
    findings = ctx.check_standards_consistency([det], declared_standard="ISO 1101")
    assert any(f["issue"] == "standard_mismatch" for f in findings)


def test_a_retired_control_is_reported_without_being_called_invalid():
    det = make("concentricity", status=STATUS_CONFIRMED)
    findings = ctx.check_standards_consistency([det], declared_standard="ASME Y14.5")
    retired = [f for f in findings if f["issue"] == "retired_control"]
    assert retired
    detail = retired[0]["detail"]
    assert "remains valid" in detail
    assert "not known" in detail


def test_no_declared_standard_means_no_mismatch_claims():
    """Without knowing the drawing's standard, a mismatch cannot be asserted."""
    det = make("independency", status=STATUS_CONFIRMED)
    findings = ctx.check_standards_consistency([det], declared_standard=None)
    assert not any(f["issue"] == "standard_mismatch" for f in findings)


def test_reconcile_records_the_conflict_and_lowers_confidence():
    det = make("perpendicularity", status=STATUS_CONFIRMED, confidence=0.9)
    ctx.reconcile(det, expected=["circularity"], reason="not expected here")
    assert det.status == STATUS_AMBIGUOUS
    assert det.confidence < 0.9
    assert det.evidence["context_conflict"]["found"] == "perpendicularity"
    # The detection survives — the drawing may simply be unusual.
    assert det.symbol_id == "perpendicularity"
