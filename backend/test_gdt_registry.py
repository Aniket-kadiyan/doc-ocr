"""
Tests for the GD&T symbol registry (backend/gdt_registry.py).

Pure data: no cv2, no model, no images. These pin down the properties the rest
of the system relies on — that ids are unique, that a concept can never be
reported as a sighting, that a retired control is recorded rather than deleted,
and that an ambiguous glyph resolves to candidates rather than to one answer.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_gdt_registry.py
"""

from __future__ import annotations

import pytest

import gdt_registry as reg


# --- structure ---------------------------------------------------------------

def test_every_symbol_id_is_unique():
    ids = [s.id for s in reg.ALL_SYMBOLS]
    assert len(ids) == len(set(ids))


def test_every_group_the_brief_names_is_populated():
    for category in (
        reg.CAT_GEOMETRIC,
        reg.CAT_DATUM,
        reg.CAT_MATERIAL_CONDITION,
        reg.CAT_DIMENSIONING,
        reg.CAT_MANUFACTURING,
        reg.CAT_FRAME,
    ):
        assert reg.by_category(category), f"{category} has no entries"


def test_all_fourteen_geometric_controls_are_present():
    """The ASME Y14.5 set, including the two retired in 2018."""
    expected = {
        "straightness", "flatness", "circularity", "cylindricity",
        "profile_of_a_line", "profile_of_a_surface", "angularity",
        "perpendicularity", "parallelism", "position", "concentricity",
        "symmetry", "circular_runout", "total_runout",
    }
    found = {s.id for s in reg.by_category(reg.CAT_GEOMETRIC)}
    assert found == expected


def test_every_geometric_control_may_open_a_frame():
    for sym in reg.by_category(reg.CAT_GEOMETRIC):
        assert sym.frame_characteristic, f"{sym.id} cannot open a frame"
    assert len(reg.FRAME_CHARACTERISTICS) == 14


def test_every_symbol_has_a_description_and_a_standard():
    for sym in reg.ALL_SYMBOLS:
        assert sym.description.strip(), f"{sym.id} has no description"
        assert sym.standards.standards, f"{sym.id} names no standard"


def test_detectable_symbols_name_a_recognition_strategy():
    for sym in reg.DETECTABLE:
        assert sym.strategy, f"{sym.id} is detectable but names no strategy"


# --- concepts are not sightings ----------------------------------------------

def test_derived_concepts_are_not_detectable():
    """The brief's rule: a concept must never be reported as a found symbol."""
    for symbol_id in (
        "virtual_condition",
        "bonus_tolerance",
        "rfs",
        "datum_reference_frame",
        "envelope_principle_rule_1",
    ):
        sym = reg.get(symbol_id)
        assert sym.kind in {reg.KIND_CONCEPT, reg.KIND_TEXTUAL}
        assert not sym.detectable, f"{symbol_id} must not be detectable"


def test_a_concept_may_only_be_derived():
    for sym in reg.ALL_SYMBOLS:
        if sym.kind == reg.KIND_CONCEPT:
            assert sym.strategy in ((), (reg.S_DERIVED,))


def test_concepts_carry_no_template():
    """Nothing without a glyph may have artwork to match against."""
    for sym in reg.NON_VISUAL:
        assert sym.template is None, f"{sym.id} is non-visual but has a template"


# --- standards ---------------------------------------------------------------

def test_retired_controls_are_recorded_not_deleted():
    legacy = {s.id for s in reg.legacy_symbols()}
    assert legacy == {"concentricity", "symmetry"}
    for symbol_id in legacy:
        sym = reg.get(symbol_id)
        assert sym.standards.retired_in == "ASME Y14.5-2018"
        # Still valid somewhere, which is why it is kept.
        assert sym.standards.current_in


def test_standard_exclusive_symbols_say_which_family():
    assert {s.id for s in reg.exclusive_to("ISO")} == {"envelope_requirement"}
    asme_only = {s.id for s in reg.exclusive_to("ASME")}
    assert {"independency", "controlled_radius"} <= asme_only


def test_a_legacy_symbol_is_not_also_claimed_current_under_asme_2018():
    for sym in reg.legacy_symbols():
        assert "ASME Y14.5-2018" not in sym.standards.current_in


# --- lookups -----------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("MMC", "mmc"),
        ("Maximum Material Condition", "mmc"),
        ("true position", "position"),
        ("ROUNDNESS", "circularity"),
        ("squareness", "perpendicularity"),
        ("CSK", "countersink"),
        ("C'BORE", "counterbore"),
        ("TP", "position"),
    ],
)
def test_aliases_resolve(name, expected):
    sym = reg.by_name(name)
    assert sym is not None and sym.id == expected


def test_a_name_two_symbols_claim_is_reported_ambiguous_not_guessed():
    """'Taylor principle' names both the ISO Ⓔ and the ASME Rule #1."""
    assert reg.by_name("TAYLOR PRINCIPLE") is None
    assert "TAYLOR PRINCIPLE" in reg.AMBIGUOUS_NAMES
    assert len(reg.candidates_for_name("TAYLOR PRINCIPLE")) == 2


def test_a_symbols_own_alias_does_not_make_it_ambiguous_with_itself():
    """Circularity's canonical name repeats its alias; that is not a clash."""
    assert reg.by_name("CIRCULARITY").id == "circularity"
    assert "CIRCULARITY" not in reg.AMBIGUOUS_NAMES


def test_one_glyph_shared_by_two_symbols_returns_both():
    arc = reg.BY_UNICODE["⌒"]
    assert {s.id for s in arc} == {"profile_of_a_line", "arc_length"}


def test_ocr_variants_generate_candidates_rather_than_an_answer():
    """A bare 'O' is produced by many symbols; the table must not pick one."""
    candidates = reg.candidates_for_ocr("O")
    ids = {s.id for s in candidates}
    assert {"circularity", "diameter", "concentricity"} <= ids
    assert len(ids) > 3


def test_unknown_lookups_are_none_not_an_exception():
    assert reg.by_name("not a symbol") is None
    assert reg.find("not_a_symbol") is None
    with pytest.raises(KeyError):
        reg.get("not_a_symbol")


def test_confusion_sets_are_closed_both_ways():
    """Declared on one side, reported on both."""
    assert "arc_length" in reg.confusion_set("profile_of_a_line")
    assert "profile_of_a_line" in reg.confusion_set("arc_length")


# --- serialization -----------------------------------------------------------

def test_to_dict_is_json_serializable_and_complete():
    import json

    payload = reg.to_dict()
    assert payload["count"] == len(reg.ALL_SYMBOLS)
    assert len(payload["symbols"]) == len(reg.ALL_SYMBOLS)
    json.dumps(payload)  # must not raise

    one = next(s for s in payload["symbols"] if s["id"] == "position")
    for key in (
        "name", "category", "kind", "unicode", "aliases", "ocr_variants",
        "description", "standards", "confusable_with", "strategy", "value",
    ):
        assert key in one


def test_value_specs_never_assert_millimetres():
    """The brief: do not invent units. No ValueSpec may name one."""
    for sym in reg.ALL_SYMBOLS:
        blob = " ".join(
            str(x) for x in (sym.value.note or "", sym.value.pattern or "")
        ).lower()
        assert "millimet" not in blob and " mm" not in blob
