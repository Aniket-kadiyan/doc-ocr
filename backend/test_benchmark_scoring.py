"""
Tests for the document benchmark's scoring (backend/benchmarks/scoring.py).

These are pure logic: no OCR, no drawings. They pin down how a reading is
compared with the ground truth, which is what every document score depends on.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_benchmark_scoring.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.scoring import (
    ExpectedItem,
    digits_of,
    load_document,
    normalize,
    score_document,
    similarity,
)

DOCUMENTS = Path(__file__).parent / "benchmarks" / "documents"


def _item(item_id, accept, type_=None):
    accept = (accept,) if isinstance(accept, str) else tuple(accept)
    return ExpectedItem(id=item_id, accept=accept, type=type_)


def _region(text, type_=None):
    return {"text": text, "type": type_}


# --- normalising a reading ---------------------------------------------------

def test_normalize_strips_space_and_uppercases():
    assert normalize("  r2 ± 0.2 ") == "R2±0.2"


@pytest.mark.parametrize("glyph", ["Ø", "ø", "⌀", "φ", "Φ", "∅"])
def test_diameter_glyphs_are_one_symbol(glyph):
    assert normalize(f"{glyph}30-0.2") == "Ø30-0.2"


@pytest.mark.parametrize("dash", ["-", "−", "–", "—"])
def test_minus_glyphs_are_one_symbol(dash):
    assert normalize(f"15{dash}0.2") == "15-0.2"


def test_degree_and_tick_marks():
    """A ground-truth list rarely carries the minute tick; OCR often does."""
    assert normalize("5°±0°30'") == normalize("5°±0°30")
    assert normalize("32˚20'40\"") == "32°2040"


def test_letter_o_is_not_a_diameter_mark():
    """Folding O into Ø would score a misread diameter as a perfect read."""
    assert normalize("O30") != normalize("Ø30")
    assert 0.5 < similarity("O30-0.2", "Ø30-0.2") < 1.0


# --- similarity --------------------------------------------------------------

def test_identical_after_normalising_is_full_score():
    assert similarity(" Ø33 −0.05 ", "Ø33-0.05") == 1.0


def test_empty_scores_zero():
    assert similarity("", "Ø30") == 0.0
    assert similarity("Ø30", "") == 0.0


def test_partial_read_scores_between():
    assert 0.6 < similarity("Ø23.5-0.05", "23.5-0.05") < 1.0


# --- pairing callouts with detections ----------------------------------------

def test_exact_run_scores_every_callout():
    expected = [_item(1, "Ø30-0.2", "diameter"), _item(2, "R10±2", "radius")]
    regions = [_region("R10±2", "radius"), _region("Ø30-0.2", "diameter")]
    report = score_document(expected, regions)
    assert report.matched == 2
    assert report.missed == 0
    assert report.mean_score == 1.0
    assert report.typed_correct == 2


def test_either_accepted_reading_satisfies_a_callout():
    """Two ground-truth entries for one callout: matching either is a match."""
    expected = [_item(1, ["Ø33-0.05-0.1", "33-0.1-0.05"], "diameter")]
    report = score_document(expected, [_region("Ø33-0.1-0.05", "diameter")])
    assert report.matched == 1
    assert report.items[0].percent > 90


def test_one_detection_cannot_satisfy_two_callouts():
    """A drawing carrying two similar values needs both of them read."""
    expected = [_item(1, "Ø30-0.2", "diameter"), _item(2, "Ø30±0.2", "diameter")]
    report = score_document(expected, [_region("Ø30-0.2", "diameter")])
    assert report.matched == 1
    assert report.items[1].verdict != "match"
    assert report.items[1].detected != report.items[0].detected


def test_missing_callout_is_a_miss_with_no_detection():
    expected = [_item(1, "114.5±0.05", "tolerance")]
    report = score_document(expected, [_region("R2±0.2", "radius")])
    assert report.missed == 1
    assert report.items[0].detected is None
    assert report.items[0].percent < 60


def test_truncated_read_is_partial_not_match():
    expected = [_item(1, "Ø23.5-0.05", "diameter")]
    report = score_document(expected, [_region("Ø23.5", "diameter")])
    assert report.partial == 1
    assert report.items[0].verdict == "partial"


def test_one_wrong_digit_is_never_a_match():
    """Ø23.5-0.05 and Ø23.5-0.06 are 90% alike and are different parts."""
    expected = [_item(1, "Ø23.5-0.05", "diameter")]
    report = score_document(expected, [_region("Ø23.5-0.06", "diameter")])
    assert report.matched == 0
    assert report.items[0].verdict == "partial"
    assert report.items[0].percent >= 90


def test_symbols_around_identical_digits_may_differ():
    """A missing diameter mark still scores as a match on the value."""
    expected = [_item(1, "Ø30-0.2", "diameter")]
    report = score_document(expected, [_region("30-0.2", "diameter")])
    assert report.matched == 1


def test_wrong_type_still_matches_text_but_is_flagged():
    expected = [_item(1, "R2±0.2", "radius")]
    report = score_document(expected, [_region("R2±0.2", "tolerance")])
    assert report.matched == 1
    assert report.typed_correct == 0
    assert report.items[0].type_ok is False


def test_thresholds_are_configurable():
    """Same digits, a missing symbol: whether that passes is the caller's call."""
    expected = [_item(1, "Ø23.5-0.05")]
    regions = [_region("23.5-0.05")]
    assert score_document(expected, regions, match_threshold=0.9).matched == 1
    assert score_document(expected, regions, match_threshold=0.99).matched == 0


# --- leftover detections -----------------------------------------------------

def test_title_block_text_is_separated_from_surprises():
    expected = [_item(1, "Ø30-0.2", "diameter")]
    regions = [
        _region("Ø30-0.2", "diameter"),
        _region("Dat.:23-11-2018", "linear"),
        _region("1P0.0", "linear"),
    ]
    report = score_document(expected, regions, noise_patterns=["^DAT"])
    assert [n["text"] for n in report.noise] == ["Dat.:23-11-2018"]
    assert [e["text"] for e in report.extras] == ["1P0.0"]


def test_a_second_read_of_the_same_callout_is_reported():
    """The duplicate is a real finding, so it must not be silently absorbed."""
    expected = [_item(1, ["Ø33-0.05-0.1", "33-0.1-0.05"], "diameter")]
    regions = [_region("Ø33-0.1-0.05", "diameter"), _region("Ø33-0.05", "diameter")]
    report = score_document(expected, regions)
    assert report.matched == 1
    assert [e["text"] for e in report.extras] == ["Ø33-0.05"]


# --- report shape ------------------------------------------------------------

def test_report_renders_and_serialises():
    expected = [_item(1, "Ø30-0.2", "diameter"), _item(2, "R10±2", "radius")]
    report = score_document(
        expected, [_region("Ø30-0.2", "diameter")], document="test sheet"
    )
    text = report.format_text()
    assert "Ø30-0.2" in text and "matched 1/2" in text
    payload = report.to_dict()
    assert payload["totals"]["expected"] == 2
    assert payload["totals"]["missed"] == 1
    assert len(payload["items"]) == 2
    json.dumps(payload)  # must be serialisable for stored results


# --- the fixtures themselves -------------------------------------------------

def _fixtures():
    return sorted(DOCUMENTS.glob("*.json"))


def test_at_least_one_document_fixture_exists():
    assert _fixtures()


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda p: p.stem)
def test_fixture_is_well_formed(fixture):
    spec = load_document(fixture)
    items = spec["expected"]
    assert items, "a fixture with no expected callouts proves nothing"
    ids = [item.id for item in items]
    assert len(ids) == len(set(ids)), f"duplicate ids in {fixture.name}"
    # Either the composed kind or a displayed category is a valid fixture type.
    known_types = {
        "linear", "diameter", "radius", "angle", "tolerance",
        "basic", "reference", "gd&t", "datum", "thread", "hole", "chamfer",
        "taper", "material", "heat treatment", "coating", "general note",
        "surface finish", "weld", "note", "unknown", None,
    }
    for item in items:
        assert item.accept and all(a.strip() for a in item.accept)
        allowed = item.type.lower() if item.type else None
        assert allowed in known_types, f"unknown type {item.type!r}"
    requires = spec.get("requires", {})
    assert requires.get("matched", 0) <= len(items)


def test_digits_of_ignores_symbols():
    assert digits_of("Ø33-0.05-0.1") == digits_of("33-0.1-0.05".replace("0.1-0.05", "0.05-0.1"))
    assert digits_of("R2±0.2") == "202"
    assert digits_of("") == ""


def test_category_name_counts_as_the_type():
    """A fixture may name what the app displays, not the composed kind."""
    expected = [_item(1, "[228.60]", "Basic")]
    region = {"text": "[228.60]", "type": "linear", "category": "Basic"}
    report = score_document(expected, [region])
    assert report.matched == 1
    assert report.typed_correct == 1


def test_wrong_category_and_kind_is_flagged():
    expected = [_item(1, "[228.60]", "Basic")]
    region = {"text": "[228.60]", "type": "linear", "category": "Reference"}
    report = score_document(expected, [region])
    assert report.matched == 1
    assert report.typed_correct == 0
