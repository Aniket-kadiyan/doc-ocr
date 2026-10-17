"""
Tests for the operator-correction store (backend/gdt_corrections.py).

Writes to a temporary directory, never to the configured data location, so
running the suite never pollutes a deployment's corrections log.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_gdt_corrections.py
"""

from __future__ import annotations

import json

import pytest

import gdt_corrections as corrections


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    """Point the store at a throwaway directory for every test."""
    monkeypatch.setenv("GDT_CORRECTIONS_DIR", str(tmp_path / "corr"))
    yield tmp_path


def make(**overrides) -> corrections.Correction:
    payload = {
        "action": corrections.ACTION_RECLASSIFY,
        "detected_symbol_id": "circularity",
        "corrected_symbol_id": "diameter",
        "page": 1,
        "bounding_box": {"x": 10.0, "y": 20.0, "width": 8.0, "height": 8.0},
        "detected_confidence": 0.61,
        "detected_status": "ambiguous",
        "document_hash": "abc123",
    }
    payload.update(overrides)
    return corrections.Correction(**payload)


# --- validation --------------------------------------------------------------


def test_a_reclassification_needs_a_target():
    with pytest.raises(corrections.CorrectionError, match="corrected_symbol_id"):
        make(corrected_symbol_id=None).validate()


def test_an_addition_needs_a_symbol():
    with pytest.raises(corrections.CorrectionError):
        make(action=corrections.ACTION_ADD, corrected_symbol_id=None).validate()


def test_a_rejection_needs_the_detection_it_rejects():
    with pytest.raises(corrections.CorrectionError):
        make(
            action=corrections.ACTION_REJECT,
            detected_symbol_id=None,
            corrected_symbol_id=None,
        ).validate()


def test_an_unknown_symbol_id_is_refused():
    """A correction naming a symbol the registry lacks is not usable later."""
    with pytest.raises(corrections.CorrectionError, match="unknown symbol id"):
        make(corrected_symbol_id="not_a_symbol").validate()


def test_an_unknown_action_is_refused():
    with pytest.raises(corrections.CorrectionError, match="action must be"):
        make(action="delete_everything").validate()


def test_a_malformed_bounding_box_is_refused():
    with pytest.raises(corrections.CorrectionError, match="bounding_box"):
        make(bounding_box={"x": 1, "y": 2}).validate()


def test_a_confirmation_needs_nothing_extra():
    make(
        action=corrections.ACTION_CONFIRM, corrected_symbol_id=None
    ).validate()


# --- persistence -------------------------------------------------------------


def test_a_correction_round_trips():
    saved = corrections.record(make())
    entries = corrections.read_all()
    assert len(entries) == 1
    assert entries[0]["id"] == saved.id
    assert entries[0]["corrected_symbol_id"] == "diameter"
    assert entries[0]["recorded_at"]


def test_corrections_append_rather_than_replace():
    corrections.record(make(note="first"))
    corrections.record(make(note="second"))
    notes = [e["note"] for e in corrections.read_all()]
    assert notes == ["first", "second"]


def test_corrections_filter_by_document():
    corrections.record(make(document_hash="aaa"))
    corrections.record(make(document_hash="bbb"))
    assert len(corrections.read_all(document_hash="aaa")) == 1
    assert len(corrections.read_all()) == 2


def test_reading_an_absent_log_is_empty_not_an_error():
    assert corrections.read_all() == []


def test_a_truncated_line_does_not_make_the_history_unreadable():
    """An interrupted write must cost one record, not the whole file."""
    corrections.record(make(note="good"))
    path = corrections.corrections_dir() / "corrections.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"action": "reclassi')  # torn write
    entries = corrections.read_all()
    assert len(entries) == 1
    assert entries[0]["note"] == "good"


def test_the_store_holds_no_image_data():
    """Retention: corrections reference a drawing, they do not copy it."""
    corrections.record(make())
    path = corrections.corrections_dir() / "corrections.jsonl"
    entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert set(entry) == {
        "action", "detected_symbol_id", "corrected_symbol_id", "page",
        "bounding_box", "detected_confidence", "detected_status",
        "document_hash", "note", "id", "recorded_at",
    }


# --- summary -----------------------------------------------------------------


def test_the_summary_counts_which_pairs_are_actually_confused():
    corrections.record(make(detected_symbol_id="circularity", corrected_symbol_id="diameter"))
    corrections.record(make(detected_symbol_id="circularity", corrected_symbol_id="diameter"))
    corrections.record(make(detected_symbol_id="counterbore", corrected_symbol_id="spotface"))
    corrections.record(
        make(action=corrections.ACTION_CONFIRM, corrected_symbol_id=None)
    )
    result = corrections.summary()
    assert result["total"] == 4
    assert result["by_action"]["reclassify"] == 3
    assert result["by_action"]["confirm"] == 1
    assert result["confusions"]["circularity->diameter"] == 2
    # Ordered most-confused first, which is what triage wants.
    assert list(result["confusions"])[0] == "circularity->diameter"


def test_the_directory_is_configurable(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere"
    monkeypatch.setenv("GDT_CORRECTIONS_DIR", str(target))
    corrections.record(make())
    assert (target / "corrections.jsonl").exists()
