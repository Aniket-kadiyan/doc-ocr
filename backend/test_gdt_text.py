"""
Tests for guarded textual GD&T recognition (backend/gdt_text.py).

Spec section 16 was switched off because its rule matched a GD&T *word*
anywhere in a string. These tests hold both halves of the replacement down:
it must still recognise a spelled-out control — which the graphical
recognizer cannot see, there being no glyph — and it must refuse the
measurement notes that caused the original false positive.

The negative corpus is drawn from real drawings in this repository, not
invented. ``test_against_the_real_corpus`` re-derives it from the benchmark
fixtures on every run, so a regression shows up against the actual text the
app sees.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_gdt_text.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

import feature_dictionary as fd
import gdt_text


# --- positives: genuine spelled-out controls ---------------------------------


@pytest.mark.parametrize(
    "text,name",
    [
        ("FLATNESS 0.05", "Flatness"),
        ("STRAIGHTNESS 0.1", "Straightness"),
        ("CIRCULARITY 0.05", "Circularity"),
        ("CYLINDRICITY 0.08", "Cylindricity"),
        ("PERPENDICULARITY 0.02 A", "Perpendicularity"),
        ("PARALLELISM 0.05 A", "Parallelism"),
        ("ANGULARITY 0.1 A B", "Angularity"),
        ("POSITION Ø0.1 A B", "Position"),
        ("TRUE POSITION Ø0.25 A B C", "Position"),
        ("SYMMETRY 0.2 A", "Symmetry"),
        ("CONCENTRICITY 0.1 A", "Concentricity"),
    ],
)
def test_a_spelled_out_control_with_a_tolerance_is_recognised(text, name):
    match, _ = gdt_text.recognize(text)
    assert match is not None, f"{text!r} was not recognised"
    assert match.name == name
    assert gdt_text.E_KEYWORD_WITH_VALUE in match.evidence


@pytest.mark.parametrize("glyph,name", [("⏥", "Flatness"), ("⌭", "Cylindricity"), ("⌖", "Position")])
def test_an_unambiguous_glyph_is_enough_on_its_own(glyph, name):
    match, _ = gdt_text.recognize(f"{glyph} 0.05 A")
    assert match is not None and match.name == name
    assert gdt_text.E_GLYPH in match.evidence
    assert match.confidence >= 0.85


def test_a_pipe_delimited_frame_is_recognised():
    match, _ = gdt_text.recognize("| Ø0.1 | A | B |")
    assert match is not None
    assert gdt_text.E_PIPE_FRAME in match.evidence


def test_datum_references_are_read_in_order():
    match, _ = gdt_text.recognize("POSITION Ø0.25 A B C")
    assert match is not None
    assert match.datums == ["A", "B", "C"]
    assert gdt_text.E_DATUM_LIST in match.evidence


def test_a_diametral_zone_is_noticed():
    match, _ = gdt_text.recognize("POSITION Ø0.1 A")
    assert match is not None and match.diameter_zone is True
    plain, _ = gdt_text.recognize("FLATNESS 0.05")
    assert plain is not None and plain.diameter_zone is False


def test_the_tolerance_value_is_read():
    match, _ = gdt_text.recognize("FLATNESS 0.05")
    assert match is not None and match.value == 0.05


def test_a_datum_list_raises_confidence_above_a_bare_value():
    with_datum, _ = gdt_text.recognize("PARALLELISM 0.05 A")
    without, _ = gdt_text.recognize("PARALLELISM 0.05")
    assert with_datum is not None and without is not None
    assert with_datum.confidence > without.confidence


def test_a_match_carries_the_stable_symbol_id():
    match, _ = gdt_text.recognize("PERPENDICULARITY 0.02 A")
    assert match is not None
    assert match.symbol_id == "perpendicularity"


def test_the_source_text_is_preserved_verbatim():
    """A GD&T label has to be arguable, which needs what it was read from."""
    text = "  POSITION Ø0.1 A B  "
    match, _ = gdt_text.recognize(text)
    assert match is not None
    assert match.source_text == text.strip()
    assert match.method == "text_rule"


# --- negatives: the false positives that kept section 16 off -----------------


@pytest.mark.parametrize(
    "text",
    [
        "ROUNDNESS(2PT./180)",
        "ROUNDNESS(2PT./180)50µm MAX.",
        "ROUNDNESS(3PT./120)50µm MAX.",
        "SQUARENESS WITH FACE 50µm MAX.",
        "ECCENTRICITY WITH 500µm MAX.",
        "TAPER 50µm MAX.",
        "SURFACE ROUGHNESS ON 6.3Ra µm",
    ],
)
def test_a_measurement_specification_is_refused(text):
    """These state how to measure, not a control. The original FP class."""
    match, _ = gdt_text.recognize(text)
    assert match is None, f"{text!r} was wrongly claimed as GD&T"


@pytest.mark.parametrize(
    "text",
    [
        "POSITION",
        "FLATNESS",
        "CIRCULARITY A",
        "SEE POSITION OF DATUM",
    ],
)
def test_a_control_name_with_no_tolerance_is_refused(text):
    match, rejections = gdt_text.recognize(text, collect_rejections=True)
    assert match is None
    assert rejections, "a refusal must say why"


@pytest.mark.parametrize(
    "text",
    ["Ø25.4", "R5.00", "45°", "3,41±0.30", "M10x1.5", "25 REF", "MATERIAL SAE52100"],
)
def test_ordinary_dimension_text_is_untouched(text):
    assert gdt_text.recognize(text)[0] is None


@pytest.mark.parametrize("text", ["http://example.com//x", "1 // 2", "A // B"])
def test_a_double_slash_never_decides(text):
    """`//` is in the display table for parallelism and is not evidence."""
    assert gdt_text.recognize(text)[0] is None


@pytest.mark.parametrize("text", ["○ 0.05", "⌒ 12.5", "∠ 45", "↗ 0.1"])
def test_ambiguous_glyphs_never_decide_alone(text):
    """A circle, an arc, an angle mark and an arrow all mean other things."""
    assert gdt_text.recognize(text)[0] is None


def test_a_long_note_is_refused_even_when_it_names_controls():
    note = (
        "BEARING NO T5232224 ACCURACIES ROUNDNESS(2PT./180)50µm MAX. "
        "TAPER 50µm MAX. SQUARENESS WITH FACE 50µm MAX. ECCENTRICITY "
        "WITH 500µm MAX. WIDTH VARIATION SURFACE ROUGHNESS ON 6.3Ra µm"
    )
    match, rejections = gdt_text.recognize(note, collect_rejections=True)
    assert match is None
    assert any(r.reason == gdt_text.R_NOTE_BLOCK for r in rejections)


def test_empty_and_whitespace_are_refused():
    assert gdt_text.recognize("")[0] is None
    assert gdt_text.recognize("   ")[0] is None


# --- the guards themselves ---------------------------------------------------


def test_text_longer_than_a_callout_is_refused():
    long_text = "FLATNESS 0.05 " + "X" * gdt_text.MAX_CALLOUT_CHARS
    assert gdt_text.recognize(long_text)[0] is None


def test_the_note_block_heuristic_is_not_applied_to_short_callouts():
    """`looks_like_note_block` calls "POSITION Ø0.1 A B" a paragraph.

    It counts any word containing a letter, and `"Ø".isalpha()` is true, so a
    diameter marker and two bare datum letters reach its four-alpha-word bar.
    It is a good paragraph test and a bad callout test, so it is only
    consulted above `_NOTE_BLOCK_MIN_WORDS`.
    """
    assert fd.looks_like_note_block("POSITION Ø0.1 A B") is True
    assert gdt_text.recognize("POSITION Ø0.1 A B")[0] is not None


def test_ambiguous_glyphs_are_excluded_from_the_strong_table():
    for glyph in ("○", "//", "∥", "⌒", "∠", "↗"):
        assert glyph not in gdt_text.UNAMBIGUOUS_GLYPHS


def test_the_strong_table_is_not_empty_and_comes_from_the_dictionary():
    assert gdt_text.UNAMBIGUOUS_GLYPHS
    for glyph in gdt_text.UNAMBIGUOUS_GLYPHS:
        assert glyph in fd.GDT_SYMBOLS


def test_describe_guards_lists_what_is_required():
    text = gdt_text.describe_guards()
    assert "note block" in text and "measurement specification" in text


# --- against the real corpus -------------------------------------------------

BACKEND = Path(__file__).resolve().parent


def _real_strings() -> list[str]:
    """Every OCR string the repository's own fixtures carry."""
    out: list[str] = []
    for path in glob.glob(str(BACKEND / "benchmarks" / "documents" / "*.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in data.get("expected", []):
            accept = item.get("accept")
            out.extend(accept if isinstance(accept, list) else [accept])
    for path in glob.glob(str(BACKEND / "benchmarks" / "results" / "*.regions.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("regions", [])
        out.extend(r["text"] for r in rows if isinstance(r, dict) and r.get("text"))
    parsed = BACKEND.parent / "TS2 32224 CUP TURNING DRAWING.parsed.json"
    if parsed.exists():
        data = json.loads(parsed.read_text(encoding="utf-8"))
        for result in data.get("results", []):
            out.extend(
                r["text"] for r in (result.get("regions") or []) if r.get("text")
            )
    return [t for t in out if t]


def test_the_real_corpus_is_actually_loaded():
    assert len(_real_strings()) > 100


def test_against_the_real_corpus():
    """No string from four real drawings may be claimed as a GD&T control.

    None of them is one. The old rule claimed exactly one — the cup-turning
    drawing's accuracy note — and that single false positive is the whole
    reason the section was switched off.
    """
    claimed = [t for t in _real_strings() if gdt_text.recognize(t)[0] is not None]
    assert claimed == [], f"false positives on real drawings: {claimed[:3]}"
