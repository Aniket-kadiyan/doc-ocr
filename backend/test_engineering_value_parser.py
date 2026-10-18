"""Fast M5 tests for the lossless engineering-value parser."""

from __future__ import annotations

import json

from engineering_value_parser import (
    engineering_parse_statistics,
    parse_engineering_value,
)


def test_tokens_reconstruct_every_normalized_character() -> None:
    parsed = parse_engineering_value("  Ø8  +0.2 / 0  ")

    assert parsed.raw_text == "  Ø8  +0.2 / 0  "
    assert "".join(token.text for token in parsed.tokens) == parsed.normalized_text
    assert parsed.unparsed_fragments == ()


def test_common_value_families_are_structured_without_disposition() -> None:
    cases = {
        "4X Ø10 THRU": ("linear", "10"),
        "2 X 45°": ("chamfer", None),
        "M8 x 1.25-6H THRU": ("thread", "8"),
        "32°20'40\"": ("angle", None),
        "2:1": ("ratio", None),
        "(25)": ("linear", "25"),
    }

    for text, (kind, nominal) in cases.items():
        parsed = parse_engineering_value(text)
        assert parsed.complete, text
        assert parsed.kind == kind, text
        if nominal is not None:
            assert parsed.components["nominal"] == nominal

    assert parse_engineering_value("4X Ø10 THRU").components["quantity"] == "4"
    assert parse_engineering_value("4X Ø10 THRU").components["qualifiers"] == [
        "THRU"
    ]
    assert parse_engineering_value("(25)").components["reference"] is True


def test_short_chamfer_keeps_its_explicit_size_and_implied_angle() -> None:
    for text, size, qualifiers in (
        ("C3", "3", []),
        ("C0.5", "0.5", []),
        ("CHF2", "2", []),
        ("C2.5 REF", "2.5", ["REF"]),
    ):
        parsed = parse_engineering_value(text)
        assert parsed.complete, text
        assert parsed.kind == "chamfer", text
        assert parsed.components["size"] == size, text
        assert parsed.components["angle_degrees"] == "45", text
        assert parsed.components["implied_angle"] is True, text
        assert parsed.components["qualifiers"] == qualifiers, text
        assert "implied_45_degree_chamfer" in parsed.warnings, text


def test_angle_minutes_tolerance_keeps_angular_magnitude() -> None:
    parsed = parse_engineering_value("60°±0°30'")

    assert parsed.complete
    assert parsed.kind == "angle"
    assert parsed.components["tolerance"]["upper"] == "+0°30'"
    assert parsed.components["tolerance"]["lower"] == "-0°30'"


def test_tolerance_bounds_are_explicit_and_missing_bound_stays_missing() -> None:
    upper = parse_engineering_value("0.1+0.1")
    lower = parse_engineering_value("15-0.2")
    pair = parse_engineering_value("Ø8 +0.2/0")
    symmetric = parse_engineering_value("25±0.05")

    assert upper.components["tolerance"] == {
        "mode": "single_deviation",
        "text": "+0.1",
        "upper": "+0.1",
        "lower": None,
        "degree": False,
    }
    assert lower.components["tolerance"]["upper"] is None
    assert lower.components["tolerance"]["lower"] == "-0.2"
    assert "single_bound_tolerance" in upper.warnings
    assert pair.components["tolerance"]["upper"] == "+0.2"
    assert pair.components["tolerance"]["lower"] == "0"
    assert symmetric.components["tolerance"]["upper"] == "+0.05"
    assert symmetric.components["tolerance"]["lower"] == "-0.05"


def test_common_ascii_plus_minus_and_compact_pairs_are_explicit_normalization() -> None:
    symmetric = parse_engineering_value("30+-3")
    pair = parse_engineering_value("25+0.1-0.2")

    assert symmetric.complete
    assert symmetric.normalized_text == "30±3"
    assert "ascii_plus_minus" in symmetric.normalization_steps
    assert pair.complete
    assert pair.components["tolerance"]["upper"] == "+0.1"
    assert pair.components["tolerance"]["lower"] == "-0.2"


def test_iso_fit_is_not_misread_as_malformed_tolerance() -> None:
    parsed = parse_engineering_value("2N9")

    assert parsed.complete
    assert parsed.kind == "linear"
    assert parsed.components["nominal"] == "2"
    assert parsed.components["fit"] == "N9"
    assert parsed.components["tolerance"] is None


def test_spaced_and_lowercase_iso_fits_preserve_their_semantics() -> None:
    cases = {
        "Ø4 H7": "H7",
        "Ø18h6": "h6",
        "20 JS9": "JS9",
    }

    for text, fit in cases.items():
        parsed = parse_engineering_value(text)
        assert parsed.complete, text
        assert parsed.kind == "linear", text
        assert parsed.components["fit"] == fit, text


def test_hole_callouts_are_complete_and_keep_modifier_values() -> None:
    depth = parse_engineering_value("Ø4 H7 ↧10")
    counterbore = parse_engineering_value("Ø6.50 ⌴Ø11 ↧4.40")
    countersink = parse_engineering_value("Ø3.50 ⌵Ø7 X 90°")
    thread = parse_engineering_value("M3x0.5 - 6H ↧3")

    for parsed in (depth, counterbore, countersink, thread):
        assert parsed.complete
        assert parsed.kind == "hole"
        assert "".join(token.text for token in parsed.tokens) == parsed.normalized_text

    assert depth.components["feature"]["fit"] == "H7"
    assert depth.components["depths"] == ["10"]
    assert counterbore.components["counterbores"] == [
        {"diameter": "11", "diameter_symbol": True}
    ]
    assert counterbore.components["depths"] == ["4.40"]
    assert countersink.components["countersinks"] == [
        {"diameter": "7", "diameter_symbol": True}
    ]
    assert thread.components["feature"]["fit"] == "6H"


def test_standalone_depth_and_hole_glyph_variants_are_lossless() -> None:
    parsed = parse_engineering_value("↓ 0.90")

    assert parsed.raw_text == "↓ 0.90"
    assert parsed.normalized_text == "↧0.90"
    assert parsed.complete
    assert parsed.kind == "depth"
    assert parsed.components["depth"] == "0.90"
    assert "hole_glyph_variant" in parsed.normalization_steps


def test_incomplete_tolerance_is_partial_and_nothing_is_deleted() -> None:
    parsed = parse_engineering_value("R0.6-0.")

    assert parsed.status == "partial"
    assert parsed.kind == "linear"
    assert parsed.normalized_text == "R0.6-0."
    assert parsed.components["prefix"] == "R"
    assert parsed.components["nominal"] == "0.6"
    assert parsed.components["tolerance"]["lower"] == "-0."
    assert parsed.unparsed_fragments == ()
    assert parsed.warnings == ("incomplete_tolerance_decimal",)


def test_unknown_suffix_is_retained_as_an_exact_fragment() -> None:
    parsed = parse_engineering_value("0.02A")

    assert parsed.status == "partial"
    assert parsed.kind == "linear"
    assert parsed.components["nominal"] == "0.02"
    assert [fragment.text for fragment in parsed.unparsed_fragments] == ["A"]
    assert "".join(token.text for token in parsed.tokens) == "0.02A"


def test_dual_unit_keeps_both_complete_component_parses() -> None:
    parsed = parse_engineering_value("2.25[57.15]REF")

    assert parsed.complete
    assert parsed.kind == "dual_unit"
    assert parsed.components["primary"]["components"]["nominal"] == "2.25"
    assert parsed.components["secondary"]["components"]["nominal"] == "57.15"
    assert parsed.components["qualifiers"] == ["REF"]


def test_parser_output_is_json_safe_and_statistics_are_deterministic() -> None:
    parses = [
        parse_engineering_value("25"),
        parse_engineering_value("R0.6-0."),
        parse_engineering_value("MATERIAL"),
        parse_engineering_value(""),
    ]

    json.dumps([parsed.to_dict() for parsed in parses])
    stats = engineering_parse_statistics(parses)

    assert stats["object_count"] == 4
    assert stats["complete_count"] == 1
    assert stats["partial_count"] == 1
    assert stats["unparsed_count"] == 1
    assert stats["empty_count"] == 1
    assert stats["kind_counts"] == {"linear": 2, "unknown": 2}
