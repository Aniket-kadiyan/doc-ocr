"""Whole-page filter regressions; no OCR models are loaded."""

from __future__ import annotations

from dataclasses import replace

from page_value_filters import (
    NEVER_BALLOON_RULES,
    PageValueCandidate,
    candidate_is_in_table,
    evaluate_page_value,
    evaluate_scan_value,
    needs_expanded_filter_context,
    normalize_page_value_text,
)


def candidate(
    text: str,
    *,
    x: float = 10,
    y: float = 10,
    width: float = 80,
    height: float = 16,
    context: str = "",
) -> PageValueCandidate:
    return PageValueCandidate(
        text=text,
        bbox={
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        context_text=context,
    )


def test_complete_engineering_values_need_neither_units_nor_geometry() -> None:
    values = (
        "50",
        ".25",
        "25.00",
        "R5.00",
        "Ø24.80",
        "15°±3°",
        "30 +/- 3",
        "30+-3",
        "32°20'40\"",
        "25 +0.1 -0.2",
        "M8",
        "M8 x 1.25",
        "2X Ø10",
        "4X Ø10 THRU",
        "Ø8 +0.2/0",
        "2 X 45°",
        "SS304",
        "TS232224",
        "R0.2 MAX",
        "2:1",
    )
    for text in values:
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, text
        assert decision.rule_name == "engineering_value"


def test_harmless_symbol_and_boundary_noise_is_normalized() -> None:
    cases = {
        "?30º +/- 3º": "30° ± 3°",
        "|R5.00;": "R5.00",
        "12˚±3˚": "12°±3°",
        "30 + - 3": "30 ± 3",
    }

    for source, expected in cases.items():
        normalized = normalize_page_value_text(source)
        decision = evaluate_page_value(candidate(source))

        assert normalized == expected
        assert decision.accepted


def test_noise_is_not_used_to_extract_a_number_from_mixed_text() -> None:
    text = "?ZONE A 25;"

    assert normalize_page_value_text(text) == text
    decision = evaluate_page_value(candidate(text))
    assert not decision.accepted
    assert decision.rule_name == "invalid_engineering_value"


def test_pure_text_is_rejected() -> None:
    decision = evaluate_page_value(candidate("MATERIAL"))

    assert not decision.accepted
    assert decision.rule_name == "no_numeric_component"


def test_direct_never_balloon_values_are_rejected_before_numeric_acceptance() -> None:
    cases = {
        "DETAIL B SCALE 2:1": "detail_view_section",
        "D3TAIL B": "detail_view_section",
        "SC4LE 2:1": "scale_information",
        "15.07.2024": "date",
        "REV1SION 3": "revision_history",
        "MOD1FICATIONS 3": "revision_history",
        "RELEA5ED 3": "revision_history",
        "N0TE 4": "note_information",
        "PART NO TS232224": "document_metadata",
        "SHEET 1 OF 1": "document_metadata",
    }

    for text, rule_name in cases.items():
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted, text
        assert decision.rule_name == rule_name


def test_split_metadata_value_is_rejected_only_when_its_label_is_nearby() -> None:
    label = candidate("PART NUMBER", x=10, y=10, width=90)
    value = candidate("TS232224", x=110, y=10, width=85)

    associated = evaluate_page_value(
        value,
        page_candidates=(label, value),
    )
    standalone = evaluate_page_value(value, page_candidates=(value,))

    assert not associated.accepted
    assert associated.rule_name == "document_metadata"
    assert standalone.accepted


def test_a_drawing_number_beside_its_label_is_not_a_dimension() -> None:
    label = candidate("DWG NO", x=10, y=10, width=60)
    number = candidate("7475870300", x=80, y=10, width=200)

    decision = evaluate_page_value(number, page_candidates=(label, number))

    assert not decision.accepted
    assert decision.rule_name == "document_metadata"


def test_a_long_bare_number_without_a_label_remains_a_value() -> None:
    number = candidate("123456", x=900, y=900, width=140)

    assert evaluate_page_value(number, page_candidates=(number,)).accepted


def test_short_integer_dimensions_are_not_treated_as_document_numbers() -> None:
    label = candidate("DWG NO", x=10, y=10, width=60)
    for text in ("21", "140", "12345"):
        value = candidate(text, x=80, y=10, width=60)
        decision = evaluate_page_value(value, page_candidates=(label, value))
        assert decision.accepted, text


def test_distant_metadata_label_does_not_exclude_a_numeric_value() -> None:
    label = candidate("PART NUMBER", x=10, y=10, width=90)
    value = candidate("50", x=700, y=500, width=30)

    decision = evaluate_page_value(
        value,
        page_candidates=(label, value),
    )

    assert decision.accepted


def test_reconstructed_context_filters_a_split_scale_value_only() -> None:
    value = candidate("2:1", context="DETAIL B SCALE 2:1")

    decision = evaluate_page_value(value, page_candidates=(value,))

    assert not decision.accepted
    assert decision.rule_name == "detail_view_section"
    assert needs_expanded_filter_context("2:1", value.bbox)
    assert not needs_expanded_filter_context("50", value.bbox)


def test_nearby_labels_do_not_exclude_strong_dimensions() -> None:
    angle = candidate("30°±3°", x=100, y=100, width=70)
    nearby_labels = (
        candidate("DETAIL B", x=175, y=100, width=70),
        candidate("SCALE", x=100, y=120, width=55),
        candidate("REVISION", x=160, y=120, width=70),
        angle,
    )

    decision = evaluate_page_value(angle, page_candidates=nearby_labels)

    assert decision.accepted
    assert decision.rule_name == "engineering_value"


def test_nearby_labels_exclude_only_context_prone_values() -> None:
    scale_label = candidate("SC4LE", x=10, y=10, width=55)
    ratio = candidate("2:1", x=70, y=10, width=35)
    revision_label = candidate("MODIFICATIONS", x=10, y=50, width=110)
    revision_number = candidate("3", x=125, y=50, width=12)

    scale_decision = evaluate_page_value(
        ratio,
        page_candidates=(scale_label, ratio),
    )
    revision_decision = evaluate_page_value(
        revision_number,
        page_candidates=(revision_label, revision_number),
    )

    assert not scale_decision.accepted
    assert scale_decision.rule_name == "scale_information"
    assert not revision_decision.accepted
    assert revision_decision.rule_name == "revision_history"


def test_isolated_confusable_digits_and_mixed_numeric_text_require_review() -> None:
    for text in ("0", "1", "8"):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted
        assert decision.rule_name == "ambiguous_single_character"

    for text in ("ZONE A 25", "MATERIAL 50", "A B 12.5"):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted
        assert decision.rule_name == "invalid_engineering_value"


def test_each_never_balloon_rule_can_be_disabled_in_code() -> None:
    scale_rules = tuple(
        replace(rule, enabled=False)
        if rule.name == "scale_information"
        else rule
        for rule in NEVER_BALLOON_RULES
    )

    label = candidate("SCALE", x=10, y=10, width=55)
    value = candidate("2:1", x=70, y=10, width=35)
    decision = evaluate_page_value(
        value,
        page_candidates=(label, value),
        rules=scale_rules,
    )

    assert decision.accepted


def test_table_region_is_available_to_both_scope_policies() -> None:
    mask = {"x": 100, "y": 100, "width": 200, "height": 120}
    inside = candidate("50", x=120, y=130, width=30, height=14)

    for scope_kind in ("page", "section"):
        decision = evaluate_scan_value(
            inside,
            scope_kind=scope_kind,
            table_masks=(mask,),
        )
        assert not decision.accepted
        assert decision.rule_name == "table_region"


def test_section_light_filter_keeps_short_values_and_rejects_garbage() -> None:
    accepted = ("0", "1", "6", "8", "50", "105", "Ø6", "R3", "30+-3")
    rejected = ("A", "B", "D", "---", "DETAIL B (1:1)", "ZONE 25 VALUE")

    for text in accepted:
        decision = evaluate_scan_value(candidate(text), scope_kind="section")
        assert decision.accepted, text
        assert decision.rule_name == "section_engineering_value"

    for text in rejected:
        decision = evaluate_scan_value(candidate(text), scope_kind="section")
        assert not decision.accepted, text


def test_mixed_section_rejects_only_candidate_inside_the_table() -> None:
    mask = {"x": 100, "y": 100, "width": 200, "height": 120}
    table_value = candidate("50", x=120, y=130, width=30, height=14)
    drawing_value = candidate("50", x=20, y=30, width=30, height=14)

    assert candidate_is_in_table(table_value.bbox, (mask,))
    assert not candidate_is_in_table(drawing_value.bbox, (mask,))
    decision = evaluate_scan_value(
        drawing_value,
        scope_kind="section",
        table_masks=(mask,),
    )
    assert decision.accepted
    assert decision.rule_name == "section_engineering_value"


def _at(text: str, x: float, y: float, width: float, height: float):
    return PageValueCandidate(
        text=text,
        bbox={"x": x, "y": y, "width": width, "height": height},
    )


TEXT_LINE = _at("", 100, 200, 460, 16)


def _letters_along_the_line(count: int = 8):
    return [
        _at("", 110 + index * 50, 201, 20, 14)
        for index in range(count)
    ]


def test_fragments_of_a_detected_text_line_are_not_values() -> None:
    others = _letters_along_the_line()
    for fragment in (
        _at("Roper0", 250, 201, 70, 15),
        _at("031", 330, 202, 22, 14),
        _at("35", 410, 203, 12, 12),
    ):
        decision = evaluate_page_value(
            fragment,
            page_candidates=[TEXT_LINE, fragment, *others],
        )
        assert not decision.accepted, fragment.text
        assert decision.rule_name == "text_line_fragment", fragment.text


def test_two_pieces_are_a_split_callout_not_a_text_line() -> None:
    fused = _at("", 100, 200, 460, 16)
    left = _at("25.4", 110, 201, 180, 14)
    right = _at("31.8", 310, 201, 180, 14)

    decision = evaluate_page_value(left, page_candidates=[fused, left, right])

    assert decision.accepted, decision.rule_name


def test_a_dimension_beside_a_text_line_remains_a_value() -> None:
    value = _at("Ø18.6", 700, 201, 60, 15)

    decision = evaluate_page_value(
        value,
        page_candidates=[TEXT_LINE, value, *_letters_along_the_line()],
    )

    assert decision.accepted, decision.rule_name


def test_misread_lowercase_words_are_not_compact_identifiers() -> None:
    for text in ("e11", "listed2", "Roper0", "mus1", "proper1"):
        assert not evaluate_page_value(candidate(text)).accepted, text


def test_uppercase_compact_identifiers_still_pass_the_page_grammar() -> None:
    for text in ("ISD-600-500Y", "ISC-A00-008Y", "SCM415", "E545", "R24"):
        assert evaluate_page_value(candidate(text)).accepted, text


def test_short_chamfer_forms_are_complete_values() -> None:
    for text in (
        "C3",
        "C0.5",
        "C 3",
        "CHF2",
        "C3MAX",
        "C2.5 REF",
        "CHF1 MIN.",
        "C4 TYP",
    ):
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, text
        assert decision.rule_name == "engineering_value", text


def test_short_chamfer_rule_does_not_accept_bare_or_zone_letters() -> None:
    for text in ("C", "CHF", "A3", "B2"):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted, text


def test_hole_callouts_and_iso_fits_are_complete_values() -> None:
    values = (
        "Ø2.50 ↧13",
        "M3x0.5 - 6H ↧3",
        "M3x0.5 - 6H THRU ALL",
        "2 x Ø2.50 ↧8.50",
        "2 x Ø2.50 THRU ALL",
        "Ø4 H7 ↧10",
        "Ø6.50 ⌴Ø11 ↧4.40",
        "Ø3.50 ⌵Ø7 X 90°",
        "Ø3.50 X 90° NEAR SIDE",
        "13 DEEP",
        "↧0.90",
        "Ø4 H7",
        "Ø18h6",
        "20 JS9",
    )

    for text in values:
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, (text, decision.rule_name)


def test_hole_glyph_normalization_preserves_meaning() -> None:
    assert normalize_page_value_text("2.50 ↓13") == "2.50 ↧13"
    assert normalize_page_value_text("2 x ∅2.50 THRU") == "2 x Ø2.50 THRU"
    assert normalize_page_value_text("↧ 0.90") == "↧0.90"
    assert normalize_page_value_text("M3x0.5-6H T4.50") == "M3x0.5-6H↧4.50"


def test_hole_repairs_do_not_change_arrows_prose_or_identifiers() -> None:
    assert normalize_page_value_text("→ 25") == "25"
    assert normalize_page_value_text("↑25") == "25"
    for text in ("Ø50T7", "ISD-600T500", "20 TO 25", "PART T5", "T10"):
        assert normalize_page_value_text(text) == text
    for text in ("3 PLACES", "DO NOT SCALE", "REMOVE ALL BURRS 0.2"):
        assert not evaluate_page_value(candidate(text)).accepted


def test_stacked_deviations_are_complete_values() -> None:
    for text in (
        "Ø33 +0.012 0",
        "Ø33 -0.05 -0.10",
        "+0.012 0",
        "+0.05 -0.05",
        "-0.05 -0.10",
    ):
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, (text, decision.rule_name)


def test_stacked_deviation_grammar_does_not_accept_two_plain_numbers() -> None:
    for text in ("33 0.5", "25 40", "12.5 0.25", "+0.0120.0"):
        assert not evaluate_page_value(candidate(text)).accepted, text


def test_diametral_tolerance_with_datums_is_a_complete_value() -> None:
    for text in (
        "Ø0.02A",
        "Ø0.02 A",
        "Ø0.1 A B",
        "Ø0.3 A B C",
        "Ø0.02Ⓜ A B",
    ):
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, (text, decision.rule_name)


def test_datum_grammar_does_not_fuse_an_ordinary_length_and_label() -> None:
    for text in ("25 A", "0.5 m", "12.5 B", "Ø0.02 a", "Ø0.02 I"):
        assert not evaluate_page_value(candidate(text)).accepted, text


def test_orphaned_closing_parenthesis_is_guarded_boundary_noise() -> None:
    assert normalize_page_value_text("Ø20.86)") == "Ø20.86"
    assert normalize_page_value_text("(20.86)") == "(20.86)"
    assert normalize_page_value_text("1.00[25.40]") == "1.00[25.40]"
    assert normalize_page_value_text("ZONE A 25)") == "ZONE A 25)"


def _detail_caption_page(value_text: str) -> tuple[PageValueCandidate, ...]:
    value = candidate(value_text, x=1520, y=390, width=160, height=34)
    return (
        candidate("DETAIL C", x=1500, y=300, width=200, height=34),
        candidate("SCALE 5:1", x=1500, y=340, width=200, height=34),
        value,
    )


def test_dimensions_inside_detail_view_survive_caption_proximity() -> None:
    for text in ("Ø20.6 -0.1", "0.30 X 45°", "Ø2.50 ↧13"):
        page = _detail_caption_page(text)
        decision = evaluate_page_value(page[-1], page_candidates=page)

        assert decision.accepted, (text, decision.rule_name)
        assert decision.rule_name == "engineering_value"


def test_damaged_dimensions_under_caption_reach_invalid_value_review_path() -> None:
    for text in ("20.6~0.1", "20.86=0.01"):
        page = _detail_caption_page(text)
        decision = evaluate_page_value(page[-1], page_candidates=page)

        assert not decision.accepted, text
        assert decision.rule_name == "invalid_engineering_value"


def test_only_split_caption_shapes_inherit_nearby_caption_exclusion() -> None:
    for text in ("C", "A-A", "(3x)", "B(3x)", "1:1", "5:1", "Scale1:1"):
        page = _detail_caption_page(text)
        decision = evaluate_page_value(page[-1], page_candidates=page)

        assert not decision.accepted, text
        assert decision.rule_name in {"detail_view_section", "scale_information"}


def test_short_real_values_ignore_view_caption_in_rebuilt_context() -> None:
    for text in ("25", "R3", "C3", "0.5", "R212.5"):
        value = candidate(text, context="SECTION VIEW B-B")
        decision = evaluate_page_value(value, page_candidates=(value,))

        assert decision.accepted, (text, decision.rule_name)
        assert decision.rule_name == "engineering_value"


def test_malformed_value_ignores_view_caption_in_rebuilt_context() -> None:
    value = candidate("20.6~0.1", context="DETAIL C SCALE 5:1")
    decision = evaluate_page_value(value, page_candidates=(value,))

    assert not decision.accepted
    assert decision.rule_name == "invalid_engineering_value"


def test_direct_view_caption_text_remains_excluded() -> None:
    cases = {
        "SECTION VIEW B-B": "detail_view_section",
        "DETAIL D": "detail_view_section",
        "SECTION A-A": "detail_view_section",
        "SCALE 5:1": "scale_information",
    }

    for text, expected_rule in cases.items():
        decision = evaluate_page_value(candidate(text))

        assert not decision.accepted, text
        assert decision.rule_name == expected_rule
