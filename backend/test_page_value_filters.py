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
        "32°20'40\"",
        "25 +0.1 -0.2",
        "M8",
        "M8 x 1.25",
        "2X Ø10",
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


def test_table_region_is_a_global_hard_exclusion() -> None:
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


def test_mixed_section_rejects_only_candidate_inside_the_table() -> None:
    mask = {"x": 100, "y": 100, "width": 200, "height": 120}
    table_value = candidate("50", x=120, y=130, width=30, height=14)
    drawing_text = candidate("DATUM A", x=20, y=30, width=70, height=14)

    assert candidate_is_in_table(table_value.bbox, (mask,))
    assert not candidate_is_in_table(drawing_text.bbox, (mask,))
    decision = evaluate_scan_value(
        drawing_text,
        scope_kind="section",
        table_masks=(mask,),
    )
    assert decision.accepted
    assert decision.rule_name == "section_passthrough"
