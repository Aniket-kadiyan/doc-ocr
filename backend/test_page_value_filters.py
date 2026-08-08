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


def test_numeric_values_need_neither_units_nor_geometry() -> None:
    for text in ("50", ".25", "M8", "SS304", "R0.2 MAX", "2:1"):
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, text
        assert decision.rule_name == "numeric_component"


def test_pure_text_is_rejected() -> None:
    decision = evaluate_page_value(candidate("MATERIAL"))

    assert not decision.accepted
    assert decision.rule_name == "no_numeric_component"


def test_direct_never_balloon_values_are_rejected_before_numeric_acceptance() -> None:
    cases = {
        "DETAIL B SCALE 2:1": "detail_view_section",
        "SCALE 2:1": "scale_information",
        "15.07.2024": "date",
        "REV 3": "revision_history",
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


def test_each_never_balloon_rule_can_be_disabled_in_code() -> None:
    scale_rules = tuple(
        replace(rule, enabled=False)
        if rule.name == "scale_information"
        else rule
        for rule in NEVER_BALLOON_RULES
    )

    decision = evaluate_page_value(
        candidate("SCALE 2:1"),
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
