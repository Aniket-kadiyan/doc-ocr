"""M6 structure-first disposition policy tests."""

from __future__ import annotations

import json

from engineering_disposition import (
    disposition_engineering_object,
    engineering_disposition_statistics,
)
from engineering_value_parser import parse_engineering_value


def decide(text: str, **kwargs: object):
    return disposition_engineering_object(
        parse_engineering_value(text),
        recognized=bool(kwargs.pop("recognized", True)),
        **kwargs,
    )


def test_complete_engineering_values_are_eligible() -> None:
    for text in (
        "25",
        "Ø8 +0.2/0",
        "0.1+0.1",
        "15-0.2",
        "M8 x 1.25-6H THRU",
        "32°20'40\"",
        "2 X 45°",
        "2.25[57.15]REF",
        "2:1",
        "2N9",
    ):
        result = decide(text)
        assert result.state == "eligible", text
        assert result.rule == "complete_engineering_object", text


def test_incomplete_or_unknown_numeric_objects_require_review() -> None:
    incomplete = decide("R0.6-0.")
    unknown = decide("12 BROKEN")

    assert incomplete.state == "review"
    assert incomplete.rule == "incomplete_engineering_object"
    assert unknown.state == "review"
    assert unknown.rule == "incomplete_engineering_object"


def test_conflicts_require_review_even_when_parse_is_complete() -> None:
    assembly = decide(
        "Ø8 +0.2/0",
        assembly_conflict=True,
        assembly_review_reason="Fragments disagree",
    )
    source = decide("25", source_conflict=True)
    numeric = decide("25", numeric_conflict=True)

    assert assembly.state == "review"
    assert assembly.reason == "Fragments disagree"
    assert source.rule == "recognition_conflict"
    assert numeric.rule == "recognition_conflict"


def test_isolated_confusable_value_requires_review() -> None:
    result = decide(
        "1",
        context_rule="ambiguous_single_character",
        context_reason="Isolated 0, 1, or 8 may be an OCR-confused label",
    )

    assert result.state == "review"
    assert result.rule == "ambiguous_single_character"


def test_hard_context_is_visible_other_and_wins_over_structure() -> None:
    for rule in (
        "table_region",
        "sheet_frame_label",
        "detail_view_section",
        "scale_information",
        "date",
        "revision_history",
        "note_information",
        "document_metadata",
    ):
        result = decide(
            "25",
            context_rule=rule,
            context_reason="Known non-inspection context",
        )
        assert result.state == "other", rule
        assert result.hard_context is True, rule


def test_text_and_identifiers_are_retained_as_other() -> None:
    plain = decide("MATERIAL")
    identifier = decide("AB12")

    assert plain.state == "other"
    assert plain.rule == "non_inspection_text"
    assert identifier.state == "other"
    assert identifier.rule == "non_inspection_identifier"


def test_empty_objects_are_reviewed_until_authoritative_then_other() -> None:
    preliminary = decide("", recognized=False, authoritative=False)
    final = decide("", recognized=False, authoritative=True)
    required = decide(
        "",
        recognized=False,
        authoritative=True,
        recognition_review_required=True,
    )

    assert preliminary.state == "review"
    assert final.state == "other"
    assert required.state == "review"


def test_authoritative_complete_structure_is_not_demoted_by_confidence() -> None:
    authoritative = decide(
        "25",
        authoritative=True,
        recognition_needs_review=True,
    )
    preliminary = decide(
        "25",
        authoritative=False,
        recognition_needs_review=True,
    )

    assert authoritative.state == "eligible"
    assert preliminary.state == "review"


def test_disposition_output_and_statistics_are_json_safe() -> None:
    dispositions = [decide("25"), decide("R0.6-0."), decide("MATERIAL")]

    json.dumps([item.to_dict() for item in dispositions])
    stats = engineering_disposition_statistics(dispositions)

    assert stats["object_count"] == 3
    assert stats["state_counts"] == {
        "eligible": 1,
        "other": 1,
        "review": 1,
    }
