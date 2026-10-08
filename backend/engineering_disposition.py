"""Structure-first disposition for assembled engineering drawing objects.

M4 assembles primitive detections and M5 parses the resulting text without
changing it.  This module is the M6 policy boundary: it decides whether that
logical object is eligible for a balloon, needs human review, or belongs in
the visible Other queue.

The policy deliberately does not interpret graphical GD&T frames, datum
symbols, or surface-finish symbols.  Those require image evidence and belong
to M7.  M6 uses only evidence already produced by recognition, assembly,
page-context filtering, and the lossless parser.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from engineering_value_parser import EngineeringValueParse


DispositionState = Literal["eligible", "review", "other"]

# These rules describe content that is known not to be an inspection value.
# They remain visible as Other outcomes; they are never silently discarded.
HARD_OTHER_RULES = frozenset(
    {
        "table_region",
        "sheet_frame_label",
        "detail_view_section",
        "scale_information",
        "date",
        "revision_history",
        "note_information",
        "document_metadata",
    }
)

# A complete identifier still needs drawing context before it can be treated
# as an inspection characteristic.  Known fit callouts such as ``2N9`` parse
# as linear values and do not enter this branch.
CONTEXT_ONLY_KINDS = frozenset({"identifier"})
REVIEW_CONTEXT_RULES = frozenset({"ambiguous_single_character"})


@dataclass(frozen=True)
class EngineeringDisposition:
    state: DispositionState
    rule: str
    reason: str
    parse_status: str
    parse_kind: str
    hard_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "state": self.state,
            "rule": self.rule,
            "reason": self.reason,
            "parse_status": self.parse_status,
            "parse_kind": self.parse_kind,
            "hard_context": self.hard_context,
        }


def _parse_field(
    parsed: EngineeringValueParse | Mapping[str, Any],
    field: str,
    default: str,
) -> str:
    if isinstance(parsed, EngineeringValueParse):
        return str(getattr(parsed, field, default) or default)
    return str(parsed.get(field) or default)


def _decision(
    state: DispositionState,
    rule: str,
    reason: str,
    *,
    parse_status: str,
    parse_kind: str,
    hard_context: bool = False,
) -> EngineeringDisposition:
    return EngineeringDisposition(
        state=state,
        rule=rule,
        reason=reason,
        parse_status=parse_status,
        parse_kind=parse_kind,
        hard_context=hard_context,
    )


def disposition_engineering_object(
    parsed: EngineeringValueParse | Mapping[str, Any],
    *,
    recognized: bool,
    authoritative: bool = True,
    context_rule: str = "",
    context_reason: str = "",
    assembly_conflict: bool = False,
    assembly_review_reason: str = "",
    source_conflict: bool = False,
    numeric_conflict: bool = False,
    recognition_review_required: bool = False,
    recognition_needs_review: bool = False,
    recognition_review_reason: str = "",
    speck: bool = False,
) -> EngineeringDisposition:
    """Return one auditable final state for an assembled object.

    Precedence is intentional: proven non-inspection context is Other;
    contradictory evidence is Review; then the structural parse decides.
    Confidence alone cannot demote a structurally complete authoritative read.
    """

    parse_status = _parse_field(parsed, "status", "unparsed")
    parse_kind = _parse_field(parsed, "kind", "unknown")
    review_reason = str(recognition_review_reason or "").strip()

    if context_rule in HARD_OTHER_RULES:
        return _decision(
            "other",
            context_rule,
            context_reason or "Recognized non-inspection drawing content",
            parse_status=parse_status,
            parse_kind=parse_kind,
            hard_context=True,
        )

    if assembly_conflict:
        return _decision(
            "review",
            "assembly_conflict",
            assembly_review_reason
            or "Assembled fragments contain conflicting recognition evidence",
            parse_status=parse_status,
            parse_kind=parse_kind,
        )

    if source_conflict or numeric_conflict:
        return _decision(
            "review",
            "recognition_conflict",
            review_reason
            or "Independent recognition evidence disagrees on this object",
            parse_status=parse_status,
            parse_kind=parse_kind,
        )

    if not recognized:
        if recognition_review_required:
            return _decision(
                "review",
                "authoritative_review_required",
                review_reason
                or "Authoritative OCR could not confirm this detector target",
                parse_status=parse_status,
                parse_kind=parse_kind,
            )
        if not authoritative and not speck:
            return _decision(
                "review",
                "recognition_pending",
                review_reason or "Accurate recognition returned no text",
                parse_status=parse_status,
                parse_kind=parse_kind,
            )
        return _decision(
            "other",
            "unread_object",
            review_reason
            or (
                "No text was read and this object is too small to hold a value"
                if speck
                else "Recognition found no text at this object"
            ),
            parse_status=parse_status,
            parse_kind=parse_kind,
        )

    if recognition_review_required:
        return _decision(
            "review",
            "authoritative_review_required",
            review_reason or "Authoritative OCR requires manual confirmation",
            parse_status=parse_status,
            parse_kind=parse_kind,
        )

    if context_rule in REVIEW_CONTEXT_RULES:
        return _decision(
            "review",
            context_rule,
            context_reason
            or "Isolated character requires confirmation before ballooning",
            parse_status=parse_status,
            parse_kind=parse_kind,
        )

    if parse_status == "complete":
        if parse_kind in CONTEXT_ONLY_KINDS:
            return _decision(
                "other",
                "non_inspection_identifier",
                "Identifier retained for reference; it is not an inspection value",
                parse_status=parse_status,
                parse_kind=parse_kind,
            )
        if recognition_needs_review and not authoritative:
            return _decision(
                "review",
                "recognition_ambiguous",
                review_reason
                or "Recognition remains genuinely ambiguous after recovery",
                parse_status=parse_status,
                parse_kind=parse_kind,
            )
        return _decision(
            "eligible",
            "complete_engineering_object",
            "Complete assembled engineering value",
            parse_status=parse_status,
            parse_kind=parse_kind,
        )

    if parse_status == "partial":
        return _decision(
            "review",
            "incomplete_engineering_object",
            review_reason
            or "Engineering value is incomplete or contains unparsed text",
            parse_status=parse_status,
            parse_kind=parse_kind,
        )

    if parse_status == "empty":
        return _decision(
            "other" if authoritative or speck else "review",
            "empty_engineering_object",
            review_reason
            or (
                "Recognition found no text at this object"
                if authoritative or speck
                else "Accurate recognition returned no text"
            ),
            parse_status=parse_status,
            parse_kind=parse_kind,
        )

    # A recognized text-only label is useful audit evidence, but it is not an
    # inspection value. Unknown text containing digits is potentially a broken
    # dimension and must be surfaced for confirmation.
    normalized_text = _parse_field(parsed, "normalized_text", "")
    if any(character.isdigit() for character in normalized_text):
        return _decision(
            "review",
            "unparsed_numeric_object",
            review_reason
            or "Numeric text could not be parsed as an engineering value",
            parse_status=parse_status,
            parse_kind=parse_kind,
        )
    return _decision(
        "other",
        "non_inspection_text",
        context_reason or "Recognized text is not an inspection value",
        parse_status=parse_status,
        parse_kind=parse_kind,
    )


def engineering_disposition_statistics(
    dispositions: Iterable[EngineeringDisposition | Mapping[str, Any]],
) -> dict[str, Any]:
    """Return deterministic, API-safe M6 state and rule counts."""

    state_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    count = 0
    for item in dispositions:
        count += 1
        if isinstance(item, EngineeringDisposition):
            state = item.state
            rule = item.rule
        else:
            state = str(item.get("state") or "other")
            rule = str(item.get("rule") or "unknown")
        state_counts[state] += 1
        rule_counts[rule] += 1
    return {
        "schema_version": 1,
        "object_count": count,
        "state_counts": dict(sorted(state_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
    }
