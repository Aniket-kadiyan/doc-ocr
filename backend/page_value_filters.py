"""Editable whole-page balloon eligibility rules.

This file is deliberately independent from detection, OCR, and annotation
code.  To change filtering, edit the enabled rules or their Python predicates
here and restart the backend.

Policy order:

1. Reject table-region content for every auto-balloon scope.
2. For section scans, accept every remaining recognized value.
3. For whole-page scans, reject an enabled never-balloon rule.
4. Reject whole-page text with no numeric component.
5. Accept every other whole-page value containing at least one digit.

Units and drawing geometry are not required.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Literal, Mapping, Sequence


BBox = Mapping[str, float]
ScanScopeKind = Literal["page", "section"]


@dataclass(frozen=True)
class PageValueCandidate:
    """One recognized whole-page candidate available to the filter."""

    text: str
    bbox: BBox


@dataclass(frozen=True)
class PageValueFilterDecision:
    """Stable decision metadata returned to the page pipeline."""

    accepted: bool
    rule_name: str
    reason: str


PageRulePredicate = Callable[
    [PageValueCandidate, Sequence[PageValueCandidate]],
    bool,
]


@dataclass(frozen=True)
class PageValueFilterRule:
    """One independently switchable never-balloon rule."""

    name: str
    enabled: bool
    reason: str
    predicate: PageRulePredicate


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


DETAIL_VIEW_SECTION = _compile(r"\b(?:DETAIL|VIEW|SECTION)\b")
SCALE_LABEL = _compile(r"\bSCALE\b")
REVISION_LABEL = _compile(
    r"\b(?:REV(?:ISION)?|CHANGE(?:\s+(?:NO|NUMBER))?)\b"
)
DOCUMENT_METADATA_LABEL = _compile(
    r"\b(?:DRAWING|DWG|DOCUMENT|DOC|PART)\s*"
    r"(?:NO\.?|NUMBER|#)\b"
    r"|\bSHEET(?:\s*(?:NO\.?|NUMBER|#))?\b"
)
DATE_VALUE = _compile(
    r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-]"
    r"(?:0?[1-9]|1[0-2])[./-](?:\d{2}|\d{4})(?!\d)"
    r"|(?<!\d)(?:19|20)\d{2}[./-](?:0?[1-9]|1[0-2])"
    r"[./-](?:0?[1-9]|[12]\d|3[01])(?!\d)"
)
NUMERIC_COMPONENT = re.compile(r"\d")


def _matches_text_or_nearby_label(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
    pattern: re.Pattern[str],
) -> bool:
    if pattern.search(candidate.text):
        return True
    return any(
        other is not candidate
        and pattern.search(other.text)
        and _is_nearby_label(candidate.bbox, other.bbox)
        for other in page_candidates
    )


def _axis_overlap(
    first_start: float,
    first_length: float,
    second_start: float,
    second_length: float,
) -> float:
    end = min(first_start + first_length, second_start + second_length)
    start = max(first_start, second_start)
    return max(0.0, end - start)


def _axis_gap(
    first_start: float,
    first_length: float,
    second_start: float,
    second_length: float,
) -> float:
    first_end = first_start + first_length
    second_end = second_start + second_length
    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0.0


def _is_nearby_label(value_box: BBox, label_box: BBox) -> bool:
    """Conservatively associate a split metadata label and value.

    Labels in drawing title blocks normally sit on the same row or immediately
    above their value.  The limits scale with detected text height so the rule
    behaves consistently across render resolutions.  A distant matching word
    elsewhere on the page cannot exclude a technical value.
    """

    vx = float(value_box["x"])
    vy = float(value_box["y"])
    vw = max(float(value_box["width"]), 1.0)
    vh = max(float(value_box["height"]), 1.0)
    lx = float(label_box["x"])
    ly = float(label_box["y"])
    lw = max(float(label_box["width"]), 1.0)
    lh = max(float(label_box["height"]), 1.0)
    text_height = max(vh, lh)

    vertical_overlap = _axis_overlap(vy, vh, ly, lh)
    horizontal_overlap = _axis_overlap(vx, vw, lx, lw)
    horizontal_gap = _axis_gap(vx, vw, lx, lw)
    vertical_gap = _axis_gap(vy, vh, ly, lh)

    same_row = (
        vertical_overlap / min(vh, lh) >= 0.35
        and horizontal_gap <= max(80.0, 8.0 * text_height)
    )
    stacked_field = (
        horizontal_overlap / min(vw, lw) >= 0.20
        and vertical_gap <= max(50.0, 4.0 * text_height)
    )
    return same_row or stacked_field


def _detail_view_section(
    candidate: PageValueCandidate,
    _page_candidates: Sequence[PageValueCandidate],
) -> bool:
    # View labels often sit close to real dimensions, so only the candidate's
    # own text may trigger this exclusion. Nearby association would be too
    # aggressive for dense engineering views.
    return bool(DETAIL_VIEW_SECTION.search(candidate.text))


def _scale_information(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        SCALE_LABEL,
    )


def _date_value(
    candidate: PageValueCandidate,
    _page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return bool(DATE_VALUE.search(candidate.text))


def _revision_history(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        REVISION_LABEL,
    )


def _document_metadata(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        DOCUMENT_METADATA_LABEL,
    )


# Edit this ordered tuple to change whole-page exclusions.  Earlier enabled
# rules win, making behavior deterministic and easy to audit in result counts.
NEVER_BALLOON_RULES: tuple[PageValueFilterRule, ...] = (
    PageValueFilterRule(
        name="detail_view_section",
        enabled=True,
        reason="Detail, view, or section label",
        predicate=_detail_view_section,
    ),
    PageValueFilterRule(
        name="scale_information",
        enabled=True,
        reason="Drawing scale information",
        predicate=_scale_information,
    ),
    PageValueFilterRule(
        name="date",
        enabled=True,
        reason="Date value",
        predicate=_date_value,
    ),
    PageValueFilterRule(
        name="revision_history",
        enabled=True,
        reason="Revision or change-history entry",
        predicate=_revision_history,
    ),
    PageValueFilterRule(
        name="document_metadata",
        enabled=True,
        reason="Drawing, document, part, or sheet metadata",
        predicate=_document_metadata,
    ),
)

REQUIRE_NUMERIC_COMPONENT = True

# Global table policy.  Changing these values affects page and section scans,
# but never manual Draw Value OCR.
EXCLUDE_TABLE_REGIONS = True
TABLE_OVERLAP_REJECTION_RATIO = 0.50
TABLE_CENTER_REJECTION = True


def _bbox_intersection_area(left: BBox, right: BBox) -> float:
    x0 = max(float(left["x"]), float(right["x"]))
    y0 = max(float(left["y"]), float(right["y"]))
    x1 = min(
        float(left["x"]) + float(left["width"]),
        float(right["x"]) + float(right["width"]),
    )
    y1 = min(
        float(left["y"]) + float(left["height"]),
        float(right["y"]) + float(right["height"]),
    )
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def candidate_is_in_table(
    bbox: BBox,
    table_masks: Sequence[BBox],
    *,
    overlap_rejection_ratio: float = TABLE_OVERLAP_REJECTION_RATIO,
    center_rejection: bool = TABLE_CENTER_REJECTION,
) -> bool:
    """Reject a candidate, not its complete mixed-content scan section."""

    width = max(float(bbox["width"]), 1.0)
    height = max(float(bbox["height"]), 1.0)
    area = width * height
    center_x = float(bbox["x"]) + width / 2
    center_y = float(bbox["y"]) + height / 2
    for mask in table_masks:
        center_inside = (
            float(mask["x"]) <= center_x <= float(mask["x"]) + float(mask["width"])
            and float(mask["y"]) <= center_y <= float(mask["y"]) + float(mask["height"])
        )
        if center_rejection and center_inside:
            return True
        if _bbox_intersection_area(bbox, mask) / area >= overlap_rejection_ratio:
            return True
    return False


def evaluate_scan_value(
    candidate: PageValueCandidate,
    *,
    scope_kind: ScanScopeKind,
    table_masks: Sequence[BBox] = (),
    page_candidates: Sequence[PageValueCandidate] = (),
    rules: Sequence[PageValueFilterRule] = NEVER_BALLOON_RULES,
    require_numeric_component: bool = REQUIRE_NUMERIC_COMPONENT,
) -> PageValueFilterDecision:
    """Apply the global table rule, then the scope-specific value policy."""

    normalized = PageValueCandidate(
        text=" ".join(candidate.text.strip().split()),
        bbox=candidate.bbox,
    )
    if EXCLUDE_TABLE_REGIONS and candidate_is_in_table(
        normalized.bbox,
        table_masks,
    ):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="table_region",
            reason="Candidate lies inside a detected table region",
        )

    if scope_kind == "section":
        return PageValueFilterDecision(
            accepted=True,
            rule_name="section_passthrough",
            reason="Section scans bypass whole-page value exclusions",
        )

    context = tuple(
        normalized if item is candidate else item
        for item in page_candidates
    )
    for rule in rules:
        if rule.enabled and rule.predicate(normalized, context):
            return PageValueFilterDecision(
                accepted=False,
                rule_name=rule.name,
                reason=rule.reason,
            )

    if require_numeric_component and not NUMERIC_COMPONENT.search(normalized.text):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="no_numeric_component",
            reason="Recognized text has no numeric component",
        )

    return PageValueFilterDecision(
        accepted=True,
        rule_name="numeric_component",
        reason="Contains a numeric component and matches no exclusion",
    )


def evaluate_page_value(
    candidate: PageValueCandidate,
    *,
    page_candidates: Sequence[PageValueCandidate] = (),
    table_masks: Sequence[BBox] = (),
    rules: Sequence[PageValueFilterRule] = NEVER_BALLOON_RULES,
    require_numeric_component: bool = REQUIRE_NUMERIC_COMPONENT,
) -> PageValueFilterDecision:
    """Evaluate one recognized value using the editable page policy."""

    return evaluate_scan_value(
        candidate,
        scope_kind="page",
        table_masks=table_masks,
        page_candidates=page_candidates,
        rules=rules,
        require_numeric_component=require_numeric_component,
    )
