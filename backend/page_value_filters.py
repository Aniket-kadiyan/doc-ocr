"""Editable whole-page balloon eligibility rules.

This file is deliberately independent from detection, OCR, and annotation
code.  To change filtering, edit the enabled rules or their Python predicates
here and restart the backend.

Policy order:

1. Reject table-region content for every auto-balloon scope.
2. For section scans, accept every remaining recognized value.
3. Normalize harmless OCR symbol and crop-boundary noise.
4. For whole-page scans, reject an enabled never-balloon rule.
5. Reject whole-page text with no numeric component.
6. Review isolated confusable digits and incomplete/mixed numeric text.
7. Accept complete engineering-value syntax.

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
    # Context is used only for exclusion decisions. It never replaces the
    # value returned to the annotation layer.
    context_text: str = ""


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
    r"\b(?:REV(?:ISION)?|MODIFICATIONS?|RELEASED|"
    r"CHANGE(?:\s+(?:NO|NUMBER))?)\b"
)
NOTE_LABEL = _compile(r"\bNOTES?\b")
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
SCALE_RATIO_VALUE = _compile(r"^\s*\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\s*$")
COMPACT_IDENTIFIER_VALUE = _compile(
    r"^(?=.{3,}$)(?=.*[A-Z])(?=.*\d)[A-Z0-9][A-Z0-9./_-]*$"
)
CONTEXT_IDENTIFIER_VALUE = _compile(
    r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9][A-Z0-9./_-]{5,}$"
)
SINGLE_CHARACTER_VALUE = _compile(r"^[A-Z0-9]$")
SUSPICIOUS_SINGLE_VALUE = _compile(r"^[018]$")

_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_ANGLE_MAGNITUDE = (
    rf"{_NUMBER}\s*°(?:\s*{_NUMBER}\s*['′]"
    rf"(?:\s*{_NUMBER}\s*[\"″])?)?"
)
_ANGLE_VALUE = _compile(
    rf"^{_ANGLE_MAGNITUDE}"
    rf"(?:\s*±\s*(?:{_ANGLE_MAGNITUDE}|{_NUMBER}\s*°?)"
    rf"|\s*\+\s*(?:{_ANGLE_MAGNITUDE}|{_NUMBER}\s*°?)"
    rf"\s*-\s*(?:{_ANGLE_MAGNITUDE}|{_NUMBER}\s*°?))?"
    r"(?:\s+(?:MAX|MIN|TYP|REF|BASIC))?$"
)
_LINEAR_VALUE = _compile(
    rf"^(?:\d+\s*[X×]\s*)?"
    rf"(?:SR|SØ|R|Ø)?\s*[+-]?{_NUMBER}"
    rf"(?:\s*±\s*{_NUMBER}"
    rf"|\s*\+\s*{_NUMBER}\s*/?\s*-\s*{_NUMBER}"
    rf"|\s*[+-]\s*{_NUMBER}"
    rf"|\s*(?:/|:|\bTO\b)\s*{_NUMBER})?"
    r"(?:\s*(?:MM|CM|IN|INCH|INCHES|\"))?"
    r"(?:\s+(?:MAX|MIN|TYP|REF|BASIC|THRU))?$"
)
_THREAD_VALUE = _compile(
    rf"^M\s*\d+(?:\.\d+)?"
    rf"(?:\s*[X×]\s*{_NUMBER})?"
    r"(?:\s*-\s*[0-9A-Z]+)?"
    r"(?:\s+(?:THRU|TYP|REF))?$"
)
_STANDALONE_TOLERANCE = _compile(
    rf"^(?:±|\+|-)\s*{_NUMBER}\s*°?$"
)
_BOUNDARY_NOISE_START = re.compile(r"^[?¦|;,]+\s*")
_BOUNDARY_NOISE_END = re.compile(r"\s*[?¦|;,]+$")
_KEYWORD_TOKEN = re.compile(r"[A-Z0-9]+")
_KEYWORD_CONFUSABLES = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "3": "E",
        "4": "A",
        "5": "S",
        "7": "T",
    }
)


def _normalize_symbols(text: str) -> str:
    normalized = str(text or "")
    normalized = re.sub(r"\+\s*/\s*[-−]", "±", normalized)
    normalized = (
        normalized.replace("º", "°")
        .replace("˚", "°")
        .replace("⁰", "°")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("‘", "′")
        .replace("’", "′")
        .replace("`", "′")
        .replace("“", "″")
        .replace("”", "″")
        .replace("ø", "Ø")
        .replace("φ", "Ø")
        .replace("Φ", "Ø")
        .replace("⌀", "Ø")
    )
    return " ".join(normalized.strip().split())


def _unwrapped_value(text: str) -> str:
    if len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _is_complete_engineering_value(text: str) -> bool:
    """Recognize a complete value without extracting digits from prose."""

    value = _unwrapped_value(text)
    if not value or not NUMERIC_COMPONENT.search(value):
        return False
    if SCALE_RATIO_VALUE.fullmatch(value):
        return True
    if _ANGLE_VALUE.fullmatch(value):
        return True
    if _LINEAR_VALUE.fullmatch(value):
        return True
    if _THREAD_VALUE.fullmatch(value):
        return True
    if _STANDALONE_TOLERANCE.fullmatch(value):
        return True
    return bool(COMPACT_IDENTIFIER_VALUE.fullmatch(value))


def normalize_page_value_text(text: str) -> str:
    """Normalize safe OCR variants while preserving the technical value.

    Crop-boundary punctuation is removed only when the remaining complete
    string satisfies the engineering grammar.  This prevents the filter from
    extracting a convenient number from a note or title-block sentence.
    """

    normalized = _normalize_symbols(text)
    stripped = _BOUNDARY_NOISE_START.sub("", normalized)
    stripped = _BOUNDARY_NOISE_END.sub("", stripped).strip()
    if stripped != normalized and _is_complete_engineering_value(stripped):
        return stripped
    return normalized


def _canonical_keyword_text(text: str) -> str:
    """Canonicalize OCR-confusable word tokens for exclusion matching only."""

    tokens = []
    for token in _KEYWORD_TOKEN.findall(str(text or "").upper()):
        tokens.append(
            token.translate(_KEYWORD_CONFUSABLES)
            if any(character.isalpha() for character in token)
            else token
        )
    return " ".join(tokens)


def _contains_direct_exclusion_keyword(text: str) -> bool:
    keyword_text = _canonical_keyword_text(text)
    return any(
        pattern.search(keyword_text)
        for pattern in (
            DETAIL_VIEW_SECTION,
            SCALE_LABEL,
            REVISION_LABEL,
            NOTE_LABEL,
            DOCUMENT_METADATA_LABEL,
        )
    )


def _candidate_search_text(candidate: PageValueCandidate) -> str:
    return " ".join(
        part
        for part in (candidate.text.strip(), candidate.context_text.strip())
        if part
    )


def _candidate_keyword_text(candidate: PageValueCandidate) -> str:
    return _canonical_keyword_text(_candidate_search_text(candidate))


def _is_compact_identifier(text: str) -> bool:
    return bool(
        COMPACT_IDENTIFIER_VALUE.fullmatch(text)
        and not _ANGLE_VALUE.fullmatch(_unwrapped_value(text))
        and not _LINEAR_VALUE.fullmatch(_unwrapped_value(text))
        and not _THREAD_VALUE.fullmatch(_unwrapped_value(text))
    )


def _is_context_exclusion_prone(text: str) -> bool:
    """Limit nearby labels to values likely to be metadata or callouts."""

    normalized = normalize_page_value_text(text)
    return bool(
        SCALE_RATIO_VALUE.fullmatch(normalized)
        or _is_compact_identifier(normalized)
        or SINGLE_CHARACTER_VALUE.fullmatch(normalized)
        or not _is_complete_engineering_value(normalized)
    )


def needs_expanded_filter_context(text: str, bbox: BBox) -> bool:
    """Limit extra context OCR to values prone to metadata false positives."""

    normalized = normalize_page_value_text(text)
    if not normalized:
        return False
    if _contains_direct_exclusion_keyword(normalized):
        return False
    height = max(float(bbox.get("height", 0.0)), 1.0)
    width = max(float(bbox.get("width", 0.0)), 1.0)
    return bool(
        SCALE_RATIO_VALUE.fullmatch(normalized)
        or DATE_VALUE.search(normalized)
        or CONTEXT_IDENTIFIER_VALUE.fullmatch(normalized)
        or (len(normalized) <= 2 and width / height >= 8.0)
    )


def _matches_text_or_nearby_label(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
    pattern: re.Pattern[str],
) -> bool:
    if pattern.search(_candidate_keyword_text(candidate)):
        return True
    if not _is_context_exclusion_prone(candidate.text):
        return False
    return any(
        other is not candidate
        and pattern.search(_canonical_keyword_text(other.text))
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
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        DETAIL_VIEW_SECTION,
    )


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
    return bool(DATE_VALUE.search(_candidate_search_text(candidate)))


def _revision_history(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        REVISION_LABEL,
    )


def _note_information(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        NOTE_LABEL,
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
        name="note_information",
        enabled=True,
        reason="Drawing note or note-associated value",
        predicate=_note_information,
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
        text=normalize_page_value_text(candidate.text),
        bbox=candidate.bbox,
        context_text=_normalize_symbols(candidate.context_text),
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

    if not require_numeric_component and not NUMERIC_COMPONENT.search(
        normalized.text
    ):
        return PageValueFilterDecision(
            accepted=True,
            rule_name="numeric_requirement_disabled",
            reason="Numeric-component requirement is disabled",
        )

    if SUSPICIOUS_SINGLE_VALUE.fullmatch(normalized.text):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="ambiguous_single_character",
            reason="Isolated 0, 1, or 8 may be an OCR-confused drawing label",
        )

    if not _is_complete_engineering_value(normalized.text):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="invalid_engineering_value",
            reason=(
                "Numeric text does not form one complete engineering value"
            ),
        )

    return PageValueFilterDecision(
        accepted=True,
        rule_name="engineering_value",
        reason="Matches complete engineering-value syntax",
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
