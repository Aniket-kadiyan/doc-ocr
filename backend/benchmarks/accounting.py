"""Candidate accounting and snapshot helpers for real-drawing benchmarks.

The production page pipeline returns three public collections:

* ``regions`` for automatically accepted balloons;
* ``review_candidates`` for unresolved values; and
* ``candidate_outcomes`` for the final state of detector objects.

Historically the benchmark saved only ``regions``. That measured recognition of
accepted balloons, but it could not prove that excluded or unread detector
objects remained accounted for. This module keeps the full scan result and
validates the lifecycle without changing production OCR behaviour.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

_OUTCOME_STATES = {"eligible", "review", "excluded"}
_DISPLAY_STATE = {
    "eligible": "accepted",
    "review": "review",
    "excluded": "other",
}


def _count(result: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(result.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _valid_bbox(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        x = float(value["x"])
        y = float(value["y"])
        width = float(value["width"])
        height = float(value["height"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(number) for number in (x, y, width, height))
        and width > 0
        and height > 0
    )


def _candidate_id(item: dict[str, Any]) -> str:
    return str(item.get("candidate_id") or "").strip()


@dataclass(frozen=True)
class AccountingIssue:
    """One reproducible violation of the candidate lifecycle contract."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass
class PageAccounting:
    page: int
    detected: int
    recognized: int
    accepted: int
    review: int
    other: int
    unread: int
    skipped_existing: int
    outcome_count: int
    issues: list[AccountingIssue] = field(default_factory=list)

    @property
    def balanced(self) -> bool:
        return self.detected == self.accepted + self.review + self.other

    @property
    def passed(self) -> bool:
        return self.balanced and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "detected": self.detected,
            "recognized": self.recognized,
            "accepted": self.accepted,
            "review": self.review,
            "other": self.other,
            "unread": self.unread,
            "skipped_existing": self.skipped_existing,
            "outcome_count": self.outcome_count,
            "balanced": self.balanced,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class AccountingReport:
    pages: list[PageAccounting]
    available: bool = True

    @property
    def issues(self) -> list[AccountingIssue]:
        return [issue for page in self.pages for issue in page.issues]

    @property
    def balanced(self) -> bool:
        return self.available and all(page.balanced for page in self.pages)

    @property
    def passed(self) -> bool:
        return self.available and all(page.passed for page in self.pages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "balanced": self.balanced,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "pages": [page.to_dict() for page in self.pages],
        }

    def format_text(self) -> str:
        if not self.available:
            return "candidate accounting: unavailable (legacy accepted-regions file)"
        lines = ["candidate accounting:"]
        for page in self.pages:
            status = "OK" if page.passed else "FAIL"
            lines.append(
                f"  {status} page {page.page}: detected {page.detected} = "
                f"accepted {page.accepted} + review {page.review} + "
                f"other {page.other}; unread {page.unread}; "
                f"skipped-existing {page.skipped_existing}; "
                f"outcomes {page.outcome_count}"
            )
            for issue in page.issues:
                lines.append(f"      {issue.code}: {issue.message}")
        return "\n".join(lines)


def evaluate_page_accounting(
    result: dict[str, Any], *, page: int = 1
) -> PageAccounting:
    """Validate counts, identities, geometry, and reasons for one scan page."""

    regions = list(result.get("regions") or [])
    reviews = list(result.get("review_candidates") or [])
    outcomes = list(result.get("candidate_outcomes") or [])

    accepted = _count(result, "eligible_count", len(regions))
    review = _count(result, "review_count", len(reviews))
    other = _count(
        result,
        "excluded_count",
        sum(1 for item in outcomes if item.get("state") == "excluded"),
    )
    detected = _count(result, "detected_count", accepted + review + other)
    unread = _count(result, "unread_count", 0)
    recognized = _count(result, "recognized_count", detected - unread)
    skipped_existing = _count(result, "skipped_existing_count", 0)
    issues: list[AccountingIssue] = []

    def issue(code: str, message: str) -> None:
        issues.append(AccountingIssue(code, message))

    required_counts = (
        "detected_count",
        "recognized_count",
        "eligible_count",
        "excluded_count",
        "review_count",
        "unread_count",
        "skipped_existing_count",
    )
    for key in required_counts:
        if key not in result:
            issue("missing_count", f"result has no {key}")
            continue
        try:
            int(result[key])
        except (TypeError, ValueError):
            issue("invalid_count", f"{key} is not an integer")

    if detected != accepted + review + other:
        issue(
            "count_not_balanced",
            f"{detected} detected but final states total "
            f"{accepted + review + other}",
        )
    if recognized != detected - unread:
        issue(
            "recognition_count_mismatch",
            f"recognized_count is {recognized}, but detected minus unread is "
            f"{detected - unread}",
        )
    if min(detected, recognized, accepted, review, other, unread, skipped_existing) < 0:
        issue("negative_count", "candidate lifecycle counts must not be negative")
    if accepted != len(regions):
        issue(
            "accepted_collection_mismatch",
            f"eligible_count is {accepted}, regions contains {len(regions)}",
        )
    if review != len(reviews):
        issue(
            "review_collection_mismatch",
            f"review_count is {review}, review_candidates contains {len(reviews)}",
        )

    states = [str(item.get("state") or "") for item in outcomes]
    invalid_states = sorted({state for state in states if state not in _OUTCOME_STATES})
    if invalid_states:
        issue("invalid_outcome_state", f"unknown states: {invalid_states}")
    excluded_outcomes = sum(state == "excluded" for state in states)
    if other != excluded_outcomes:
        issue(
            "other_collection_mismatch",
            f"excluded_count is {other}, excluded outcomes total {excluded_outcomes}",
        )

    ids = [_candidate_id(item) for item in outcomes]
    nonempty_ids = [candidate_id for candidate_id in ids if candidate_id]
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(nonempty_ids).items()
        if count > 1
    )
    if duplicate_ids:
        issue("duplicate_candidate_id", f"duplicates: {duplicate_ids}")

    outcome_by_id = {
        _candidate_id(item): item for item in outcomes if _candidate_id(item)
    }
    for collection_name, items, wanted_state in (
        ("regions", regions, "eligible"),
        ("review_candidates", reviews, "review"),
    ):
        for index, item in enumerate(items):
            if not _valid_bbox(item.get("bbox")):
                issue(
                    "invalid_bbox",
                    f"{collection_name}[{index}] has no usable bbox",
                )
            candidate_id = _candidate_id(item)
            if not candidate_id:
                # Synthetic notes and recovered angled values may not originate
                # from one detector object. They still count as accepted but do
                # not invalidate detector-object accounting.
                continue
            outcome = outcome_by_id.get(candidate_id)
            if outcome is None:
                issue(
                    "published_without_outcome",
                    f"{collection_name} candidate {candidate_id} has no outcome",
                )
            elif outcome.get("state") != wanted_state:
                issue(
                    "published_state_mismatch",
                    f"{candidate_id} is in {collection_name} but outcome state "
                    f"is {outcome.get('state')!r}",
                )

    for index, outcome in enumerate(outcomes):
        if not _valid_bbox(outcome.get("bbox")):
            issue("invalid_bbox", f"candidate_outcomes[{index}] has no usable bbox")
        state = str(outcome.get("state") or "")
        if state in {"review", "excluded"} and not str(
            outcome.get("reason") or ""
        ).strip():
            issue(
                "missing_disposition_reason",
                f"candidate {_candidate_id(outcome) or index} in state {state} "
                "has no reason",
            )

    return PageAccounting(
        page=page,
        detected=detected,
        recognized=recognized,
        accepted=accepted,
        review=review,
        other=other,
        unread=unread,
        skipped_existing=skipped_existing,
        outcome_count=len(outcomes),
        issues=issues,
    )


def evaluate_snapshot_accounting(snapshot: object) -> AccountingReport:
    """Evaluate a full scan snapshot; legacy region arrays have no accounting."""

    if isinstance(snapshot, list):
        return AccountingReport(pages=[], available=False)
    if not isinstance(snapshot, dict):
        raise ValueError("benchmark result must be a region list or scan snapshot")
    raw_pages = snapshot.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("scan snapshot must contain at least one page")
    pages = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict) or not isinstance(
            raw_page.get("result"), dict
        ):
            raise ValueError(f"snapshot page {index} has no result object")
        pages.append(
            evaluate_page_accounting(
                raw_page["result"], page=int(raw_page.get("page", index))
            )
        )
    return AccountingReport(pages=pages, available=True)


def _with_benchmark_fields(
    item: dict[str, Any],
    *,
    page: int,
    disposition: str,
    page_width: float | None,
    page_height: float | None,
) -> dict[str, Any]:
    return {
        **item,
        "page": page,
        "benchmark_disposition": disposition,
        "_benchmark_page_width": page_width,
        "_benchmark_page_height": page_height,
    }


def candidates_from_snapshot(snapshot: object) -> list[dict[str, Any]]:
    """Flatten accepted, review, and other candidates for ground-truth scoring."""

    if isinstance(snapshot, list):
        return [
            {**item, "benchmark_disposition": "accepted"}
            for item in snapshot
            if isinstance(item, dict)
        ]
    if not isinstance(snapshot, dict):
        raise ValueError("benchmark result must be a region list or scan snapshot")
    raw_pages = snapshot.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("scan snapshot must contain at least one page")

    candidates: list[dict[str, Any]] = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict) or not isinstance(
            raw_page.get("result"), dict
        ):
            raise ValueError(f"snapshot page {index} has no result object")
        page = int(raw_page.get("page", index))
        width = raw_page.get("width")
        height = raw_page.get("height")
        page_width = float(width) if isinstance(width, (int, float)) else None
        page_height = float(height) if isinstance(height, (int, float)) else None
        result = raw_page["result"]
        for item in list(result.get("regions") or []):
            candidates.append(
                _with_benchmark_fields(
                    item,
                    page=page,
                    disposition="accepted",
                    page_width=page_width,
                    page_height=page_height,
                )
            )
        for item in list(result.get("review_candidates") or []):
            candidates.append(
                _with_benchmark_fields(
                    item,
                    page=page,
                    disposition="review",
                    page_width=page_width,
                    page_height=page_height,
                )
            )
        for item in list(result.get("candidate_outcomes") or []):
            state = str(item.get("state") or "")
            if state != "excluded":
                continue
            candidates.append(
                _with_benchmark_fields(
                    item,
                    page=page,
                    disposition=_DISPLAY_STATE[state],
                    page_width=page_width,
                    page_height=page_height,
                )
            )
    return candidates


def aggregate_counts(pages: Iterable[PageAccounting]) -> dict[str, int]:
    """Small helper for reports that need whole-document lifecycle totals."""

    fields = (
        "detected",
        "recognized",
        "accepted",
        "review",
        "other",
        "unread",
        "skipped_existing",
        "outcome_count",
    )
    return {
        name: sum(int(getattr(page, name)) for page in pages)
        for name in fields
    }
