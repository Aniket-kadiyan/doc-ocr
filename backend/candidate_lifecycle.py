"""Final candidate lifecycle reconciliation for page scans.

This module changes bookkeeping only.  It does not alter OCR text, confidence,
geometry, detection, recognition, or which accepted/review items are published.
Objects removed by post-processing are retained as explicit ``excluded``
outcomes so the UI can show them under Other and the final counts cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def candidate_id(item: dict[str, Any]) -> str:
    """Return a normalized candidate identity, or an empty string."""

    return str(item.get("candidate_id") or "").strip()


def retire_candidate_outcomes(
    outcomes: Iterable[dict[str, Any]],
    candidate_ids: Iterable[str],
    *,
    reason: str,
    overlays: Iterable[dict[str, Any]] = (),
    rule: str = "post_processing_retirement",
) -> set[str]:
    """Move active outcomes to Other without deleting their evidence."""

    wanted = {str(value).strip() for value in candidate_ids if str(value).strip()}
    retired: set[str] = set()
    if not wanted:
        return retired

    for outcome in outcomes:
        identity = candidate_id(outcome)
        if identity not in wanted or outcome.get("state") == "excluded":
            continue
        outcome["state"] = "excluded"
        outcome["reason"] = reason
        outcome["rule"] = rule
        retired.add(identity)

    for overlay in overlays:
        identity = str(overlay.get("id") or "").strip()
        if identity not in retired:
            continue
        overlay["state"] = "excluded"
        overlay["reason"] = reason
        overlay["rule"] = rule
    return retired


def reconcile_unpublished_outcomes(
    regions: Iterable[dict[str, Any]],
    reviews: Iterable[dict[str, Any]],
    outcomes: Iterable[dict[str, Any]],
    *,
    overlays: Iterable[dict[str, Any]] = (),
) -> set[str]:
    """Retire active detector outcomes absent from their public collection.

    Stage-specific code supplies a more precise retirement reason where it can.
    This final conservative pass prevents any later post-processing path from
    silently dropping a candidate.
    """

    outcome_items = list(outcomes)
    accepted_ids = {candidate_id(item) for item in regions}
    review_ids = {candidate_id(item) for item in reviews}
    accepted_ids.discard("")
    review_ids.discard("")
    unpublished = {
        candidate_id(outcome)
        for outcome in outcome_items
        if candidate_id(outcome)
        and (
            (
                outcome.get("state") == "eligible"
                and candidate_id(outcome) not in accepted_ids
            )
            or (
                outcome.get("state") == "review"
                and candidate_id(outcome) not in review_ids
            )
        )
    }
    return retire_candidate_outcomes(
        outcome_items,
        unpublished,
        reason="Superseded during final page post-processing",
        overlays=overlays,
    )


@dataclass(frozen=True)
class FinalCandidateCounts:
    detected: int
    recognized: int
    eligible: int
    review: int
    excluded: int
    unread: int


def final_candidate_counts(
    regions: Iterable[dict[str, Any]],
    reviews: Iterable[dict[str, Any]],
    outcomes: Iterable[dict[str, Any]],
) -> FinalCandidateCounts:
    """Derive balanced lifecycle totals from final collections and outcomes."""

    region_items = list(regions)
    review_items = list(reviews)
    outcome_items = list(outcomes)
    outcome_ids = {candidate_id(item) for item in outcome_items}
    outcome_ids.discard("")

    # Recovered stacked/slanted values can be real published values without a
    # detector identity. A non-empty identity, however, must resolve to exactly
    # one outcome; treating an orphaned identity as synthetic would hide a bug.
    for collection_name, items in (
        ("regions", region_items),
        ("review_candidates", review_items),
    ):
        orphaned = sorted(
            identity
            for item in items
            if (identity := candidate_id(item)) and identity not in outcome_ids
        )
        if orphaned:
            raise AssertionError(
                f"{collection_name} contains candidates without outcomes: {orphaned}"
            )

    synthetic_regions = [item for item in region_items if not candidate_id(item)]
    synthetic_reviews = [item for item in review_items if not candidate_id(item)]
    eligible_outcomes = [
        item for item in outcome_items if item.get("state") == "eligible"
    ]
    review_outcomes = [
        item for item in outcome_items if item.get("state") == "review"
    ]
    excluded_outcomes = [
        item for item in outcome_items if item.get("state") == "excluded"
    ]

    if len(region_items) != len(eligible_outcomes) + len(synthetic_regions):
        raise AssertionError("Accepted regions do not match eligible outcomes")
    if len(review_items) != len(review_outcomes) + len(synthetic_reviews):
        raise AssertionError("Review candidates do not match review outcomes")

    synthetic_items = synthetic_regions + synthetic_reviews
    detected = len(outcome_items) + len(synthetic_items)
    recognized = sum(
        bool(item.get("recognized", True)) for item in outcome_items
    ) + sum(bool(item.get("recognized", True)) for item in synthetic_items)
    unread = detected - recognized
    counts = FinalCandidateCounts(
        detected=detected,
        recognized=recognized,
        eligible=len(region_items),
        review=len(review_items),
        excluded=len(excluded_outcomes),
        unread=unread,
    )
    if counts.unread < 0:
        raise AssertionError("Recognized candidates cannot exceed detected candidates")
    if counts.detected != counts.eligible + counts.review + counts.excluded:
        raise AssertionError("Every detected page object must have one final state")
    return counts
