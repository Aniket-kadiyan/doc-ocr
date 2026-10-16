"""Model-free tests for post-processing candidate accounting."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from benchmarks.accounting import evaluate_page_accounting
from candidate_lifecycle import (
    final_candidate_counts,
    reconcile_unpublished_outcomes,
    retire_candidate_outcomes,
)


def _outcome(candidate_id: str, state: str, *, recognized: bool = True) -> dict:
    return {
        "candidate_id": candidate_id,
        "state": state,
        "text": "25",
        "bbox": {"x": 10, "y": 20, "width": 30, "height": 12},
        "recognized": recognized,
        "reason": "Initial decision",
        "rule": "initial_rule",
        "recognition_evidence": {"source": "test"},
    }


def test_retirement_preserves_evidence_and_updates_overlay():
    outcome = _outcome("C0001", "eligible")
    overlay = {"id": "C0001", "state": "eligible", "text": "25"}

    retired = retire_candidate_outcomes(
        [outcome],
        {"C0001"},
        reason="Duplicate of another published value",
        overlays=[overlay],
    )

    assert retired == {"C0001"}
    assert outcome["state"] == "excluded"
    assert outcome["reason"] == "Duplicate of another published value"
    assert outcome["rule"] == "post_processing_retirement"
    assert outcome["text"] == "25"
    assert outcome["recognized"] is True
    assert outcome["recognition_evidence"] == {"source": "test"}
    assert overlay["state"] == "excluded"


def test_reconciliation_retires_unpublished_active_outcome_instead_of_deleting_it():
    accepted = _outcome("C0001", "eligible")
    unpublished = _outcome("C0002", "eligible")
    outcomes = [accepted, unpublished]

    retired = reconcile_unpublished_outcomes([accepted], [], outcomes)

    assert retired == {"C0002"}
    assert len(outcomes) == 2
    assert unpublished["state"] == "excluded"
    assert unpublished["reason"] == "Superseded during final page post-processing"


def test_final_counts_include_idless_recoveries_and_remain_balanced():
    accepted = _outcome("C0001", "eligible")
    unread_other = _outcome("C0002", "excluded", recognized=False)
    recovered = {
        "text": "R5",
        "bbox": {"x": 50, "y": 20, "width": 30, "height": 12},
        "recognized": True,
    }

    counts = final_candidate_counts(
        [accepted, recovered],
        [],
        [accepted, unread_other],
    )

    assert counts.detected == 3
    assert counts.recognized == 2
    assert counts.eligible == 2
    assert counts.review == 0
    assert counts.excluded == 1
    assert counts.unread == 1


def test_final_counts_reject_published_candidate_without_outcome():
    orphan = _outcome("C9999", "eligible")
    try:
        final_candidate_counts([orphan], [], [])
    except AssertionError as exc:
        assert "without outcomes" in str(exc)
    else:
        raise AssertionError("Expected orphaned published identity to fail")


def test_m5_snapshots_reconcile_without_changing_published_values():
    expected_retired = {
        "129E01-13300-ID.scan.json": 5,
        "47630.scan.json": 7,
        "503645_DES001_CF_BALLOON_DRAWING.scan.json": 0,
        "56103-0182B.scan.json": 0,
        "BS1801006.020.scan.json": 0,
        "ballwoon_drawing.scan.json": 6,
    }
    snapshot_dir = Path(__file__).parent / "benchmarks" / "results" / "m5"

    for filename, expected_count in expected_retired.items():
        snapshot = json.loads((snapshot_dir / filename).read_text(encoding="utf-8"))
        result = snapshot["pages"][0]["result"]
        original_regions = copy.deepcopy(result.get("regions") or [])
        original_reviews = copy.deepcopy(result.get("review_candidates") or [])

        retired = reconcile_unpublished_outcomes(
            result.get("regions") or [],
            result.get("review_candidates") or [],
            result.get("candidate_outcomes") or [],
        )
        counts = final_candidate_counts(
            result.get("regions") or [],
            result.get("review_candidates") or [],
            result.get("candidate_outcomes") or [],
        )
        result.update(
            detected_count=counts.detected,
            recognized_count=counts.recognized,
            eligible_count=counts.eligible,
            review_count=counts.review,
            excluded_count=counts.excluded,
            unread_count=counts.unread,
        )

        assert len(retired) == expected_count
        assert result.get("regions") == original_regions
        assert result.get("review_candidates") == original_reviews
        assert evaluate_page_accounting(result).passed
