"""Fast, model-free tests for full candidate lifecycle accounting."""

from __future__ import annotations

import pytest

from benchmarks.accounting import (
    candidates_from_snapshot,
    evaluate_page_accounting,
    evaluate_snapshot_accounting,
)


def _bbox(x: int) -> dict[str, int]:
    return {"x": x, "y": 20, "width": 40, "height": 12}


def _result() -> dict:
    return {
        "detected_count": 3,
        "recognized_count": 3,
        "eligible_count": 1,
        "excluded_count": 1,
        "review_count": 1,
        "unread_count": 0,
        "skipped_existing_count": 2,
        "regions": [
            {
                "candidate_id": "C0001",
                "bbox": _bbox(10),
                "text": "25±0.1",
                "type": "Tolerance",
            }
        ],
        "review_candidates": [
            {
                "candidate_id": "C0002",
                "bbox": _bbox(60),
                "text": "R0.3+0.05",
                "review_reason": "Tolerance grouping is incomplete",
            }
        ],
        "candidate_outcomes": [
            {
                "candidate_id": "C0001",
                "bbox": _bbox(10),
                "state": "eligible",
                "text": "25±0.1",
                "reason": "Complete engineering value",
            },
            {
                "candidate_id": "C0002",
                "bbox": _bbox(60),
                "state": "review",
                "text": "R0.3+0.05",
                "reason": "Tolerance grouping is incomplete",
            },
            {
                "candidate_id": "C0003",
                "bbox": _bbox(110),
                "state": "excluded",
                "text": "SCALE 1:1",
                "reason": "Scale information",
            },
        ],
    }


def _snapshot(result: dict | None = None) -> dict:
    return {
        "schema_version": 2,
        "route": "page",
        "dpi": 250,
        "pages": [
            {
                "page": 1,
                "width": 1000,
                "height": 700,
                "result": result or _result(),
            }
        ],
    }


def test_balanced_page_accounts_for_all_three_final_states():
    report = evaluate_page_accounting(_result(), page=1)
    assert report.passed
    assert report.detected == report.accepted + report.review + report.other
    assert report.recognized == report.detected - report.unread
    assert report.skipped_existing == 2
    assert report.to_dict()["issues"] == []


def test_snapshot_exposes_accepted_review_and_other_for_scoring():
    candidates = candidates_from_snapshot(_snapshot())
    assert [item["benchmark_disposition"] for item in candidates] == [
        "accepted",
        "review",
        "other",
    ]
    assert [item["text"] for item in candidates] == [
        "25±0.1",
        "R0.3+0.05",
        "SCALE 1:1",
    ]
    assert all(item["page"] == 1 for item in candidates)


def test_count_imbalance_and_missing_reason_are_explicit_failures():
    result = _result()
    result["detected_count"] = 4
    result["recognized_count"] = 4
    result["candidate_outcomes"][2]["reason"] = ""
    report = evaluate_page_accounting(result)
    assert not report.passed
    assert {issue.code for issue in report.issues} == {
        "count_not_balanced",
        "missing_disposition_reason",
    }


def test_duplicate_candidate_identity_is_rejected():
    result = _result()
    result["candidate_outcomes"][2]["candidate_id"] = "C0002"
    report = evaluate_page_accounting(result)
    assert not report.passed
    assert any(issue.code == "duplicate_candidate_id" for issue in report.issues)


def test_recognition_count_mismatch_is_rejected():
    result = _result()
    result["recognized_count"] = 2
    report = evaluate_page_accounting(result)
    assert not report.passed
    assert any(
        issue.code == "recognition_count_mismatch" for issue in report.issues
    )


def test_missing_lifecycle_count_is_explicit_failure():
    result = _result()
    del result["review_count"]
    report = evaluate_page_accounting(result)
    assert not report.passed
    assert any(issue.code == "missing_count" for issue in report.issues)


def test_empty_snapshot_is_invalid():
    with pytest.raises(ValueError, match="at least one page"):
        evaluate_snapshot_accounting({"schema_version": 2, "pages": []})
    with pytest.raises(ValueError, match="at least one page"):
        candidates_from_snapshot({"schema_version": 2, "pages": []})


def test_legacy_regions_remain_scorable_but_accounting_is_unavailable():
    legacy = [{"text": "25", "bbox": _bbox(10)}]
    accounting = evaluate_snapshot_accounting(legacy)
    candidates = candidates_from_snapshot(legacy)
    assert not accounting.available
    assert candidates[0]["benchmark_disposition"] == "accepted"
