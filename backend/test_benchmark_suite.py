"""Fast tests for the combined M1-M4 benchmark summary."""

from benchmarks.suite import (
    build_suite_report, format_suite_markdown, format_suite_text,
    snapshot_execution_summary,
)


def _report(name, *, expected, matched, mean, failures=(), seconds=1.0):
    zero = {"expected": 0, "matched": 0, "semantic_exact": 0,
            "disposition_correct": 0}
    return {
        "fixture": f"{name}.json", "document": name, "seconds": seconds,
        "totals": {
            "expected": expected, "matched": matched,
            "partial": expected - matched, "missed": 0, "mean_percent": mean,
            "type_correct": matched, "type_total": expected,
            "semantic_exact": matched, "disposition_correct": matched,
            "disposition_total": expected, "noise_detections": 0,
            "unexpected_detections": 1, "unmatched_review_or_other": 2,
            "by_expected_disposition": {
                "accepted": {"expected": expected, "matched": matched,
                             "semantic_exact": matched,
                             "disposition_correct": matched},
                "review": dict(zero), "other": dict(zero),
            },
        },
        "accounting": {
            "available": True, "passed": not failures,
            "issue_count": len(failures), "pages": [{"passed": not failures}],
        },
        "execution": {
            "source_profile_pages": {"raster": 1},
            "text_layer_strategy_pages": {"raster_ocr": 1},
            "recognition_source_counts": {"ocr": matched},
        },
        "gate_failures": list(failures),
    }


def test_combined_totals_use_expected_weighted_mean_and_keep_failures():
    suite = build_suite_report([
        _report("A", expected=2, matched=2, mean=100.0),
        _report("B", expected=6, matched=3, mean=50.0,
                failures=("matched 3 < 4",), seconds=3.0),
    ], ocr_changeset="mvp1-m4-test")
    summary = suite["summary"]
    assert summary["documents_passed"] == 1
    assert summary["documents_failed"] == 1
    assert summary["gate_failures"] == 1
    assert summary["seconds"] == 4.0
    assert summary["totals"]["expected"] == 8
    assert summary["totals"]["matched"] == 5
    assert summary["totals"]["mean_percent"] == 62.5
    assert summary["totals"]["unexpected_detections"] == 2
    assert summary["accounting"]["pages_passed"] == 1
    assert summary["execution"]["recognition_source_counts"] == {"ocr": 5}
    assert suite["ocr_changeset"] == "mvp1-m4-test"
    assert "combined benchmark: FAIL" in format_suite_text(suite)
    markdown = format_suite_markdown(suite)
    assert "# QTracX combined OCR benchmark" in markdown
    assert "matched 3 < 4" in markdown


def test_snapshot_summary_records_native_and_ocr_sources():
    snapshot = {
        "execution_policy": "native_pdf_first", "source_format": "pdf",
        "pages": [
            {"page": 1, "result": {
                "source_profile": {"kind": "vector",
                                   "text_layer_strategy": "native_text_first"},
                "recognition_source_counts": {"native_pdf": 4},
            }},
            {"page": 2, "result": {
                "source_profile": {"kind": "hybrid",
                    "text_layer_strategy": "native_text_first_hybrid_ocr"},
                "recognition_source_counts": {"native_pdf": 2, "ocr": 3},
            }},
        ],
    }
    summary = snapshot_execution_summary(snapshot)
    assert summary["policy"] == "native_pdf_first"
    assert summary["source_profile_pages"] == {"hybrid": 1, "vector": 1}
    assert summary["recognition_source_counts"] == {"native_pdf": 6, "ocr": 3}
