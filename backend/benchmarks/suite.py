"""Combined M1-M4 benchmark reporting."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


_SUM_FIELDS = (
    "expected", "matched", "partial", "missed", "type_correct", "type_total",
    "semantic_exact", "disposition_correct", "disposition_total",
    "noise_detections", "unexpected_detections", "unmatched_review_or_other",
)

_GDT_COUNT_FIELDS = (
    "detected_frame_count",
    "published_frame_count",
    "eligible_frame_count",
    "review_frame_count",
    "absorbed_candidate_count",
    "skipped_existing_count",
    "duplicate_frame_count",
    "native_evidence_frame_count",
)


def snapshot_execution_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize which recognition source actually produced each page."""
    profiles: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    recognition_sources: Counter[str] = Counter()
    gdt_statuses: Counter[str] = Counter()
    gdt_totals: Counter[str] = Counter()
    gdt_policy = snapshot.get("gdt_page_scan") or {}
    gdt_enabled = bool(gdt_policy.get("enabled"))
    pages: list[dict[str, Any]] = []
    for page in snapshot.get("pages", ()):
        result = page.get("result") or {}
        source_profile = result.get("source_profile") or {}
        kind = str(source_profile.get("kind") or "unknown")
        strategy = str(source_profile.get("text_layer_strategy") or "raster_ocr")
        profiles[kind] += 1
        strategies[strategy] += 1
        for source, count in (result.get("recognition_source_counts") or {}).items():
            recognition_sources[str(source)] += int(count)
        gdt_integration = result.get("gdt_integration") or {}
        gdt_status = str(
            gdt_integration.get("status")
            or ("disabled" if not gdt_enabled else "missing")
        )
        gdt_statuses[gdt_status] += 1
        for field in _GDT_COUNT_FIELDS:
            gdt_totals[field] += int(gdt_integration.get(field, 0) or 0)
        pages.append({
            "page": int(page.get("page", len(pages) + 1)),
            "source_profile": kind,
            "text_layer_strategy": strategy,
            "recognition_source_counts": dict(sorted(
                (result.get("recognition_source_counts") or {}).items()
            )),
            "gdt_integration": {
                "status": gdt_status,
                **{
                    field: int(gdt_integration.get(field, 0) or 0)
                    for field in _GDT_COUNT_FIELDS
                },
                **(
                    {"error": str(gdt_integration.get("error") or "")}
                    if gdt_status == "error"
                    else {}
                ),
            },
        })
    return {
        "policy": snapshot.get("execution_policy", "unknown"),
        "source_format": snapshot.get("source_format", "unknown"),
        "source_profile_pages": dict(sorted(profiles.items())),
        "text_layer_strategy_pages": dict(sorted(strategies.items())),
        "recognition_source_counts": dict(sorted(recognition_sources.items())),
        "gdt_page_scan": {
            "policy": str(gdt_policy.get("policy") or "unknown"),
            "enabled": gdt_enabled,
            "status_pages": dict(sorted(gdt_statuses.items())),
            "completed_pages": gdt_statuses["completed"],
            "error_pages": gdt_statuses["error"],
            "missing_pages": gdt_statuses["missing"],
            **{field: gdt_totals[field] for field in _GDT_COUNT_FIELDS},
        },
        "pages": pages,
    }


def required_gdt_execution_failures(
    execution: Mapping[str, Any],
) -> list[str]:
    """Return gate failures for a benchmark that explicitly requires GD&T."""

    gdt = execution.get("gdt_page_scan") or {}
    if not bool(gdt.get("enabled")):
        return ["GD&T production integration was not enabled for this run"]

    page_count = len(execution.get("pages") or ())
    if page_count == 0:
        return ["GD&T production integration did not execute on any pages"]

    completed = int(gdt.get("completed_pages", 0) or 0)
    incomplete = max(0, page_count - completed)
    if incomplete:
        return [
            "GD&T production integration did not complete on "
            f"{incomplete} page(s)"
        ]
    return []


def build_suite_report(
    reports: Iterable[Mapping[str, Any]],
    *,
    ocr_changeset: str | None = None,
) -> dict[str, Any]:
    """Aggregate detailed document payloads into one regression report."""
    documents = [dict(report) for report in reports]
    totals = {field: 0 for field in _SUM_FIELDS}
    weighted_score = 0.0
    seconds = 0.0
    timed_documents = 0
    accounting_pages = 0
    accounting_issues = 0
    accounting_passed = 0
    accounting_unavailable = 0
    profiles: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    recognition_sources: Counter[str] = Counter()
    gdt_statuses: Counter[str] = Counter()
    gdt_totals: Counter[str] = Counter()
    gdt_enabled_documents = 0
    dispositions = {
        state: {"expected": 0, "matched": 0, "semantic_exact": 0,
                "disposition_correct": 0}
        for state in ("accepted", "review", "other")
    }
    rows: list[dict[str, Any]] = []

    for report in documents:
        document_totals = report.get("totals") or {}
        for field in _SUM_FIELDS:
            totals[field] += int(document_totals.get(field, 0))
        expected = int(document_totals.get("expected", 0))
        weighted_score += float(document_totals.get("mean_percent", 0.0)) * expected
        if isinstance(report.get("seconds"), (int, float)):
            seconds += float(report["seconds"])
            timed_documents += 1
        for state, values in dispositions.items():
            source = (document_totals.get("by_expected_disposition") or {}).get(state, {})
            for field in values:
                values[field] += int(source.get(field, 0))

        accounting = report.get("accounting") or {}
        if not bool(accounting.get("available", True)):
            accounting_unavailable += 1
        pages = list(accounting.get("pages") or ())
        accounting_pages += len(pages)
        accounting_issues += int(accounting.get("issue_count", 0))
        accounting_passed += sum(bool(page.get("passed")) for page in pages)

        execution = report.get("execution") or {}
        profiles.update(execution.get("source_profile_pages") or {})
        strategies.update(execution.get("text_layer_strategy_pages") or {})
        recognition_sources.update(execution.get("recognition_source_counts") or {})
        gdt_execution = execution.get("gdt_page_scan") or {}
        gdt_enabled_documents += int(bool(gdt_execution.get("enabled")))
        gdt_statuses.update(gdt_execution.get("status_pages") or {})
        for field in _GDT_COUNT_FIELDS:
            gdt_totals[field] += int(gdt_execution.get(field, 0) or 0)
        failures = list(report.get("gate_failures") or ())
        rows.append({
            "fixture": report.get("fixture", ""),
            "document": report.get("document", ""),
            "passed": not failures,
            "gate_failures": failures,
            "seconds": report.get("seconds"),
            "totals": document_totals,
            "accounting": {
                "passed": bool(accounting.get("passed")),
                "issue_count": int(accounting.get("issue_count", 0)),
            },
            "execution": execution,
        })

    failed_documents = sum(not row["passed"] for row in rows)
    totals["mean_percent"] = round(
        weighted_score / totals["expected"] if totals["expected"] else 0.0, 1
    )
    totals["by_expected_disposition"] = dispositions
    return {
        "schema_version": 1,
        "milestone": "M5 combined M1-M4 regression",
        "ocr_changeset": ocr_changeset or "unknown",
        "summary": {
            "documents": len(rows),
            "documents_passed": len(rows) - failed_documents,
            "documents_failed": failed_documents,
            "gate_failures": sum(len(row["gate_failures"]) for row in rows),
            "seconds": round(seconds, 3) if timed_documents else None,
            "totals": totals,
            "accounting": {
                "pages": accounting_pages,
                "pages_passed": accounting_passed,
                "issue_count": accounting_issues,
                "unavailable_documents": accounting_unavailable,
                "passed": (
                    bool(accounting_pages)
                    and accounting_passed == accounting_pages
                    and accounting_issues == 0
                    and accounting_unavailable == 0
                ),
            },
            "execution": {
                "source_profile_pages": dict(sorted(profiles.items())),
                "text_layer_strategy_pages": dict(sorted(strategies.items())),
                "recognition_source_counts": dict(sorted(recognition_sources.items())),
                "gdt_page_scan": {
                    "enabled_documents": gdt_enabled_documents,
                    "status_pages": dict(sorted(gdt_statuses.items())),
                    "completed_pages": gdt_statuses["completed"],
                    "error_pages": gdt_statuses["error"],
                    "missing_pages": gdt_statuses["missing"],
                    **{field: gdt_totals[field] for field in _GDT_COUNT_FIELDS},
                },
            },
        },
        "documents": rows,
    }


def format_suite_text(suite: Mapping[str, Any]) -> str:
    summary = suite["summary"]
    totals = summary["totals"]
    accounting = summary["accounting"]
    status = "PASS" if summary["documents_failed"] == 0 else "FAIL"
    return "\n".join((
        f"combined benchmark: {status}",
        f"  documents: {summary['documents_passed']}/{summary['documents']} passed",
        f"  recognition: {totals['matched']}/{totals['expected']} matched; "
        f"{totals['partial']} partial; {totals['missed']} missed; "
        f"mean {totals['mean_percent']:.1f}%",
        f"  semantics: {totals['semantic_exact']}/{totals['expected']} exact; "
        f"type {totals['type_correct']}/{totals['type_total']}; "
        f"state {totals['disposition_correct']}/{totals['disposition_total']}",
        f"  false positives: {totals['unexpected_detections']} unexpected accepted; "
        f"{totals['unmatched_review_or_other']} unmatched review/other retained",
        f"  accounting: {accounting['pages_passed']}/{accounting['pages']} pages; "
        f"{accounting['issue_count']} issue(s); "
        f"{accounting['unavailable_documents']} unavailable document(s)",
        (
            "  GD&T frames: "
            f"{summary['execution']['gdt_page_scan']['published_frame_count']} published; "
            f"{summary['execution']['gdt_page_scan']['eligible_frame_count']} accepted; "
            f"{summary['execution']['gdt_page_scan']['review_frame_count']} review; "
            f"{summary['execution']['gdt_page_scan']['error_pages']} error page(s)"
        ),
        f"  runtime: {summary['seconds']:.1f}s"
        if summary["seconds"] is not None else "  runtime: unavailable",
    ))


def format_suite_markdown(suite: Mapping[str, Any]) -> str:
    """Produce a shareable, diff-friendly Markdown benchmark summary."""
    summary = suite["summary"]
    totals = summary["totals"]
    accounting = summary["accounting"]
    status = "PASS" if summary["documents_failed"] == 0 else "FAIL"
    lines = [
        "# QTracX combined OCR benchmark", "", f"**Suite status:** {status}", "",
        "## Combined results", "", "| Measure | Result |", "|---|---:|",
        f"| Documents passing | {summary['documents_passed']} / {summary['documents']} |",
        f"| Expected callouts matched | {totals['matched']} / {totals['expected']} |",
        f"| Partial / missed | {totals['partial']} / {totals['missed']} |",
        f"| Mean recognition score | {totals['mean_percent']:.1f}% |",
        f"| Semantically exact | {totals['semantic_exact']} / {totals['expected']} |",
        f"| Correct type | {totals['type_correct']} / {totals['type_total']} |",
        f"| Correct final state | {totals['disposition_correct']} / {totals['disposition_total']} |",
        f"| Unexpected accepted detections | {totals['unexpected_detections']} |",
        f"| Accounting pages passing | {accounting['pages_passed']} / {accounting['pages']} |",
        f"| Accounting issues | {accounting['issue_count']} |",
        f"| Accounting unavailable documents | {accounting['unavailable_documents']} |",
        f"| Total runtime | {summary['seconds']:.1f} s |"
        if summary["seconds"] is not None else "| Total runtime | unavailable |",
        "", "## Per drawing", "",
        "| Drawing | Status | Matched | Exact | State | Unexpected accepted | Runtime |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in suite["documents"]:
        values = row["totals"]
        seconds = row.get("seconds")
        runtime = f"{float(seconds):.1f} s" if isinstance(seconds, (int, float)) else "-"
        lines.append(
            f"| {row['document']} | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{values.get('matched', 0)} / {values.get('expected', 0)} | "
            f"{values.get('semantic_exact', 0)} / {values.get('expected', 0)} | "
            f"{values.get('disposition_correct', 0)} / {values.get('disposition_total', 0)} | "
            f"{values.get('unexpected_detections', 0)} | {runtime} |"
        )
    lines.extend(("", "## Recognition sources", ""))
    execution = summary["execution"]
    for heading, key in (
        ("Page profiles", "source_profile_pages"),
        ("Text-layer strategies", "text_layer_strategy_pages"),
        ("Recognized candidates", "recognition_source_counts"),
    ):
        values = execution.get(key) or {}
        rendered = ", ".join(f"{name}: {count}" for name, count in values.items())
        lines.append(f"- **{heading}:** {rendered or 'unavailable'}")
    gdt = execution.get("gdt_page_scan") or {}
    lines.append(
        "- **GD&T production pass:** "
        f"{int(gdt.get('published_frame_count', 0))} published "
        f"({int(gdt.get('eligible_frame_count', 0))} accepted, "
        f"{int(gdt.get('review_frame_count', 0))} review); "
        f"{int(gdt.get('native_evidence_frame_count', 0))} used native evidence; "
        f"{int(gdt.get('error_pages', 0))} error page(s)"
    )
    failed = [row for row in suite["documents"] if not row["passed"]]
    if failed:
        lines.extend(("", "## Gate failures", ""))
        for row in failed:
            lines.append(f"- **{row['document']}**")
            lines.extend(f"  - {failure}" for failure in row["gate_failures"])
    lines.append("")
    return "\n".join(lines)
