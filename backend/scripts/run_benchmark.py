#!/usr/bin/env python3
"""
Run the auto-segment accuracy benchmark for one or more drawings.

Each drawing has a fixture in ``backend/benchmarks/documents/`` listing the
callouts it is known to carry. This renders the drawing, segments every page,
and reports per callout how much of the expected string came back.

    # the whole suite (renders and OCRs, minutes per drawing)
    .venv/bin/python scripts/run_benchmark.py

    # one drawing, saving its complete candidate lifecycle for re-scoring
    .venv/bin/python scripts/run_benchmark.py ballwoon_drawing \\
        --gdt-page-scan --save-snapshot

    # the same drawing with the optional hook forced off for comparison
    .venv/bin/python scripts/run_benchmark.py ballwoon_drawing \\
        --no-gdt-page-scan --save-snapshot

    # re-score a saved snapshot after changing scoring or ground truth
    .venv/bin/python scripts/run_benchmark.py BS1801006.020 --regions \\
        benchmarks/results/BS1801006.020.scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from benchmarks.accounting import (  # noqa: E402
    candidates_from_snapshot,
    evaluate_snapshot_accounting,
)
from benchmarks.execution import segment_document  # noqa: E402
from benchmarks.scoring import (  # noqa: E402
    load_document,
    requirement_failures,
    score_document,
)
from benchmarks.suite import (  # noqa: E402
    build_suite_report,
    format_suite_markdown,
    format_suite_text,
    required_gdt_execution_failures,
    snapshot_execution_summary,
)
from ocr_pipeline import OCR_CHANGESET_ID  # noqa: E402

DOCUMENTS = BACKEND / "benchmarks" / "documents"
RESULTS = BACKEND / "benchmarks" / "results"


def resolve_document(relative: str) -> Path:
    """Find the drawing file, whether given absolute or relative to the repo."""
    candidate = Path(relative)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    base = BACKEND
    for _ in range(6):
        probe = base / relative
        if probe.exists():
            return probe
        base = base.parent
    raise FileNotFoundError(f"cannot find {relative} above {BACKEND}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("names", nargs="*", help="fixture names (default: all)")
    parser.add_argument(
        "--regions",
        help="score a legacy regions JSON or full scan snapshot instead of OCR",
    )
    parser.add_argument(
        "--save-regions",
        action="store_true",
        help="write flattened candidates for legacy re-scoring",
    )
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="write accepted, review, other, counts, reasons, and geometry",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--output-dir",
        help="write combined-report.json and combined-report.md to this directory",
    )
    parser.add_argument(
        "--ignore-gates",
        action="store_true",
        help="report scores without returning failure for fixture gates",
    )
    parser.add_argument("--dpi", type=int, default=None, help="override the fixture's render DPI")
    parser.add_argument(
        "--route",
        choices=("page", "legacy_segment"),
        help="override the fixture route; page is the production Auto-Balloon route",
    )
    gdt_group = parser.add_mutually_exclusive_group()
    gdt_group.add_argument(
        "--gdt-page-scan",
        dest="gdt_page_scan",
        action="store_true",
        default=None,
        help=(
            "force the graphical GD&T production hook on, independently of "
            "GDT_PAGE_SCAN_ENABLED"
        ),
    )
    gdt_group.add_argument(
        "--no-gdt-page-scan",
        dest="gdt_page_scan",
        action="store_false",
        help=(
            "force the graphical GD&T production hook off for a comparison run"
        ),
    )
    parser.add_argument(
        "--page",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    fixtures = sorted(DOCUMENTS.glob("*.json"))
    if args.names:
        wanted = {name.removesuffix(".json") for name in args.names}
        fixtures = [f for f in fixtures if f.stem in wanted]
        missing = wanted - {f.stem for f in fixtures}
        if missing:
            print(f"no fixture for: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    if not fixtures:
        print(f"no fixtures in {DOCUMENTS}", file=sys.stderr)
        return 2
    if args.regions and len(fixtures) != 1:
        print("--regions scores a single document", file=sys.stderr)
        return 2

    reports = []
    failed = False
    for fixture in fixtures:
        spec = load_document(fixture)
        if args.regions:
            snapshot = json.loads(
                Path(args.regions).read_text(encoding="utf-8")
            )
        else:
            path = resolve_document(spec["document"])
            route = args.route or (
                "page" if args.page else spec.get("route", "page")
            )
            snapshot = segment_document(
                path,
                args.dpi or spec.get("dpi", 250),
                route=route,
                gdt_page_scan=args.gdt_page_scan,
            )
            snapshot["fixture"] = fixture.name
            snapshot["document"] = spec["document"]
            snapshot["title"] = spec.get("title") or fixture.stem

        candidates = candidates_from_snapshot(snapshot)
        accounting = evaluate_snapshot_accounting(snapshot)
        seconds = (
            float(snapshot["seconds"])
            if isinstance(snapshot, dict)
            and isinstance(snapshot.get("seconds"), (int, float))
            else None
        )

        if not args.regions:
            if args.save_regions:
                snapshot_dir = Path(args.output_dir) if args.output_dir else RESULTS
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                target = snapshot_dir / f"{fixture.stem}.regions.json"
                legacy_candidates = [
                    {
                        key: value
                        for key, value in candidate.items()
                        if not key.startswith("_benchmark_")
                    }
                    for candidate in candidates
                ]
                target.write_text(
                    json.dumps(legacy_candidates, indent=1, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"wrote {target}")
            if args.save_snapshot:
                snapshot_dir = Path(args.output_dir) if args.output_dir else RESULTS
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                target = snapshot_dir / f"{fixture.stem}.scan.json"
                target.write_text(
                    json.dumps(snapshot, indent=1, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"wrote {target}")

        report = score_document(
            spec["expected"],
            candidates,
            document=spec.get("title") or fixture.stem,
            noise_patterns=spec.get("noise_patterns", ()),
            match_threshold=spec.get("match_threshold", 0.9),
            partial_threshold=spec.get("partial_threshold", 0.6),
            seconds=seconds,
        )
        execution_summary = (
            snapshot_execution_summary(snapshot)
            if isinstance(snapshot, dict) and isinstance(snapshot.get("pages"), list)
            else {}
        )
        failures = requirement_failures(
            report,
            spec.get("requires", {}),
            accounting=accounting,
        )
        if args.gdt_page_scan is True:
            failures.extend(required_gdt_execution_failures(execution_summary))
        payload = report.to_dict()
        payload["fixture"] = fixture.name
        payload["document_path"] = spec["document"]
        payload["route"] = snapshot.get("route") if isinstance(snapshot, dict) else None
        payload["dpi"] = snapshot.get("dpi") if isinstance(snapshot, dict) else None
        payload["execution"] = execution_summary
        payload["accounting"] = accounting.to_dict()
        payload["gate_failures"] = failures
        reports.append(payload)
        if not args.json:
            print(report.format_text())
            print(accounting.format_text())
            if failures:
                print("gate failures:")
                for failure in failures:
                    print(f"  - {failure}")
            print()
        if failures and not args.ignore_gates:
            failed = True

    suite = build_suite_report(reports, ocr_changeset=OCR_CHANGESET_ID)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_target = output_dir / "combined-report.json"
        markdown_target = output_dir / "combined-report.md"
        json_target.write_text(
            json.dumps(suite, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        markdown_target.write_text(format_suite_markdown(suite), encoding="utf-8")
        if not args.json:
            print(f"wrote {json_target}")
            print(f"wrote {markdown_target}")

    if args.json:
        print(json.dumps(reports, indent=1, ensure_ascii=False))
    else:
        print(format_suite_text(suite))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
