#!/usr/bin/env python3
"""
Run the auto-segment accuracy benchmark for one or more drawings.

Each drawing has a fixture in ``backend/benchmarks/documents/`` listing the
callouts it is known to carry. This renders the drawing, segments every page,
and reports per callout how much of the expected string came back.

    # the whole suite (renders and OCRs, minutes per drawing)
    .venv/bin/python scripts/run_benchmark.py

    # one drawing, saving its complete candidate lifecycle for re-scoring
    .venv/bin/python scripts/run_benchmark.py BS1801006.020 --save-snapshot

    # re-score a saved snapshot after changing scoring or ground truth
    .venv/bin/python scripts/run_benchmark.py BS1801006.020 --regions \\
        benchmarks/results/BS1801006.020.scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from benchmarks.accounting import (  # noqa: E402
    candidates_from_snapshot,
    evaluate_snapshot_accounting,
)
from benchmarks.scoring import (  # noqa: E402
    load_document,
    requirement_failures,
    score_document,
)

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


def render_pages(path: Path, dpi: int):
    if path.suffix.lower() != ".pdf":
        from PIL import Image

        yield Image.open(path).convert("RGB")
        return
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    for index in range(len(pdf)):
        yield pdf[index].render(scale=dpi / 72).to_pil().convert("RGB")


def _json_safe(value):
    """Convert NumPy scalars, tuples, and nested scan data to JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def segment_document(path: Path, dpi: int, *, route: str = "page") -> dict:
    """Run every page and retain the complete result for lifecycle scoring."""

    from ocr_pipeline import get_pipeline

    pipeline = get_pipeline()
    pages: list[dict] = []
    started = time.time()
    for page_no, image in enumerate(render_pages(path, dpi), start=1):
        result = (
            pipeline.segment_page(image)
            if route == "page"
            else pipeline.segment(image)
        )
        pages.append(
            {
                "page": page_no,
                "width": image.width,
                "height": image.height,
                "result": _json_safe(result),
            }
        )
    return {
        "schema_version": 2,
        "route": route,
        "dpi": dpi,
        "seconds": time.time() - started,
        "pages": pages,
    }


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
                RESULTS.mkdir(parents=True, exist_ok=True)
                target = RESULTS / f"{fixture.stem}.regions.json"
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
                RESULTS.mkdir(parents=True, exist_ok=True)
                target = RESULTS / f"{fixture.stem}.scan.json"
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
        failures = requirement_failures(
            report,
            spec.get("requires", {}),
            accounting=accounting,
        )
        payload = report.to_dict()
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

    if args.json:
        print(json.dumps(reports, indent=1, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
