#!/usr/bin/env python3
"""
Run the auto-segment accuracy benchmark for one or more drawings.

Each drawing has a fixture in ``backend/benchmarks/documents/`` listing the
callouts it is known to carry. This renders the drawing, segments every page,
and reports per callout how much of the expected string came back.

    # the whole suite (renders and OCRs, minutes per drawing)
    .venv/bin/python scripts/run_benchmark.py

    # one drawing, saving its regions so it can be re-scored for free
    .venv/bin/python scripts/run_benchmark.py BS1801006.020 --save-regions

    # re-score a saved run after changing the scoring or the fixture
    .venv/bin/python scripts/run_benchmark.py BS1801006.020 --regions \\
        benchmarks/results/BS1801006.020.regions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from benchmarks.scoring import load_document, score_document  # noqa: E402

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


def segment_document(path: Path, dpi: int) -> tuple[list[dict], float]:
    from ocr_pipeline import get_pipeline

    pipeline = get_pipeline()
    regions: list[dict] = []
    started = time.time()
    for page_no, image in enumerate(render_pages(path, dpi), start=1):
        result = pipeline.segment(image)
        for region in result["regions"]:
            regions.append({**region, "page": page_no})
    return regions, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("names", nargs="*", help="fixture names (default: all)")
    parser.add_argument("--regions", help="score this saved regions JSON instead of running OCR")
    parser.add_argument("--save-regions", action="store_true", help="write the regions for re-scoring")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--dpi", type=int, default=None, help="override the fixture's render DPI")
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
            regions = json.loads(Path(args.regions).read_text(encoding="utf-8"))
            seconds = None
        else:
            path = resolve_document(spec["document"])
            regions, seconds = segment_document(path, args.dpi or spec.get("dpi", 250))
            if args.save_regions:
                RESULTS.mkdir(parents=True, exist_ok=True)
                target = RESULTS / f"{fixture.stem}.regions.json"
                target.write_text(json.dumps(regions, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"wrote {target}")

        report = score_document(
            spec["expected"],
            regions,
            document=spec.get("title") or fixture.stem,
            noise_patterns=spec.get("noise_patterns", ()),
            match_threshold=spec.get("match_threshold", 0.9),
            partial_threshold=spec.get("partial_threshold", 0.6),
            seconds=seconds,
        )
        reports.append(report.to_dict())
        if not args.json:
            print(report.format_text())
            print()
        if report.missed:
            failed = True

    if args.json:
        print(json.dumps(reports, indent=1, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
