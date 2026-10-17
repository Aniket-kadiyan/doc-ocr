#!/usr/bin/env python
"""
Measure the GD&T recognizer against the synthetic suite and print a baseline.

    cd backend
    PYTHONPATH=. .venv/bin/python scripts/run_gdt_eval.py
    PYTHONPATH=. .venv/bin/python scripts/run_gdt_eval.py --conditions clean,scan
    PYTHONPATH=. .venv/bin/python scripts/run_gdt_eval.py --ocr --save

``--ocr`` loads PaddleOCR so the feature-control-frame *fields* are scored as
well. Without it the frames are still found and their symbols classified, but
the tolerance and datum compartments have no text, so those fields are
reported as declined rather than wrong. The symbol numbers do not need OCR and
do not change between the two modes.

Read the macro F1, not the micro one, and read it per condition. The suite is
synthetic: see the module docstring of ``benchmarks/gdt_synthetic.py`` for what
that does and does not entitle anyone to claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from benchmarks import gdt_eval, gdt_synthetic as syn  # noqa: E402


def build_ocr():
    """Load the project's own OCR pipeline as the frame-text reader."""
    from ocr_pipeline import get_pipeline

    pipeline = get_pipeline()

    def ocr_fn(image):
        try:
            result = pipeline.recognize(image)
            return (result.get("text") or "", float(result.get("confidence") or 0.0))
        except Exception:  # noqa: BLE001 - a bad cell is not a failed run
            return ("", 0.0)

    return ocr_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        default="clean,blurred,noisy,low_res,skewed,scan,poor_scan",
        help="comma-separated conditions from gdt_synthetic.CONDITIONS",
    )
    parser.add_argument(
        "--scenes",
        default=",".join(syn.BUILDERS),
        help="comma-separated scene names",
    )
    parser.add_argument("--ocr", action="store_true", help="load PaddleOCR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iou", type=float, default=gdt_eval.DEFAULT_IOU)
    parser.add_argument(
        "--save",
        nargs="?",
        const="benchmarks/results/gdt_baseline.json",
        help="write the reports as JSON",
    )
    parser.add_argument("--quiet", action="store_true", help="summary only")
    args = parser.parse_args()

    import gdt_pipeline

    ocr_fn = build_ocr() if args.ocr else None

    def analyze(image):
        return gdt_pipeline.analyze_page(
            image, ocr_fn=ocr_fn, source="synthetic-eval"
        )

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    reports: list[gdt_eval.Report] = []
    for condition in conditions:
        per_condition: list[gdt_eval.Report] = []
        for name in scenes:
            base = syn.build(name, seed=args.seed)
            scene = (
                base
                if condition == "clean"
                else syn.with_condition(base, condition, seed=args.seed)
            )
            report = gdt_eval.evaluate_scene(scene, analyze, iou_threshold=args.iou)
            per_condition.append(report)
            reports.append(report)
            if not args.quiet:
                print(report.format_text())
                print()
        pooled = gdt_eval.merge(per_condition, name=f"{condition} (all scenes)")
        pooled.condition = condition
        mp, mr, mf = pooled.macro
        print(
            f"== {condition:<10} macro P {mp:.3f} R {mr:.3f} F1 {mf:.3f}"
            f"   macro top-k R {pooled.macro_top_k_recall:.3f}"
            f"   declined {pooled.declined_rate:.1%}"
            f"   FP rate {pooled.false_positive_rate:.1%}"
            f"   frames {pooled.frames.frames_found}/{pooled.frames.frames_expected}"
            f"   {pooled.elapsed_ms / max(pooled.pages, 1):.0f}ms/page"
        )

    if args.save:
        target = gdt_eval.save(reports, BACKEND / args.save)
        print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
