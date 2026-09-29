"""
End-to-end accuracy benchmark: segment a real drawing and score the result.

Slow and model-dependent, so it is opt-in. Every fixture in
``benchmarks/documents/`` becomes a test case, checked against the ``requires``
block it carries.

    RUN_DOC_BENCHMARKS=1 .venv/bin/python -m pytest -q test_benchmark_documents.py

Without the flag these skip, and the fast suite stays fast. The fixtures
themselves are still validated on every run by test_benchmark_scoring.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmarks.scoring import load_document, score_document

BACKEND = Path(__file__).resolve().parent
DOCUMENTS = BACKEND / "benchmarks" / "documents"
FIXTURES = sorted(DOCUMENTS.glob("*.json"))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOC_BENCHMARKS", "").strip().lower() not in {"1", "true", "yes"},
    reason="set RUN_DOC_BENCHMARKS=1 to segment real drawings (minutes per document)",
)


def _resolve(relative: str) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    base = BACKEND
    for _ in range(6):
        probe = base / relative
        if probe.exists():
            return probe
        base = base.parent
    return None


def _render(path: Path, dpi: int):
    if path.suffix.lower() != ".pdf":
        from PIL import Image

        yield Image.open(path).convert("RGB")
        return
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    for index in range(len(pdf)):
        yield pdf[index].render(scale=dpi / 72).to_pil().convert("RGB")


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_document_accuracy(fixture):
    spec = load_document(fixture)
    drawing = _resolve(spec["document"])
    if drawing is None:
        pytest.skip(f"drawing not available: {spec['document']}")

    from ocr_pipeline import get_pipeline

    pipeline = get_pipeline()
    regions = []
    for image in _render(drawing, spec.get("dpi", 250)):
        regions.extend(pipeline.segment(image)["regions"])

    report = score_document(
        spec["expected"],
        regions,
        document=spec.get("title") or fixture.stem,
        noise_patterns=spec.get("noise_patterns", ()),
        match_threshold=spec.get("match_threshold", 0.9),
        partial_threshold=spec.get("partial_threshold", 0.6),
    )
    detail = "\n" + report.format_text()

    requires = spec.get("requires", {})
    if "matched" in requires:
        assert report.matched >= requires["matched"], detail
    if "mean_percent" in requires:
        assert report.mean_score * 100 >= requires["mean_percent"], detail
    if "type_correct" in requires:
        assert report.typed_correct >= requires["type_correct"], detail
    if "max_unexpected" in requires:
        assert len(report.extras) <= requires["max_unexpected"], detail
