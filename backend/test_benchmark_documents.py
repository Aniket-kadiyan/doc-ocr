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

from benchmarks.accounting import (
    candidates_from_snapshot,
    evaluate_snapshot_accounting,
)
from benchmarks.execution import segment_document
from benchmarks.scoring import (
    load_document,
    requirement_failures,
    score_document,
)

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


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_document_accuracy(fixture):
    spec = load_document(fixture)
    drawing = _resolve(spec["document"])
    if drawing is None:
        pytest.skip(f"drawing not available: {spec['document']}")

    route = spec.get("route", "page")
    snapshot = segment_document(
        drawing,
        spec.get("dpi", 250),
        route=route,
    )
    candidates = candidates_from_snapshot(snapshot)
    accounting = evaluate_snapshot_accounting(snapshot)

    report = score_document(
        spec["expected"],
        candidates,
        document=spec.get("title") or fixture.stem,
        noise_patterns=spec.get("noise_patterns", ()),
        match_threshold=spec.get("match_threshold", 0.9),
        partial_threshold=spec.get("partial_threshold", 0.6),
    )
    detail = "\n" + report.format_text() + "\n" + accounting.format_text()

    failures = requirement_failures(
        report,
        spec.get("requires", {}),
        accounting=accounting,
    )
    assert not failures, "\n".join([detail, "gate failures:", *failures])
