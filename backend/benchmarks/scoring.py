"""
Score an auto-segment run against the dimensions a drawing is known to carry.

A run is judged per expected callout, not by counting regions: every callout
gets its best detection, a percentage for how much of the string came back, and
a verdict. Detections that match nothing are reported separately, split into
the drawing's known non-dimension text (title block, part number) and genuine
surprises, so a regression shows up as a changed number rather than a wall of
output to re-read.

One expected callout may list several acceptable strings. Two entries in a
ground-truth list sometimes describe the same physical callout read two ways —
``Ø33-0.05-0.1`` and ``33-0.1-0.05`` are one dimension — and either read
satisfies it.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Glyph families that mean the same thing on a drawing. Comparing raw strings
# would fail a correct read for spelling Ø with a different code point.
# Note: the letter O is deliberately NOT folded into Ø. A leading O is often a
# misread diameter mark, but treating them as identical would score that misread
# as a perfect read.
_DIAMETER_GLYPHS = "ø⌀φΦ∅"
_MINUS_GLYPHS = "−–—‒"
_DEGREE_GLYPHS = "˚⁰º"
# Prime and quote marks for minutes and seconds. OCR emits them inconsistently
# and a ground-truth list rarely includes them, so they are dropped on both
# sides rather than counted as a difference.
_TICK_GLYPHS = "′ʹ'’‘″\"”“`"

_DEFAULT_MATCH = 0.90
_DEFAULT_PARTIAL = 0.60


def normalize(text: str) -> str:
    """Reduce a reading to what a comparison should actually care about."""
    out = []
    for char in (text or "").strip().upper():
        if char.isspace() or char in _TICK_GLYPHS:
            continue
        if char in _DIAMETER_GLYPHS.upper():
            out.append("Ø")
        elif char in _MINUS_GLYPHS:
            out.append("-")
        elif char in _DEGREE_GLYPHS.upper():
            out.append("°")
        else:
            out.append(char)
    return "".join(out)


def digits_of(text: str) -> str:
    """Just the digits of a reading, in order."""
    return "".join(c for c in (text or "") if c.isdigit())


def similarity(a: str, b: str) -> float:
    """How much of two readings agree, 0.0 to 1.0, after normalising."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


@dataclass(frozen=True)
class ExpectedItem:
    """One callout the drawing is known to carry."""

    id: int
    accept: tuple[str, ...]
    type: str | None = None
    note: str | None = None

    @property
    def label(self) -> str:
        return " | ".join(self.accept)


@dataclass
class ItemResult:
    expected: ExpectedItem
    detected: str | None
    detected_type: str | None
    score: float
    verdict: str          # "match" | "partial" | "miss"
    type_ok: bool | None  # None when the fixture states no type

    @property
    def percent(self) -> float:
        return round(self.score * 100, 1)


@dataclass
class Report:
    document: str
    items: list[ItemResult]
    noise: list[dict[str, Any]] = field(default_factory=list)
    extras: list[dict[str, Any]] = field(default_factory=list)
    match_threshold: float = _DEFAULT_MATCH
    partial_threshold: float = _DEFAULT_PARTIAL
    seconds: float | None = None

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def matched(self) -> int:
        return sum(1 for i in self.items if i.verdict == "match")

    @property
    def partial(self) -> int:
        return sum(1 for i in self.items if i.verdict == "partial")

    @property
    def missed(self) -> int:
        return sum(1 for i in self.items if i.verdict == "miss")

    @property
    def mean_score(self) -> float:
        if not self.items:
            return 0.0
        return sum(i.score for i in self.items) / len(self.items)

    @property
    def typed_correct(self) -> int:
        return sum(1 for i in self.items if i.type_ok)

    @property
    def typed_total(self) -> int:
        return sum(1 for i in self.items if i.type_ok is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "seconds": self.seconds,
            "totals": {
                "expected": self.total,
                "matched": self.matched,
                "partial": self.partial,
                "missed": self.missed,
                "mean_percent": round(self.mean_score * 100, 1),
                "type_correct": self.typed_correct,
                "type_total": self.typed_total,
                "noise_detections": len(self.noise),
                "unexpected_detections": len(self.extras),
            },
            "items": [
                {
                    "id": i.expected.id,
                    "expected": i.expected.label,
                    "detected": i.detected,
                    "percent": i.percent,
                    "verdict": i.verdict,
                    "expected_type": i.expected.type,
                    "detected_type": i.detected_type,
                    "type_ok": i.type_ok,
                }
                for i in self.items
            ],
            "noise": self.noise,
            "unexpected": self.extras,
        }

    def format_text(self) -> str:
        mark = {"match": "OK  ", "partial": "~   ", "miss": "MISS"}
        width = max((len(i.expected.label) for i in self.items), default=10)
        lines = [f"document: {self.document}"]
        if self.seconds is not None:
            lines.append(f"segment:  {self.seconds:.0f}s")
        lines.append("")
        lines.append(f"{'':4} {'#':>2}  {'expected'.ljust(width)}  {'detected'.ljust(width)}   %     type")
        lines.append("-" * (width * 2 + 34))
        for item in self.items:
            detected = item.detected if item.detected is not None else "-"
            if item.type_ok is None:
                type_note = "-"
            else:
                type_note = "ok" if item.type_ok else f"{item.detected_type or '-'} != {item.expected.type}"
            lines.append(
                f"{mark[item.verdict]} {item.expected.id:>2}  "
                f"{item.expected.label.ljust(width)}  {detected.ljust(width)}  "
                f"{item.percent:5.1f}  {type_note}"
            )
        lines.append("-" * (width * 2 + 34))
        lines.append(
            f"matched {self.matched}/{self.total}"
            f"   partial {self.partial}   missed {self.missed}"
            f"   mean {self.mean_score * 100:.1f}%"
            f"   type {self.typed_correct}/{self.typed_total}"
        )
        if self.noise:
            lines.append(f"known non-dimension text ignored: {len(self.noise)}")
        if self.extras:
            lines.append("unexpected detections:")
            for extra in self.extras:
                lines.append(f"    {extra['text']!r} ({extra.get('type')})")
        return "\n".join(lines)


def load_document(path: str | Path) -> dict[str, Any]:
    """Read a document fixture and build its expected items."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for raw in data["expected"]:
        accept = raw.get("accept")
        if isinstance(accept, str):
            accept = [accept]
        if not accept:
            raise ValueError(f"expected item {raw.get('id')} lists no accepted text")
        items.append(
            ExpectedItem(
                id=int(raw["id"]),
                accept=tuple(accept),
                type=raw.get("type"),
                note=raw.get("note"),
            )
        )
    data["expected"] = items
    return data


def _is_noise(text: str, patterns: Iterable[str]) -> bool:
    normalized = normalize(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def score_document(
    expected: list[ExpectedItem],
    regions: list[dict[str, Any]],
    *,
    document: str = "",
    noise_patterns: Iterable[str] = (),
    match_threshold: float = _DEFAULT_MATCH,
    partial_threshold: float = _DEFAULT_PARTIAL,
    seconds: float | None = None,
) -> Report:
    """
    Pair every expected callout with its best detection and score the run.

    Pairing is one-to-one and greedy from the strongest pair down, so a single
    detection cannot be credited for two different callouts — which matters
    exactly where a drawing carries two similar values.
    """
    texts = [str(r.get("text") or "") for r in regions]
    pairs: list[tuple[float, int, int]] = []
    for ei, item in enumerate(expected):
        for ri, text in enumerate(texts):
            score = max(similarity(candidate, text) for candidate in item.accept)
            if score > 0:
                pairs.append((score, ei, ri))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    taken_expected: dict[int, tuple[float, int]] = {}
    taken_region: set[int] = set()
    for score, ei, ri in pairs:
        if ei in taken_expected or ri in taken_region:
            continue
        taken_expected[ei] = (score, ri)
        taken_region.add(ri)

    results: list[ItemResult] = []
    for ei, item in enumerate(expected):
        score, ri = taken_expected.get(ei, (0.0, -1))
        region = regions[ri] if ri >= 0 else None
        detected_type = (region or {}).get("type")
        # A dimension whose digits differ is not a match however close the
        # strings look: Ø23.5-0.05 and Ø23.5-0.06 are 90% alike and are
        # different parts. Only the symbols around the digits may vary.
        digits_agree = region is not None and any(
            digits_of(candidate) == digits_of(texts[ri]) for candidate in item.accept
        )
        if score >= match_threshold and digits_agree:
            verdict = "match"
        elif score >= partial_threshold:
            verdict = "partial"
        else:
            verdict = "miss"
            region, detected_type = None, None
        # The fixture may name either the composed kind (linear, diameter,
        # radius, angle, tolerance) or the category the app displays (Basic,
        # Reference, GD&T …). They come from different fields, so either
        # naming counts.
        type_ok = None
        if item.type:
            wanted = item.type.strip().lower()
            got_kind = (detected_type or "").strip().lower()
            got_category = ((region or {}).get("category") or "").strip().lower()
            type_ok = bool(region) and wanted in {got_kind, got_category}
        results.append(
            ItemResult(
                expected=item,
                detected=(region or {}).get("text") if region else None,
                detected_type=(
                    (region or {}).get("category") or detected_type
                    if item.type
                    and detected_type
                    and item.type.strip().lower() != detected_type.strip().lower()
                    else detected_type
                ),
                score=score,
                verdict=verdict,
                type_ok=type_ok,
            )
        )

    # A detection counts as used only if the callout it was paired with actually
    # credited it; a pairing too weak to be a partial leaves the region free to
    # be reported as noise or as a surprise.
    consumed = {ri for score, ri in taken_expected.values() if score >= partial_threshold}
    noise, extras = [], []
    for ri, region in enumerate(regions):
        if ri in consumed:
            continue
        entry = {"text": texts[ri], "type": region.get("type")}
        (noise if _is_noise(texts[ri], noise_patterns) else extras).append(entry)

    return Report(
        document=document,
        items=results,
        noise=noise,
        extras=extras,
        match_threshold=match_threshold,
        partial_threshold=partial_threshold,
        seconds=seconds,
    )
