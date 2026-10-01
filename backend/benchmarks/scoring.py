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
import math
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
    disposition: str = "accepted"
    page: int | None = None
    # Optional normalized page box: x, y, width, height in the range 0..1.
    bbox: tuple[float, float, float, float] | None = None
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
    semantic_exact: bool
    detected_disposition: str | None
    disposition_ok: bool

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

    @property
    def semantic_exact(self) -> int:
        return sum(1 for i in self.items if i.semantic_exact)

    @property
    def disposition_correct(self) -> int:
        return sum(1 for i in self.items if i.disposition_ok)

    @property
    def disposition_total(self) -> int:
        return sum(1 for i in self.items if i.expected.disposition)

    def _items_for(self, disposition: str) -> list[ItemResult]:
        return [
            item
            for item in self.items
            if item.expected.disposition == disposition
        ]

    def disposition_summary(self, disposition: str) -> dict[str, int]:
        """Recognition and routing totals for one expected final state."""

        items = self._items_for(disposition)
        return {
            "expected": len(items),
            "matched": sum(item.verdict == "match" for item in items),
            "semantic_exact": sum(item.semantic_exact for item in items),
            "disposition_correct": sum(item.disposition_ok for item in items),
        }

    @property
    def unexpected_accepted(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.extras
            if item.get("disposition", "accepted") == "accepted"
        ]

    @property
    def unmatched_other(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.extras
            if item.get("disposition", "accepted") != "accepted"
        ]

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
                "semantic_exact": self.semantic_exact,
                "disposition_correct": self.disposition_correct,
                "disposition_total": self.disposition_total,
                "noise_detections": len(self.noise),
                "unexpected_detections": len(self.unexpected_accepted),
                "unmatched_review_or_other": len(self.unmatched_other),
                "by_expected_disposition": {
                    disposition: self.disposition_summary(disposition)
                    for disposition in ("accepted", "review", "other")
                },
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
                    "semantic_exact": i.semantic_exact,
                    "expected_disposition": i.expected.disposition,
                    "detected_disposition": i.detected_disposition,
                    "disposition_ok": i.disposition_ok,
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
        lines.append(
            f"{'':4} {'#':>2}  {'expected'.ljust(width)}  "
            f"{'detected'.ljust(width)}   %     type/state"
        )
        lines.append("-" * (width * 2 + 46))
        for item in self.items:
            detected = item.detected if item.detected is not None else "-"
            if item.type_ok is None:
                type_note = "-"
            else:
                type_note = "ok" if item.type_ok else f"{item.detected_type or '-'} != {item.expected.type}"
            state_note = (
                "ok"
                if item.disposition_ok
                else f"{item.detected_disposition or '-'} != {item.expected.disposition}"
            )
            lines.append(
                f"{mark[item.verdict]} {item.expected.id:>2}  "
                f"{item.expected.label.ljust(width)}  {detected.ljust(width)}  "
                f"{item.percent:5.1f}  {type_note}/{state_note}"
            )
        lines.append("-" * (width * 2 + 46))
        lines.append(
            f"matched {self.matched}/{self.total}"
            f"   partial {self.partial}   missed {self.missed}"
            f"   mean {self.mean_score * 100:.1f}%"
            f"   type {self.typed_correct}/{self.typed_total}"
            f"   exact {self.semantic_exact}/{self.total}"
            f"   state {self.disposition_correct}/{self.disposition_total}"
        )
        if self.noise:
            lines.append(f"known non-dimension text ignored: {len(self.noise)}")
        if self.unexpected_accepted:
            lines.append("unexpected accepted detections:")
            for extra in self.unexpected_accepted:
                lines.append(f"    {extra['text']!r} ({extra.get('type')})")
        if self.unmatched_other:
            lines.append(
                f"unmatched review/other detections retained: "
                f"{len(self.unmatched_other)}"
            )
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
        disposition = str(raw.get("disposition") or "accepted").strip().lower()
        if disposition not in {"accepted", "review", "other"}:
            raise ValueError(
                f"expected item {raw.get('id')} has invalid disposition "
                f"{disposition!r}"
            )
        page = raw.get("page")
        if page is not None:
            page = int(page)
            if page < 1:
                raise ValueError(
                    f"expected item {raw.get('id')} has invalid page {page}"
                )
        raw_bbox = raw.get("bbox")
        bbox = None
        if raw_bbox is not None:
            if not isinstance(raw_bbox, dict):
                raise ValueError(
                    f"expected item {raw.get('id')} bbox must be an object"
                )
            try:
                bbox = tuple(
                    float(raw_bbox[name])
                    for name in ("x", "y", "width", "height")
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"expected item {raw.get('id')} has invalid bbox"
                ) from exc
            if (
                bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] <= 0
                or bbox[3] <= 0
                or bbox[0] + bbox[2] > 1
                or bbox[1] + bbox[3] > 1
            ):
                raise ValueError(
                    f"expected item {raw.get('id')} bbox must be normalized "
                    "inside the page"
                )
        items.append(
            ExpectedItem(
                id=int(raw["id"]),
                accept=tuple(accept),
                type=raw.get("type"),
                disposition=disposition,
                page=page,
                bbox=bbox,
                note=raw.get("note"),
            )
        )
    data["expected"] = items
    return data


def _is_noise(text: str, patterns: Iterable[str]) -> bool:
    normalized = normalize(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def _candidate_disposition(region: dict[str, Any]) -> str:
    explicit = str(
        region.get("benchmark_disposition")
        or region.get("disposition")
        or ""
    ).strip().lower()
    if explicit in {"accepted", "review", "other"}:
        return explicit
    state = str(region.get("state") or "").strip().lower()
    return {
        "eligible": "accepted",
        "review": "review",
        "excluded": "other",
    }.get(state, "accepted")


def _normalized_region_bbox(
    region: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    bbox = region.get("bbox")
    width = region.get("_benchmark_page_width")
    height = region.get("_benchmark_page_height")
    if not isinstance(bbox, dict) or not isinstance(width, (int, float)) or not isinstance(
        height, (int, float)
    ):
        return None
    if width <= 0 or height <= 0:
        return None
    try:
        return (
            float(bbox["x"]) / float(width),
            float(bbox["y"]) / float(height),
            float(bbox["width"]) / float(width),
            float(bbox["height"]) / float(height),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _anchor_score(
    expected: ExpectedItem, region: dict[str, Any]
) -> float:
    """Spatial tiebreaker for repeated values; text remains the primary score."""

    if expected.bbox is None:
        return 0.0
    actual = _normalized_region_bbox(region)
    if actual is None:
        return 0.0
    expected_centre = (
        expected.bbox[0] + expected.bbox[2] / 2,
        expected.bbox[1] + expected.bbox[3] / 2,
    )
    actual_centre = (actual[0] + actual[2] / 2, actual[1] + actual[3] / 2)
    distance = math.dist(expected_centre, actual_centre)
    return max(0.0, 1.0 - distance / math.sqrt(2.0))


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
    pairs: list[tuple[float, float, int, int]] = []
    for ei, item in enumerate(expected):
        for ri, text in enumerate(texts):
            region_page = regions[ri].get("page")
            if item.page is not None and region_page is not None:
                try:
                    if int(region_page) != item.page:
                        continue
                except (TypeError, ValueError):
                    continue
            score = max(similarity(candidate, text) for candidate in item.accept)
            if score > 0:
                pairs.append((score, _anchor_score(item, regions[ri]), ei, ri))
    pairs.sort(key=lambda p: (-p[0], -p[1], p[2], p[3]))

    taken_expected: dict[int, tuple[float, int]] = {}
    taken_region: set[int] = set()
    for score, _spatial, ei, ri in pairs:
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
        detected_disposition = (
            _candidate_disposition(region) if region is not None else None
        )
        disposition_ok = bool(region) and (
            detected_disposition == item.disposition
        )
        semantic_exact = bool(region) and any(
            normalize(candidate) == normalize(str((region or {}).get("text") or ""))
            for candidate in item.accept
        )
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
                semantic_exact=semantic_exact,
                detected_disposition=detected_disposition,
                disposition_ok=disposition_ok,
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
        entry = {
            "text": texts[ri],
            "type": region.get("type"),
            "disposition": _candidate_disposition(region),
            "page": region.get("page"),
            "reason": region.get("reason") or region.get("review_reason"),
        }
        if entry["disposition"] == "accepted" and _is_noise(
            texts[ri], noise_patterns
        ):
            noise.append(entry)
        else:
            extras.append(entry)

    return Report(
        document=document,
        items=results,
        noise=noise,
        extras=extras,
        match_threshold=match_threshold,
        partial_threshold=partial_threshold,
        seconds=seconds,
    )


def requirement_failures(
    report: Report,
    requires: dict[str, Any],
    *,
    accounting: Any | None = None,
) -> list[str]:
    """Return human-readable benchmark gate failures without raising early."""

    failures: list[str] = []

    def minimum(key: str, actual: float, label: str) -> None:
        if key in requires and actual < float(requires[key]):
            failures.append(
                f"{label} {actual:g} is below required {requires[key]}"
            )

    minimum("matched", report.matched, "matched")
    minimum("mean_percent", report.mean_score * 100, "mean percent")
    minimum("type_correct", report.typed_correct, "type correct")
    minimum("semantic_exact", report.semantic_exact, "semantic exact")
    minimum(
        "disposition_correct",
        report.disposition_correct,
        "disposition correct",
    )
    accepted = report.disposition_summary("accepted")
    review = report.disposition_summary("review")
    other = report.disposition_summary("other")
    minimum("accepted_matched", accepted["matched"], "accepted matched")
    minimum(
        "accepted_semantic_exact",
        accepted["semantic_exact"],
        "accepted semantic exact",
    )
    minimum(
        "accepted_disposition_correct",
        accepted["disposition_correct"],
        "accepted disposition correct",
    )
    minimum("review_matched", review["matched"], "review matched")
    minimum(
        "review_disposition_correct",
        review["disposition_correct"],
        "review disposition correct",
    )
    minimum("other_matched", other["matched"], "other matched")
    minimum(
        "other_disposition_correct",
        other["disposition_correct"],
        "other disposition correct",
    )

    if "max_unexpected" in requires and len(report.unexpected_accepted) > int(
        requires["max_unexpected"]
    ):
        failures.append(
            f"unexpected accepted detections {len(report.unexpected_accepted)} "
            f"exceed maximum {requires['max_unexpected']}"
        )
    if "max_seconds" in requires and report.seconds is not None:
        if report.seconds > float(requires["max_seconds"]):
            failures.append(
                f"runtime {report.seconds:.1f}s exceeds maximum "
                f"{requires['max_seconds']}s"
            )

    wants_accounting = bool(requires.get("accounting_balanced"))
    max_accounting_errors = requires.get("max_accounting_errors")
    if wants_accounting or max_accounting_errors is not None:
        if accounting is None or not getattr(accounting, "available", False):
            failures.append("candidate accounting is unavailable")
        else:
            if wants_accounting and not getattr(accounting, "balanced", False):
                failures.append("candidate final-state counts are not balanced")
            issue_count = len(getattr(accounting, "issues", ()))
            if (
                max_accounting_errors is not None
                and issue_count > int(max_accounting_errors)
            ):
                failures.append(
                    f"candidate accounting issues {issue_count} exceed maximum "
                    f"{max_accounting_errors}"
                )
    return failures
