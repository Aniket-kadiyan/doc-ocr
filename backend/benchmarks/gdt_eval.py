"""
Scoring for the GD&T recognizer.

Follows the shape of ``benchmarks/scoring.py`` — a report object that knows how
to print itself and how to serialize — but measures different things, because
a symbol is not a dimension string. Detection, localisation and classification
are scored separately, since a system can be good at one and bad at another
and an aggregate would hide it.

What is deliberately *not* here
-------------------------------
There is no single headline accuracy number. The brief is explicit that rare
symbols must not be hidden behind common ones, and on a drawing the class
balance is extreme: a sheet may carry forty diameters and one cylindricity.
A micro-average would be a report on how well the recognizer reads Ø. So
per-class precision and recall are the primary output, the macro average is
reported beside the micro one, and the formatter prints every class including
the ones with a single instance.

Matching rule
-------------
A detection matches a ground-truth symbol when their boxes overlap by at least
``iou_threshold`` **and** the class agrees. Matching is one-to-one and greedy
from the best overlap down, so one detection cannot satisfy two truths — the
same rule ``scoring.py`` uses for dimension strings, and it matters in the same
place: where a drawing carries the same symbol twice, close together.

Unknown and ambiguous are counted, not discarded
------------------------------------------------
A recognizer that answers "unknown" to everything has perfect precision and is
useless. A recognizer that guesses confidently on everything has good recall
and is dangerous. Both numbers are reported, along with the rate at which the
system declines to answer, so the trade is visible rather than tuned away.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_IOU = 0.4


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class ClassScore:
    """Precision and recall for one symbol class."""

    symbol_id: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    #: Mean IoU over the matched instances.
    iou_sum: float = 0.0
    #: Matches whose class was right but which the system called ambiguous.
    ambiguous_hits: int = 0
    #: Misses where a detection sat on the symbol and listed the right class
    #: among its alternatives. These are false negatives under strict scoring
    #: and they should be — the system did not commit to the right answer —
    #: but they are a different failure from "saw nothing" or "said something
    #: unrelated", and the registry predicts most of them: they are the pairs
    #: drawn with one glyph, where shape cannot decide and only context can.
    recovered_in_alternatives: int = 0

    @property
    def support(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        return self.iou_sum / self.true_positives if self.true_positives else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "support": self.support,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "ambiguous_hits": self.ambiguous_hits,
            "recovered_in_alternatives": self.recovered_in_alternatives,
        }


@dataclass
class FrameScore:
    """Field-level accuracy for feature control frames."""

    frames_expected: int = 0
    frames_found: int = 0
    field_correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    field_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    #: Fields the system declined to read rather than guessing.
    field_declined: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    datum_sequences_correct: int = 0
    datum_sequences_total: int = 0
    composite_groups_correct: int = 0
    composite_groups_total: int = 0

    @property
    def detection_rate(self) -> float:
        return (
            self.frames_found / self.frames_expected if self.frames_expected else 0.0
        )

    def accuracy(self, name: str) -> float:
        total = self.field_total.get(name, 0)
        return self.field_correct.get(name, 0) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_expected": self.frames_expected,
            "frames_found": self.frames_found,
            "detection_rate": round(self.detection_rate, 4),
            "fields": {
                name: {
                    "correct": self.field_correct.get(name, 0),
                    "total": total,
                    "declined": self.field_declined.get(name, 0),
                    "accuracy": round(self.accuracy(name), 4),
                }
                for name, total in sorted(self.field_total.items())
            },
            "datum_sequence_accuracy": round(
                self.datum_sequences_correct / self.datum_sequences_total, 4
            )
            if self.datum_sequences_total
            else 0.0,
            "datum_sequences": [
                self.datum_sequences_correct,
                self.datum_sequences_total,
            ],
            "composite_grouping": [
                self.composite_groups_correct,
                self.composite_groups_total,
            ],
        }


@dataclass
class Report:
    """One scored run."""

    name: str
    condition: str = "clean"
    classes: dict[str, ClassScore] = field(default_factory=dict)
    frames: FrameScore = field(default_factory=FrameScore)
    unknown_detections: int = 0
    ambiguous_detections: int = 0
    total_detections: int = 0
    elapsed_ms: float = 0.0
    pages: int = 1
    #: Set when the scene's boxes cannot be trusted for localisation.
    localization_scored: bool = True
    notes: list[str] = field(default_factory=list)

    def cls(self, symbol_id: str) -> ClassScore:
        if symbol_id not in self.classes:
            self.classes[symbol_id] = ClassScore(symbol_id=symbol_id)
        return self.classes[symbol_id]

    # ── aggregates ───────────────────────────────────────────────────────────

    @property
    def micro(self) -> tuple[float, float, float]:
        """Precision, recall, F1 pooled over instances (favours common classes)."""
        tp = sum(c.true_positives for c in self.classes.values())
        fp = sum(c.false_positives for c in self.classes.values())
        fn = sum(c.false_negatives for c in self.classes.values())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    @property
    def macro(self) -> tuple[float, float, float]:
        """Precision, recall, F1 averaged over classes (every class counts once).

        This is the number to read. A drawing's class balance is extreme, and
        the micro average mostly reports how well the common symbols are read.
        """
        present = [c for c in self.classes.values() if c.support > 0]
        if not present:
            return 0.0, 0.0, 0.0
        p = sum(c.precision for c in present) / len(present)
        r = sum(c.recall for c in present) / len(present)
        f = sum(c.f1 for c in present) / len(present)
        return p, r, f

    @property
    def mean_iou(self) -> float:
        matched = [c for c in self.classes.values() if c.true_positives]
        if not matched:
            return 0.0
        return sum(c.iou_sum for c in matched) / sum(
            c.true_positives for c in matched
        )

    @property
    def recovered_in_alternatives(self) -> int:
        return sum(c.recovered_in_alternatives for c in self.classes.values())

    @property
    def macro_top_k_recall(self) -> float:
        """:attr:`top_k_recall`, averaged per class rather than per instance.

        Kept distinct because mixing the two is how a report ends up claiming
        a "top-k recall" lower than its own recall: the headline recall here
        is macro, and a micro top-k sitting beside it compares two different
        denominators. Both are published, each labelled.
        """
        present = [c for c in self.classes.values() if c.support > 0]
        if not present:
            return 0.0
        return sum(
            min(
                1.0,
                (c.true_positives + c.recovered_in_alternatives) / c.support,
            )
            for c in present
        ) / len(present)

    @property
    def top_k_recall(self) -> float:
        """Recall counting a miss as a hit when the right class was offered.

        Reported *beside* strict recall, never instead of it. A system that
        offers the right answer among several has done something useful for an
        operator who will arbitrate — and something useless for an automated
        consumer that needs one answer. Both readers are real.
        """
        tp = sum(c.true_positives for c in self.classes.values())
        fn = sum(c.false_negatives for c in self.classes.values())
        denom = tp + fn
        return (tp + self.recovered_in_alternatives) / denom if denom else 0.0

    @property
    def declined_rate(self) -> float:
        """How often the system refused to name something it detected."""
        if not self.total_detections:
            return 0.0
        return (self.unknown_detections + self.ambiguous_detections) / self.total_detections

    @property
    def false_positive_rate(self) -> float:
        fp = sum(c.false_positives for c in self.classes.values())
        return fp / self.total_detections if self.total_detections else 0.0

    def to_dict(self) -> dict[str, Any]:
        mp, mr, mf = self.macro
        ip, ir, if_ = self.micro
        return {
            "name": self.name,
            "condition": self.condition,
            "pages": self.pages,
            "aggregate": {
                "macro_precision": round(mp, 4),
                "macro_recall": round(mr, 4),
                "macro_f1": round(mf, 4),
                "micro_precision": round(ip, 4),
                "micro_recall": round(ir, 4),
                "micro_f1": round(if_, 4),
                "mean_iou": round(self.mean_iou, 4) if self.localization_scored else None,
                "declined_rate": round(self.declined_rate, 4),
                "false_positive_rate": round(self.false_positive_rate, 4),
                "micro_top_k_recall": round(self.top_k_recall, 4),
                "macro_top_k_recall": round(self.macro_top_k_recall, 4),
                "recovered_in_alternatives": self.recovered_in_alternatives,
                "classes_with_support": sum(
                    1 for c in self.classes.values() if c.support
                ),
            },
            "per_class": [
                c.to_dict()
                for c in sorted(
                    self.classes.values(), key=lambda c: (-c.support, c.symbol_id)
                )
            ],
            "frames": self.frames.to_dict(),
            "latency": {
                "total_ms": round(self.elapsed_ms, 2),
                "ms_per_page": round(self.elapsed_ms / max(self.pages, 1), 2),
            },
            "localization_scored": self.localization_scored,
            "notes": self.notes,
        }

    def format_text(self) -> str:
        mp, mr, mf = self.macro
        ip, ir, if_ = self.micro
        lines = [
            f"scene: {self.name}   condition: {self.condition}",
            "",
            f"{'class':<30}{'sup':>4}{'TP':>4}{'FP':>4}{'FN':>4}"
            f"{'prec':>7}{'rec':>7}{'F1':>7}{'IoU':>7}{'amb':>5}",
            "-" * 84,
        ]
        for score in sorted(
            self.classes.values(), key=lambda c: (-c.support, c.symbol_id)
        ):
            if not score.support and not score.false_positives:
                continue
            lines.append(
                f"{score.symbol_id:<30}{score.support:>4}{score.true_positives:>4}"
                f"{score.false_positives:>4}{score.false_negatives:>4}"
                f"{score.precision:>7.2f}{score.recall:>7.2f}{score.f1:>7.2f}"
                f"{score.mean_iou:>7.2f}{score.ambiguous_hits:>5}"
            )
        lines.append("-" * 84)
        lines.append(
            f"macro  P {mp:.3f}  R {mr:.3f}  F1 {mf:.3f}"
            f"      (every class counted once — read this one)"
        )
        lines.append(f"micro  P {ip:.3f}  R {ir:.3f}  F1 {if_:.3f}")
        if self.recovered_in_alternatives:
            lines.append(
                f"recall counting the right class offered among alternatives: "
                f"macro {self.macro_top_k_recall:.3f} / micro {self.top_k_recall:.3f}"
                f"  (+{self.recovered_in_alternatives} where shape alone cannot decide)"
            )
        if self.localization_scored:
            lines.append(f"mean IoU over matches: {self.mean_iou:.3f}")
        else:
            lines.append("localisation not scored for this condition")
        lines.append(
            f"declined (unknown or ambiguous): {self.declined_rate:.1%}"
            f"   false-positive rate: {self.false_positive_rate:.1%}"
        )
        if self.frames.frames_expected:
            frames = self.frames
            lines.append("")
            lines.append(
                f"frames found {frames.frames_found}/{frames.frames_expected}"
                f"  ({frames.detection_rate:.0%})"
            )
            for name, total in sorted(frames.field_total.items()):
                declined = frames.field_declined.get(name, 0)
                tail = f"  declined {declined}" if declined else ""
                lines.append(
                    f"    {name:<22}{frames.field_correct.get(name, 0):>3}/{total:<3}"
                    f"  {frames.accuracy(name):>6.1%}{tail}"
                )
            if frames.datum_sequences_total:
                lines.append(
                    f"    {'datum sequence':<22}"
                    f"{frames.datum_sequences_correct:>3}/"
                    f"{frames.datum_sequences_total:<3}"
                    f"  {frames.datum_sequences_correct / frames.datum_sequences_total:>6.1%}"
                )
        lines.append("")
        lines.append(
            f"latency {self.elapsed_ms:.0f}ms over {self.pages} page(s)"
            f"  ({self.elapsed_ms / max(self.pages, 1):.0f}ms/page)"
        )
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


# ── Scoring a scene ──────────────────────────────────────────────────────────


def score_symbols(
    report: Report,
    truth: Sequence[Any],
    detections: Sequence[Any],
    *,
    iou_threshold: float = DEFAULT_IOU,
) -> None:
    """Match detections to ground truth and fill in the per-class scores.

    ``truth`` items need ``symbol_id`` and ``bbox``; ``detections`` are
    ``gdt_detect.Detection``. Greedy one-to-one matching from the best overlap
    down, so one detection cannot be credited against two truths.
    """
    pairs: list[tuple[float, int, int]] = []
    for ti, want in enumerate(truth):
        for di, det in enumerate(detections):
            if det.symbol_id != want.symbol_id:
                continue
            overlap = iou(det.bbox, want.bbox)
            if overlap >= iou_threshold:
                pairs.append((overlap, ti, di))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    matched_truth: dict[int, tuple[float, int]] = {}
    matched_det: set[int] = set()
    for overlap, ti, di in pairs:
        if ti in matched_truth or di in matched_det:
            continue
        matched_truth[ti] = (overlap, di)
        matched_det.add(di)

    for ti, want in enumerate(truth):
        score = report.cls(want.symbol_id)
        hit = matched_truth.get(ti)
        if hit is None:
            score.false_negatives += 1
            # Did something land on this symbol and merely fail to commit?
            for det in detections:
                if iou(det.bbox, want.bbox) < iou_threshold:
                    continue
                alternatives = {
                    getattr(a, "symbol_id", None)
                    for a in getattr(det, "alternatives", ())
                }
                if want.symbol_id in alternatives:
                    score.recovered_in_alternatives += 1
                    break
            continue
        overlap, di = hit
        score.true_positives += 1
        score.iou_sum += overlap
        if getattr(detections[di], "status", "") == "ambiguous":
            score.ambiguous_hits += 1

    for di, det in enumerate(detections):
        if di in matched_det:
            continue
        if det.symbol_id is None:
            report.unknown_detections += 1
            continue
        report.cls(det.symbol_id).false_positives += 1


def score_frames(report: Report, truth_frames: Sequence[Any], found: Sequence[Any]) -> None:
    """Score frame detection, field extraction and datum precedence."""
    report.frames.frames_expected += len(truth_frames)
    if not truth_frames:
        return

    used: set[int] = set()
    for want in truth_frames:
        best_index, best_overlap = None, 0.0
        for index, frame in enumerate(found):
            if index in used:
                continue
            overlap = iou(frame.bbox, want.bbox)
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is None or best_overlap < DEFAULT_IOU:
            continue
        used.add(best_index)
        report.frames.frames_found += 1
        got = found[best_index]

        # characteristic symbol
        report.frames.field_total["symbol"] += 1
        got_symbol = got.symbol.symbol_id if got.symbol else None
        if got_symbol == want.symbol_id:
            report.frames.field_correct["symbol"] += 1
        elif got_symbol is None:
            report.frames.field_declined["symbol"] += 1

        # tolerance
        report.frames.field_total["tolerance"] += 1
        if got.tolerance.state == "present":
            if _close(got.tolerance.value, want.tolerance):
                report.frames.field_correct["tolerance"] += 1
        else:
            report.frames.field_declined["tolerance"] += 1

        # diametral zone
        report.frames.field_total["diameter_zone"] += 1
        if got.diameter_zone.state == "present":
            if bool(got.diameter_zone.value) == bool(want.diameter_zone):
                report.frames.field_correct["diameter_zone"] += 1
        else:
            report.frames.field_declined["diameter_zone"] += 1

        # material condition. "absent" is the correct answer when the truth
        # has none — declining is only declining when something is there.
        report.frames.field_total["material_condition"] += 1
        state = got.material_condition.state
        if state == "unreadable":
            report.frames.field_declined["material_condition"] += 1
        elif got.material_condition.value == want.material_condition:
            report.frames.field_correct["material_condition"] += 1

        # datum sequence, in precedence order
        report.frames.datum_sequences_total += 1
        got_labels = [d.label for d in sorted(got.datums, key=lambda d: d.precedence)]
        if got_labels == list(want.datums):
            report.frames.datum_sequences_correct += 1

    # composite grouping
    want_groups = {
        f.composite_id for f in truth_frames if f.composite_id
    }
    if want_groups:
        report.frames.composite_groups_total += len(want_groups)
        got_groups = {f.composite_id for f in found if f.composite_id}
        report.frames.composite_groups_correct += min(
            len(got_groups), len(want_groups)
        )


def _close(got: Any, want: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(got) - float(want)) <= tolerance
    except (TypeError, ValueError):
        return got == want


# ── Running the suite ────────────────────────────────────────────────────────

#: Conditions whose geometry is preserved exactly, so IoU is meaningful.
#: A skewed page's ground-truth boxes are axis-aligned and the symbol is not,
#: so scoring localisation there would measure the rotation, not the detector.
_LOCALIZATION_SAFE = frozenset({"clean", "blurred", "noisy", "low_res", "scan"})


def evaluate_scene(
    scene: Any,
    analyze: Any,
    *,
    iou_threshold: float = DEFAULT_IOU,
) -> Report:
    """Run ``analyze(image)`` over one scene and score the result.

    ``analyze`` is a callable taking a PIL image and returning a
    ``gdt_pipeline.PageResult`` — injected rather than imported so the scorer
    can be exercised against a stub.
    """
    report = Report(name=scene.name, condition=scene.condition)
    report.localization_scored = scene.condition in _LOCALIZATION_SAFE
    if not report.localization_scored:
        report.notes.append(
            f"localisation not scored under {scene.condition!r}: the page is "
            "rotated or resampled, so axis-aligned truth boxes no longer "
            "describe the symbols"
        )

    started = time.perf_counter()
    result = analyze(scene.image)
    report.elapsed_ms = (time.perf_counter() - started) * 1000

    detections = list(result.symbols)
    report.total_detections = len(detections)
    report.ambiguous_detections = sum(
        1 for d in detections if d.status == "ambiguous"
    )
    report.unknown_detections += sum(1 for d in detections if d.symbol_id is None)

    score_symbols(report, scene.symbols, detections, iou_threshold=iou_threshold)
    score_frames(report, scene.frames, result.frames)
    return report


def merge(reports: Iterable[Report], *, name: str = "all") -> Report:
    """Pool several reports into one, keeping per-class detail."""
    out = Report(name=name, condition="mixed")
    for other in reports:
        out.pages += other.pages
        out.elapsed_ms += other.elapsed_ms
        out.total_detections += other.total_detections
        out.unknown_detections += other.unknown_detections
        out.ambiguous_detections += other.ambiguous_detections
        out.localization_scored = out.localization_scored and other.localization_scored
        for symbol_id, score in other.classes.items():
            target = out.cls(symbol_id)
            target.true_positives += score.true_positives
            target.false_positives += score.false_positives
            target.false_negatives += score.false_negatives
            target.iou_sum += score.iou_sum
            target.ambiguous_hits += score.ambiguous_hits
            target.recovered_in_alternatives += score.recovered_in_alternatives
        out.frames.frames_expected += other.frames.frames_expected
        out.frames.frames_found += other.frames.frames_found
        for key, value in other.frames.field_correct.items():
            out.frames.field_correct[key] += value
        for key, value in other.frames.field_total.items():
            out.frames.field_total[key] += value
        for key, value in other.frames.field_declined.items():
            out.frames.field_declined[key] += value
        out.frames.datum_sequences_correct += other.frames.datum_sequences_correct
        out.frames.datum_sequences_total += other.frames.datum_sequences_total
        out.frames.composite_groups_correct += other.frames.composite_groups_correct
        out.frames.composite_groups_total += other.frames.composite_groups_total
    out.pages = max(out.pages - 1, 1)
    return out


def save(reports: Sequence[Report], path: str | Path) -> Path:
    """Write a run's reports as JSON, for tracking a baseline over time."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "reports": [r.to_dict() for r in reports],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target
