"""Human-readable diagnostics for long-running QTracX OCR scans.

The normal backend log remains newline-delimited JSON for support tooling.
This module adds one plain-text report per auto-balloon scan so an operator can
see where time was spent without exposing drawing content. Reports contain only
operational stage names, counts, timings, resource observations and job IDs.
"""

from __future__ import annotations

import os
import platform
import re
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter, time
from typing import Any, Iterator

from app_logging import get_logger

try:  # Optional at import time; declared in the production requirements.
    import psutil
except ImportError:  # pragma: no cover - bare developer interpreters only
    psutil = None  # type: ignore[assignment]


RUNS_DIR_NAME = "runs"
INDEX_FILE_NAME = "index.log"
DEFAULT_KEEP_REPORTS = 300

logger = get_logger("run_report")

_CURRENT: ContextVar["RunReport | None"] = ContextVar(
    "qtracx_run_report",
    default=None,
)
_ACTIVE: dict[str, "RunReport"] = {}
_RECENT: deque["RunReport"] = deque(maxlen=50)
_REGISTRY_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_SETTINGS: dict[str, Any] = {
    "directory": None,
    "keep": DEFAULT_KEEP_REPORTS,
}


STEP_DESCRIPTIONS: dict[str, str] = {
    "waiting": "Waiting for a previous scan belonging to this user to stop.",
    "preparing": "Decoding the page and preparing the requested scan.",
    "text-layer": "Reading exact values available in the PDF text layer.",
    "layout": "Finding tables and the drawing areas that should be scanned.",
    "detecting": "Finding printed text candidates in the drawing.",
    "grouping": "Combining detector boxes into engineering callouts.",
    "recognizing": "Reading candidate callouts with the OCR models.",
    "recovering": "Running second-chance reads for uncertain candidates.",
    "context": "Checking surrounding notes, tables and titles.",
    "filtering": "Applying page rules to the detected values.",
    "rereading": "Performing the authoritative read of final values.",
    "splitting": "Separating stacked values that were detected as one object.",
    "slanted": "Reading angled callouts from rotated local views.",
    "notes": "Reading drawing note regions.",
    "gdt": "Reading GD&T feature-control frames.",
    "deduplicating": "Removing duplicate values before publication.",
    "finalizing": "Preparing the balloon and review results.",
    "cancelling": "Stopping after the current OCR operation.",
    "cancelled": "The scan was cancelled without publishing results.",
    "complete": "The completed scan result was published.",
    "failed": "The scan stopped because an operation failed.",
}


def configure_run_reports(
    log_dir: Path,
    *,
    keep: int = DEFAULT_KEEP_REPORTS,
) -> Path:
    """Create and select the report directory for this backend process."""

    directory = log_dir / RUNS_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    _SETTINGS["directory"] = directory
    _SETTINGS["keep"] = max(1, int(keep))
    return directory


def disable_run_reports() -> None:
    """Disable report writes; used by isolated unit tests."""

    _SETTINGS["directory"] = None


def report_status() -> dict[str, Any]:
    directory = _SETTINGS["directory"]
    return {
        "enabled": directory is not None,
        "directory": str(directory) if directory is not None else None,
        "retained_reports": int(_SETTINGS["keep"]),
    }


def current_run() -> "RunReport | None":
    return _CURRENT.get()


@contextmanager
def activate(report: "RunReport | None") -> Iterator[None]:
    token = _CURRENT.set(report)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def record_model_call(
    operation: str,
    *,
    waited_seconds: float,
    active_seconds: float,
) -> None:
    """Charge one inference-gate lease to the active scan report."""

    report = _CURRENT.get()
    if report is not None:
        report.model_call(
            operation,
            max(0.0, waited_seconds),
            max(0.0, active_seconds),
        )


@dataclass(frozen=True)
class _ResourceSample:
    cpu_seconds: float
    rss_bytes: int | None
    available_bytes: int | None
    swap_bytes: int | None
    threads: int | None


_PROCESS = psutil.Process() if psutil is not None else None


def _resources() -> _ResourceSample:
    if _PROCESS is None:
        return _ResourceSample(0.0, None, None, None, None)
    try:
        cpu = _PROCESS.cpu_times()
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return _ResourceSample(
            cpu_seconds=float(cpu.user + cpu.system),
            rss_bytes=int(_PROCESS.memory_info().rss),
            available_bytes=int(memory.available),
            swap_bytes=int(swap.used),
            threads=int(_PROCESS.num_threads()),
        )
    except Exception:  # noqa: BLE001 - diagnostics must never break OCR
        return _ResourceSample(0.0, None, None, None, None)


@dataclass
class _Operation:
    count: int = 0
    seconds: float = 0.0
    slowest: float = 0.0


@dataclass
class _Stage:
    index: int
    name: str
    description: str
    started: float
    start_resources: _ResourceSample
    ended: float | None = None
    end_resources: _ResourceSample | None = None
    first_message: str = ""
    last_message: str = ""
    completed: int = 0
    total: int = 0
    candidates: int = 0
    model_calls: int = 0
    model_seconds: float = 0.0
    model_wait_seconds: float = 0.0
    max_rss_bytes: int | None = None
    min_available_bytes: int | None = None
    max_swap_bytes: int | None = None
    operations: dict[str, _Operation] = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return max(0.0, (self.ended or perf_counter()) - self.started)

    @property
    def cpu_seconds(self) -> float:
        if self.end_resources is None:
            return 0.0
        return max(
            0.0,
            self.end_resources.cpu_seconds - self.start_resources.cpu_seconds,
        )

    def observe(self, sample: _ResourceSample) -> None:
        if sample.rss_bytes is not None:
            self.max_rss_bytes = max(self.max_rss_bytes or 0, sample.rss_bytes)
        if sample.available_bytes is not None:
            self.min_available_bytes = (
                sample.available_bytes
                if self.min_available_bytes is None
                else min(self.min_available_bytes, sample.available_bytes)
            )
        if sample.swap_bytes is not None:
            self.max_swap_bytes = max(self.max_swap_bytes or 0, sample.swap_bytes)


_DIGITS = re.compile(r"\d+")
_UNSAFE_TEXT = re.compile(r"[\r\n\t]+")


def _safe_text(value: Any, *, limit: int = 120) -> str:
    return _UNSAFE_TEXT.sub(" ", str(value)).strip()[:limit]


def _operation_key(label: str) -> str:
    """Collapse numbered work items, such as Object 17 and Object 18."""

    return _DIGITS.sub("#", _safe_text(label, limit=80))


class RunReport:
    """Collect one scan's timings and write a readable report at completion."""

    def __init__(
        self,
        kind: str,
        run_id: str,
        title: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.kind = _safe_text(kind, limit=40)
        self.run_id = _safe_text(run_id, limit=80)
        self.title = _safe_text(title, limit=160)
        self.details = {
            _safe_text(key, limit=50): _safe_text(value, limit=160)
            for key, value in (details or {}).items()
        }
        self.outcome: dict[str, Any] = {}
        self.status = "running"
        self.error = ""
        self.wall_started = time()
        self.started = perf_counter()
        self.ended: float | None = None
        self.stages: list[_Stage] = []
        self.model_calls = 0
        self.model_seconds = 0.0
        self.model_wait_seconds = 0.0
        self.path: Path | None = None
        self._lock = threading.Lock()
        self._stage: _Stage | None = None
        self._operation_label = ""
        self._operation_started = self.started
        self._start_resources: _ResourceSample | None = None
        self._end_resources: _ResourceSample | None = None

    def start(self) -> "RunReport":
        sample = _resources()
        with self._lock:
            self._start_resources = sample
            self._end_resources = sample
        with _REGISTRY_LOCK:
            _ACTIVE[self.run_id] = self
        return self

    def progress(
        self,
        *,
        stage: str,
        message: str = "",
        operation_label: str = "",
        completed: int = 0,
        total: int = 0,
        candidate_count: int = 0,
    ) -> None:
        stage = _safe_text(stage, limit=40) or "working"
        now = perf_counter()
        sample = _resources()
        with self._lock:
            if self.ended is not None:
                return
            if self._stage is None or self._stage.name != stage:
                self._close_stage(now, sample)
                self._stage = _Stage(
                    index=len(self.stages) + 1,
                    name=stage,
                    description=STEP_DESCRIPTIONS.get(stage, ""),
                    started=now,
                    start_resources=sample,
                )
                self._stage.observe(sample)
                self.stages.append(self._stage)
                self._operation_label = ""
                self._operation_started = now
            self._switch_operation(operation_label, now)
            current = self._stage
            assert current is not None
            current.observe(sample)
            safe_message = _safe_text(message, limit=180)
            if safe_message:
                current.first_message = current.first_message or safe_message
                current.last_message = safe_message
            current.completed = max(0, int(completed))
            current.total = max(0, int(total))
            current.candidates = max(
                current.candidates,
                max(0, int(candidate_count)),
            )
            self._end_resources = sample

    def model_call(self, operation: str, waited: float, active: float) -> None:
        sample = _resources()
        with self._lock:
            if self.ended is not None:
                return
            self.model_calls += 1
            self.model_seconds += active
            self.model_wait_seconds += waited
            if self._stage is not None:
                self._stage.model_calls += 1
                self._stage.model_seconds += active
                self._stage.model_wait_seconds += waited
                self._stage.observe(sample)
                key = f"OCR model: {_operation_key(operation) or 'inference'}"
                item = self._stage.operations.setdefault(key, _Operation())
                item.count += 1
                item.seconds += active
                item.slowest = max(item.slowest, active)
            self._end_resources = sample

    def finish(
        self,
        status: str,
        *,
        error: str = "",
        outcome: dict[str, Any] | None = None,
    ) -> Path | None:
        now = perf_counter()
        sample = _resources()
        with self._lock:
            if self.ended is not None:
                return self.path
            self._close_stage(now, sample)
            self.ended = now
            self._end_resources = sample
            self.status = _safe_text(status, limit=30) or "unknown"
            self.error = _safe_text(error, limit=120)
            self.outcome = {
                _safe_text(key, limit=50): _safe_text(value, limit=80)
                for key, value in (outcome or {}).items()
            }
        with _REGISTRY_LOCK:
            _ACTIVE.pop(self.run_id, None)
            related = [
                other
                for other in [*_ACTIVE.values(), *_RECENT]
                if other is not self
            ]
            _RECENT.append(self)
        try:
            return self._write(related)
        except Exception:  # noqa: BLE001 - report failure must not fail a scan
            logger.exception(
                "Could not write scan run report",
                extra={"event": "run_report_failed"},
            )
            return None

    @property
    def total_seconds(self) -> float:
        return max(0.0, (self.ended or perf_counter()) - self.started)

    def _switch_operation(self, label: str, now: float) -> None:
        safe_label = _safe_text(label, limit=80)
        if safe_label == self._operation_label:
            return
        self._flush_operation(now)
        self._operation_label = safe_label
        self._operation_started = now

    def _flush_operation(self, now: float) -> None:
        if not self._operation_label or self._stage is None:
            return
        key = _operation_key(self._operation_label)
        if not key:
            return
        elapsed = max(0.0, now - self._operation_started)
        item = self._stage.operations.setdefault(key, _Operation())
        item.count += 1
        item.seconds += elapsed
        item.slowest = max(item.slowest, elapsed)

    def _close_stage(self, now: float, sample: _ResourceSample) -> None:
        if self._stage is None:
            return
        self._flush_operation(now)
        self._stage.ended = now
        self._stage.end_resources = sample
        self._stage.observe(sample)
        self._stage = None
        self._operation_label = ""
        self._operation_started = now

    def _write(self, related: list["RunReport"]) -> Path | None:
        directory = _SETTINGS["directory"]
        if directory is None:
            return None
        timestamp = datetime.fromtimestamp(self.wall_started).strftime(
            "%Y%m%d-%H%M%S"
        )
        safe_id = re.sub(r"[^A-Za-z0-9-]", "", self.run_id)[:8] or "run"
        safe_kind = re.sub(r"[^A-Za-z0-9-]", "-", self.kind)[:40]
        path = directory / f"{timestamp}_{safe_kind}_{safe_id}.log"
        with _WRITE_LOCK:
            path.write_text(self.render(related), encoding="utf-8")
            with (directory / INDEX_FILE_NAME).open("a", encoding="utf-8") as index:
                index.write(self._index_line(path) + "\n")
            _prune(directory, int(_SETTINGS["keep"]))
        self.path = path
        logger.info(
            "Scan run report written",
            extra={
                "event": "run_report_written",
                "duration_ms": round(self.total_seconds * 1000, 2),
                "details": {
                    "report": path.name,
                    "status": self.status,
                },
            },
        )
        return path

    def _index_line(self, path: Path) -> str:
        slowest = max(self.stages, key=lambda item: item.seconds, default=None)
        slow = (
            f"slowest={slowest.name} {_duration(slowest.seconds)}"
            if slowest is not None
            else ""
        )
        started = datetime.fromtimestamp(self.wall_started).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return (
            f"{started}  {self.status.upper():<10} "
            f"{_duration(self.total_seconds):>10}  {self.title}  "
            f"{slow}  -> {path.name}"
        )

    def render(self, related: list["RunReport"] | None = None) -> str:
        with self._lock:
            return self._render(related or [])

    def _render(self, related: list["RunReport"]) -> str:
        rule = "=" * 96
        started = datetime.fromtimestamp(self.wall_started)
        finished = datetime.fromtimestamp(
            self.wall_started + self.total_seconds
        )
        lines = [
            rule,
            f"QTracX scan run report: {self.title}",
            rule,
            f"{'Job id':<20}: {self.run_id}",
            f"{'Status':<20}: {self.status.upper()}"
            + (f" ({self.error})" if self.error else ""),
            f"{'Started':<20}: {started:%Y-%m-%d %H:%M:%S}",
            f"{'Finished':<20}: {finished:%Y-%m-%d %H:%M:%S}",
            f"{'Processing time':<20}: {_duration(self.total_seconds)}",
            f"{'Host':<20}: {platform.system()} {platform.machine()}, "
            f"{os.cpu_count() or 0} logical CPU cores, "
            f"Python {platform.python_version()}",
        ]
        for key, value in self.details.items():
            lines.append(f"{key.replace('_', ' ').capitalize():<20}: {value}")
        if self.outcome:
            result = ", ".join(
                f"{key.replace('_', ' ')}={value}"
                for key, value in self.outcome.items()
            )
            lines.append(f"{'Result':<20}: {result}")

        lines.extend(["", "SUMMARY", "-" * 96])
        slowest = max(self.stages, key=lambda item: item.seconds, default=None)
        if slowest is not None:
            lines.append(
                f"Slowest stage: {slowest.name} — {_duration(slowest.seconds)} "
                f"({slowest.seconds / max(self.total_seconds, 1e-9) * 100:.1f}% of processing)"
            )
        lines.append(
            "OCR model use: "
            f"{self.model_calls} calls, {_duration(self.model_seconds)} inside models, "
            f"{_duration(self.model_wait_seconds)} waiting for model capacity"
        )
        start_sample = self._start_resources
        end_sample = self._end_resources
        if start_sample is not None and end_sample is not None:
            cpu = max(0.0, end_sample.cpu_seconds - start_sample.cpu_seconds)
            lines.append(
                "Backend resources observed during this window: "
                f"{_duration(cpu)} CPU time, "
                f"RAM {_size(start_sample.rss_bytes)} → {_size(end_sample.rss_bytes)}, "
                f"threads {start_sample.threads or 'n/a'} → {end_sample.threads or 'n/a'}"
            )
        overlaps = self._overlaps(related)
        if overlaps:
            lines.append(
                "Ran alongside: "
                + "; ".join(
                    f"{title} for {_duration(seconds)}"
                    for title, seconds in overlaps[:5]
                )
            )

        if self.stages:
            lines.extend(["", "STAGES", "-" * 96])
            lines.append(
                f"  {'#':>2}  {'stage':<16} {'time':>11} {'share':>7} "
                f"{'work':>13} {'candidates':>11} {'model':>10} {'wait':>10}"
            )
            for stage in self.stages:
                work = (
                    f"{stage.completed}/{stage.total}"
                    if stage.total
                    else "-"
                )
                lines.append(
                    f"  {stage.index:>2}  {stage.name[:16]:<16} "
                    f"{_duration(stage.seconds):>11} "
                    f"{stage.seconds / max(self.total_seconds, 1e-9) * 100:6.1f}% "
                    f"{work:>13} {stage.candidates:>11} "
                    f"{_duration(stage.model_seconds) if stage.model_calls else '-':>10} "
                    f"{_duration(stage.model_wait_seconds) if stage.model_calls else '-':>10}"
                )

            lines.extend(["", "STAGE DETAILS", "-" * 96])
            for stage in self.stages:
                lines.append(
                    f"[{stage.index}] {stage.name} — {_duration(stage.seconds)}"
                )
                if stage.description:
                    lines.append(f"    Purpose      : {stage.description}")
                if stage.first_message:
                    lines.append(f"    Began with   : {stage.first_message}")
                if (
                    stage.last_message
                    and stage.last_message != stage.first_message
                ):
                    lines.append(f"    Ended with   : {stage.last_message}")
                lines.append(
                    "    Resources    : "
                    f"backend CPU {_duration(stage.cpu_seconds)}, "
                    f"highest observed RAM {_size(stage.max_rss_bytes)}, "
                    f"lowest free RAM {_size(stage.min_available_bytes)}"
                )
                operations = sorted(
                    stage.operations.items(),
                    key=lambda item: -item[1].seconds,
                )
                if operations:
                    lines.append("    Operations (slowest first):")
                    for name, operation in operations[:10]:
                        average = operation.seconds / max(operation.count, 1)
                        lines.append(
                            f"      {name[:48]:<48} {operation.count:>4} x  "
                            f"total {_duration(operation.seconds):>10}  "
                            f"avg {_duration(average):>9}  "
                            f"max {_duration(operation.slowest):>9}"
                        )
                lines.append("")

        lines.extend(
            [
                "PRIVACY",
                "-" * 96,
                "This report intentionally excludes uploaded files, filenames, drawing text, "
                "recognized values, coordinates, user identities, credentials and raw exceptions.",
                "CPU and RAM are backend-process observations. When scans overlap, those resource "
                "figures include shared backend activity; stage and inference timings remain per job.",
                "",
            ]
        )
        return "\n".join(lines)

    def _overlaps(
        self,
        related: list["RunReport"],
    ) -> list[tuple[str, float]]:
        start = self.started
        end = self.ended or perf_counter()
        overlaps: list[tuple[str, float]] = []
        for other in related:
            other_end = other.ended or perf_counter()
            seconds = min(end, other_end) - max(start, other.started)
            if seconds >= 0.5:
                overlaps.append((other.title, seconds))
        return sorted(overlaps, key=lambda item: -item[1])


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remaining = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {remaining:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def _size(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GB"
    return f"{value / 1024**2:.0f} MB"


def _prune(directory: Path, keep: int) -> None:
    reports = sorted(
        (
            path
            for path in directory.glob("*.log")
            if path.name != INDEX_FILE_NAME
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    for stale in reports[:-keep]:
        try:
            stale.unlink()
        except OSError:
            continue
