"""One plain-text report per pipeline run, written for people to read.

``qtracx-backend.log`` is one JSON event per line: right for tooling, hard to
answer "why did that scan take 17 minutes?" from. Each scan job and each OCR
request also gets a file under ``<log dir>/runs/`` that says, in order, which
step ran, how long it took, what it worked on, what it cost in CPU and memory
(the scan worker processes included), how long it spent inside the OCR models
or waiting for another run to release them, and what else was running beside
it. ``runs/index.log`` holds one line per run.

Like the JSON log this records operational facts only: step names, counts,
timings and resources. The progress messages it quotes are the pipeline's own
count-only wording; recognized drawing text never reaches a report.
"""

from __future__ import annotations

import os
import platform
import re
import sys
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

try:  # Present in the OCR venv; without it the report omits worker and RAM figures.
    import psutil
except ImportError:  # pragma: no cover - exercised only on a bare interpreter
    psutil = None  # type: ignore[assignment]

RUNS_DIR_NAME = "runs"
INDEX_FILE_NAME = "index.log"
DEFAULT_KEEP_REPORTS = 300
SAMPLE_INTERVAL_SECONDS = 0.5
logger = get_logger("run_report")

_CURRENT: ContextVar["RunReport | None"] = ContextVar("qtracx_run_report", default=None)
_ACTIVE: dict[str, "RunReport"] = {}
_RECENT: deque["RunReport"] = deque(maxlen=50)
_REGISTRY_LOCK = threading.Lock()
_settings: dict[str, Any] = {"directory": None, "keep": DEFAULT_KEEP_REPORTS}

# What each scan stage does, in the words a reader of the report needs.
STEP_DESCRIPTIONS: dict[str, str] = {
    "queued": "Waiting for the scan worker thread to become free.",
    "waiting": "Waiting for a previous scan to stop before this one starts.",
    "preparing": "Planning the work: panels, table masks and detector passes.",
    "layout": "Finding tables, title block and processing panels on the page "
              "(cached per page, so a re-scan of the same page skips it).",
    "detecting": "Running the text detector over each panel and rotation to "
                 "find every place text is printed.",
    "grouping": "Merging detector boxes into candidate callouts (bridge "
                "filtering, spatial grouping, fragment and overlap merging).",
    "recognizing": "Reading each candidate with the OCR recognizer.",
    "recovering": "Second-chance reads for candidates the first read was "
                  "unsure of.",
    "context": "Reading surrounding text for values that might be excluded "
               "(titles, tables, notes).",
    "filtering": "Applying page rules that decide which reads are balloon "
                 "values and which are excluded.",
    "rereading": "Authoritative re-read of every final value (Draw Value OCR).",
    "slanted": "Reading slanted callouts from rotated neighbourhood views.",
    "notes": "Reading the notes paragraphs.",
    "gdt": "Reading GD&T feature control frames.",
    "finalizing": "Building the balloon list returned to the browser.",
    "cancelling": "Stopping after the current OCR operation.",
    "request": "Handling the HTTP request.",
}


def configure_run_reports(log_dir: Path, *, keep: int = DEFAULT_KEEP_REPORTS) -> Path:
    directory = log_dir / RUNS_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    _settings["directory"] = directory
    _settings["keep"] = max(1, keep)
    return directory


def current_run() -> "RunReport | None":
    return _CURRENT.get()


@contextmanager
def activate(report: "RunReport | None") -> Iterator[None]:
    token = _CURRENT.set(report)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@contextmanager
def run_step(name: str, description: str = "") -> Iterator[None]:
    """Mark a step of the current run; a no-op when no run is active."""

    report = _CURRENT.get()
    if report is None:
        yield
        return
    report.begin_step(name, description)
    try:
        yield
    finally:
        report.end_step()


def note(**values: Any) -> None:
    """Add input facts (sizes, options) to the current run's header."""
    report = _CURRENT.get()
    if report is not None:
        report.note(**values)


def count(name: str, amount: int = 1) -> None:
    report = _CURRENT.get()
    if report is not None:
        report.count(name, amount)


class TimedLock:
    """``threading.Lock`` that charges wait and hold time to the current run.

    The OCR models take turns behind one lock, so time a run spends waiting
    here is time another run (or request) was using the models.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        started = perf_counter()
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._local.acquired_at = perf_counter()
            self._local.waited = self._local.acquired_at - started
        return acquired

    def release(self) -> None:
        acquired_at = getattr(self._local, "acquired_at", None)
        waited = getattr(self._local, "waited", 0.0)
        self._lock.release()
        report = _CURRENT.get()
        if report is not None and acquired_at is not None:
            report.model_call(waited, perf_counter() - acquired_at)

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------


@dataclass
class _Sample:
    at: float
    cpu_seconds: float  # cumulative: this process plus its worker processes
    main_rss: int
    workers_rss: int
    workers: int
    available: int | None
    swap_used: int | None
    threads: int


class _ResourceReader:
    """Cumulative CPU across the backend and its children, plus memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._child_cpu: dict[int, float] = {}
        self._process = psutil.Process() if psutil is not None else None

    def read(self) -> _Sample:
        with self._lock:
            return self._read_locked()

    def _read_locked(self) -> _Sample:
        now = perf_counter()
        if self._process is None:
            return self._read_stdlib(now)
        try:
            times = self._process.cpu_times()
            main_cpu = times.user + times.system
            main_rss = self._process.memory_info().rss
            threads = self._process.num_threads()
            workers_rss = 0
            workers = 0
            for child in self._process.children(recursive=True):
                try:
                    child_times = child.cpu_times()
                    self._child_cpu[child.pid] = child_times.user + child_times.system
                    workers_rss += child.memory_info().rss
                    workers += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return _Sample(
                now, main_cpu + sum(self._child_cpu.values()), main_rss,
                workers_rss, workers, memory.available, swap.used, threads,
            )
        except Exception:  # noqa: BLE001 - a report must never fail a scan
            return self._read_stdlib(now)

    @staticmethod
    def _read_stdlib(now: float) -> _Sample:
        cpu = 0.0
        rss = 0
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu = usage.ru_utime + usage.ru_stime
            # ru_maxrss is bytes on macOS and kilobytes on Linux.
            rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        except Exception:  # noqa: BLE001 - Windows has no resource module
            pass
        return _Sample(now, cpu, rss, 0, 0, None, None, threading.active_count())


_READER = _ResourceReader()


class _Sampler:
    """One background thread feeding samples to every active run."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def ensure_running(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="run-report-sampler", daemon=True,
            )
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(SAMPLE_INTERVAL_SECONDS):
            with _REGISTRY_LOCK:
                active = list(_ACTIVE.values())
            if not active:
                with self._lock:
                    self._thread = None
                return
            sample = _READER.read()
            for report in active:
                report.add_sample(sample)


_SAMPLER = _Sampler()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class _Operation:
    count: int = 0
    seconds: float = 0.0
    slowest: float = 0.0


@dataclass
class _Step:
    index: int
    name: str
    description: str
    started: float
    start_sample: _Sample
    ended: float | None = None
    end_sample: _Sample | None = None
    first_message: str = ""
    last_message: str = ""
    completed: int = 0
    total: int = 0
    candidates: int = 0
    model_calls: int = 0
    model_seconds: float = 0.0
    model_wait_seconds: float = 0.0
    peak_main_rss: int = 0
    peak_workers_rss: int = 0
    peak_workers: int = 0
    min_available: int | None = None
    max_swap: int | None = None
    operations: dict[str, _Operation] = field(default_factory=dict)

    def take(self, sample: _Sample) -> None:
        self.peak_main_rss = max(self.peak_main_rss, sample.main_rss)
        self.peak_workers_rss = max(self.peak_workers_rss, sample.workers_rss)
        self.peak_workers = max(self.peak_workers, sample.workers)
        if sample.available is not None:
            self.min_available = (
                sample.available if self.min_available is None
                else min(self.min_available, sample.available)
            )
        if sample.swap_used is not None:
            self.max_swap = (
                sample.swap_used if self.max_swap is None
                else max(self.max_swap, sample.swap_used)
            )

    @property
    def seconds(self) -> float:
        return max(0.0, (self.ended or perf_counter()) - self.started)

    @property
    def cpu_seconds(self) -> float:
        if self.end_sample is None:
            return 0.0
        return max(0.0, self.end_sample.cpu_seconds - self.start_sample.cpu_seconds)


_DIGITS = re.compile(r"\d+")


def _operation_key(label: str) -> str:
    """'Object 17' and 'Object 18' are one kind of operation."""
    return _DIGITS.sub("#", label.strip())[:80]


class RunReport:
    """Collects one run's steps and resources, then writes its report."""

    def __init__(self, kind: str, run_id: str, title: str, details: dict[str, Any] | None = None) -> None:
        self.kind = kind
        self.run_id = run_id
        self.title = title
        self.details: dict[str, Any] = dict(details or {})
        self.counters: dict[str, int] = {}
        self.steps: list[_Step] = []
        self.status = "running"
        self.error = ""
        self.outcome: dict[str, Any] = {}
        self.wall_started = time()
        self.started = perf_counter()
        self.ended: float | None = None
        self._lock = threading.Lock()
        self._step: _Step | None = None
        self._op_label = ""
        self._op_started = self.started
        self._first_sample: _Sample | None = None
        self._last_sample: _Sample | None = None
        self.path: Path | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "RunReport":
        sample = _READER.read()
        with self._lock:
            self._first_sample = sample
            self._last_sample = sample
        with _REGISTRY_LOCK:
            _ACTIVE[self.run_id] = self
        _SAMPLER.ensure_running()
        return self

    def finish(self, status: str, *, error: str = "", outcome: dict[str, Any] | None = None) -> Path | None:
        sample = _READER.read()
        with self._lock:
            if self.ended is not None:
                return self.path
            self._close_step_locked(sample)
            self._last_sample = sample
            self.ended = perf_counter()
            self.status = status
            self.error = error
            self.outcome.update(outcome or {})
        with _REGISTRY_LOCK:
            _ACTIVE.pop(self.run_id, None)
            others = [r for r in [*_ACTIVE.values(), *_RECENT] if r is not self]
            _RECENT.append(self)
        try:
            return self._write(others)
        except Exception:  # noqa: BLE001 - a report must never fail a scan
            logger.exception("Could not write run report", extra={"event": "run_report_failed"})
            return None

    # -- recording ---------------------------------------------------------

    def begin_step(self, name: str, description: str = "") -> None:
        sample = _READER.read()
        with self._lock:
            self._close_step_locked(sample)
            self._open_step_locked(name, description, sample)

    def end_step(self) -> None:
        sample = _READER.read()
        with self._lock:
            self._close_step_locked(sample)

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
        """Feed one scan progress event; a new stage starts a new step."""

        stage_changed = self._step is None or self._step.name != stage
        sample = _READER.read() if stage_changed else None
        with self._lock:
            if self.ended is not None:
                return
            if sample is not None:
                self._close_step_locked(sample)
                self._open_step_locked(stage, "", sample)
            step = self._step
            assert step is not None
            self._switch_operation_locked(operation_label)
            if message:
                step.first_message = step.first_message or message
                step.last_message = message
            step.completed, step.total = completed, total
            step.candidates = max(step.candidates, candidate_count)

    def model_call(self, waited: float, held: float) -> None:
        with self._lock:
            self.counters["OCR model calls"] = self.counters.get("OCR model calls", 0) + 1
            step = self._step
            if step is not None:
                step.model_calls += 1
                step.model_seconds += held
                step.model_wait_seconds += waited
            else:
                self.details["_unstepped_model_seconds"] = (
                    float(self.details.get("_unstepped_model_seconds", 0.0)) + held
                )
                self.details["_unstepped_model_wait"] = (
                    float(self.details.get("_unstepped_model_wait", 0.0)) + waited
                )

    def count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def note(self, **values: Any) -> None:
        with self._lock:
            self.details.update(values)

    def add_sample(self, sample: _Sample) -> None:
        with self._lock:
            if self.ended is not None:
                return
            self._last_sample = sample
            if self._step is not None:
                self._step.take(sample)

    # -- internals ---------------------------------------------------------

    def _open_step_locked(self, name: str, description: str, sample: _Sample) -> None:
        step = _Step(
            index=len(self.steps) + 1,
            name=name,
            description=description or STEP_DESCRIPTIONS.get(name, ""),
            started=perf_counter(),
            start_sample=sample,
        )
        step.take(sample)
        self.steps.append(step)
        self._step = step
        self._op_label = ""
        self._op_started = step.started

    def _close_step_locked(self, sample: _Sample) -> None:
        step = self._step
        if step is None:
            return
        self._switch_operation_locked("")
        step.ended = perf_counter()
        step.end_sample = sample
        step.take(sample)
        self._step = None

    def _switch_operation_locked(self, label: str) -> None:
        # Time between two progress events belongs to the operation the
        # first one announced.
        now = perf_counter()
        if label == self._op_label:
            return
        if self._op_label and self._step is not None:
            op = self._step.operations.setdefault(_operation_key(self._op_label), _Operation())
            elapsed = now - self._op_started
            op.count += 1
            op.seconds += elapsed
            op.slowest = max(op.slowest, elapsed)
        self._op_label = label
        self._op_started = now

    # -- writing -----------------------------------------------------------

    def _write(self, others: list["RunReport"]) -> Path | None:
        directory = _settings["directory"]
        if directory is None:
            return None
        stamp = datetime.fromtimestamp(self.wall_started).strftime("%Y%m%d-%H%M%S")
        safe_id = re.sub(r"[^A-Za-z0-9-]", "", self.run_id)[:8] or "run"
        safe_kind = re.sub(r"[^A-Za-z0-9-]", "-", self.kind)[:40]
        path = directory / f"{stamp}_{safe_kind}_{safe_id}.log"
        path.write_text(self.render(others), encoding="utf-8")
        self.path = path
        with (directory / INDEX_FILE_NAME).open("a", encoding="utf-8") as index:
            index.write(self._index_line(path) + "\n")
        _prune(directory)
        logger.info(
            "Run report written",
            extra={
                "event": "run_report_written",
                "duration_ms": round(self.total_seconds * 1000, 2),
                "details": {"report": path.name, "status": self.status},
            },
        )
        return path

    @property
    def total_seconds(self) -> float:
        return max(0.0, (self.ended or perf_counter()) - self.started)

    def _index_line(self, path: Path) -> str:
        slowest = max(self.steps, key=lambda s: s.seconds, default=None)
        slow = f"slowest={slowest.name} {_duration(slowest.seconds)}" if slowest else ""
        started = datetime.fromtimestamp(self.wall_started).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{started}  {self.status.upper():<9} {_duration(self.total_seconds):>10}  "
            f"{self.title}  {slow}  -> {path.name}"
        )

    def render(self, others: list["RunReport"] | None = None) -> str:
        with self._lock:
            return _Renderer(self, others or []).render()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {rest:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def _size(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GB"
    return f"{value / 1024**2:.0f} MB"


def _machine_summary() -> str:
    cores = os.cpu_count() or 0
    parts = [f"{cores} CPU cores"]
    if psutil is not None:
        try:
            parts.append(f"{_size(psutil.virtual_memory().total)} RAM")
        except Exception:  # noqa: BLE001
            pass
    parts.append(f"Python {platform.python_version()}")
    parts.append(f"{platform.system()} {platform.machine()}")
    return ", ".join(parts)


class _Renderer:
    def __init__(self, report: RunReport, others: list[RunReport]) -> None:
        self.r = report
        self.others = others
        self.total = max(report.total_seconds, 1e-9)

    def render(self) -> str:
        lines: list[str] = []
        lines += self._header()
        lines += self._where_time_went()
        lines += self._why()
        lines += self._step_table()
        lines += self._step_details()
        lines += self._counters()
        lines.append("")
        lines.append(
            "How to read this: each step starts when the pipeline reports a new "
            "stage. Time inside a step is attributed to the operation the "
            "pipeline announced until its next progress update. 'CPU cores' is "
            "CPU time / wall time across the backend and its worker "
            "processes: 1.0 means one core busy the whole step. 'Model' is time "
            "inside the OCR models; 'waiting for model' is time another run "
            "held them."
        )
        return "\n".join(lines) + "\n"

    def _header(self) -> list[str]:
        r = self.r
        started = datetime.fromtimestamp(r.wall_started)
        finished = datetime.fromtimestamp(r.wall_started + r.total_seconds)
        rule = "=" * 88
        lines = [
            rule,
            f"QTracX run report: {r.title}",
            rule,
            f"{'Run id':<16}: {r.run_id}",
            f"{'Kind':<16}: {r.kind}",
            f"{'Status':<16}: {r.status.upper()}" + (f"  ({r.error})" if r.error else ""),
            f"{'Started':<16}: {started:%Y-%m-%d %H:%M:%S}",
            f"{'Finished':<16}: {finished:%Y-%m-%d %H:%M:%S}",
            f"{'Total time':<16}: {_duration(r.total_seconds)}",
            f"{'Machine':<16}: {_machine_summary()}",
        ]
        for key, value in r.details.items():
            if key.startswith("_"):
                continue
            label = key.replace("_", " ")
            lines.append(f"{label[:1].upper() + label[1:]:<16}: {value}")
        if r.outcome:
            outcome = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in r.outcome.items())
            lines.append(f"{'Result':<16}: {outcome}")
        lines.append("")
        return lines

    def _where_time_went(self) -> list[str]:
        if not self.r.steps:
            return []
        by_name: dict[str, float] = {}
        for step in self.r.steps:
            by_name[step.name] = by_name.get(step.name, 0.0) + step.seconds
        lines = ["WHERE THE TIME WENT", "-" * 88]
        for name, seconds in sorted(by_name.items(), key=lambda item: -item[1]):
            share = seconds / self.total
            bar = "#" * max(1 if seconds > 0 else 0, int(round(share * 40)))
            lines.append(f"  {name:<14} {_duration(seconds):>12}  {share * 100:5.1f}%  {bar}")
        lines.append("")
        return lines

    def _why(self) -> list[str]:
        r = self.r
        reasons: list[str] = []
        steps = sorted(r.steps, key=lambda s: -s.seconds)
        for step in steps[:3]:
            if step.seconds < 0.05 * self.total and step is not steps[0]:
                break
            text = (
                f"Step {step.index} '{step.name}' took {_duration(step.seconds)} "
                f"({step.seconds / self.total * 100:.0f}% of the run)"
            )
            work = self._work_phrase(step)
            if work:
                text += f"; {work}"
            slowest_op = max(step.operations.items(), key=lambda kv: kv[1].seconds, default=None)
            if slowest_op and slowest_op[1].seconds >= 0.25 * step.seconds:
                name, op = slowest_op
                text += (
                    f". Mostly '{name}': {op.count} x, avg {_duration(op.seconds / op.count)}, "
                    f"slowest {_duration(op.slowest)}"
                )
            reasons.append(text + ".")

        model = sum(s.model_seconds for s in r.steps) + float(r.details.get("_unstepped_model_seconds", 0.0))
        wait = sum(s.model_wait_seconds for s in r.steps) + float(r.details.get("_unstepped_model_wait", 0.0))
        calls = r.counters.get("OCR model calls", 0)
        if calls:
            reasons.append(
                f"OCR model work in this process: {calls} calls, {_duration(model)} "
                f"({model / self.total * 100:.0f}% of the run)."
                + (" Worker processes ran more model calls in parallel (see CPU)." if self._max_workers() else "")
            )
        if wait >= max(1.0, 0.05 * self.total):
            reasons.append(
                f"Waited {_duration(wait)} for the OCR models while another run used them "
                f"({wait / self.total * 100:.0f}% of the run). See 'Ran alongside'."
            )

        cpu = self._cpu_seconds()
        cores = os.cpu_count() or 1
        if cpu is not None and self.total >= 2:
            used = cpu / self.total
            text = f"CPU: averaged {used:.1f} of {cores} cores"
            if used < 0.6:
                text += " (mostly idle: waiting on the model lock, disk, or memory paging)"
            elif used < 1.4:
                text += " (essentially single-threaded)"
            reasons.append(text + ".")

        swap_growth = self._swap_growth()
        min_free = min((s.min_available for s in r.steps if s.min_available is not None), default=None)
        if swap_growth is not None and swap_growth > 256 * 1024**2:
            reasons.append(
                f"Memory pressure: swap grew by {_size(swap_growth)} during the run, so the "
                "machine was paging to disk, which slows every step."
            )
        elif min_free is not None and min_free < 512 * 1024**2:
            reasons.append(f"Memory was tight: free RAM fell to {_size(min_free)}.")

        overlaps = self._overlaps()
        if overlaps:
            joined = "; ".join(f"{title} for {_duration(sec)}" for title, sec in overlaps[:5])
            reasons.append(f"Ran alongside: {joined}.")

        if r.status == "cancelled":
            reasons.append("The run was cancelled, so later steps never ran.")
        elif r.status == "failed":
            reasons.append(f"The run failed in step '{r.steps[-1].name if r.steps else '?'}'.")

        if not reasons:
            return []
        lines = ["WHY IT TOOK THIS LONG", "-" * 88]
        lines += [f"  * {reason}" for reason in reasons]
        lines.append("")
        return lines

    @staticmethod
    def _work_phrase(step: _Step) -> str:
        parts = []
        if step.total:
            parts.append(f"{step.completed}/{step.total} work units")
            if step.completed:
                parts.append(f"{_duration(step.seconds / step.completed)} per unit")
        if step.candidates:
            parts.append(f"{step.candidates} candidates")
        if step.model_calls:
            parts.append(f"{step.model_calls} model calls")
        return ", ".join(parts)

    def _cpu_seconds(self) -> float | None:
        first, last = self.r._first_sample, self.r._last_sample
        if first is None or last is None or (first.cpu_seconds == 0 and last.cpu_seconds == 0):
            return None
        return max(0.0, last.cpu_seconds - first.cpu_seconds)

    def _swap_growth(self) -> int | None:
        first = self.r._first_sample
        peaks = [s.max_swap for s in self.r.steps if s.max_swap is not None]
        if first is None or first.swap_used is None or not peaks:
            return None
        return max(peaks) - first.swap_used

    def _max_workers(self) -> int:
        return max((s.peak_workers for s in self.r.steps), default=0)

    def _overlaps(self) -> list[tuple[str, float]]:
        start = self.r.started
        end = self.r.ended or perf_counter()
        found = []
        for other in self.others:
            o_end = other.ended or perf_counter()
            overlap = min(end, o_end) - max(start, other.started)
            if overlap >= 0.5:
                found.append((other.title, overlap))
        return sorted(found, key=lambda item: -item[1])

    def _step_table(self) -> list[str]:
        if not self.r.steps:
            return []
        lines = [
            "STEPS",
            "-" * 88,
            f"  {'#':>2}  {'step':<12} {'starts at':>10} {'time':>11} {'share':>6} "
            f"{'CPU cores':>9} {'model':>10} {'peak RAM':>9} {'workers':>9}",
        ]
        for step in self.r.steps:
            offset = step.started - self.r.started
            cores = step.cpu_seconds / step.seconds if step.seconds > 0 else 0.0
            workers = f"{step.peak_workers}/{_size(step.peak_workers_rss)}" if step.peak_workers else "-"
            lines.append(
                f"  {step.index:>2}  {step.name[:12]:<12} {'+' + _duration(offset):>10} "
                f"{_duration(step.seconds):>11} {step.seconds / self.total * 100:5.1f}% "
                f"{cores:9.2f} {_duration(step.model_seconds) if step.model_calls else '-':>10} "
                f"{_size(step.peak_main_rss):>9} {workers:>9}"
            )
        lines.append("")
        return lines

    def _step_details(self) -> list[str]:
        if not self.r.steps:
            return []
        lines = ["STEP DETAILS", "-" * 88]
        for step in self.r.steps:
            lines.append(f"[{step.index}] {step.name} - {_duration(step.seconds)}")
            if step.description:
                lines.append(f"    What it does : {step.description}")
            if step.first_message:
                lines.append(f"    Began with   : {step.first_message}")
            if step.last_message and step.last_message != step.first_message:
                lines.append(f"    Ended with   : {step.last_message}")
            work = self._work_phrase(step)
            if work:
                lines.append(f"    Work done    : {work}")
            if step.model_calls:
                lines.append(
                    f"    OCR models   : {step.model_calls} calls, {_duration(step.model_seconds)} inside, "
                    f"{_duration(step.model_wait_seconds)} waiting for another run"
                )
            resources = [f"CPU {step.cpu_seconds:.1f} s"]
            resources.append(f"backend RAM peak {_size(step.peak_main_rss)}")
            if step.peak_workers:
                resources.append(
                    f"{step.peak_workers} workers peak {_size(step.peak_workers_rss)}"
                )
            if step.min_available is not None:
                resources.append(f"free RAM low {_size(step.min_available)}")
            if step.max_swap is not None and step.start_sample.swap_used is not None:
                growth = step.max_swap - step.start_sample.swap_used
                if growth > 64 * 1024**2:
                    resources.append(f"swap +{_size(growth)}")
            lines.append(f"    Resources    : {', '.join(resources)}")
            ops = sorted(step.operations.items(), key=lambda kv: -kv[1].seconds)
            if ops:
                lines.append("    Operations (slowest first):")
                for name, op in ops[:8]:
                    lines.append(
                        f"      {name[:46]:<46} {op.count:>4} x  total {_duration(op.seconds):>10}"
                        f"  avg {_duration(op.seconds / op.count):>9}  max {_duration(op.slowest):>9}"
                    )
                if len(ops) > 8:
                    rest = sum(op.seconds for _, op in ops[8:])
                    lines.append(f"      ... {len(ops) - 8} more operations, {_duration(rest)} together")
            lines.append("")
        return lines

    def _counters(self) -> list[str]:
        if not self.r.counters:
            return []
        lines = ["COUNTERS", "-" * 88]
        for name, value in sorted(self.r.counters.items()):
            lines.append(f"  {name:<50} {value:>8}")
        return lines


def _prune(directory: Path) -> None:
    keep = int(_settings["keep"])
    reports = sorted(
        (p for p in directory.glob("*.log") if p.name != INDEX_FILE_NAME),
        key=lambda p: p.name,
    )
    for stale in reports[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass
