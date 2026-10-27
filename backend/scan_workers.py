"""Worker processes that share a page scan's independent OCR work.

Paddle inference on a CPU runs one predictor on one core, and threads do not
help: the Python work around each call holds the GIL. Measured on an M1, 24
recognize() calls took 26.4 s serially, 26.6 s on three threads sharing the
class-level inference lock, 20.6 s on three threads with a lock each, and
14.1 s on three worker processes. Hence processes.

The whole-page route has work that needs only the page image -- the notes
paragraphs and the slanted-callout neighbourhoods -- and a batch of
independent per-object reads (the authoritative rereads and stacked splits).
On 113-M02-292A those were 235 s, 1724 s, 436 s and 231 s of a 50 minute
scan, every second of it on one core. The pool runs them beside the main
pipeline: background work at low priority, rereads at high priority so the
critical path is never queued behind a slanted neighbourhood.

Every job has a local fallback. A pool that cannot start, a worker that
dies, or ``OCR_SCAN_WORKERS=0`` leaves the scan exactly as serial as before.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import multiprocessing as mp
import os
import tempfile
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

PRIORITY_CRITICAL = 0
PRIORITY_BACKGROUND = 10

DEFAULT_MAX_WORKERS = 3
# What the pool may not touch: the main process (1.8 GB on a 24 MP page) and
# the browser and dev server beside it. Each worker then needs its own Paddle
# models plus its working page, which reached 1.3 GB on 113-M02-292A. Three
# workers on an 8 GB M1 pushed 4 GB into swap and spent 42% of the CPU in the
# kernel: detection alone went from 2 s a panel to over 10 minutes a page.
RESERVED_MEMORY_BYTES = 6 * 1024**3
WORKER_MEMORY_BYTES = 1300 * 1024**2


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def configured_worker_count() -> int:
    """``OCR_SCAN_WORKERS``, else what the cores and the memory allow, capped."""

    raw = os.environ.get("OCR_SCAN_WORKERS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("Ignoring non-integer OCR_SCAN_WORKERS=%r", raw)
    workers = min(DEFAULT_MAX_WORKERS, (os.cpu_count() or 1) - 1)
    memory = _physical_memory_bytes()
    if memory is not None:
        workers = min(
            workers,
            (memory - RESERVED_MEMORY_BYTES) // WORKER_MEMORY_BYTES,
        )
    return max(0, int(workers))


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

_WORKER_PIPELINE: Any = None
_WORKER_PAGE: tuple[str, Image.Image] | None = None


def _exit_with_parent(parent: int) -> None:
    """Leave as soon as the backend that started this worker is gone.

    A backend killed or reloaded mid-scan left its workers running, each
    holding its models: one sat at 1.1 GB for half an hour and pushed the
    next scan into swap.
    """
    import time

    while os.getppid() == parent:
        time.sleep(2.0)
    os._exit(0)


def _init_worker() -> None:
    global _WORKER_PIPELINE
    threading.Thread(
        target=_exit_with_parent,
        args=(os.getppid(),),
        name="scan-worker-parent-watch",
        daemon=True,
    ).start()
    from ocr_pipeline import OcrPipeline

    _WORKER_PIPELINE = OcrPipeline()
    _WORKER_PIPELINE.load()


def _page(path: str) -> Image.Image:
    """The scan's page, loaded once per worker per scan."""

    global _WORKER_PAGE
    if _WORKER_PAGE is None or _WORKER_PAGE[0] != path:
        _WORKER_PAGE = None
        _WORKER_PAGE = (path, Image.fromarray(np.load(path)))
    return _WORKER_PAGE[1]


def _job(path: str, method: str, args: tuple, kwargs: dict) -> Any:
    """Run one ``OcrPipeline`` method on the shared page in this worker."""

    return getattr(_WORKER_PIPELINE, method)(_page(path), *args, **kwargs)


# ---------------------------------------------------------------------------
# Main-process side
# ---------------------------------------------------------------------------


class ScanWorkerPool:
    """A process pool fed from a priority queue.

    ``ProcessPoolExecutor`` is first-in first-out and pulls work eagerly, so
    the queue lives here and at most ``workers`` jobs are handed over at once.
    """

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
        )
        self._lock = threading.RLock()
        self._queue: list[tuple[int, int, Future, tuple]] = []
        self._order = itertools.count()
        self._in_flight = 0
        self.broken = False

    def submit(self, priority: int, path: str, method: str, *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        with self._lock:
            heapq.heappush(
                self._queue,
                (priority, next(self._order), future, (path, method, args, kwargs)),
            )
        self._pump()
        return future

    def _pump(self) -> None:
        with self._lock:
            while self._queue and self._in_flight < self.workers:
                _, _, future, job = heapq.heappop(self._queue)
                # A job the caller took back to run itself is skipped here.
                if not future.set_running_or_notify_cancel():
                    continue
                if self.broken:
                    future.set_exception(RuntimeError("scan worker pool is broken"))
                    continue
                try:
                    inner = self._executor.submit(_job, *job)
                except Exception as exc:  # noqa: BLE001 - BrokenProcessPool et al.
                    self.broken = True
                    future.set_exception(exc)
                    continue
                self._in_flight += 1
                inner.add_done_callback(
                    lambda done, outer=future: self._finished(done, outer)
                )

    def _finished(self, inner: Future, outer: Future) -> None:
        with self._lock:
            self._in_flight -= 1
        exc = inner.exception()
        if exc is not None:
            if type(exc).__name__ == "BrokenProcessPool":
                self.broken = True
            outer.set_exception(exc)
        else:
            outer.set_result(inner.result())
        self._pump()


_POOL: ScanWorkerPool | None = None
_POOL_LOCK = threading.Lock()


def get_scan_pool() -> ScanWorkerPool | None:
    """The shared pool, started on first use; ``None`` when disabled."""

    global _POOL
    with _POOL_LOCK:
        if _POOL is not None and _POOL.broken:
            _POOL = None
        if _POOL is None:
            workers = configured_worker_count()
            if workers < 1:
                return None
            try:
                _POOL = ScanWorkerPool(workers)
            except Exception:  # noqa: BLE001 - the scan runs serially instead
                logger.exception("Could not start scan worker processes")
                return None
        return _POOL


def shutdown_scan_pool() -> None:
    """Stop the worker processes, abandoning any queued work."""

    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        pool.broken = True
        pool._executor.shutdown(wait=False, cancel_futures=True)


class ScanWork:
    """One scan's share of the pool: the page file and its outstanding jobs.

    Without a pool every job runs in the calling process, so callers write
    one code path for both.
    """

    def __init__(self, pipeline: Any, image: Image.Image, pool: ScanWorkerPool | None) -> None:
        self._pipeline = pipeline
        self._image = image
        self._pool = pool
        self._futures: list[Future] = []
        self._path: str | None = None
        self._closed = False
        if pool is not None:
            handle, self._path = tempfile.mkstemp(prefix="scan-page-", suffix=".npy")
            os.close(handle)
            np.save(self._path, np.asarray(image.convert("RGB")))

    @property
    def parallel(self) -> bool:
        return self._path is not None and not self._pool.broken

    def submit(self, priority: int, method: str, *args: Any, **kwargs: Any) -> Future:
        """Queue ``pipeline.<method>(page, *args, **kwargs)``."""

        if self._closed:
            # A follow-up queued from a callback after the scan ended.
            future = Future()
            future.cancel()
        elif self.parallel:
            assert self._path is not None
            future = self._pool.submit(priority, self._path, method, *args, **kwargs)
        else:
            future = Future()
            future.set_running_or_notify_cancel()
            future.set_result(_LOCAL)
        future._scan_call = (method, args, kwargs)  # type: ignore[attr-defined]
        self._futures.append(future)
        return future

    def _local(self, future: Future) -> Any:
        method, args, kwargs = future._scan_call  # type: ignore[attr-defined]
        return getattr(self._pipeline, method)(self._image, *args, **kwargs)

    def result(self, future: Future, heartbeat: Callable[[], None] | None = None) -> Any:
        """Wait for one job; run it here if it never started or its worker failed."""

        if future.cancel():
            return self._local(future)
        while True:
            try:
                value = future.result(timeout=1.0)
            except FutureTimeout:
                if heartbeat is not None:
                    heartbeat()
                continue
            except Exception:  # noqa: BLE001 - a worker failure is not the scan's
                logger.exception("Scan worker job failed; running it locally")
                return self._local(future)
            return self._local(future) if value is _LOCAL else value

    def gather(
        self,
        futures: Sequence[Future],
        heartbeat: Callable[[int], None] | None = None,
    ) -> list[Any]:
        """Results in order, with this process taking queued jobs from the back.

        The workers start from the front, so the main process is never idle
        while there is work it could do itself.
        """

        results: list[Any] = [None] * len(futures)
        done = [False] * len(futures)
        finished = 0
        for index in reversed(range(len(futures))):
            if futures[index].cancel():
                results[index] = self._local(futures[index])
                done[index] = True
                finished += 1
                if heartbeat is not None:
                    heartbeat(finished)
        for index, future in enumerate(futures):
            if done[index]:
                continue
            results[index] = self.result(
                future,
                (lambda: heartbeat(finished)) if heartbeat is not None else None,
            )
            finished += 1
            if heartbeat is not None:
                heartbeat(finished)
        return results

    def close(self) -> None:
        """Withdraw this scan's queued jobs and delete the page file."""

        self._closed = True
        for future in self._futures:
            future.cancel()
        self._futures.clear()
        if self._path is not None:
            try:
                os.unlink(self._path)
            except OSError:
                pass
            self._path = None


class _LocalSentinel:
    """Result placeholder meaning 'run this job in the calling process'."""


_LOCAL = _LocalSentinel()
