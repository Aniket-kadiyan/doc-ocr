"""A small pool of OCR worker processes for the independent parts of a scan.

Paddle inference on this platform holds the GIL and uses one core per process:
three PaddleOCR instances driven from three threads measured 5.2 predictions/s,
exactly the same as one, while three processes measured 10.6. The stages of a
page scan that are many independent reads — the per-object authoritative
reread, the levelled callout neighbourhoods, the recognition batches — are
therefore farmed out to a few worker processes, each holding its own pipeline,
with the calling process taking a share of the work itself.

Everything crossing the process boundary is plain data: images travel as PNG
bytes, results as dicts of Python scalars. A worker failure never fails a scan;
the items it was holding are re-run in the calling process.

``OCR_WORKERS`` sets the pool size (default 2, ``0`` disables the pool). The
pool is started once at server start-up so the first scan does not pay the
model-load time; a scan that arrives before a worker is ready simply runs
without it.
"""

from __future__ import annotations

import io
import os
import threading
import time
import uuid
from typing import Any, Callable, Sequence

from config_env import get_int

DEFAULT_WORKERS = 2
WORKER_PROCESS_PREFIX = "ocr-worker-"
MAX_WORKERS = 4
# How long a whole job may run before the caller stops waiting for the workers
# and finishes the remaining items itself. Generous: it is a safety net for a
# hung worker, not a performance knob.
JOB_BASE_TIMEOUT_SECONDS = 60.0
JOB_PER_ITEM_TIMEOUT_SECONDS = 8.0


def configured_worker_count() -> int:
    """Pool size from ``OCR_WORKERS``, bounded to what this machine can hold."""

    try:
        count = get_int("OCR_WORKERS", DEFAULT_WORKERS)
    except Exception:  # noqa: BLE001 - a bad setting means "default", not "crash"
        count = DEFAULT_WORKERS
    cpu = os.cpu_count() or 2
    return max(0, min(count, MAX_WORKERS, cpu - 1))


def image_to_png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def png_to_image(data: bytes) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def plain(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays so a result pickles cleanly."""

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


# --------------------------------------------------------------------------- #
# Task execution — the same code runs in a worker and, for the caller's own
# share, in the calling process.
# --------------------------------------------------------------------------- #


def run_task(pipeline: Any, task: str, payload: dict[str, Any]) -> Any:
    """Execute one task against ``pipeline``; images arrive as PNG bytes."""

    if task == "recognize":
        image = png_to_image(payload["png"])
        return plain(pipeline.recognize(image, **dict(payload.get("kwargs") or {})))
    if task == "authoritative":
        from page_candidate_recovery import AuthoritativeCrop

        crops = []
        for item in payload["crops"]:
            crops.append(
                AuthoritativeCrop(
                    profile=str(item["profile"]),
                    image=png_to_image(item["png"]),
                    target_bbox=dict(item["target_bbox"]),
                    target_polygon=tuple(
                        (float(x), float(y)) for x, y in item["target_polygon"]
                    ),
                    polygon_usable=bool(item.get("polygon_usable", False)),
                )
            )
        return plain(
            pipeline._authoritative_attempts(  # noqa: SLF001 - pool is an internal helper
                crops[0],
                crops[1],
                debug_dump=bool(payload.get("debug_dump", False)),
                debug_dump_force=bool(payload.get("debug_dump_force", False)),
                max_paddle_predictions=payload.get("max_paddle_predictions"),
                preliminary_text=str(payload.get("preliminary_text") or ""),
            )
        )
    if task == "batch":
        images = [png_to_image(data) for data in payload["pngs"]]
        return plain(
            pipeline._recognize_page_batch(  # noqa: SLF001
                images,
                batch_size=int(payload.get("batch_size") or 16),
                profile=str(payload.get("profile") or "batch_recognition"),
            )
        )
    if task == "angled_roi":
        image = png_to_image(payload["png"])
        return plain(
            pipeline._angled_in_roi(  # noqa: SLF001
                image, **dict(payload.get("kwargs") or {})
            )
        )
    if task == "split":
        image = png_to_image(payload["png"])
        pieces = pipeline._split_stacked_cluster(  # noqa: SLF001
            image,
            (0, 0, image.width, image.height),
            max_paddle_predictions=payload.get("max_paddle_predictions"),
        )
        return plain(pieces) if pieces else None
    if task == "detect_tile":
        image = png_to_image(payload["png"])
        return plain(
            pipeline._detector_only_boxes(  # noqa: SLF001
                image, target_long_edge=int(payload.get("target_long_edge") or 1100)
            )
        )
    if task == "ping":
        return os.getpid()
    raise ValueError(f"Unknown OCR worker task: {task!r}")


def _worker_main(task_queue: Any, result_queue: Any, parent_pid: int) -> None:
    """Entry point of one worker process."""

    import warnings

    warnings.filterwarnings("ignore")

    def watch_parent() -> None:
        # A worker must never outlive the server that spawned it (uvicorn's
        # reloader kills the app process without running its shutdown hooks).
        while True:
            time.sleep(2.0)
            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()

    try:
        from ocr_pipeline import get_pipeline

        pipeline = get_pipeline()
    except Exception as exc:  # noqa: BLE001
        result_queue.put(("__init__", os.getpid(), False, repr(exc)))
        return
    result_queue.put(("__init__", os.getpid(), True, os.getpid()))

    while True:
        item = task_queue.get()
        if item is None:
            return
        job_id, index, task, payload = item
        try:
            result_queue.put((job_id, index, True, run_task(pipeline, task, payload)))
        except Exception as exc:  # noqa: BLE001 - reported to the caller, which re-runs the item
            result_queue.put((job_id, index, False, repr(exc)))


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #


class OcrWorkerPool:
    """Spawned worker processes plus the calling process, sharing one job."""

    def __init__(self, size: int) -> None:
        self.size = max(0, int(size))
        self._processes: list[Any] = []
        self._ready: set[int] = set()
        self._failed: set[int] = set()
        self._task_queue: Any = None
        self._result_queue: Any = None
        # One job at a time holds ``_job_lock`` for its whole duration; the
        # readiness bookkeeping has its own lock so /health and a second
        # caller can ask "how many workers" without waiting for that job.
        self._job_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._started = False
        self._broken = False

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Spawn the workers. Returns at once; readiness arrives asynchronously."""

        if self._started or self.size <= 0:
            return
        self._started = True
        try:
            import multiprocessing as mp

            # A worker is itself a spawned process; it must never grow a pool
            # of its own (spawn re-imports the parent's __main__ module, so a
            # script without a main guard would recurse). Workers are named,
            # because the server itself is a spawned child under uvicorn's
            # reloader and must still be allowed to start one.
            if mp.current_process().name.startswith(WORKER_PROCESS_PREFIX):
                self._broken = True
                return
            context = mp.get_context("spawn")
            self._task_queue = context.Queue()
            self._result_queue = context.Queue()
            for index in range(self.size):
                process = context.Process(
                    target=_worker_main,
                    args=(self._task_queue, self._result_queue, os.getpid()),
                    name=f"{WORKER_PROCESS_PREFIX}{index + 1}",
                    daemon=True,
                )
                process.start()
                self._processes.append(process)
        except Exception:  # noqa: BLE001 - no pool is a slower scan, not a failure
            self._broken = True
            self._processes = []

    def shutdown(self) -> None:
        if not self._started or self._task_queue is None:
            return
        try:
            for _ in self._processes:
                self._task_queue.put(None)
        except Exception:  # noqa: BLE001
            pass
        for process in self._processes:
            try:
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
            except Exception:  # noqa: BLE001
                pass
        self._processes = []
        self._broken = True

    def _drain_control_messages(self) -> None:
        """Absorb readiness messages without blocking."""

        if self._result_queue is None:
            return
        while True:
            try:
                message = self._result_queue.get_nowait()
            except Exception:  # noqa: BLE001 - queue.Empty or a closed queue
                return
            self._absorb(message)

    def _absorb(self, message: tuple[Any, ...]) -> tuple[Any, ...] | None:
        """Handle a control message; return job messages to the caller."""

        job_id, index, ok, value = message
        if job_id == "__init__":
            with self._control_lock:
                if ok:
                    self._ready.add(int(value))
                else:
                    self._failed.add(int(index))
            return None
        return message

    def ready_workers(self) -> int:
        if self._broken or not self._started:
            return 0
        # Only the caller holding the job lock reads the result queue (it
        # would otherwise steal job results); anyone else counts what is
        # already known.
        if self._job_lock.acquire(blocking=False):
            try:
                self._drain_control_messages()
            finally:
                self._job_lock.release()
        with self._control_lock:
            ready = set(self._ready)
        return sum(1 for p in self._processes if p.is_alive() and p.pid in ready)

    @property
    def available(self) -> bool:
        return self.ready_workers() > 0

    def status(self) -> dict[str, Any]:
        ready = self.ready_workers()
        return {
            "configured": self.size,
            "started": self._started,
            "ready": ready,
            "broken": self._broken,
        }

    # -- work --------------------------------------------------------------- #

    def map(
        self,
        task: str,
        payloads: Sequence[dict[str, Any]],
        local: Callable[[dict[str, Any]], Any],
        *,
        on_item_done: Callable[[int, Any], None] | None = None,
    ) -> list[Any]:
        """
        Run ``task`` over ``payloads`` and return the results in order.

        The payloads are dealt round-robin over the ready workers and the
        calling process, which runs its share through ``local`` while the
        workers run theirs. Anything a worker does not return — it crashed, or
        the job timed out — is run through ``local`` afterwards, so the result
        list is always complete. ``on_item_done`` is called (in the calling
        thread) as each result becomes available, for progress reporting.
        """

        payloads = list(payloads)
        results: list[Any] = [None] * len(payloads)
        done = [False] * len(payloads)
        if not payloads:
            return results

        workers = self.ready_workers()
        if workers <= 0:
            for index, payload in enumerate(payloads):
                results[index] = local(payload)
                done[index] = True
                if on_item_done is not None:
                    on_item_done(index, results[index])
            return results

        with self._job_lock:
            self._drain_control_messages()
            job_id = uuid.uuid4().hex
            slots = workers + 1
            local_indexes: list[int] = []
            remote_count = 0
            for index, payload in enumerate(payloads):
                if index % slots == 0:
                    local_indexes.append(index)
                else:
                    try:
                        self._task_queue.put((job_id, index, task, payload))
                        remote_count += 1
                    except Exception:  # noqa: BLE001
                        local_indexes.append(index)

            for index in local_indexes:
                results[index] = local(payloads[index])
                done[index] = True
                if on_item_done is not None:
                    on_item_done(index, results[index])

            deadline = time.monotonic() + JOB_BASE_TIMEOUT_SECONDS + (
                JOB_PER_ITEM_TIMEOUT_SECONDS * remote_count
            )
            received = 0
            while received < remote_count:
                if time.monotonic() > deadline:
                    self._broken = True
                    break
                if not any(p.is_alive() for p in self._processes):
                    self._broken = True
                    break
                try:
                    message = self._result_queue.get(timeout=0.5)
                except Exception:  # noqa: BLE001 - queue.Empty
                    continue
                message = self._absorb(message)
                if message is None:
                    continue
                msg_job, index, ok, value = message
                if msg_job != job_id:
                    continue  # a straggler from an abandoned job
                received += 1
                if ok:
                    results[index] = value
                    done[index] = True
                    if on_item_done is not None:
                        on_item_done(index, value)

            # Whatever did not come back is finished here.
            for index, finished in enumerate(done):
                if finished:
                    continue
                results[index] = local(payloads[index])
                done[index] = True
                if on_item_done is not None:
                    on_item_done(index, results[index])
        return results


_pool: OcrWorkerPool | None = None
_pool_lock = threading.Lock()


def get_worker_pool(*, start: bool = True) -> OcrWorkerPool:
    """The process-wide pool, created (and optionally started) on first use."""

    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = OcrWorkerPool(configured_worker_count())
        if start:
            _pool.start()
        return _pool


def shutdown_worker_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown()
            _pool = None
