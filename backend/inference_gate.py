"""Priority-aware capacity gate for Paddle model inference.

HTTP request threads and auto-balloon workers may prepare images concurrently,
but only the verified number of calls may enter the model pool at once.
Interactive work is admitted before waiting background scans whenever a slot
becomes available.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Condition
from time import perf_counter
from typing import Any, Iterator, Literal

from ocr_capacity import AUTO_BALLOON_SCAN_CAPACITY
import run_report


InferencePriority = Literal["interactive", "background"]
_VALID_PRIORITIES: tuple[InferencePriority, ...] = (
    "interactive",
    "background",
)
_inference_priority: ContextVar[InferencePriority] = ContextVar(
    "qtracx_inference_priority",
    default="interactive",
)


def current_inference_priority() -> InferencePriority:
    """Return the priority inherited by the current request or scan worker."""

    return _inference_priority.get()


@contextmanager
def inference_priority(priority: InferencePriority) -> Iterator[None]:
    """Temporarily assign inference priority to all calls in this context."""

    if priority not in _VALID_PRIORITIES:
        raise ValueError(f"Unknown inference priority: {priority}")
    token = _inference_priority.set(priority)
    try:
        yield
    finally:
        _inference_priority.reset(token)


class InferenceCapacityGate:
    """Limit and observe concurrent model calls within one backend process."""

    def __init__(self, slots: int) -> None:
        if slots < 0:
            raise ValueError("Inference capacity cannot be negative")
        self._slots = slots
        self._condition = Condition()
        self._active_calls = 0
        self._peak_active_calls = 0
        self._total_acquisitions = 0
        self._waiting: dict[InferencePriority, int] = {
            "interactive": 0,
            "background": 0,
        }
        self._active_operations: dict[str, int] = {}

    @contextmanager
    def slot(
        self,
        operation: str,
        *,
        priority: InferencePriority | None = None,
    ) -> Iterator[None]:
        """Wait for capacity, preferring interactive work between scan calls."""

        selected = priority or current_inference_priority()
        if selected not in _VALID_PRIORITIES:
            raise ValueError(f"Unknown inference priority: {selected}")
        if self._slots == 0:
            raise RuntimeError("No OCR model bundle is available for inference")

        wait_started = perf_counter()
        with self._condition:
            self._waiting[selected] += 1
            try:
                while (
                    self._active_calls >= self._slots
                    or (
                        selected == "background"
                        and self._waiting["interactive"] > 0
                    )
                ):
                    self._condition.wait()
            except BaseException:
                self._waiting[selected] -= 1
                self._condition.notify_all()
                raise

            self._waiting[selected] -= 1
            self._active_calls += 1
            self._peak_active_calls = max(
                self._peak_active_calls,
                self._active_calls,
            )
            self._total_acquisitions += 1
            self._active_operations[operation] = (
                self._active_operations.get(operation, 0) + 1
            )
        acquired_at = perf_counter()
        waited = acquired_at - wait_started

        try:
            yield
        finally:
            held = perf_counter() - acquired_at
            with self._condition:
                self._active_calls -= 1
                remaining = self._active_operations.get(operation, 0) - 1
                if remaining > 0:
                    self._active_operations[operation] = remaining
                else:
                    self._active_operations.pop(operation, None)
                self._condition.notify_all()
            run_report.model_call(waited, held)

    def status(self) -> dict[str, Any]:
        """Return live counts without request content or drawing data."""

        with self._condition:
            return {
                "configured_slots": self._slots,
                "active_calls": self._active_calls,
                "available_slots": max(
                    0,
                    self._slots - self._active_calls,
                ),
                "waiting_interactive": self._waiting["interactive"],
                "waiting_background": self._waiting["background"],
                "peak_active_calls": self._peak_active_calls,
                "total_acquisitions": self._total_acquisitions,
                "active_operations": dict(self._active_operations),
            }


_inference_gate = InferenceCapacityGate(AUTO_BALLOON_SCAN_CAPACITY)


def get_inference_gate() -> InferenceCapacityGate:
    """Return the one-slot default used before a model pool is loaded."""

    return _inference_gate

