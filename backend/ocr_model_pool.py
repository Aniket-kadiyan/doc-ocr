"""Priority-aware pool of independently loaded Paddle model bundles."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator, Sequence

from inference_gate import InferenceCapacityGate


class OcrModelBundlePool:
    """Lease one concrete model bundle for the duration of an inference call.

    The capacity gate decides *when* a caller may enter. The deque decides
    *which* independently loaded bundle it owns. A bundle is returned before
    the gate releases its slot, so the two capacities cannot drift apart.
    """

    def __init__(self, bundles: Sequence[Any]) -> None:
        if not bundles:
            raise ValueError("At least one loaded OCR model bundle is required")
        self._all_bundles = tuple(bundles)
        self._available = deque(self._all_bundles)
        self._bundle_lock = Lock()
        self._gate = InferenceCapacityGate(len(self._all_bundles))

    @property
    def capacity(self) -> int:
        return len(self._all_bundles)

    @property
    def inference_gate(self) -> InferenceCapacityGate:
        return self._gate

    @contextmanager
    def lease(self, operation: str) -> Iterator[Any]:
        """Yield one exclusive bundle using the caller's inherited priority."""

        with self._gate.slot(operation):
            with self._bundle_lock:
                # The gate and deque have identical capacity. Reaching this
                # point without a bundle would therefore be an internal bug,
                # not a condition callers should wait around and conceal.
                if not self._available:
                    raise RuntimeError("OCR model pool capacity became inconsistent")
                bundle = self._available.popleft()
            try:
                yield bundle
            finally:
                with self._bundle_lock:
                    self._available.append(bundle)

    def status(self) -> dict[str, Any]:
        with self._bundle_lock:
            available = len(self._available)
        return {
            "loaded_bundles": self.capacity,
            "leased_bundles": self.capacity - available,
            "available_bundles": available,
        }
