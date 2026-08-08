"""Small thread-safe cache for reusable page layout geometry."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Callable

from page_layout import PageLayout


class PageLayoutCache:
    """Bounded least-recently-used cache keyed by full-page fingerprint."""

    def __init__(self, *, max_entries: int = 8) -> None:
        if max_entries < 1:
            raise ValueError("Page layout cache must retain at least one entry")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, PageLayout] = OrderedDict()
        self._lock = Lock()

    def get(self, fingerprint: str) -> PageLayout | None:
        with self._lock:
            layout = self._entries.get(fingerprint)
            if layout is not None:
                self._entries.move_to_end(fingerprint)
            return layout

    def put(self, fingerprint: str, layout: PageLayout) -> None:
        with self._lock:
            self._entries[fingerprint] = layout
            self._entries.move_to_end(fingerprint)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def get_or_create(
        self,
        fingerprint: str,
        create: Callable[[], PageLayout],
    ) -> tuple[PageLayout, bool]:
        """Return ``(layout, cache_hit)`` without holding the lock in analysis."""

        cached = self.get(fingerprint)
        if cached is not None:
            return cached, True
        layout = create()
        # Scan workers are serialized today; the second lookup also makes this
        # safe if that changes and two equal pages finish analysis together.
        existing = self.get(fingerprint)
        if existing is not None:
            return existing, True
        self.put(fingerprint, layout)
        return layout, False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
