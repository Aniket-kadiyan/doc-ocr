"""Time-limited access window for temporary QTracX demonstrations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable


Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


IST = timezone(timedelta(hours=5, minutes=30), name="IST")
DEMO_EXPIRED_STATUS_CODE = 423
DEMO_EXPIRED_CODE = "demo_expired"


def parse_demo_start(value: str) -> datetime:
    """Parse an ISO timestamp; values without an offset are India local time."""

    candidate = value.strip()
    if not candidate:
        raise ValueError("QTRACX_DEMO_START_AT is required when demo lock is enabled")
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError(
            "QTRACX_DEMO_START_AT must be an ISO date/time, for example "
            "2026-11-01T10:00:00"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST)


def parse_demo_duration(value: str) -> float:
    """Accept positive whole or decimal hours, including short test windows."""

    candidate = value.strip()
    try:
        duration = float(candidate)
    except ValueError as error:
        raise ValueError(
            "QTRACX_DEMO_DURATION_HOURS must be a positive number"
        ) from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("QTRACX_DEMO_DURATION_HOURS must be a positive number")
    if duration > 8760:
        raise ValueError("QTRACX_DEMO_DURATION_HOURS must not exceed 8760")
    return duration


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_ist(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(IST).isoformat(timespec="seconds")


@dataclass(frozen=True)
class DemoAccessWindow:
    """Compute the current state of a configured demonstration window."""

    enabled: bool
    start_at: datetime | None
    duration_hours: float
    now_provider: Callable[[], datetime] = field(
        default=_utc_now,
        repr=False,
        compare=False,
    )

    @property
    def expires_at(self) -> datetime | None:
        if self.start_at is None:
            return None
        return self.start_at + timedelta(hours=self.duration_hours)

    def snapshot(self, *, now: datetime | None = None) -> dict[str, object]:
        current = now or self.now_provider()
        if current.tzinfo is None:
            raise ValueError("The current time must include a timezone")
        current = current.astimezone(timezone.utc)
        start = self.start_at.astimezone(timezone.utc) if self.start_at else None
        expires = self.expires_at
        expiry_utc = expires.astimezone(timezone.utc) if expires else None

        if not self.enabled:
            state = "disabled"
            remaining = None
        elif start is not None and current < start:
            state = "scheduled"
            remaining = (expiry_utc - current).total_seconds() if expiry_utc else None
        elif expiry_utc is not None and current < expiry_utc:
            state = "active"
            remaining = (expiry_utc - current).total_seconds()
        else:
            state = "expired"
            remaining = 0.0

        return {
            "enabled": self.enabled,
            "state": state,
            "expired": state == "expired",
            "timezone": "Asia/Kolkata",
            "start_at": _iso_ist(self.start_at),
            "expires_at": _iso_ist(expires),
            "duration_hours": self.duration_hours,
            "remaining_seconds": (
                round(max(0.0, remaining), 3) if remaining is not None else None
            ),
        }

    def is_expired(self) -> bool:
        return bool(self.snapshot()["expired"])


class DemoLockMiddleware:
    """Reject every functional backend request after the demo expires."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        access_window: DemoAccessWindow,
        exempt_paths: tuple[str, ...] = ("/health",),
    ) -> None:
        self.app = app
        self.access_window = access_window
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope.get("path", "") not in self.exempt_paths
        ):
            snapshot = self.access_window.snapshot()
            if snapshot["expired"]:
                # Kept lazy so config parsing remains a standard-library-only
                # operation for setup and management commands.
                from starlette.responses import JSONResponse

                response = JSONResponse(
                    status_code=DEMO_EXPIRED_STATUS_CODE,
                    content={
                        "detail": "The QTracX demonstration period has expired.",
                        "code": DEMO_EXPIRED_CODE,
                        "demo": snapshot,
                    },
                    headers={"Cache-Control": "no-store"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
