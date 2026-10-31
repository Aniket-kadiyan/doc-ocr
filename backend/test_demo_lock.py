"""Tests for the configurable QTracX demonstration expiry gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from demo_lock import (
    DEMO_EXPIRED_CODE,
    DemoAccessWindow,
    DemoLockMiddleware,
    IST,
    parse_demo_duration,
    parse_demo_start,
)


def test_start_without_offset_is_interpreted_as_ist() -> None:
    parsed = parse_demo_start("2026-11-01T10:00:00")

    assert parsed.utcoffset() == timedelta(hours=5, minutes=30)
    assert parsed.astimezone(timezone.utc) == datetime(
        2026, 11, 1, 4, 30, tzinfo=timezone.utc
    )


def test_explicit_offset_is_converted_to_ist() -> None:
    parsed = parse_demo_start("2026-11-01T04:30:00Z")

    assert parsed == datetime(2026, 11, 1, 10, 0, tzinfo=IST)


def test_decimal_demo_duration_is_accepted() -> None:
    assert parse_demo_duration("0.25") == pytest.approx(0.25)
    with pytest.raises(ValueError, match="positive number"):
        parse_demo_duration("0")


def test_window_moves_from_scheduled_to_active_to_expired() -> None:
    start = parse_demo_start("2026-11-01T10:00:00")
    window = DemoAccessWindow(
        enabled=True,
        start_at=start,
        duration_hours=0.5,
    )

    scheduled = window.snapshot(now=start - timedelta(seconds=1))
    active = window.snapshot(now=start + timedelta(minutes=15))
    expired = window.snapshot(now=start + timedelta(minutes=30))

    assert scheduled["state"] == "scheduled"
    assert active["state"] == "active"
    assert active["remaining_seconds"] == pytest.approx(900)
    assert expired["state"] == "expired"
    assert expired["expired"] is True
    assert expired["expires_at"] == "2026-11-01T10:30:00+05:30"


def test_middleware_blocks_all_functions_but_keeps_health_available() -> None:
    start = parse_demo_start("2026-11-01T10:00:00")
    clock = {"now": start + timedelta(minutes=15)}
    window = DemoAccessWindow(
        enabled=True,
        start_at=start,
        duration_hours=0.5,
        now_provider=lambda: clock["now"],
    )
    app = FastAPI()
    app.add_middleware(DemoLockMiddleware, access_window=window)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/auth/me")
    def authenticated_function() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/auth/me").status_code == 200
        clock["now"] = start + timedelta(minutes=30)

        blocked = client.get("/auth/me")
        health = client.get("/health")

    assert blocked.status_code == 423
    assert blocked.json()["code"] == DEMO_EXPIRED_CODE
    assert blocked.json()["demo"]["state"] == "expired"
    assert blocked.headers["cache-control"] == "no-store"
    assert health.status_code == 200


def test_disabled_window_never_locks_the_backend() -> None:
    window = DemoAccessWindow(
        enabled=False,
        start_at=parse_demo_start("2020-01-01T00:00:00"),
        duration_hours=0.001,
    )

    assert window.snapshot()["state"] == "disabled"
    assert window.is_expired() is False
