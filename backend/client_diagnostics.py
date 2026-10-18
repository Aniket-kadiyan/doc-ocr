"""Validated, content-free diagnostics reported by the QTracX frontend."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClientDiagnosticEvent(BaseModel):
    """Small allow-listed browser event; free-form messages are forbidden."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["runtime_error", "unhandled_rejection", "api_error"]
    operation: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    error_type: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    http_status: int | None = Field(default=None, ge=100, le=599)


def record_client_diagnostic(
    event: ClientDiagnosticEvent,
    logger: logging.Logger,
) -> None:
    """Write only the fields admitted by :class:`ClientDiagnosticEvent`."""

    details: dict[str, object] = {
        "kind": event.kind,
        "operation": event.operation,
        "error_type": event.error_type,
    }
    if event.http_status is not None:
        details["http_status"] = event.http_status
    logger.warning(
        "Frontend diagnostic reported",
        extra={
            "event": "frontend_diagnostic",
            "status_code": event.http_status,
            "details": details,
        },
    )
