"""Structured, rotating diagnostics for the QTracX backend.

The logger records operational facts only. Request bodies, uploaded document
contents, recognized drawing text, passwords, tokens, and database connection
strings are deliberately outside the event schema.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from config_env import RuntimeConfig

LOG_FILE_NAME = "qtracx-backend.log"
_HANDLER_MARKER = "_qtracx_managed_handler"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LOG_CONTEXT: ContextVar[dict[str, object]] = ContextVar(
    "qtracx_log_context",
    default={},
)
_BASE_LOGGER = logging.getLogger("qtracx")
if not _BASE_LOGGER.handlers:
    # Library-style imports (notably unit tests) must stay silent until the
    # application explicitly configures its console and file destinations.
    _BASE_LOGGER.addHandler(logging.NullHandler())
_BASE_LOGGER.propagate = False
_EVENT_FIELDS = (
    "request_id",
    "job_id",
    "user_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "client_ip",
    "scope_kind",
    "page",
    "candidate_count",
    "error_type",
    "error_location",
    "details",
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return str(value)


def safe_exception_details(error: BaseException) -> dict[str, str]:
    """Describe an exception without copying its potentially sensitive text."""

    location = ""
    frames = traceback.extract_tb(error.__traceback__)
    if frames:
        last = frames[-1]
        location = f"{Path(last.filename).name}:{last.lineno}:{last.name}"
    return {
        "error_type": error.__class__.__name__,
        "error_location": location,
    }


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _LOG_CONTEXT.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line, suitable for local support collection."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": "qtracx-backend",
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            "process_id": record.process,
            "thread": record.threadName,
        }
        for field in _EVENT_FIELDS:
            value = getattr(record, field, None)
            if value is None or value == "":
                continue
            payload[field] = _json_value(value)
        if record.exc_info and record.exc_info[1] is not None:
            payload.update(safe_exception_details(record.exc_info[1]))
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _managed_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        handler
        for handler in logger.handlers
        if bool(getattr(handler, _HANDLER_MARKER, False))
    ]


def configure_logging(config: RuntimeConfig) -> Path:
    """Configure console and rotating-file output, safely and idempotently."""

    config.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.log_dir / LOG_FILE_NAME
    logger = logging.getLogger("qtracx")
    logger.setLevel(config.log_level)
    logger.propagate = False

    for handler in _managed_handlers(logger):
        logger.removeHandler(handler)
        handler.close()

    formatter = JsonLogFormatter()
    context_filter = _ContextFilter()

    console = logging.StreamHandler(sys.stderr)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=config.log_max_mb * 1024 * 1024,
        backupCount=config.log_backup_count,
        encoding="utf-8",
        delay=True,
    )
    for handler in (console, file_handler):
        setattr(handler, _HANDLER_MARKER, True)
        handler.setLevel(config.log_level)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        logger.addHandler(handler)
    return log_path


def shutdown_logging() -> None:
    """Flush and close only handlers installed by :func:`configure_logging`."""

    logger = logging.getLogger("qtracx")
    for handler in _managed_handlers(logger):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def get_logger(component: str) -> logging.Logger:
    cleaned = component.strip(".") or "backend"
    return logging.getLogger(f"qtracx.{cleaned}")


def request_id(value: str | None = None) -> str:
    """Keep a safe caller correlation ID, otherwise issue a new UUID."""

    candidate = (value or "").strip()
    return candidate if _REQUEST_ID_PATTERN.fullmatch(candidate) else str(uuid4())


def current_log_context() -> dict[str, object]:
    return dict(_LOG_CONTEXT.get())


@contextmanager
def log_context(**values: object) -> Iterator[None]:
    """Temporarily add correlation fields, including across nested calls."""

    merged = dict(_LOG_CONTEXT.get())
    merged.update(
        {
            key: value
            for key, value in values.items()
            if value is not None and value != ""
        }
    )
    token = _LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)
