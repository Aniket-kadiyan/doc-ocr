"""
Lightweight .env loader + typed getters — no third-party dependency.

Reads the project's root .env.local / .env so the Python backend draws from the
SAME config file as the Next.js frontend. Precedence matches Next.js:

    real environment  >  .env.local  >  .env

(an already-set variable is never overwritten — that's why we use setdefault).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

_REPO_ROOT = Path(__file__).resolve().parent.parent


class RuntimeConfigurationError(ValueError):
    """A configured runtime value is invalid or unsafe to interpret."""


def _configured_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _strict_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _strict_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeConfigurationError(
        f"{name} must be one of: 1, 0, true, false, yes, no, on, off"
    )


def _validate_origin(origin: str) -> str:
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeConfigurationError(
            "OCR_CORS_ORIGINS entries must be complete http:// or https:// origins"
        )
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RuntimeConfigurationError(
            "OCR_CORS_ORIGINS entries must not contain paths, queries, or fragments"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated runtime settings shared by startup and diagnostics."""

    environment: str
    ui_port: int
    backend_host: str
    backend_port: int
    backend_reload: bool
    cors_origins: tuple[str, ...] | None
    checksheet_data_dir: Path
    checksheet_max_document_mb: int
    database_url: str = field(repr=False)
    log_dir: Path
    log_level: str
    log_max_mb: int
    log_backup_count: int

    def public_summary(self) -> dict[str, object]:
        """Return startup facts without exposing connection secrets."""

        return {
            "environment": self.environment,
            "ui_port": self.ui_port,
            "backend_host": self.backend_host,
            "backend_port": self.backend_port,
            "backend_reload": self.backend_reload,
            "cors_origins": list(self.cors_origins or ()),
            "checksheet_data_dir": str(self.checksheet_data_dir),
            "checksheet_max_document_mb": self.checksheet_max_document_mb,
            "database_configured": bool(self.database_url),
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "log_max_mb": self.log_max_mb,
            "log_backup_count": self.log_backup_count,
        }


def load_env_files() -> None:
    """Populate os.environ from .env.local then .env (without overriding)."""
    for name in (".env.local", ".env"):
        path = _REPO_ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if not key:
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            os.environ.setdefault(key, val)


def get_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def get_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val.strip() if val and val.strip() else default


def get_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    try:
        return int(val) if val and val.strip() else default
    except ValueError:
        return default


def get_cors_origins() -> list[str] | None:
    """Explicit comma-separated allow-list, or None to fall back to a dev regex."""
    raw = os.environ.get("OCR_CORS_ORIGINS", "").strip()
    if not raw:
        return None
    return [_validate_origin(o.strip()) for o in raw.split(",") if o.strip()]


def load_runtime_config() -> RuntimeConfig:
    """Read and validate the complete QTracX runtime configuration."""

    environment = get_str("QTRACX_ENV", "development").lower()
    if environment not in {"development", "test", "production"}:
        raise RuntimeConfigurationError(
            "QTRACX_ENV must be development, test, or production"
        )

    log_level = get_str("QTRACX_LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise RuntimeConfigurationError(
            "QTRACX_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
        )

    return RuntimeConfig(
        environment=environment,
        ui_port=_strict_int("PORT", 3002, minimum=1, maximum=65535),
        backend_host=get_str("OCR_HOST", "127.0.0.1"),
        backend_port=_strict_int("OCR_PORT", 8000, minimum=1, maximum=65535),
        backend_reload=_strict_bool("OCR_RELOAD", True),
        cors_origins=(
            tuple(origins) if (origins := get_cors_origins()) is not None else None
        ),
        checksheet_data_dir=_configured_path(
            "CHECKSHEET_DATA_DIR",
            _REPO_ROOT / "backend" / "data" / "checksheets",
        ),
        checksheet_max_document_mb=_strict_int(
            "CHECKSHEET_MAX_DOCUMENT_MB", 200, minimum=1, maximum=2048
        ),
        database_url=os.environ.get("DATABASE_URL", "").strip(),
        log_dir=_configured_path(
            "QTRACX_LOG_DIR", _REPO_ROOT / "backend" / "logs"
        ),
        log_level=log_level,
        log_max_mb=_strict_int(
            "QTRACX_LOG_MAX_MB", 25, minimum=1, maximum=2048
        ),
        log_backup_count=_strict_int(
            "QTRACX_LOG_BACKUP_COUNT", 10, minimum=1, maximum=1000
        ),
    )
