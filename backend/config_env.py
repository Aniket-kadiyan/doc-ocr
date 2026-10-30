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

_REPO_ROOT = Path(__file__).resolve().parent.parent


class RuntimeConfigurationError(ValueError):
    """A configured runtime value is invalid or unsafe to interpret."""


def _configured_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _optional_configured_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
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


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated runtime settings shared by startup and diagnostics."""

    environment: str
    ui_host: str
    ui_port: int
    backend_host: str
    backend_port: int
    backend_reload: bool
    ocr_model_instances: int
    ocr_scan_workers: int
    https_enabled: bool
    tls_cert_file: Path | None = field(repr=False)
    tls_key_file: Path | None = field(repr=False)
    checksheet_data_dir: Path
    checksheet_max_document_mb: int
    cad_max_document_mb: int
    cad_convert_timeout_seconds: int
    dwg_converter_path: Path | None = field(repr=False)
    database_url: str = field(repr=False)
    auth_session_hours: int
    log_dir: Path
    log_level: str
    log_max_mb: int
    log_backup_count: int
    run_reports_enabled: bool
    run_report_keep: int
    run_report_sample_ms: int

    def public_summary(self) -> dict[str, object]:
        """Return startup facts without exposing connection secrets."""

        return {
            "environment": self.environment,
            "ui_host": self.ui_host,
            "ui_port": self.ui_port,
            "backend_host": self.backend_host,
            "backend_port": self.backend_port,
            "backend_reload": self.backend_reload,
            "ocr_model_instances": self.ocr_model_instances,
            "ocr_scan_workers": self.ocr_scan_workers,
            "https_enabled": self.https_enabled,
            "tls_cert_configured": self.tls_cert_file is not None,
            "tls_key_configured": self.tls_key_file is not None,
            "cors_policy": "allow_all_origins",
            "checksheet_data_dir": str(self.checksheet_data_dir),
            "checksheet_max_document_mb": self.checksheet_max_document_mb,
            "cad_max_document_mb": self.cad_max_document_mb,
            "cad_convert_timeout_seconds": self.cad_convert_timeout_seconds,
            "dwg_converter_configured": self.dwg_converter_path is not None,
            "database_configured": bool(self.database_url),
            "auth_session_hours": self.auth_session_hours,
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "log_max_mb": self.log_max_mb,
            "log_backup_count": self.log_backup_count,
            "run_reports_enabled": self.run_reports_enabled,
            "run_report_keep": self.run_report_keep,
            "run_report_sample_ms": self.run_report_sample_ms,
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
        ui_host=get_str("UI_HOST", "127.0.0.1"),
        ui_port=_strict_int("PORT", 3002, minimum=1, maximum=65535),
        backend_host=get_str("OCR_HOST", "127.0.0.1"),
        backend_port=_strict_int("OCR_PORT", 8000, minimum=1, maximum=65535),
        backend_reload=_strict_bool("OCR_RELOAD", True),
        ocr_model_instances=_strict_int(
            "OCR_MODEL_INSTANCES", 1, minimum=1, maximum=8
        ),
        ocr_scan_workers=_strict_int(
            "OCR_SCAN_WORKERS", 2, minimum=1, maximum=32
        ),
        https_enabled=_strict_bool("HTTPS_ENABLED", False),
        tls_cert_file=_optional_configured_path("TLS_CERT_FILE"),
        tls_key_file=_optional_configured_path("TLS_KEY_FILE"),
        checksheet_data_dir=_configured_path(
            "CHECKSHEET_DATA_DIR",
            _REPO_ROOT / "backend" / "data" / "checksheets",
        ),
        checksheet_max_document_mb=_strict_int(
            "CHECKSHEET_MAX_DOCUMENT_MB", 200, minimum=1, maximum=2048
        ),
        cad_max_document_mb=_strict_int(
            "CAD_MAX_DOCUMENT_MB", 128, minimum=1, maximum=2048
        ),
        cad_convert_timeout_seconds=_strict_int(
            "CAD_CONVERT_TIMEOUT_SECONDS", 180, minimum=5, maximum=1800
        ),
        dwg_converter_path=_optional_configured_path("DWG2DXF_ACADSHARP"),
        database_url=os.environ.get("DATABASE_URL", "").strip(),
        auth_session_hours=_strict_int(
            "AUTH_SESSION_HOURS", 8, minimum=1, maximum=168
        ),
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
        run_reports_enabled=_strict_bool(
            "QTRACX_RUN_REPORTS_ENABLED", True
        ),
        run_report_keep=_strict_int(
            "QTRACX_RUN_REPORT_KEEP", 300, minimum=1, maximum=10000
        ),
        run_report_sample_ms=_strict_int(
            "QTRACX_RUN_REPORT_SAMPLE_MS", 500, minimum=100, maximum=10000
        ),
    )

