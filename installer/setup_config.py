"""Validated setup selections and QTracX environment-file generation.

The future graphical installer and unattended installer both use this module.
Database-administrator passwords are deliberately not represented here: they
are transient setup inputs and must never be written to the generated config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from urllib.parse import quote_plus


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
_HOST = re.compile(r"^[A-Za-z0-9_.:-]+$")


class SetupConfigurationError(ValueError):
    """A setup selection is incomplete or unsafe to render."""


def _require_port(name: str, value: int) -> None:
    if not 1 <= value <= 65535:
        raise SetupConfigurationError(f"{name} must be between 1 and 65535")


def _require_identifier(name: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise SetupConfigurationError(
            f"{name} must start with a letter and contain only letters, "
            "numbers, or underscores"
        )


def _require_host(name: str, value: str) -> None:
    if not value or not _HOST.fullmatch(value):
        raise SetupConfigurationError(f"{name} is not a valid host or IP address")


@dataclass(frozen=True)
class DatabaseSetup:
    engine: str = "mssql"
    host: str = "localhost"
    port: int = 1433
    name: str = "QTracX"
    application_user: str = "qtracx_app"
    application_password: str = field(default="", repr=False)
    odbc_driver: str = "ODBC Driver 18 for SQL Server"
    encrypt: bool = True
    trust_server_certificate: bool = True

    def validate(self) -> None:
        if self.engine not in {"mssql", "postgresql"}:
            raise SetupConfigurationError(
                "database engine must be mssql or postgresql"
            )
        _require_host("database host", self.host)
        _require_port("database port", self.port)
        _require_identifier("database name", self.name)
        _require_identifier("application database user", self.application_user)
        if len(self.application_password) < 4:
            raise SetupConfigurationError(
                "application database password must contain at least 4 characters"
            )
        if self.engine == "mssql" and not self.odbc_driver.strip():
            raise SetupConfigurationError("an MSSQL ODBC driver is required")

    def url(self) -> str:
        self.validate()
        user = quote_plus(self.application_user)
        password = quote_plus(self.application_password)
        host = self.host
        database = quote_plus(self.name)
        if self.engine == "postgresql":
            return (
                f"postgresql+pg8000://{user}:{password}@{host}:{self.port}/"
                f"{database}"
            )
        driver = quote_plus(self.odbc_driver)
        encrypt = "yes" if self.encrypt else "no"
        trust = "yes" if self.trust_server_certificate else "no"
        return (
            f"mssql+pyodbc://{user}:{password}@{host}:{self.port}/{database}"
            f"?driver={driver}&Encrypt={encrypt}&TrustServerCertificate={trust}"
        )


@dataclass(frozen=True)
class InstallerSetup:
    deployment_mode: str = "local"
    server_address: str = "localhost"
    ui_port: int = 3002
    backend_port: int = 8000
    ocr_device: str = "cpu"
    model_instances: int = 1
    scan_workers: int = 2
    https_enabled: bool = False
    certificate_file: str = ""
    private_key_file: str = ""
    checksheet_data_directory: str = "backend/data/checksheets"
    log_directory: str = "backend/logs"
    database: DatabaseSetup = field(default_factory=DatabaseSetup)

    def validate(self) -> None:
        if self.deployment_mode not in {"local", "lan"}:
            raise SetupConfigurationError("deployment mode must be local or lan")
        _require_host("server address", self.server_address)
        _require_port("UI port", self.ui_port)
        _require_port("backend port", self.backend_port)
        if self.ui_port == self.backend_port:
            raise SetupConfigurationError("UI and backend ports must be different")
        if self.ocr_device != "cpu" and not re.fullmatch(r"gpu:\d+", self.ocr_device):
            raise SetupConfigurationError("OCR device must be cpu or gpu:<index>")
        if not 1 <= self.model_instances <= 8:
            raise SetupConfigurationError("model instances must be between 1 and 8")
        if not 1 <= self.scan_workers <= 32:
            raise SetupConfigurationError("scan workers must be between 1 and 32")
        if self.https_enabled:
            if not self.certificate_file or not self.private_key_file:
                raise SetupConfigurationError(
                    "HTTPS requires both certificate and private-key files"
                )
        if not self.checksheet_data_directory.strip():
            raise SetupConfigurationError("checksheet data directory is required")
        if not self.log_directory.strip():
            raise SetupConfigurationError("log directory is required")
        self.database.validate()

    @property
    def scheme(self) -> str:
        return "https" if self.https_enabled else "http"

    @property
    def bind_host(self) -> str:
        return "127.0.0.1" if self.deployment_mode == "local" else "0.0.0.0"

    def environment_text(self) -> str:
        """Return an ordered production `.env.local` without admin secrets."""

        self.validate()
        rows = [
            "# Generated by the QTracX deployment installer.",
            "# Database administrator credentials are never stored here.",
            "",
            "# Application",
            "QTRACX_ENV=production",
            f"UI_HOST={self.bind_host}",
            f"PORT={self.ui_port}",
            "",
            "# Transport",
            f"HTTPS_ENABLED={'true' if self.https_enabled else 'false'}",
        ]
        if self.https_enabled:
            rows.extend([
                f"TLS_CERT_FILE={self.certificate_file}",
                f"TLS_KEY_FILE={self.private_key_file}",
            ])
        rows.extend([
            "",
            "# Browser to backend",
            (
                "NEXT_PUBLIC_OCR_API_URL="
                f"{self.scheme}://{self.server_address}:{self.backend_port}"
            ),
            "",
            "# Backend and OCR capacity",
            f"OCR_HOST={self.bind_host}",
            f"OCR_PORT={self.backend_port}",
            f"OCR_DEVICE={self.ocr_device}",
            f"OCR_MODEL_INSTANCES={self.model_instances}",
            f"OCR_SCAN_WORKERS={self.scan_workers}",
            "OCR_RELOAD=false",
            "",
            "# Database and authentication",
            f"DATABASE_URL={self.database.url()}",
            "AUTH_SESSION_HOURS=8",
            "",
            "# Storage",
            f"CHECKSHEET_DATA_DIR={self.checksheet_data_directory}",
            "CHECKSHEET_MAX_DOCUMENT_MB=200",
            "",
            "# Logging",
            f"QTRACX_LOG_DIR={self.log_directory}",
            "QTRACX_LOG_LEVEL=INFO",
            "QTRACX_LOG_MAX_MB=25",
            "QTRACX_LOG_BACKUP_COUNT=10",
            "",
        ])
        return "\n".join(rows)

    def write_environment(self, destination: Path, *, overwrite: bool = False) -> None:
        self.validate()
        if destination.exists() and not overwrite:
            raise SetupConfigurationError(
                f"configuration already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.environment_text(), encoding="utf-8")

