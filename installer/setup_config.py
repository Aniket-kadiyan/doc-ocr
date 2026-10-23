"""Validated setup selections and QTracX environment-file generation.

The future graphical installer and unattended installer both use this module.
Database-administrator passwords are deliberately not represented here: they
are transient setup inputs and must never be written to the generated config.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
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


def _integer(payload: Mapping[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise SetupConfigurationError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise SetupConfigurationError(f"{key} must be an integer") from error


def _boolean(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise SetupConfigurationError(f"{key} must be true or false")
    return value


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise SetupConfigurationError(f"{key} must be an object")
    return value


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

    def safe_summary(self) -> dict[str, object]:
        """Return installer selections without credentials or connection URLs."""

        return {
            "deployment_mode": self.deployment_mode,
            "server_address": self.server_address,
            "ui_port": self.ui_port,
            "backend_port": self.backend_port,
            "protocol": self.scheme,
            "ocr_device": self.ocr_device,
            "model_instances": self.model_instances,
            "scan_workers": self.scan_workers,
            "database_engine": self.database.engine,
            "database_host": self.database.host,
            "database_port": self.database.port,
            "database_name": self.database.name,
            "database_application_user": self.database.application_user,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "InstallerSetup":
        database_payload = _mapping(payload, "database")
        https_payload = _mapping(payload, "https")
        storage_payload = _mapping(payload, "storage")
        engine = str(database_payload.get("engine", "mssql")).strip().lower()
        runtime = str(payload.get("runtime", "cpu")).strip().lower()
        ocr_device = str(
            payload.get(
                "ocrDevice",
                "gpu:0" if runtime == "gpu" else "cpu",
            )
        ).strip().lower()
        database = DatabaseSetup(
            engine=engine,
            host=str(database_payload.get("host", "localhost")).strip(),
            port=_integer(
                database_payload,
                "port",
                1433 if engine == "mssql" else 5432,
            ),
            name=str(database_payload.get("name", "QTracX")).strip(),
            application_user=str(
                database_payload.get("applicationUser", "qtracx_app")
            ).strip(),
            application_password=str(
                database_payload.get("applicationPassword", "")
            ),
            odbc_driver=str(
                database_payload.get(
                    "odbcDriver",
                    "ODBC Driver 18 for SQL Server",
                )
            ).strip(),
            encrypt=_boolean(database_payload, "encrypt", True),
            trust_server_certificate=_boolean(
                database_payload,
                "trustServerCertificate",
                True,
            ),
        )
        setup = cls(
            deployment_mode=str(
                payload.get("deploymentMode", "local")
            ).strip().lower(),
            server_address=str(
                payload.get("serverAddress", "localhost")
            ).strip(),
            ui_port=_integer(payload, "uiPort", 3002),
            backend_port=_integer(payload, "backendPort", 8000),
            ocr_device=ocr_device,
            model_instances=_integer(payload, "modelInstances", 1),
            scan_workers=_integer(payload, "scanWorkers", 2),
            https_enabled=_boolean(https_payload, "enabled", False),
            certificate_file=str(
                https_payload.get("certificateFile", "")
            ).strip(),
            private_key_file=str(
                https_payload.get("privateKeyFile", "")
            ).strip(),
            checksheet_data_directory=str(
                storage_payload.get(
                    "checksheetDataDirectory",
                    "backend/data/checksheets",
                )
            ).strip(),
            log_directory=str(
                storage_payload.get("logDirectory", "backend/logs")
            ).strip(),
            database=database,
        )
        setup.validate()
        return setup

    def write_environment(self, destination: Path, *, overwrite: bool = False) -> None:
        self.validate()
        if destination.exists() and not overwrite:
            raise SetupConfigurationError(
                f"configuration already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.environment_text(), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the validated QTracX deployment environment"
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, Mapping):
            raise SetupConfigurationError("stdin must contain a setup object")
        setup = InstallerSetup.from_mapping(payload)
        setup.write_environment(
            arguments.destination,
            overwrite=arguments.overwrite,
        )
    except (OSError, json.JSONDecodeError, SetupConfigurationError) as error:
        print(json.dumps({
            "status": "failed",
            "message": str(error),
        }))
        return 1
    print(json.dumps({
        "status": "ok",
        "destination": str(arguments.destination.resolve()),
        "configuration": setup.safe_summary(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

