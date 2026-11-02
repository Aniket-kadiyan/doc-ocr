from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import unquote_plus

from installer.setup_config import (
    DatabaseSetup,
    InstallerSetup,
    SetupConfigurationError,
)


class InstallerSetupTests(unittest.TestCase):
    def test_checked_in_defaults_match_approved_values(self) -> None:
        path = Path(__file__).parent / "config" / "installer.defaults.json"
        defaults = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(defaults["network"]["serverAddress"], "localhost")
        self.assertEqual(defaults["network"]["uiPort"], 3002)
        self.assertEqual(defaults["network"]["backendPort"], 8000)
        self.assertEqual(defaults["database"]["name"], "QTracX")
        self.assertEqual(defaults["ocr"]["modelInstances"], 1)
        self.assertEqual(defaults["ocr"]["scanWorkers"], 2)
        self.assertFalse(defaults["demo"]["enabled"])
        self.assertEqual(defaults["demo"]["timezone"], "Asia/Kolkata")
        self.assertEqual(defaults["demo"]["durationHours"], 48)
        self.assertEqual(
            defaults["prerequisites"]["python"]["wingetId"],
            "Python.Python.3.13",
        )
        self.assertEqual(
            defaults["prerequisites"]["mssqlOdbc"]["supportedMajorVersions"],
            [18, 17],
        )

    def test_localhost_is_valid_and_generates_local_bindings(self) -> None:
        setup = InstallerSetup(
            database=DatabaseSetup(application_password="safe password"),
        )
        text = setup.environment_text()
        self.assertIn("UI_HOST=127.0.0.1", text)
        self.assertIn("OCR_HOST=127.0.0.1", text)
        self.assertIn("NEXT_PUBLIC_OCR_API_URL=http://localhost:8000", text)
        self.assertIn("OCR_MODEL_INSTANCES=1", text)
        self.assertIn("OCR_SCAN_WORKERS=2", text)
        self.assertIn("QTRACX_DEMO_LOCK_ENABLED=false", text)
        self.assertIn("QTRACX_DEMO_DURATION_HOURS=48", text)
        self.assertNotIn("QTRACX_DEMO_START_AT=", text)

    def test_demo_selection_writes_ist_start_and_decimal_duration(self) -> None:
        setup = InstallerSetup.from_mapping({
            "database": {"applicationPassword": "safe password"},
            "demo": {
                "enabled": True,
                "startAt": "2026-11-01T10:00:00+05:30",
                "durationHours": "0.25",
            },
        })

        text = setup.environment_text()

        self.assertIn("QTRACX_DEMO_LOCK_ENABLED=true", text)
        self.assertIn(
            "QTRACX_DEMO_START_AT=2026-11-01T10:00:00+05:30",
            text,
        )
        self.assertIn("QTRACX_DEMO_DURATION_HOURS=0.25", text)
        self.assertTrue(setup.safe_summary()["demo_enabled"])

    def test_demo_requires_a_valid_start_time(self) -> None:
        setup = InstallerSetup(
            demo_enabled=True,
            demo_start_at="not-a-date",
            database=DatabaseSetup(application_password="safe password"),
        )

        with self.assertRaisesRegex(SetupConfigurationError, "ISO date/time"):
            setup.validate()

    def test_demo_duration_accepts_decimal_but_rejects_nonpositive_values(self) -> None:
        valid = InstallerSetup(
            demo_enabled=True,
            demo_start_at="2026-11-01T10:00:00+05:30",
            demo_duration_hours=1.5,
            database=DatabaseSetup(application_password="safe password"),
        )
        valid.validate()

        invalid = InstallerSetup(
            demo_duration_hours=0,
            database=DatabaseSetup(application_password="safe password"),
        )
        with self.assertRaisesRegex(SetupConfigurationError, "greater than 0"):
            invalid.validate()

    def test_lan_mode_binds_all_interfaces_but_advertises_server(self) -> None:
        setup = InstallerSetup(
            deployment_mode="lan",
            server_address="qtracx-server",
            database=DatabaseSetup(application_password="safe password"),
        )
        text = setup.environment_text()
        self.assertIn("UI_HOST=0.0.0.0", text)
        self.assertIn("OCR_HOST=0.0.0.0", text)
        self.assertIn(
            "NEXT_PUBLIC_OCR_API_URL=http://qtracx-server:8000",
            text,
        )

    def test_database_password_is_url_encoded(self) -> None:
        database = DatabaseSetup(application_password="a b@c/1")
        self.assertIn("a+b%40c%2F1", database.url())

    def test_mssql_named_instance_can_use_sql_browser_without_a_port(self) -> None:
        database = DatabaseSetup(
            host=r"jains\SQLEXPRESS",
            port=None,
            application_password="safe password",
            odbc_driver="ODBC Driver 17 for SQL Server",
        )

        decoded = unquote_plus(database.url())

        self.assertEqual(database.mssql_server, r"jains\SQLEXPRESS")
        self.assertIn(r"SERVER={jains\SQLEXPRESS}", decoded)
        self.assertNotIn(r"SQLEXPRESS,None", decoded)

    def test_mssql_explicit_port_is_preserved_in_odbc_target(self) -> None:
        database = DatabaseSetup(
            host="sql-server",
            port=51433,
            application_password="safe password",
        )

        self.assertIn("SERVER={sql-server,51433}", unquote_plus(database.url()))

    def test_postgresql_requires_a_port(self) -> None:
        database = DatabaseSetup(
            engine="postgresql",
            port=None,
            application_password="safe password",
        )

        with self.assertRaisesRegex(SetupConfigurationError, "port is required"):
            database.validate()

    def test_postgresql_uses_permissive_pg8000_driver(self) -> None:
        database = DatabaseSetup(
            engine="postgresql",
            port=5432,
            application_password="safe password",
        )
        self.assertTrue(database.url().startswith("postgresql+pg8000://"))

    def test_ui_and_backend_ports_must_be_different(self) -> None:
        setup = InstallerSetup(
            ui_port=8000,
            backend_port=8000,
            database=DatabaseSetup(application_password="safe password"),
        )
        with self.assertRaisesRegex(SetupConfigurationError, "different"):
            setup.validate()

    def test_https_requires_both_files(self) -> None:
        setup = InstallerSetup(
            https_enabled=True,
            certificate_file="certificate.pem",
            database=DatabaseSetup(application_password="safe password"),
        )
        with self.assertRaisesRegex(SetupConfigurationError, "both"):
            setup.validate()

    def test_existing_environment_is_not_overwritten_without_permission(self) -> None:
        setup = InstallerSetup(
            database=DatabaseSetup(application_password="safe password"),
        )
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / ".env.local"
            destination.write_text("existing=true\n", encoding="utf-8")
            with self.assertRaisesRegex(SetupConfigurationError, "already exists"):
                setup.write_environment(destination)

    def test_wizard_mapping_generates_gpu_lan_configuration(self) -> None:
        setup = InstallerSetup.from_mapping({
            "deploymentMode": "lan",
            "serverAddress": "10.10.20.15",
            "uiPort": "3002",
            "backendPort": "8000",
            "runtime": "gpu",
            "modelInstances": "2",
            "scanWorkers": "3",
            "database": {
                "engine": "mssql",
                "host": "sql-server",
                "port": "1433",
                "name": "QTracX",
                "applicationUser": "qtracx_app",
                "applicationPassword": "safe password",
                "odbcDriver": "ODBC Driver 17 for SQL Server",
                "encrypt": True,
                "trustServerCertificate": True,
            },
            "https": {"enabled": False},
        })

        text = setup.environment_text()
        self.assertIn("UI_HOST=0.0.0.0", text)
        self.assertIn("NEXT_PUBLIC_OCR_API_URL=http://10.10.20.15:8000", text)
        self.assertIn("OCR_DEVICE=gpu:0", text)
        self.assertIn("OCR_MODEL_INSTANCES=2", text)
        self.assertIn("odbc_connect=", text)
        self.assertIn(
            "DRIVER={ODBC Driver 17 for SQL Server}",
            unquote_plus(text),
        )

    def test_wizard_mapping_accepts_a_named_instance_without_a_port(self) -> None:
        setup = InstallerSetup.from_mapping({
            "database": {
                "engine": "mssql",
                "host": r"jains\SQLEXPRESS",
                "port": None,
                "applicationPassword": "safe password",
            },
        })

        self.assertIsNone(setup.database.port)
        self.assertIn(
            r"SERVER={jains\SQLEXPRESS}",
            unquote_plus(setup.database.url()),
        )

    def test_safe_summary_never_contains_database_password(self) -> None:
        password = "do-not-show-this"
        setup = InstallerSetup(
            database=DatabaseSetup(application_password=password),
        )

        summary = json.dumps(setup.safe_summary())

        self.assertNotIn(password, summary)
        self.assertNotIn("DATABASE_URL", summary)


if __name__ == "__main__":
    unittest.main()

