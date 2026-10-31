from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from installer.database_bootstrap import (
    DatabaseBootstrapError,
    DatabaseBootstrapRequest,
    _parser,
    _detected_mssql_driver,
    _diagnose_database_error,
    _mssql_connection_string,
    _odbc_value,
    _postgresql_string,
    _sql_string,
    bootstrap_database,
    check_database_connection,
)
from installer.setup_config import SetupConfigurationError


def _payload(engine: str = "mssql") -> dict[str, object]:
    return {
        "database": {
            "engine": engine,
            "provisioningMode": "installer",
            "host": "localhost",
            "port": 1433 if engine == "mssql" else 5432,
            "name": "QTracX",
            "administratorUser": "sa" if engine == "mssql" else "postgres",
            "administratorPassword": "admin-secret",
            "applicationUser": "qtracx_app",
            "applicationPassword": "app-secret",
            "odbcDriver": "ODBC Driver 17 for SQL Server",
            "encrypt": True,
            "trustServerCertificate": True,
        },
        "initialAdministrator": {
            "username": "admin",
            "displayName": "QTracX Administrator",
            "password": "login-secret",
        },
    }


class DatabaseBootstrapTests(unittest.TestCase):
    def test_cli_accepts_an_explicit_application_root(self) -> None:
        arguments = _parser().parse_args(["--repo-root", "C:/QTracX"])
        self.assertEqual(arguments.repo_root, Path("C:/QTracX"))

    def test_detected_odbc_driver_is_used_for_interactive_default(self) -> None:
        with patch(
            "installer.preflight.detect_database_drivers",
            return_value=(["ODBC Driver 17 for SQL Server"], True),
        ):
            self.assertEqual(
                _detected_mssql_driver(),
                "ODBC Driver 17 for SQL Server",
            )

    def test_request_contains_approved_defaults_without_exposing_secrets(self) -> None:
        request = DatabaseBootstrapRequest.from_mapping(_payload())
        summary = request.safe_summary()
        self.assertEqual(summary["database"], "QTracX")
        self.assertEqual(summary["application_user"], "qtracx_app")
        combined = repr(request) + repr(summary)
        self.assertNotIn("admin-secret", combined)
        self.assertNotIn("app-secret", combined)
        self.assertNotIn("login-secret", combined)

    def test_database_admin_and_application_user_must_differ(self) -> None:
        payload = _payload()
        payload["database"]["administratorUser"] = "qtracx_app"
        with self.assertRaisesRegex(SetupConfigurationError, "must be different"):
            DatabaseBootstrapRequest.from_mapping(payload)

    def test_preconfigured_mode_does_not_require_sa_credentials(self) -> None:
        payload = _payload()
        payload["database"]["provisioningMode"] = "preconfigured"
        payload["database"].pop("administratorUser")
        payload["database"].pop("administratorPassword")
        request = DatabaseBootstrapRequest.from_mapping(payload)
        self.assertEqual(request.provisioning_mode, "preconfigured")
        self.assertEqual(request.administrator_user, "")
        self.assertNotIn("admin-secret", repr(request))

    def test_named_instance_connection_omits_a_fake_none_port(self) -> None:
        payload = _payload()
        payload["database"]["host"] = r"jains\SQLEXPRESS"
        payload["database"]["port"] = None
        request = DatabaseBootstrapRequest.from_mapping(payload)

        connection_string = _mssql_connection_string(
            request,
            database="master",
        )

        self.assertIn(r"SERVER={jains\SQLEXPRESS}", connection_string)
        self.assertNotIn("None", connection_string)
        self.assertIsNone(request.safe_summary()["port"])

    def test_explicit_mssql_port_uses_odbc_host_comma_port_syntax(self) -> None:
        payload = _payload()
        payload["database"]["host"] = "sql-server"
        payload["database"]["port"] = 51433
        request = DatabaseBootstrapRequest.from_mapping(payload)

        connection_string = _mssql_connection_string(
            request,
            database="master",
        )

        self.assertIn("SERVER={sql-server,51433}", connection_string)

    def test_common_database_errors_get_actionable_sanitized_guidance(self) -> None:
        cases = (
            (RuntimeError("IM002 selected driver missing"), "OdbcDriverUnavailable"),
            (RuntimeError("08001 ERROR 26"), "DatabaseServerUnreachable"),
            (RuntimeError("28000 18456"), "DatabaseLoginRejected"),
            (PermissionError("secret detail"), "DatabasePermissionDenied"),
        )
        for error, expected_type in cases:
            with self.subTest(expected_type=expected_type):
                error_type, guidance = _diagnose_database_error(error)
                self.assertEqual(error_type, expected_type)
                self.assertTrue(guidance)
                self.assertNotIn("secret detail", guidance)

    def test_sql_secret_quoting_does_not_break_literals(self) -> None:
        self.assertEqual(_sql_string("pa'ss"), "N'pa''ss'")
        self.assertEqual(_postgresql_string("pa'ss"), "'pa''ss'")
        self.assertEqual(_odbc_value("pa}ss"), "{pa}}ss}")

    def test_coordinator_runs_setup_migrations_verification_and_admin(self) -> None:
        request = DatabaseBootstrapRequest.from_mapping(_payload())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = root / "backend"
            backend.mkdir()
            (backend / "alembic.ini").write_text("[alembic]\n", encoding="utf-8")
            events: list[tuple[str, str]] = []
            with (
                patch(
                    "installer.database_bootstrap._test_mssql_administrator",
                ) as connection_test,
                patch(
                    "installer.database_bootstrap._bootstrap_mssql",
                    return_value="created",
                ) as database_setup,
                patch("installer.database_bootstrap._upgrade_schema") as upgrade,
                patch(
                    "installer.database_bootstrap._missing_schema_tables",
                    return_value=[],
                ) as schema_check,
                patch(
                    "installer.database_bootstrap._ensure_initial_admin",
                    return_value="created",
                ) as initial_admin,
            ):
                result = bootstrap_database(
                    request,
                    repo_root=root,
                    progress=lambda stage, message: events.append((stage, message)),
                )
        database_setup.assert_called_once_with(request)
        connection_test.assert_called_once_with(request)
        upgrade.assert_called_once()
        schema_check.assert_called_once()
        initial_admin.assert_called_once()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["schema"], "up-to-date")
        self.assertEqual(events[-1][0], "complete")
        self.assertNotIn("app-secret", repr(result))

    def test_connection_only_does_not_run_creation(self) -> None:
        request = DatabaseBootstrapRequest.from_mapping(_payload())
        with (
            patch(
                "installer.database_bootstrap._test_mssql_administrator",
            ) as connection_test,
            patch("installer.database_bootstrap._bootstrap_mssql") as bootstrap,
        ):
            result = check_database_connection(request)
        connection_test.assert_called_once_with(request)
        bootstrap.assert_not_called()
        self.assertEqual(result["connection"], "administrator-verified")

    def test_preconfigured_mode_checks_app_account_without_bootstrap(self) -> None:
        payload = _payload()
        payload["database"]["provisioningMode"] = "preconfigured"
        payload["database"].pop("administratorUser")
        payload["database"].pop("administratorPassword")
        request = DatabaseBootstrapRequest.from_mapping(payload)
        with (
            patch(
                "installer.database_bootstrap._test_mssql_application_database",
            ) as connection_test,
            patch("installer.database_bootstrap._bootstrap_mssql") as bootstrap,
        ):
            result = check_database_connection(request)
        connection_test.assert_called_once_with(request)
        bootstrap.assert_not_called()
        self.assertEqual(
            result["connection"],
            "preconfigured-application-database-verified",
        )

    def test_preconfigured_bootstrap_skips_server_provisioning(self) -> None:
        payload = _payload()
        payload["database"]["provisioningMode"] = "preconfigured"
        payload["database"].pop("administratorUser")
        payload["database"].pop("administratorPassword")
        request = DatabaseBootstrapRequest.from_mapping(payload)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = root / "backend"
            backend.mkdir()
            (backend / "alembic.ini").write_text("[alembic]\n", encoding="utf-8")
            with (
                patch(
                    "installer.database_bootstrap._test_mssql_application_database",
                ) as connection_test,
                patch("installer.database_bootstrap._bootstrap_mssql") as provision,
                patch("installer.database_bootstrap._upgrade_schema"),
                patch(
                    "installer.database_bootstrap._missing_schema_tables",
                    return_value=[],
                ),
                patch(
                    "installer.database_bootstrap._ensure_initial_admin",
                    return_value="created",
                ),
            ):
                result = bootstrap_database(request, repo_root=root)
        connection_test.assert_called_once_with(request)
        provision.assert_not_called()
        self.assertEqual(result["application_database_account"], "preconfigured")

    def test_driver_failure_is_sanitized(self) -> None:
        request = DatabaseBootstrapRequest.from_mapping(_payload())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = root / "backend"
            backend.mkdir()
            (backend / "alembic.ini").write_text("[alembic]\n", encoding="utf-8")
            with (
                patch("installer.database_bootstrap._test_mssql_administrator"),
                patch(
                    "installer.database_bootstrap._bootstrap_mssql",
                    side_effect=RuntimeError("connection included admin-secret"),
                ),
            ):
                with self.assertRaises(DatabaseBootstrapError) as captured:
                    bootstrap_database(request, repo_root=root)
        self.assertEqual(captured.exception.stage, "mssql-database-setup")
        self.assertNotIn("admin-secret", str(captured.exception))
        self.assertIn("RuntimeError", str(captured.exception))


if __name__ == "__main__":
    unittest.main()

