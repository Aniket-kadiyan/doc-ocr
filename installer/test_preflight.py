from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from installer.preflight import collect_preflight, select_mssql_driver


class PreflightTests(unittest.TestCase):
    def test_newest_supported_mssql_driver_is_selected(self) -> None:
        self.assertEqual(
            select_mssql_driver([
                "SQL Server",
                "ODBC Driver 17 for SQL Server",
                "ODBC Driver 18 for SQL Server",
            ]),
            "ODBC Driver 18 for SQL Server",
        )

    def test_existing_driver_17_is_supported(self) -> None:
        self.assertEqual(
            select_mssql_driver(["ODBC Driver 17 for SQL Server"]),
            "ODBC Driver 17 for SQL Server",
        )

    def test_legacy_mssql_driver_is_not_silently_selected(self) -> None:
        self.assertIsNone(select_mssql_driver(["SQL Server"]))

    def test_unvalidated_future_mssql_driver_is_not_silently_selected(self) -> None:
        self.assertIsNone(select_mssql_driver(["ODBC Driver 19 for SQL Server"]))

    def test_missing_application_layout_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            report = collect_preflight(
                Path(temporary),
                ui_port=0,
                backend_port=0,
            )
        self.assertEqual(report["status"], "failed")

    def test_repository_layout_reports_approved_defaults(self) -> None:
        report = collect_preflight(
            Path(__file__).resolve().parent.parent,
            ui_port=0,
            backend_port=0,
        )
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["defaults"]["database_name"], "QTracX")
        self.assertEqual(report["defaults"]["model_instances"], 1)
        self.assertEqual(report["defaults"]["scan_workers"], 2)
        self.assertTrue(report["code_source_modes"]["archive"])
        self.assertIn("mssql", report["database_options"])
        self.assertIn("postgresql", report["database_options"])
        self.assertIn("selected_driver", report["database_options"]["mssql"])
        self.assertIn("install_command", report["database_options"]["mssql"])


if __name__ == "__main__":
    unittest.main()

