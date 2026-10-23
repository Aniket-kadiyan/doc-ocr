from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from installer.preflight import collect_preflight


class PreflightTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

