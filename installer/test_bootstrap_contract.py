from __future__ import annotations

from pathlib import Path
import unittest


class BootstrapContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (
            Path(__file__).resolve().parent / "bootstrap.ps1"
        ).read_text(encoding="utf-8")

    def test_plan_is_the_safe_default(self) -> None:
        self.assertIn('[string]$Mode = "Plan"', self.script)
        self.assertIn('changesApplied = $false', self.script)

    def test_prerequisite_package_ids_are_fixed(self) -> None:
        for package_id in (
            "Python.Python.3.13",
            "OpenJS.NodeJS.LTS",
            "Git.Git",
            "Microsoft.msodbcsql.18",
        ):
            self.assertIn(package_id, self.script)

    def test_mssql_driver_detection_is_64_bit_and_version_aware(self) -> None:
        self.assertIn('$_.Platform -eq "64-bit"', self.script)
        self.assertIn(
            '"^ODBC Driver ([0-9]+) for SQL Server$"',
            self.script,
        )
        self.assertIn('supportedMajorVersions = @(18, 17)', self.script)

    def test_runtime_setup_is_delegated_after_python_detection(self) -> None:
        self.assertIn('$PSScriptRoot "runtime_setup.py"', self.script)
        self.assertIn('Invoke-QTracXRuntimeSetup', self.script)

    def test_missing_gpu_is_not_silently_changed_to_cpu(self) -> None:
        self.assertIn('Supported NVIDIA GPU and driver', self.script)
        self.assertNotIn('$Runtime = "cpu" # fallback', self.script)

    def test_host_only_mode_can_prepare_a_machine_before_source_acquisition(self) -> None:
        self.assertIn("[switch]$HostOnly", self.script)
        self.assertIn(
            'applicationSetup = $(if ($HostOnly) { "skipped" }',
            self.script,
        )
        self.assertIn(
            'if (-not $HostOnly) {\n        Invoke-QTracXRuntimeSetup',
            self.script,
        )


if __name__ == "__main__":
    unittest.main()
