from __future__ import annotations

from pathlib import Path
import unittest


class WizardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (
            Path(__file__).resolve().parent / "wizard.ps1"
        ).read_text(encoding="utf-8")

    def test_wizard_is_built_from_windows_inbox_components(self) -> None:
        self.assertIn("System.Windows.Forms", self.script)
        self.assertIn("System.Drawing", self.script)
        self.assertNotIn("Install-Module", self.script)

    def test_wizard_offers_required_source_database_and_runtime_modes(self) -> None:
        for expected in (
            "Offline source ZIP",
            "Predefined Git branch",
            "Microsoft SQL Server",
            "PostgreSQL",
            "CPU",
            "GPU 0",
        ):
            self.assertIn(expected, self.script)

    def test_host_prerequisites_run_before_source_acquisition(self) -> None:
        host_stage = self.script.index(
            "Installing or verifying host prerequisites"
        )
        source_stage = self.script.index("Acquiring QTracX application source")
        self.assertLess(host_stage, source_stage)
        self.assertIn(
            'Get-QTracXBootstrapArguments -Mode "Install" -HostOnly',
            self.script,
        )

    def test_secrets_are_passed_through_standard_input(self) -> None:
        self.assertIn("-InputText $payloadJson", self.script)
        self.assertNotIn('"--administrator-password"', self.script)
        self.assertNotIn('"--application-password"', self.script)
        self.assertIn("UseSystemPasswordChar", self.script)

    def test_mssql_named_instances_can_omit_an_explicit_port(self) -> None:
        self.assertIn("Database server / instance", self.script)
        self.assertIn("Use explicit TCP port", self.script)
        self.assertIn(r"jains\SQLEXPRESS", self.script)
        self.assertIn("$null", self.script)
        self.assertIn("SQL Server Browser", self.script)

    def test_installation_finishes_with_verification(self) -> None:
        database_stage = self.script.index(
            "Creating and migrating the QTracX database"
        )
        verify_stage = self.script.index(
            "Verifying the completed QTracX installation"
        )
        self.assertLess(database_stage, verify_stage)

    def test_installation_output_is_streamed_without_freezing_the_form(self) -> None:
        self.assertIn("ReadLineAsync", self.script)
        self.assertIn("-OnOutput", self.script)
        self.assertIn("Application]::DoEvents()", self.script)
        self.assertNotIn("ReadToEndAsync", self.script)

    def test_installation_has_visible_overall_progress_and_a_persistent_log(self) -> None:
        self.assertIn("System.Windows.Forms.ProgressBar", self.script)
        self.assertIn("Stage $StageNumber of $StageCount", self.script)
        self.assertIn("-StageNumber 6 -StageCount 6", self.script)
        self.assertIn('"QTracX\\installer-logs"', self.script)
        self.assertIn("Installer log:", self.script)

    def test_wizard_cannot_close_or_reenter_during_a_running_stage(self) -> None:
        self.assertIn("$form.Add_FormClosing", self.script)
        self.assertIn("$tabs.Add_Selecting", self.script)
        self.assertIn("$EventArguments.Cancel = $true", self.script)
        self.assertIn("$script:Busy = $true", self.script)

    def test_acquired_source_drives_application_setup(self) -> None:
        self.assertIn(
            '$installedBootstrap = Join-Path $installedInstaller "bootstrap.ps1"',
            self.script,
        )
        self.assertIn('"-File", $installedBootstrap', self.script)
        self.assertIn('$installedConfiguration,', self.script)
        self.assertIn('$installedDatabase,', self.script)

    def test_layout_is_dpi_safe_and_numeric_values_are_visible(self) -> None:
        self.assertIn("AutoScaleMode]::Dpi", self.script)
        self.assertIn("$page.AutoScroll = $true", self.script)
        self.assertIn("Drawing.SystemColors]::WindowText", self.script)
        self.assertIn("$number.Value = [decimal]$Value", self.script)
        self.assertIn("$tabs.Anchor", self.script)


if __name__ == "__main__":
    unittest.main()
