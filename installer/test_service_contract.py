from __future__ import annotations

from pathlib import Path
import unittest


class ServiceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parent
        cls.manager = (root / "service_manager.ps1").read_text(encoding="utf-8")
        cls.host = (root / "service_host.ps1").read_text(encoding="utf-8")

    def test_tasks_start_at_boot_and_restart_after_failure(self) -> None:
        self.assertIn("New-ScheduledTaskTrigger -AtStartup", self.manager)
        self.assertIn("-RestartCount 10", self.manager)
        self.assertIn("-RestartInterval (New-TimeSpan -Minutes 1)", self.manager)
        self.assertIn('"QTracX Backend"', self.manager)
        self.assertIn('"QTracX Frontend"', self.manager)

    def test_system_tasks_are_protected_from_unprivileged_source_tampering(self) -> None:
        self.assertIn("Protect-QTracXApplicationRoot", self.manager)
        self.assertIn('"S-1-5-18"', self.manager)
        self.assertIn('"S-1-5-32-544"', self.manager)
        self.assertIn("SetAccessRuleProtection($true, $false)", self.manager)

    def test_health_checks_cover_backend_and_frontend(self) -> None:
        self.assertIn('TrimEnd(\'/\') + "/health"', self.manager)
        self.assertIn("$frontendHealthy", self.manager)
        self.assertIn("$backendHealthy", self.manager)
        self.assertIn("Assert-QTracXTasksRegistered", self.manager)

    def test_lan_install_adds_private_network_firewall_rules(self) -> None:
        self.assertIn("New-NetFirewallRule", self.manager)
        self.assertIn("-Profile Domain,Private", self.manager)
        self.assertIn('$environment["UI_HOST"] -ne "0.0.0.0"', self.manager)

    def test_uninstall_only_unregisters_runtime_integration(self) -> None:
        self.assertIn('"Uninstall" { Unregister-QTracXTasks }', self.manager)
        self.assertIn("application_data_removed = $false", self.manager)
        self.assertNotIn("Remove-Item -Recurse", self.manager)

    def test_service_host_uses_mode_specific_python_and_bounded_logs(self) -> None:
        self.assertIn('if ($Runtime -eq "gpu") { ".venv-gpu" }', self.host)
        self.assertIn("MaximumBytes = 26214400", self.host)
        self.assertIn("Keep = 5", self.host)


if __name__ == "__main__":
    unittest.main()
