from __future__ import annotations

import json
from pathlib import Path
import re
import unittest
from urllib.parse import urlparse


class PrerequisiteCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.installer_root = Path(__file__).resolve().parent
        cls.catalog = json.loads(
            (cls.installer_root / "config" / "prerequisites.windows-x64.json")
            .read_text(encoding="utf-8")
        )

    def test_catalog_contains_every_clean_host_prerequisite(self) -> None:
        self.assertEqual(
            set(self.catalog["packages"]),
            {"python", "node", "git", "mssqlOdbc"},
        )

    def test_downloads_are_exact_https_urls(self) -> None:
        for key, package in self.catalog["packages"].items():
            with self.subTest(package=key):
                parsed = urlparse(package["officialDownloadUrl"])
                self.assertEqual(parsed.scheme, "https")
                self.assertTrue(parsed.netloc)
                self.assertIn(package["version"], package["displayName"])
                self.assertFalse(package["officialDownloadUrl"].endswith("/download"))

    def test_published_hashes_are_valid_sha256_values(self) -> None:
        for key, package in self.catalog["packages"].items():
            digest = package["sha256"]
            if not digest:
                self.assertEqual(key, "mssqlOdbc")
                self.assertEqual(
                    package["authenticodePublisher"],
                    "Microsoft Corporation",
                )
                continue
            self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))

    def test_node_fresh_install_uses_supported_lts(self) -> None:
        node = self.catalog["packages"]["node"]
        major = int(node["version"].split(".", 1)[0])
        self.assertGreaterEqual(major, 24)

    def test_bundle_preparation_script_does_not_run_by_default(self) -> None:
        script = (self.installer_root / "prepare-prerequisites.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('[string]$Mode = "Plan"', script)
        self.assertIn("Get-FileHash", script)
        self.assertIn("prerequisites.lock.json", script)


if __name__ == "__main__":
    unittest.main()
