from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from installer.source_acquisition import (
    SourceAcquisitionError,
    install_from_archive,
    load_source_defaults,
)


class SourceAcquisitionTests(unittest.TestCase):
    def test_predefined_git_source_is_fixed_in_build_defaults(self) -> None:
        defaults = load_source_defaults()
        self.assertEqual(
            defaults.repository_url,
            "https://github.com/Aniket-kadiyan/doc-ocr.git",
        )
        self.assertEqual(defaults.branch, "deployment-readiness")

    def test_source_archive_extracts_a_single_root_folder(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "qtracx.zip"
            destination = root / "installed"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("QTracX/package.json", json.dumps({"name": "qtracx"}))
                bundle.writestr("QTracX/backend/main.py", "app = None\n")
                bundle.writestr("QTracX/src/example.ts", "export {};\n")
            result = install_from_archive(archive, destination)
            self.assertEqual(result["mode"], "archive")
            self.assertTrue((destination / "backend" / "main.py").is_file())
            self.assertTrue((destination / "package.json").is_file())

    def test_archive_rejects_dependencies(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "qtracx.zip"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("package.json", "{}")
                bundle.writestr("backend/main.py", "")
                bundle.writestr("node_modules/example/index.js", "")
            with self.assertRaisesRegex(SourceAcquisitionError, "dependencies"):
                install_from_archive(archive, root / "installed")

    def test_archive_rejects_path_traversal(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "qtracx.zip"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "unsafe")
            with self.assertRaisesRegex(SourceAcquisitionError, "unsafe path"):
                install_from_archive(archive, root / "installed")

    def test_existing_installation_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "qtracx.zip"
            destination = root / "installed"
            destination.mkdir()
            (destination / "customer-data.txt").write_text("keep", encoding="utf-8")
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("package.json", "{}")
                bundle.writestr("backend/main.py", "")
            with self.assertRaisesRegex(SourceAcquisitionError, "not empty"):
                install_from_archive(archive, destination)


if __name__ == "__main__":
    unittest.main()

