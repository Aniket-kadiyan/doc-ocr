from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from installer.runtime_setup import (
    RuntimeSetupError,
    RuntimeSetupRequest,
    _display_command,
    install_runtime_setup,
    parse_node_version,
    plan_runtime_setup,
    verify_runtime_setup,
)


def _source_tree(root: Path) -> None:
    backend = root / "backend"
    backend.mkdir(parents=True)
    (backend / "requirements.txt").write_text("example\n", encoding="utf-8")
    (backend / "requirements-gpu-cu126.txt").write_text(
        "example-gpu\n",
        encoding="utf-8",
    )
    (backend / "run_ocr.py").write_text("pass\n", encoding="utf-8")
    (root / "package.json").write_text("{}\n", encoding="utf-8")
    (root / "package-lock.json").write_text("{}\n", encoding="utf-8")


def _supported_python() -> dict[str, object]:
    return {
        "available": True,
        "supported": True,
        "version": "3.13.14",
        "architecture_bits": 64,
        "executable": "python.exe",
        "required": "CPython 3.13 x64",
    }


def _supported_node() -> dict[str, object]:
    return {
        "available": True,
        "supported": True,
        "version": "24.19.0",
        "node": "node.exe",
        "npm": "npm.cmd",
        "npm_version": "11.9.0",
        "npm_command": ["node.exe", "npm-cli.js"],
        "npm_source": "node-bundled",
        "required": "Node.js 20.19 or newer with npm 10 or newer",
    }


class RuntimeSetupTests(unittest.TestCase):
    def test_node_version_parser(self) -> None:
        self.assertEqual(parse_node_version("v20.19.5"), (20, 19, 5))
        self.assertEqual(parse_node_version("24.1.0"), (24, 1, 0))
        self.assertIsNone(parse_node_version("unknown"))

    def test_plan_is_read_only_and_selects_cpu_requirements(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            request = RuntimeSetupRequest(application_root=root, runtime="cpu")
            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
            ):
                result = plan_runtime_setup(request)

            self.assertEqual(result["status"], "ready")
            self.assertFalse(result["changes_applied"])
            self.assertEqual(result["environment_action"], "create")
            self.assertEqual(request.environment, root / "backend" / ".venv")
            self.assertTrue(str(result["requirements"]).endswith("requirements.txt"))
            self.assertFalse(request.environment.exists())
            stages = [action["stage"] for action in result["actions"]]
            self.assertEqual(
                stages,
                [
                    "python-environment",
                    "python-dependencies",
                    "frontend-dependencies",
                ],
            )

    def test_gpu_defaults_to_a_separate_environment_that_will_be_created(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            request = RuntimeSetupRequest(application_root=root, runtime="gpu")
            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
            ):
                result = plan_runtime_setup(request)

            self.assertEqual(request.environment, root / "backend" / ".venv-gpu")
            self.assertEqual(result["environment_action"], "create")
            self.assertEqual(result["actions"][0]["stage"], "python-environment")

    def test_plan_reports_when_an_existing_environment_will_be_reused(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            request = RuntimeSetupRequest(application_root=root, runtime="gpu")
            request.environment_python.parent.mkdir(parents=True)
            request.environment_python.touch()
            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
            ):
                result = plan_runtime_setup(request)

            self.assertEqual(result["environment_action"], "reuse")
            stages = [action["stage"] for action in result["actions"]]
            self.assertNotIn("python-environment", stages)

    def test_plan_blocks_an_existing_non_environment_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            request = RuntimeSetupRequest(application_root=root, runtime="cpu")
            request.environment.mkdir()
            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
            ):
                result = plan_runtime_setup(request)

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(
                result["environment_action"],
                "blocked-invalid-directory",
            )
            self.assertEqual(result["actions"], [])

    def test_display_command_separates_and_quotes_paths_with_spaces(self) -> None:
        command = _display_command(
            [
                r"G:\job related works\Python313\python.exe",
                "-m",
                "pip",
                "-r",
                r"G:\job related works\doc-ocr\backend\requirements.txt",
            ]
        )

        self.assertEqual(
            command,
            '"G:\\job related works\\Python313\\python.exe" -m pip -r '
            '"G:\\job related works\\doc-ocr\\backend\\requirements.txt"',
        )

    def test_gpu_plan_uses_gpu_manifest_and_can_include_warmup(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            request = RuntimeSetupRequest(
                application_root=root,
                runtime="gpu",
                warm_models=True,
            )
            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
            ):
                result = plan_runtime_setup(request)

            self.assertTrue(
                str(result["requirements"]).endswith(
                    "requirements-gpu-cu126.txt"
                )
            )
            self.assertEqual(result["actions"][-1]["stage"], "ocr-model-warmup")

    def test_install_orchestration_uses_venv_pip_and_npm_ci(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            request = RuntimeSetupRequest(application_root=root, runtime="cpu")
            calls: list[tuple[list[str], Path, object]] = []

            def runner(arguments, cwd, environment) -> None:
                calls.append((list(arguments), cwd, environment))

            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
            ):
                result = install_runtime_setup(request, runner=runner)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0][0][1:3], ["-m", "venv"])
            self.assertIn("requirements.txt", calls[1][0][-1])
            self.assertEqual(
                calls[2][0],
                [
                    "node.exe",
                    "npm-cli.js",
                    "ci",
                    "--no-audit",
                    "--no-fund",
                ],
            )

    def test_existing_non_environment_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            environment = root / "backend" / ".venv"
            environment.mkdir()
            marker = environment / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            request = RuntimeSetupRequest(application_root=root)
            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeSetupError,
                    "not a valid virtual environment",
                ):
                    install_runtime_setup(request, runner=lambda *_args: None)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_frontend_build_requires_generated_environment(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            request = RuntimeSetupRequest(
                application_root=root,
                build_frontend=True,
            )
            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeSetupError, "generate .env.local"):
                    install_runtime_setup(request, runner=lambda *_args: None)

    def test_verify_requires_frontend_dependencies(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_tree(root)
            request = RuntimeSetupRequest(application_root=root)
            with (
                patch(
                    "installer.runtime_setup.host_python_status",
                    return_value=_supported_python(),
                ),
                patch(
                    "installer.runtime_setup.node_status",
                    return_value=_supported_node(),
                ),
                patch(
                    "installer.runtime_setup._environment_probe",
                    return_value={"python": "3.13.14", "bits": 64},
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeSetupError,
                    "frontend dependencies are not installed",
                ):
                    verify_runtime_setup(request)


if __name__ == "__main__":
    unittest.main()
