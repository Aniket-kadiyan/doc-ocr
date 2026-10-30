"""Create the QTracX Python/Node runtime after host prerequisites exist.

The Windows ``bootstrap.ps1`` layer installs or locates Python, Node and the
MSSQL ODBC driver. This module owns the repeatable application-level work:
virtual-environment creation, Python dependencies, npm dependencies, optional
frontend build, and optional OCR-model warmup.

Plan and verify modes are read-only. Install mode never deletes or recreates an
existing environment; an incompatible environment is reported for the user or
future graphical installer to resolve explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
from typing import Callable, Mapping, Sequence


MINIMUM_NODE_VERSION = (20, 19, 0)
MINIMUM_NPM_VERSION = (10, 0, 0)
REQUIRED_PYTHON_VERSION = (3, 13)
_NODE_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")
CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str] | None], None]


class RuntimeSetupError(RuntimeError):
    """A safe, user-facing runtime setup failure."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(message)


@dataclass(frozen=True)
class RuntimeSetupRequest:
    application_root: Path
    runtime: str = "cpu"
    environment_dir: Path | None = None
    build_frontend: bool = False
    warm_models: bool = False

    @property
    def root(self) -> Path:
        return self.application_root.resolve()

    @property
    def backend_dir(self) -> Path:
        return self.root / "backend"

    @property
    def environment(self) -> Path:
        configured = self.environment_dir
        if configured is None:
            default_name = ".venv-gpu" if self.runtime == "gpu" else ".venv"
            return self.backend_dir / default_name
        if configured.is_absolute():
            return configured.resolve()
        return (self.root / configured).resolve()

    @property
    def requirements(self) -> Path:
        filename = (
            "requirements-gpu-cu126.txt"
            if self.runtime == "gpu"
            else "requirements.txt"
        )
        return self.backend_dir / filename

    @property
    def environment_python(self) -> Path:
        if os.name == "nt":
            return self.environment / "Scripts" / "python.exe"
        return self.environment / "bin" / "python"

    @property
    def cad_project(self) -> Path:
        return self.root / "tools" / "dwg2dxf" / "Dwg2Dxf.csproj"

    @property
    def cad_runtime_dir(self) -> Path:
        return self.root / "tools" / "dwg2dxf" / "dist" / "win-x64"

    @property
    def cad_converter(self) -> Path:
        return self.cad_runtime_dir / "dwg2dxf-acadsharp.exe"

    def validate_layout(self) -> None:
        if self.runtime not in {"cpu", "gpu"}:
            raise RuntimeSetupError("validation", "runtime must be cpu or gpu")
        required = [
            self.backend_dir,
            self.requirements,
            self.root / "package.json",
            self.root / "package-lock.json",
            self.cad_project,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeSetupError(
                "validation",
                "QTracX source is incomplete: " + ", ".join(missing),
            )
        if self.environment == self.root or self.environment == self.backend_dir:
            raise RuntimeSetupError(
                "validation",
                "the virtual environment must be a dedicated child directory",
            )


def parse_node_version(value: str) -> tuple[int, int, int] | None:
    match = _NODE_VERSION.match(value.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def host_python_status() -> dict[str, object]:
    bits = struct.calcsize("P") * 8
    version = sys.version_info[:3]
    supported = sys.version_info[:2] == REQUIRED_PYTHON_VERSION and bits == 64
    return {
        "available": True,
        "supported": supported,
        "version": ".".join(str(part) for part in version),
        "architecture_bits": bits,
        "executable": sys.executable,
        "required": "CPython 3.13 x64",
    }


def node_status() -> dict[str, object]:
    node = shutil.which("node")
    npm = shutil.which("npm")
    if node is None:
        return {
            "available": False,
            "supported": False,
            "version": None,
            "node": node,
            "npm": npm,
            "npm_version": None,
            "npm_command": None,
            "npm_source": None,
            "required": "Node.js 20.19 or newer with npm 10 or newer",
        }

    # npm.cmd can deliberately redirect to a globally installed npm under
    # %APPDATA%.  A stale user-level npm then overrides the modern npm shipped
    # with Node.js.  Prefer Node's own npm-cli.js so setup is deterministic and
    # does not alter or depend on the user's global npm installation.
    bundled_npm_cli = (
        Path(node).resolve().parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
    )
    npm_command: list[str] | None = None
    npm_source: str | None = None
    if bundled_npm_cli.is_file():
        npm_command = [node, str(bundled_npm_cli)]
        npm_source = "node-bundled"
    elif npm is not None:
        npm_command = [npm]
        npm_source = "path"

    try:
        result = subprocess.run(
            [node, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        version = None
    else:
        version = parse_node_version(result.stdout) if result.returncode == 0 else None

    npm_version = None
    if npm_command is not None:
        try:
            npm_result = subprocess.run(
                [*npm_command, "--version"],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            if npm_result.returncode == 0:
                npm_version = parse_node_version(npm_result.stdout)

    supported = (
        version is not None
        and version >= MINIMUM_NODE_VERSION
        and npm_version is not None
        and npm_version >= MINIMUM_NPM_VERSION
    )
    return {
        "available": version is not None,
        "supported": supported,
        "version": (
            ".".join(str(part) for part in version)
            if version is not None
            else None
        ),
        "node": node,
        "npm": npm,
        "npm_version": (
            ".".join(str(part) for part in npm_version)
            if npm_version is not None
            else None
        ),
        "npm_command": npm_command,
        "npm_source": npm_source,
        "required": "Node.js 20.19 or newer with npm 10 or newer",
    }


def _display_command(arguments: Sequence[str]) -> str:
    def quote(value: str) -> str:
        if not value or any(character.isspace() for character in value):
            return '"' + value.replace('"', '\\"') + '"'
        return value

    return " ".join(quote(str(argument)) for argument in arguments)


def _environment_action(request: RuntimeSetupRequest) -> str:
    if request.environment_python.is_file():
        return "reuse"
    if request.environment.exists():
        return "blocked-invalid-directory"
    return "create"


def _cad_action(request: RuntimeSetupRequest) -> tuple[str, str | None]:
    """Return how the permissively licensed DWG helper will be supplied."""

    if request.cad_converter.is_file():
        return "reuse-bundled", None
    dotnet = shutil.which("dotnet")
    if dotnet:
        return "build", dotnet
    return "blocked-missing-runtime", None


def _npm_command(node: Mapping[str, object]) -> list[str]:
    command = node.get("npm_command")
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise RuntimeSetupError(
            "node",
            "a supported npm command could not be resolved from the Node.js installation",
        )
    return list(command)


def _commands(
    request: RuntimeSetupRequest,
    node: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    environment_python = str(request.environment_python)
    npm = _npm_command(node or node_status())
    actions: list[tuple[str, list[str], Path]] = []
    environment_action = _environment_action(request)
    if environment_action == "blocked-invalid-directory":
        return []
    cad_action, dotnet = _cad_action(request)
    if cad_action == "build" and dotnet is not None:
        actions.append((
            "cad-converter",
            [
                dotnet,
                "publish",
                str(request.cad_project),
                "--configuration",
                "Release",
                "--runtime",
                "win-x64",
                "--self-contained",
                "true",
                "-p:PublishSingleFile=true",
                "-p:DebugType=None",
                "-p:DebugSymbols=false",
                "--output",
                str(request.cad_runtime_dir),
            ],
            request.root,
        ))
    if environment_action == "create":
        actions.append((
            "python-environment",
            [sys.executable, "-m", "venv", str(request.environment)],
            request.root,
        ))
    actions.append((
        "python-dependencies",
        [
            environment_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(request.requirements),
        ],
        request.root,
    ))
    actions.append((
        "frontend-dependencies",
        [*npm, "ci", "--no-audit", "--no-fund"],
        request.root,
    ))
    if request.build_frontend:
        actions.append((
            "frontend-build",
            [*npm, "run", "build"],
            request.root,
        ))
    if request.warm_models:
        actions.append((
            "ocr-model-warmup",
            [environment_python, str(request.backend_dir / "run_ocr.py"), "health"],
            request.backend_dir,
        ))
    return [
        {
            "stage": stage,
            "command": _display_command(argv),
            "cwd": str(cwd),
        }
        for stage, argv, cwd in actions
    ]


def plan_runtime_setup(request: RuntimeSetupRequest) -> dict[str, object]:
    """Return a read-only setup plan for an existing or fresh host."""

    request.validate_layout()
    python = host_python_status()
    node = node_status()
    environment_action = _environment_action(request)
    cad_action, _dotnet = _cad_action(request)
    blockers: list[str] = []
    if not python["supported"]:
        blockers.append("CPython 3.13 x64 is not available")
    if not node["supported"]:
        blockers.append("Node.js 20.19 or newer with npm 10 or newer is not available")
    if environment_action == "blocked-invalid-directory":
        blockers.append(
            "the selected environment directory exists but is not a valid "
            "Python virtual environment"
        )
    if cad_action == "blocked-missing-runtime":
        blockers.append(
            "the bundled Windows DWG converter is missing and a .NET SDK is not available to build it"
        )
    return {
        "status": "ready" if not blockers else "blocked",
        "mode": "plan",
        "changes_applied": False,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "runtime": request.runtime,
        "application_root": str(request.root),
        "environment": str(request.environment),
        "environment_exists": request.environment_python.is_file(),
        "environment_action": environment_action,
        "cad_converter": str(request.cad_converter),
        "cad_converter_action": cad_action,
        "configuration_exists": (request.root / ".env.local").is_file(),
        "requirements": str(request.requirements),
        "python": python,
        "node": node,
        "blockers": blockers,
        "actions": _commands(request, node) if node["supported"] else [],
    }


def _run_command(
    arguments: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    try:
        result = subprocess.run(
            list(arguments),
            cwd=cwd,
            env=None if environment is None else dict(environment),
            check=False,
        )
    except OSError as error:
        raise RuntimeSetupError("command", error.__class__.__name__) from error
    if result.returncode != 0:
        raise RuntimeSetupError("command", f"command exited with {result.returncode}")


def _run_stage(
    stage: str,
    runner: CommandRunner,
    arguments: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    try:
        runner(arguments, cwd, environment)
    except RuntimeSetupError as error:
        raise RuntimeSetupError(stage, str(error)) from error
    except Exception as error:
        raise RuntimeSetupError(stage, error.__class__.__name__) from error


def _validate_prerequisites() -> tuple[dict[str, object], dict[str, object]]:
    python = host_python_status()
    node = node_status()
    if not python["supported"]:
        raise RuntimeSetupError(
            "python",
            "QTracX requires 64-bit CPython 3.13",
        )
    if not node["supported"]:
        raise RuntimeSetupError(
            "node",
            "QTracX requires Node.js 20.19 or newer with npm 10 or newer",
        )
    return python, node


def _environment_probe(request: RuntimeSetupRequest) -> dict[str, object]:
    python = request.environment_python
    if not python.is_file():
        raise RuntimeSetupError(
            "verification",
            f"virtual-environment Python was not found: {python}",
        )
    probe = (
        "import importlib, importlib.metadata, json, struct, sys; "
        "modules=['fastapi','paddle','paddleocr','sqlalchemy','alembic','pyodbc','pg8000']; "
        "[importlib.import_module(name) for name in modules]; "
        "import paddle; "
        "print(json.dumps({'python':sys.version.split()[0],"
        "'bits':struct.calcsize('P')*8,"
        "'paddle':getattr(paddle,'__version__','unknown'),"
        "'paddle_distributions':importlib.metadata.packages_distributions().get('paddle',[]),"
        "'cuda_compiled':bool(paddle.device.is_compiled_with_cuda())}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", probe],
            cwd=request.root,
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeSetupError("verification", error.__class__.__name__) from error
    if result.returncode != 0:
        raise RuntimeSetupError(
            "verification",
            "one or more Python runtime dependencies could not be imported",
        )
    try:
        details = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeSetupError(
            "verification",
            "the Python runtime probe returned an invalid result",
        ) from error
    if details.get("bits") != 64:
        raise RuntimeSetupError("verification", "the virtual environment is not 64-bit")
    if not str(details.get("python", "")).startswith("3.13."):
        raise RuntimeSetupError(
            "verification",
            "the virtual environment does not use Python 3.13",
        )
    if request.runtime == "gpu" and not details.get("cuda_compiled"):
        raise RuntimeSetupError(
            "verification",
            "GPU mode was selected but PaddlePaddle has no CUDA support",
        )
    distributions = {
        str(name).casefold()
        for name in (details.get("paddle_distributions") or [])
    }
    if request.runtime == "gpu" and "paddlepaddle-gpu" not in distributions:
        raise RuntimeSetupError(
            "verification",
            "GPU mode does not contain the paddlepaddle-gpu package",
        )
    if request.runtime == "cpu" and (
        "paddlepaddle" not in distributions or "paddlepaddle-gpu" in distributions
    ):
        raise RuntimeSetupError(
            "verification",
            "CPU mode does not contain the CPU-only paddlepaddle package",
        )
    return details


def verify_runtime_setup(request: RuntimeSetupRequest) -> dict[str, object]:
    """Verify installed packages without modifying the installation."""

    request.validate_layout()
    _python, node = _validate_prerequisites()
    python_details = _environment_probe(request)
    next_package = request.root / "node_modules" / "next" / "package.json"
    if not next_package.is_file():
        raise RuntimeSetupError(
            "verification",
            "frontend dependencies are not installed",
        )
    if request.build_frontend and not (request.root / ".next" / "BUILD_ID").is_file():
        raise RuntimeSetupError(
            "verification",
            "the production frontend build was not found",
        )
    if not request.cad_converter.is_file():
        raise RuntimeSetupError(
            "verification",
            "the self-contained Windows DWG converter was not found",
        )
    return {
        "status": "ok",
        "mode": "verify",
        "changes_applied": False,
        "runtime": request.runtime,
        "environment": str(request.environment),
        "python": python_details,
        "node": node,
        "frontend_dependencies": "installed",
        "frontend_build": (
            "verified" if request.build_frontend else "not-requested"
        ),
        "cad_converter": str(request.cad_converter),
        "cad_converter_status": "verified",
    }


def install_runtime_setup(
    request: RuntimeSetupRequest,
    *,
    runner: CommandRunner = _run_command,
) -> dict[str, object]:
    """Create/reuse the environment and install application dependencies."""

    request.validate_layout()
    _validate_prerequisites()
    if request.build_frontend and not (request.root / ".env.local").is_file():
        raise RuntimeSetupError(
            "configuration",
            "generate .env.local before building the production frontend",
        )
    node = node_status()
    npm = _npm_command(node)

    cad_action, dotnet = _cad_action(request)
    if cad_action == "blocked-missing-runtime":
        raise RuntimeSetupError(
            "cad-converter",
            "the release does not contain the DWG converter and dotnet is unavailable",
        )
    if cad_action == "build" and dotnet is not None:
        request.cad_runtime_dir.mkdir(parents=True, exist_ok=True)
        _run_stage(
            "cad-converter",
            runner,
            [
                dotnet,
                "publish",
                str(request.cad_project),
                "--configuration",
                "Release",
                "--runtime",
                "win-x64",
                "--self-contained",
                "true",
                "-p:PublishSingleFile=true",
                "-p:DebugType=None",
                "-p:DebugSymbols=false",
                "--output",
                str(request.cad_runtime_dir),
            ],
            request.root,
        )

    if request.environment.exists() and not request.environment_python.is_file():
        raise RuntimeSetupError(
            "python-environment",
            "the environment directory exists but is not a valid virtual environment",
        )
    if not request.environment_python.is_file():
        _run_stage(
            "python-environment",
            runner,
            [sys.executable, "-m", "venv", str(request.environment)],
            request.root,
        )

    _run_stage(
        "python-dependencies",
        runner,
        [
            str(request.environment_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(request.requirements),
        ],
        request.root,
    )
    _run_stage(
        "frontend-dependencies",
        runner,
        [*npm, "ci", "--no-audit", "--no-fund"],
        request.root,
    )
    if request.build_frontend:
        _run_stage(
            "frontend-build",
            runner,
            [*npm, "run", "build"],
            request.root,
        )
    if request.warm_models:
        environment = dict(os.environ)
        environment.update({
            "OCR_DEVICE": "gpu:0" if request.runtime == "gpu" else "cpu",
            "OCR_MODEL_INSTANCES": "1",
            "OCR_SCAN_WORKERS": "1",
            "OCR_RELOAD": "false",
        })
        _run_stage(
            "ocr-model-warmup",
            runner,
            [
                str(request.environment_python),
                str(request.backend_dir / "run_ocr.py"),
                "health",
            ],
            request.backend_dir,
            environment,
        )

    # The injected runner used by unit tests deliberately does not create an
    # environment. Production uses the default runner and receives full probe
    # verification here.
    if runner is _run_command:
        verified = verify_runtime_setup(request)
    else:
        verified = {"status": "deferred-to-runner-test"}
    return {
        "status": "ok",
        "mode": "install",
        "changes_applied": True,
        "runtime": request.runtime,
        "environment": str(request.environment),
        "requirements": str(request.requirements),
        "cad_converter": (
            "bundled" if cad_action == "reuse-bundled" else "built"
        ),
        "frontend_build": (
            "created" if request.build_frontend else "not-requested"
        ),
        "ocr_models": "warmed" if request.warm_models else "not-requested",
        "verification": verified,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the QTracX application runtime")
    parser.add_argument(
        "--mode",
        choices=("plan", "install", "verify"),
        default="plan",
    )
    parser.add_argument(
        "--application-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--runtime", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--environment-dir", type=Path)
    parser.add_argument("--build-frontend", action="store_true")
    parser.add_argument("--warm-models", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    request = RuntimeSetupRequest(
        application_root=arguments.application_root,
        runtime=arguments.runtime,
        environment_dir=arguments.environment_dir,
        build_frontend=arguments.build_frontend,
        warm_models=arguments.warm_models,
    )
    try:
        if arguments.mode == "plan":
            result = plan_runtime_setup(request)
        elif arguments.mode == "verify":
            result = verify_runtime_setup(request)
        else:
            result = install_runtime_setup(request)
    except RuntimeSetupError as error:
        print(json.dumps({
            "status": "failed",
            "stage": error.stage,
            "message": str(error),
        }, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"ok", "ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

