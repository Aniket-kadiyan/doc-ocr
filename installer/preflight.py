"""Read-only host inventory for the QTracX deployment installer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str
    details: dict[str, object] | None = None


def _command_version(command: str, *arguments: str) -> tuple[bool, str]:
    executable = shutil.which(command)
    if not executable:
        return False, "not found"
    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, error.__class__.__name__
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, output[0] if output else "no version output"


def detect_nvidia_gpus() -> Check:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return Check(
            "nvidia_gpu",
            "not-detected",
            "No NVIDIA GPU was detected; the installer will offer CPU mode only.",
        )
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return Check(
            "nvidia_gpu",
            "warning",
            "NVIDIA tooling was found but GPU detection failed.",
            {"error_type": error.__class__.__name__},
        )
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not devices:
        return Check(
            "nvidia_gpu",
            "warning",
            "NVIDIA tooling was found but did not report an available GPU.",
        )
    return Check(
        "nvidia_gpu",
        "ok",
        "A supported NVIDIA GPU candidate was detected; GPU mode may be selected.",
        {"devices": devices},
    )


def _port_check(host: str, port: int, label: str) -> Check:
    bind_host = "127.0.0.1" if host.lower() == "localhost" else host
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((bind_host, port))
    except OSError as error:
        return Check(
            f"{label}_port",
            "warning",
            f"{label} port {port} is currently unavailable on {host}.",
            {"error_type": error.__class__.__name__},
        )
    finally:
        sock.close()
    return Check(
        f"{label}_port",
        "ok",
        f"{label} port {port} is available on {host}.",
    )


def collect_preflight(
    repo_root: Path,
    *,
    host: str = "localhost",
    ui_port: int = 3002,
    backend_port: int = 8000,
) -> dict[str, object]:
    root = repo_root.resolve()
    checks: list[Check] = []
    required = [root / "backend", root / "package.json", root / "installer"]
    missing = [str(path) for path in required if not path.exists()]
    checks.append(Check(
        "application_layout",
        "failed" if missing else "ok",
        "QTracX application files are incomplete." if missing else "QTracX application layout is present.",
        {"missing": missing} if missing else None,
    ))

    is_windows = platform.system().lower() == "windows"
    checks.append(Check(
        "operating_system",
        "ok" if is_windows else "warning",
        f"Detected {platform.system()} {platform.release()} ({platform.machine()}).",
    ))
    checks.append(Check(
        "python",
        "ok" if sys.version_info[:2] == (3, 13) else "warning",
        f"Detected Python {platform.python_version()} at {sys.executable}.",
        {"required": "CPython 3.13 x64"},
    ))

    node_ok, node_version = _command_version("node", "--version")
    checks.append(Check(
        "node",
        "ok" if node_ok else "warning",
        f"Node.js {node_version}" if node_ok else "Node.js was not detected.",
    ))
    git_ok, git_version = _command_version("git", "--version")
    checks.append(Check(
        "git",
        "ok" if git_ok else "not-installed",
        (
            f"{git_version}; predefined-branch installation is available."
            if git_ok
            else "Git was not detected; offline source ZIP installation remains available."
        ),
    ))
    checks.append(detect_nvidia_gpus())

    paddle_installed = importlib.util.find_spec("paddle") is not None
    checks.append(Check(
        "paddle_runtime",
        "ok" if paddle_installed else "not-installed",
        "Paddle is installed in the selected Python environment."
        if paddle_installed
        else "Paddle is not installed in the selected Python environment yet.",
    ))
    checks.append(_port_check(host, ui_port, "ui"))
    checks.append(_port_check(host, backend_port, "backend"))

    failed = sum(check.status == "failed" for check in checks)
    warnings = sum(check.status == "warning" for check in checks)
    gpu_available = any(
        check.name == "nvidia_gpu" and check.status == "ok" for check in checks
    )
    return {
        "status": "failed" if failed else "ready",
        "gpu_option_available": gpu_available,
        "code_source_modes": {
            "archive": True,
            "git": git_ok,
        },
        "defaults": {
            "server_address": host,
            "ui_port": ui_port,
            "backend_port": backend_port,
            "database_name": "QTracX",
            "model_instances": 1,
            "scan_workers": 2,
        },
        "summary": {
            "failed": failed,
            "warnings": warnings,
            "checks": len(checks),
        },
        "checks": [asdict(check) for check in checks],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a QTracX deployment host")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--ui-port", type=int, default=3002)
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = collect_preflight(
        arguments.repo_root,
        host=arguments.host,
        ui_port=arguments.ui_port,
        backend_port=arguments.backend_port,
    )
    if arguments.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            print(f"[{check['status'].upper():>13}] {check['message']}")
        print(
            f"\nResult: {report['status']} | "
            f"GPU option: {'available' if report['gpu_option_available'] else 'hidden'}"
        )
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

