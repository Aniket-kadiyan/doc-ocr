"""Acquire QTracX application source from an offline ZIP or fixed Git branch."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Sequence
from zipfile import BadZipFile, ZipFile, ZipInfo


_DEFAULTS_PATH = Path(__file__).resolve().parent / "config" / "installer.defaults.json"
_REQUIRED_PATHS = ("backend", "package.json")
_FORBIDDEN_PARTS = {
    ".git",
    ".next",
    ".venv",
    ".venv-gpu",
    "__pycache__",
    "node_modules",
}


class SourceAcquisitionError(RuntimeError):
    """The selected application source cannot be installed safely."""


@dataclass(frozen=True)
class SourceDefaults:
    repository_url: str
    branch: str


def load_source_defaults(path: Path = _DEFAULTS_PATH) -> SourceDefaults:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))["codeSource"]
        repository_url = str(raw["repositoryUrl"]).strip()
        branch = str(raw["branch"]).strip()
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SourceAcquisitionError(
            "The installer code-source defaults are missing or invalid"
        ) from error
    if not repository_url.startswith("https://"):
        raise SourceAcquisitionError("The predefined repository must use HTTPS")
    if not branch or any(character.isspace() for character in branch):
        raise SourceAcquisitionError("The predefined Git branch is invalid")
    return SourceDefaults(repository_url=repository_url, branch=branch)


def _zip_path(info: ZipInfo) -> PurePosixPath:
    normalized = info.filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise SourceAcquisitionError(
            f"The source ZIP contains an unsafe path: {info.filename}"
        )
    if any(part in _FORBIDDEN_PARTS for part in path.parts):
        raise SourceAcquisitionError(
            "The source ZIP contains generated dependencies or repository data: "
            f"{info.filename}"
        )
    unix_mode = info.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise SourceAcquisitionError(
            f"The source ZIP contains an unsupported symbolic link: {info.filename}"
        )
    return path


def _application_root(extracted: Path) -> Path:
    candidates = [extracted]
    children = [
        child
        for child in extracted.iterdir()
        if child.name != "__MACOSX"
    ]
    if len(children) == 1 and children[0].is_dir():
        candidates.insert(0, children[0])
    for candidate in candidates:
        if all((candidate / required).exists() for required in _REQUIRED_PATHS):
            return candidate
    raise SourceAcquisitionError(
        "The application source must contain backend/ and package.json"
    )


def _require_empty_destination(destination: Path) -> None:
    if not destination.exists():
        return
    if not destination.is_dir() or any(destination.iterdir()):
        raise SourceAcquisitionError(
            f"The installation directory is not empty: {destination}"
        )


def install_from_archive(archive: Path, destination: Path) -> dict[str, object]:
    """Validate and safely extract a dependency-free application ZIP."""

    archive = archive.resolve()
    destination = destination.resolve()
    if not archive.is_file():
        raise SourceAcquisitionError(f"Source ZIP was not found: {archive}")
    _require_empty_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="qtracx-source-",
            dir=destination.parent,
        ) as temporary:
            staging = Path(temporary)
            with ZipFile(archive) as bundle:
                infos = bundle.infolist()
                if not infos:
                    raise SourceAcquisitionError("The source ZIP is empty")
                for info in infos:
                    _zip_path(info)
                bundle.extractall(staging)
            source_root = _application_root(staging)
            shutil.copytree(source_root, destination, dirs_exist_ok=True)
    except BadZipFile as error:
        raise SourceAcquisitionError("The selected source file is not a valid ZIP") from error
    return {
        "status": "ok",
        "mode": "archive",
        "source": str(archive),
        "destination": str(destination),
    }


def install_from_predefined_git(
    destination: Path,
    *,
    defaults_path: Path = _DEFAULTS_PATH,
) -> dict[str, object]:
    """Clone the build-configured repository branch into the destination."""

    defaults = load_source_defaults(defaults_path)
    git = shutil.which("git")
    if not git:
        raise SourceAcquisitionError(
            "Git is not installed; use the offline source ZIP mode"
        )
    destination = destination.resolve()
    _require_empty_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="qtracx-git-",
        dir=destination.parent,
    ) as temporary:
        checkout = Path(temporary) / "checkout"
        result = subprocess.run(
            [
                git,
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                defaults.branch,
                "--no-tags",
                defaults.repository_url,
                str(checkout),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            # Git output can include credential-bearing URLs. Do not echo it.
            raise SourceAcquisitionError(
                "Git could not retrieve the predefined QTracX release branch"
            )
        source_root = _application_root(checkout)
        shutil.rmtree(source_root / ".git", ignore_errors=True)
        shutil.copytree(source_root, destination, dirs_exist_ok=True)
    return {
        "status": "ok",
        "mode": "git",
        "repository": defaults.repository_url,
        "branch": defaults.branch,
        "destination": str(destination),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire QTracX application source")
    parser.add_argument("--mode", required=True, choices=("archive", "git"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.mode == "archive":
            if arguments.archive is None:
                raise SourceAcquisitionError(
                    "--archive is required when --mode archive is selected"
                )
            result = install_from_archive(arguments.archive, arguments.destination)
        else:
            result = install_from_predefined_git(arguments.destination)
    except SourceAcquisitionError as error:
        print(json.dumps({"status": "failed", "message": str(error)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

