"""
Lightweight .env loader + typed getters — no third-party dependency.

Reads the project's root .env.local / .env so the Python backend draws from the
SAME config file as the Next.js frontend. Precedence matches Next.js:

    real environment  >  .env.local  >  .env

(an already-set variable is never overwritten — that's why we use setdefault).
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env_files() -> None:
    """Populate os.environ from .env.local then .env (without overriding)."""
    for name in (".env.local", ".env"):
        path = _REPO_ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if not key:
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            os.environ.setdefault(key, val)


def get_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def get_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val.strip() if val and val.strip() else default


def get_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    try:
        return int(val) if val and val.strip() else default
    except ValueError:
        return default


def get_cors_origins() -> list[str] | None:
    """Explicit comma-separated allow-list, or None to fall back to a dev regex."""
    raw = os.environ.get("OCR_CORS_ORIGINS", "").strip()
    if not raw:
        return None
    return [o.strip() for o in raw.split(",") if o.strip()]
