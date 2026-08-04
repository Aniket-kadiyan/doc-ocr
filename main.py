"""
ASGI entry when running from project root:

    uvicorn main:app --reload --port 8000

Implementation lives in backend/main.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_backend_main = _backend_dir / "main.py"
_spec = importlib.util.spec_from_file_location("doc_ocr_backend", _backend_main)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load backend from {_backend_main}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

app = _module.app
