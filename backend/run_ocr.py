"""
Child-process OCR runner — invoked by the (external) Node Express backend.

Instead of running the FastAPI server and talking HTTP, Node spawns this script
as a child process:

    python run_ocr.py recognize /path/to/crop.png
    python run_ocr.py segment   /path/to/crop.png
    python run_ocr.py health

It calls the SAME pipeline functions the FastAPI endpoints use (see main.py)
and writes the result to stdout as **JSON and only JSON** — every other byte
(model banners, PaddleOCR/Paddle logs, warnings, tracebacks) is forced to
stderr so Node can `JSON.parse` stdout safely.

Exit code 0 on success, non-zero on failure. On failure stdout still carries a
single JSON object: {"ok": false, "error": "..."}.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from typing import Any

# Make the flat backend imports (ocr_pipeline, debug_dump, ...) resolve no
# matter what working directory Node launches us from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_env import load_env_files  # noqa: E402  (after sys.path setup)

# Same root .env.local / .env the server and frontend use.
load_env_files()


@contextlib.contextmanager
def _stdout_to_stderr():
    """Redirect fd 1 -> fd 2 so C-level library prints never touch real stdout.

    Yields the saved original stdout as a text stream to write the final JSON to.
    """
    sys.stdout.flush()
    saved_fd = os.dup(1)
    saved_out = os.fdopen(saved_fd, "w", encoding="utf-8", closefd=True)
    os.dup2(2, 1)  # anything written to fd 1 now goes to stderr
    try:
        yield saved_out
    finally:
        sys.stdout.flush()
        os.dup2(saved_out.fileno(), 1)  # restore (harmless; we're about to exit)
        saved_out.flush()


# Mirrors main.classify_dimension — duplicated here to keep this runner free of
# any FastAPI dependency (the whole point: Node must not depend on the API).
def _classify_dimension(text: str) -> str:
    t = text.strip()
    if re.search(r"[Øø]", t):
        return "Diameter"
    if re.search(r"\bR\d", t) or re.match(r"^R[\d.]", t):
        return "Radius"
    if re.search(r"°|['\"]", t) and re.search(r"\d", t):
        return "Angle"
    if "±" in t:
        return "Tolerance"
    if re.match(r"^\d+(\.\d+)?", t):
        return "Linear"
    if t:
        return "Note"
    return "Unknown"


def _load_image(path: str):
    from PIL import Image

    with open(path, "rb") as fh:
        raw = fh.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image, raw


def op_recognize(args: argparse.Namespace) -> dict[str, Any]:
    from ocr_pipeline import get_pipeline

    image, raw = _load_image(args.image_path)
    pipeline = get_pipeline()
    result = pipeline.recognize(
        image,
        image_bytes=raw,
        debug_dump=args.debug_dump,
        debug_dump_force=args.debug_dump_force,
    )
    dim_type = result.get("type") or _classify_dimension(result["text"])
    return {**result, "type": dim_type}


def op_segment(args: argparse.Namespace) -> dict[str, Any]:
    from ocr_pipeline import get_pipeline

    image, _raw = _load_image(args.image_path)
    pipeline = get_pipeline()
    seg = pipeline.segment(
        image,
        debug_dump=args.debug_dump,
        debug_dump_force=args.debug_dump_force,
    )
    regions = []
    for r in seg["regions"]:
        dim_type = r.get("type") or _classify_dimension(r["text"])
        regions.append({**r, "type": dim_type})
    return {"count": len(regions), "regions": regions}


def op_health(_args: argparse.Namespace) -> dict[str, Any]:
    from debug_dump import dump_status
    from ocr_pipeline import get_pipeline

    pipeline = get_pipeline()
    return {"status": "ok", **pipeline.status, "debug_dump": dump_status()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ocr.py",
        description="OCR runner for the Node backend (JSON-only stdout).",
    )
    sub = parser.add_subparsers(dest="op", required=True)

    for name in ("recognize", "segment"):
        p = sub.add_parser(name, help=f"{name} an image region")
        p.add_argument("image_path", help="path to the cropped image file")
        p.add_argument("--debug-dump", action="store_true")
        p.add_argument("--debug-dump-force", action="store_true")

    sub.add_parser("health", help="pipeline status + model warmup")
    return parser


_DISPATCH = {
    "recognize": op_recognize,
    "segment": op_segment,
    "health": op_health,
}


def main() -> int:
    # argparse errors go to stderr and exit(2) — stdout stays clean.
    args = build_parser().parse_args()

    with _stdout_to_stderr() as real_stdout:
        try:
            result = _DISPATCH[args.op](args)
            payload, code = result, 0
        except Exception as exc:  # noqa: BLE001 — emit any failure as JSON
            import traceback

            traceback.print_exc()  # full trace -> stderr (fd 1 is stderr here)
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            code = 1

        json.dump(payload, real_stdout)
        real_stdout.write("\n")
        real_stdout.flush()

    return code


if __name__ == "__main__":
    sys.exit(main())
