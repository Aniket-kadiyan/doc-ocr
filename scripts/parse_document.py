#!/usr/bin/env python3
"""
Process and parse an ENTIRE document (PDF or image) in one shot.

Renders every page, sends each full page to the backend's /ocr/segment
endpoint (auto text-region detection + clustering + full recognize pipeline),
and writes an aggregated JSON of all detected dimensions.

Prereq: the OCR backend must be running (npm run ocr-api, or uvicorn on :8000).

Usage:
    python scripts/parse_document.py "TS2 32224 CUP TURNING DRAWING.pdf"
    python scripts/parse_document.py drawing.png --dpi 300 --out result.json
    python scripts/parse_document.py doc.pdf --api http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import requests


def render_pages(path: Path, dpi: int):
    """Yield (page_index, PIL.Image) for each page. PDFs via pdfium, else one image."""
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        for i in range(len(pdf)):
            yield i, pdf[i].render(scale=dpi / 72).to_pil().convert("RGB")
    else:
        from PIL import Image

        yield 0, Image.open(path).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse an entire document via /ocr/segment")
    ap.add_argument("document", help="Path to a PDF or image file")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="Backend base URL")
    ap.add_argument("--dpi", type=int, default=250, help="Render DPI for PDF pages")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <name>.parsed.json)")
    ap.add_argument("--timeout", type=int, default=600, help="Per-page HTTP timeout (s)")
    args = ap.parse_args()

    src = Path(args.document)
    if not src.exists():
        print(f"error: file not found: {src}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else src.with_suffix(".parsed.json")

    pages_out = []
    total = 0
    for idx, img in render_pages(src, args.dpi):
        buf = io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        t = time.time()
        resp = requests.post(
            f"{args.api}/ocr/segment",
            files={"file": (f"page{idx + 1}.png", buf, "image/png")},
            timeout=args.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        n = data.get("count", 0)
        total += n
        print(f"page {idx + 1}: {n} dimensions  ({time.time() - t:.1f}s)")
        pages_out.append(
            {"page": idx + 1, "size": img.size, "count": n, "regions": data.get("regions", [])}
        )

    result = {"source": src.name, "pages": len(pages_out), "total_dimensions": total, "results": pages_out}
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{total} dimensions across {len(pages_out)} page(s) → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
