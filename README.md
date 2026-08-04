

## Full workflow

```text
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER (Next.js)                                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Open PDF/image (PDF.js → canvas)                            │
│  2. Draw Box (Konva) around one dimension                       │
│  3. Crop region + white padding → PNG                           │
│  4. POST /ocr/recognize  ──────────────────────────────┐        │
│  5. Popup: value, type, label → Save                    │        │
│  6. Balloon #1,2,3… + IndexedDB + export JSON/CSV     │        │
└───────────────────────────────────────────────────────│────────┘
                                                        │
                        LOCAL HTTP (no cloud)           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PYTHON API  localhost:8000  (PaddleOCR + OpenCV)               │
├─────────────────────────────────────────────────────────────────┤
│  A. Receive crop PNG                                            │
│  B. Preprocess                                                  │
│       • Upscale small regions                                   │
│       • CAD blue/cyan ink → high-contrast gray                  │
│       • If vertical → rotate 90° for OCR passes                   │
│  C. PaddleOCR (multi-pass)                                      │
│       • Several gray variants × det on/off                      │
│       • Parse text + confidence                                 │
│  D. Symbol vision (OpenCV)                                      │
│       • Template match: ±, °, Ø                                 │
│       • Hough circles for diameter symbol                       │
│       • Run on rotated + original views                         │
│  E. Merge                                                       │
│       • Insert Ø / ± / ° OCR missed                             │
│       • Vertical tolerance heuristic → Ø174.07±0.05             │
│  F. Return JSON → browser popup                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Tesseract is removed.** OCR is **PaddleOCR only** via the local API.

---

## Quick start

### Terminal 1 — OCR API (required)

```bash
cd doc-ocr-box
cd backend && pip install -r requirements.txt && cd ..
uvicorn main:app --reload --port 8000
```

Or, cross-platform, from the repo root: `npm run ocr-api` (auto-picks the
venv Python for your OS).

Check: http://127.0.0.1:8000/health → `"paddleocr": true`

### Terminal 2 — UI

```bash
npm install
npm run dev
```

Open http://localhost:3000 → banner: **PaddleOCR + symbol vision**

---

## Windows

The `.venv` is OS-specific and gitignored — a macOS venv will **not** run on
Windows, so create a fresh one. From the repo root in **PowerShell / cmd**:

```bat
REM One-shot: create venv, install deps, and start the OCR API
scripts\win-ocr-setup.bat
```

Or manually:

```bat
REM Terminal 1 — OCR API (required)
py -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
npm run ocr-api

REM Terminal 2 — UI
npm install
npm run dev
```

`npm run ocr-api` / `ocr-api:debug` / `ocr-api:debug:force` work on Windows,
macOS and Linux (they use `scripts/run-ocr.mjs`, which selects
`.venv\Scripts\python.exe` on Windows and `.venv/bin/python` elsewhere).

> If OCR produces no output but you see **no error**, the backend isn't
> reachable. Open http://127.0.0.1:8000/health — if it doesn't return
> `"paddleocr": true`, the Python API in Terminal 1 isn't running.

---

## Tips for Ø and ±

1. Include the **full symbol** in the box (slightly left of digits for Ø).  
2. For **vertical** dimensions, a tall narrow box is correct — we rotate internally.  
3. If Ø still missing once, type it in the popup — field is editable.

---

## Project layout

```text
src/           Next.js UI (Konva, PDF.js, Dexie)
backend/       PaddleOCR + symbol_vision + image_preprocess
main.py        Run uvicorn from project root
```
