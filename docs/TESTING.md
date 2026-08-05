# Testing

The test strategy has two levels:

1. Fast automated regressions that do not load OCR models.
2. Manual and model-loading tests on the validated Windows OCR environment.

## Setup

Install JavaScript and Python development dependencies from the repository
root. Runtime deployments can continue using `backend/requirements.txt`; only
developers need `requirements-dev.txt`. Use Node.js 20.19 or newer so the
Vitest/Vite test toolchain is within its supported engine range.

```bash
npm install
backend/.venv/bin/python -m pip install -r backend/requirements-dev.txt
```

Windows:

```bat
npm install
backend\.venv\Scripts\python -m pip install -r backend\requirements-dev.txt
```

## Commands

| Command | Coverage |
|---|---|
| `npm run lint` | Next.js/TypeScript source through ESLint CLI |
| `npm run typecheck` | TypeScript without emitting files |
| `npm run test:frontend` | Value/tolerance, numbering, state, migration, exports, and balloon geometry |
| `npm run test:backend` | Fast Python parsing, conversion, clustering, and balloon-builder regressions |
| `npm run test:ocr` | OpenCV region refinement and synthetic PaddleOCR integration; may load models |
| `npm run verify` | Lint, typecheck, frontend tests, and fast Python tests |
| `npm run build` | Production Next.js build; run before a release |

`scripts/run-python-tests.mjs` selects
`backend\.venv\Scripts\python.exe` on Windows and
`backend/.venv/bin/python` on macOS/Linux. It falls back to the system Python
only when the repository virtual environment is absent.

## Automated regression boundaries

The fast suite verifies:

- Numeric values default to tolerance `0`.
- Decimal, asymmetric, and DMS tolerances are derived consistently.
- Incomplete tolerance intent remains invalid instead of silently becoming `0`.
- Balloon numbers normalize to a contiguous `1…N` sequence after load/delete.
- Stable annotation IDs, array order, and manually drawn bounding boxes survive
  numbering operations.
- Optional Label, Method, and Tool fields export as empty values when omitted.
- Legacy label-first projects migrate to values-only records and report orphans.
- JSON, CSV, XML, checksheet, and verification data remain values-only.
- The deployed balloon JSON validates, stays anchored at every zoom level, and
  fits multi-digit numbers.
- The developer balloon builder preserves enclosed white regions, round-trips
  one-file styles, and rejects corrupt or unsafe files.

Automated tests do not claim OCR accuracy on real manufacturing drawings or
pixel-perfect browser rendering. Those remain manual acceptance checks.

## Windows manual acceptance checklist

Use representative horizontal, vertical, angular, and low-contrast drawings.
For calculation checks, manually enter the exact Value text so OCR accuracy and
range parsing can be diagnosed independently.

### Value lifecycle

- Draw one value and confirm that the final box exactly matches the selection.
- Create several values; confirm drawing/sidebar numbers are `1…N`.
- Delete a middle value; confirm all remaining values renumber immediately.
- Leave Label, Method, and Tool blank, save, reload, and export successfully.
- Add and remove optional metadata without creating a separate label balloon.
- Save `.docbox.json`, remove the drawing, load the project, and compare boxes,
  page numbers, rotations, values, tolerances, and metadata.
- Load a pre–value-first project and verify mapped labels migrate into value
  metadata while orphan labels produce one warning.

### Tolerance and angle cases

| Value | Expected automatic Tolerance |
|---|---:|
| `0.02` | `0` |
| `58.21±0.05` | `+0.05, -0.05` |
| `25 +0.1 -0.2` | `+0.1, -0.2` |
| `90°±0°30` | Empty/incomplete; final export blocked after save |
| `90°±0°30′` | `+0.5, -0.5` |
| `90°±0°30′15″` | `+0.504167, -0.504167` |

Generate a Digital Checksheet and verify the numeric ranges as well as the
displayed text. Malformed cases such as `25 ±`, `+0.1 +0.2`, and `+0.1 -` must
identify the offending balloon and block final output once the edit is saved.

### Balloon rendering

- Confirm the white number circle, black outline/text, and orange pointer match
  in the drawing and sidebar.
- Confirm the pointer tip touches the value box's top centre.
- Check 50%, 100%, 200%, and 400% zoom; screen size stays approximately
  constant and the pointer remains anchored.
- Confirm numbers `1`, `35`, and `128` fit without clipping.
- Confirm selection adds a blue halo without recolouring the artwork.
- Replace only `public/balloon-style.json`, refresh, and verify old, new, and
  loaded-project balloons all change without changing their data.
- Test a missing, malformed, and corrupt-artwork style file separately; the app
  must remain usable with the built-in fallback.

### OCR integration

- Run `npm run test:ocr` with the complete Windows OCR environment installed.
- Start `npm run ocr-api` and verify `/health` reports PaddleOCR available.
- Read at least one horizontal dimension, vertical dimension, diameter,
  decimal tolerance, and DMS angle from representative drawings.
- Verify an OCR failure still opens an editable empty Value dialog without
  moving the user-drawn box.

## Explicitly deferred until after auto-ballooning

These observed behaviors are recorded but intentionally not changed in the
current cleanup milestone:

1. Editing the separate Tolerance field does not yet rewrite a tolerance already
   embedded in Value.
2. Export validation evaluates committed annotation data; exporting while the
   latest editor changes are still unsaved can miss those pending changes.

Do not mark tests for these two behaviors as passing until their later fixes are
implemented.

## Reporting a failure

Record the test command or manual case, expected behavior, actual behavior,
screenshot/error, and affected project/export file. Keep the original project
file unchanged and reproduce with a copy whenever possible.
