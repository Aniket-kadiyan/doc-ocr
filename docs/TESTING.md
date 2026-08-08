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
- Section auto-balloon localization retains its morphology proposal pass,
  bounded detector passes, grouping, and capped local refinements.
- Whole-page auto-ballooning bypasses the section cascade: every page is split
  into four to eight adaptive overlapping panels, with two bounded standalone
  detector calls (`0°` and `90°`) per panel.
- Repeated-cell table grids produce page-coordinate hard-exclusion masks while
  isolated drawing rectangles remain unmasked. Mixed section selections reject
  only candidates inside those masks.
- Whole-page crops use standalone orientation and recognition modules in
  batches; the slow full OCR pipeline is not a permitted fallback.
- Whole-page eligibility rules are isolated in
  `backend/page_value_filters.py`; enabled never-balloon exclusions run before
  the numeric-component requirement.
- Scan results remain hidden until success while heartbeat, liveness,
  panel/pass/batch counters, elapsed time, stage-rate ETA, and temporary debug
  geometry remain observable.

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
- Start `npm run ocr-api` and verify `/health` reports `paddleocr: true`,
  `text_detector: true`, `text_recognizer: true`,
  `text_line_orientation: true`, and `page_batch_recognition: true`. Whole-page
  scanning must fail clearly if a standalone page model is unavailable; it
  must not fall back to per-object full-pipeline calls.
- Read at least one horizontal dimension, vertical dimension, diameter,
  decimal tolerance, and DMS angle from representative drawings.
- Verify an OCR failure still opens an editable empty Value dialog without
  moving the user-drawn box.

### Auto-balloon progress and liveness

- Scan the same small and large sections used before the page-orchestration
  change. Confirm their object boxes and OCR results are unchanged and each
  blocking detection pass is named before PaddleOCR begins it.
- Scan a whole page and confirm layout analysis visibly creates 4–8 adaptive
  overlapping panels regardless of render resolution. Each panel runs `0°`
  then `90°`; no page may be processed as one full-page panel. Whole-page mode
  must not show morphology, section grouping, local coverage refinement, or
  cluster refinement stages.
- Confirm the temporary overlay shows red table masks, blue pending panels,
  one bright-green active panel, muted completed panels, amber overlap, neutral
  detections, green eligible values, orange exclusions, and grey unread values.
  It must stay aligned at every zoom, disappear after success, and remain with
  a Dismiss action after failure.
- Confirm neighbouring atomic boxes are never merged; only similar-size,
  same-position repeats across panel overlap or rotation may be deduplicated.
- Confirm elapsed time and current-step time continue advancing throughout the
  job. Before a comparable unit completes, the UI must say **Calculating
  estimate**. Afterwards it shows **Stage ETA**; it must not extrapolate total
  runtime from the global weighted percentage.
- Confirm the heartbeat normally stays current. A slow unit may change to
  **Long-running step**. A current heartbeat with unusually old progress must
  read **Service responsive — slow progress**; **Heartbeat missing — possibly
  stalled** is reserved for a stale service heartbeat. None of these warnings
  cancels the scan or commits partial balloons.
- Confirm cross-tile deduplication removes repeated overlap views only when the
  boxes have similar position and size. A large containing box and its smaller
  child objects must all survive for review.
- Confirm recognition reports `Batch N/M` and uses the `batch_recognition`
  profile with an initial batch size of 16. Standalone orientation and
  recognition receive crop lists; the complete PaddleOCR pipeline, prefix OCR,
  consensus variants, text-bbox fallback, and angle completion remain disabled.
- Confirm the completion banner reports detected, recognized, eligible,
  excluded, and unread counts, with
  `detected = eligible + excluded + unread` and
  `recognized = eligible + excluded`. Unread detections remain diagnostic and
  must not create balloons in whole-page mode.
- Confirm whole-page filtering accepts values such as `50`, `.25`, `M8`,
  `SS304`, `R0.2 MAX`, and standalone `2:1` without requiring units, leaders,
  arrows, or other geometry. Pure text must be excluded.
- Confirm `DETAIL B SCALE 2:1`, dates, revision entries, and drawing/document/
  part/sheet metadata are excluded. A split value such as `TS232224` is
  excluded when its nearby recognized label is `PART NUMBER`, but accepted
  without that association.
- Confirm the upper specification grid, revision history, title block, and
  bottom tolerance grid produce no balloons in whole-page or section mode.
  Select an area containing both a table and a legitimate drawing value: only
  the table part is red/masked, and the drawing value must still be processed.
- Disable one whole-page rule in `backend/page_value_filters.py`, restart the
  backend, and confirm only that rule changes. Section scans bypass those
  whole-page rules but still obey the global table exclusion.
- Confirm finalization is visible and all balloons still appear together only
  after the complete job succeeds.
- During a section scan, confirm the banner advances through bridge filtering,
  spatial grouping, fragment merging, orientation splitting, bounded local
  refinement, and overlap merging. The current candidate count must remain
  visible instead of leaving the banner unchanged for the entire stage.
- Oversized-cluster refinement may run at most eight standalone detector calls.
  If more oversized candidates exist, or the standalone detector is unavailable,
  the remaining candidates must be retained without launching full OCR inside
  grouping.

## Milestone 2B performance gate

Performance is now a prerequisite for the remaining auto-ballooning accuracy
milestones. On the actual deployment Windows machine, complete click-to-balloon
insertion must take no more than **10 minutes** for the agreed representative
worst-case drawing and maximum page resolution; **5 minutes or less** is the
preferred target. Use a warm OCR service, exclude manual review time, and retain
at least the accepted detection/recognition accuracy. Neither limit is an
automatic cancellation timeout. For whole-page runs record total time plus
layout analysis, masked panel detection, atomic deduplication, batched
recognition, filtering, and finalization time so the next optimization targets measured work. The
section-only morphology and refinement timings are not part of this route.

Frontend job cancellation remains queued after this detection correction and
is not part of the current acceptance run.

## Explicitly deferred until after auto-ballooning

These observed behaviors are recorded but intentionally not changed during the
current auto-ballooning milestones:

1. Editing the separate Tolerance field does not yet rewrite a tolerance already
   embedded in Value.
2. Export validation evaluates committed annotation data; exporting while the
   latest editor changes are still unsaved can miss those pending changes.
3. Clicking the currently selected balloon does not deselect it.
4. Balloons cannot yet be reordered by the user. Reordering must preserve
   contiguous `1...N` numbering in the drawing, sidebar, project, and exports.
Do not mark tests for these four behaviors as passing until their later fixes are
implemented.

## Reporting a failure

Record the test command or manual case, expected behavior, actual behavior,
screenshot/error, and affected project/export file. Keep the original project
file unchanged and reproduce with a copy whenever possible.
