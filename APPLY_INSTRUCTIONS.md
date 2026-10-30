# QTracX MVP2 — Chitra Port Batch 1

Apply this package to the `MVP2` branch at commit:

`f2278c41034bb1c5ad4e34caec53a696f992c6df`

No commit has been created or pushed.

## What this batch adds

- **Read GD&T frames** under the existing **Settings** menu. It is enabled by
  default, saved per browser, and passed through the queued scan API. Turning
  it off skips only feature-control-frame reading.
- Exact native-PDF recovery for CAD symbol encodings, special-characteristic
  marks, rotated/diagonal text, stacked deviations and previously rejected but
  clearly dimensional values.
- Safer tolerance extraction for stacked and one-sided deviations, angular
  tolerances, count suffixes, dual units and fit/thread notation.
- Selected-balloon/sidebar synchronization and click-to-edit behavior.
- Escape closes the value editor or pending-value popup without saving.

MVP2's existing model pool, scan scheduler, queue ownership, authentication,
network configuration, checksheets, references, exports and CAD support are
not replaced by Chitra's older equivalents.

## Apply in VS Code / PowerShell

1. Check out the same MVP2 base and make sure unrelated local edits are saved:

   ```powershell
   git switch MVP2
   git status
   ```

2. Extract this ZIP.

3. Copy the extracted `backend` and `src` folders into the repository root and
   allow Windows to replace files with the same names. The two new
   `scanSettings` files will be added automatically.

4. Review the result before committing:

   ```powershell
   git status --short
   git diff --check
   git diff
   ```

## Verification

From the repository root:

```powershell
npm run typecheck
npm run lint
npm run test:frontend
npm run build
```

From `backend`, using your configured environment:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_dimension_digits.py test_pdf_text_layer.py `
  test_text_layer_scan.py test_page_scan.py `
  test_gdt_text.py test_gdt_context.py
```

Validated before packaging:

- frontend typecheck and lint: passed;
- frontend tests: 187 passed;
- production Next.js build: passed;
- targeted native-PDF tests: 80 passed, 9 fixture-based skips;
- text-layer/page/GD&T tests: 107 passed.

The untouched MVP2 base already has three failing assertions in
`test_scan_progress.py`; the same three failures were reproduced in a clean
worktree at the base commit and were not caused by this package.

## Suggested commit message

`feat(detection): port native PDF recovery and configurable GD&T scanning`
