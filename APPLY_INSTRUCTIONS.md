# QTracX MVP2 — Chitra Port Batch 2

This ZIP contains the safe detection-performance changes only. It is intended
for branch `mvp2-chitra-new-feature-port` after commit `ccd4a25`.

It does **not** replace MVP2's model pool, inference gate, job scheduler, queue,
authentication, networking, exports, or frontend.

## Changes included

- Faster GD&T image cleanup without changing detection rules.
- Faster diameter-symbol scoring with an early decisive topology check.
- Avoid the costly whole-page slant pass on very large sheets while retaining
  local angled-region recovery.
- Avoid redundant frame discovery and deskew work for vector-PDF GD&T regions.
- Bound small symbol/frame OCR calls so they do not run unnecessary fallback
  predictions.

## Apply

1. Confirm you are on `mvp2-chitra-new-feature-port` and your work is saved.
2. Extract this ZIP into the repository root.
3. Allow the six files under `backend` to overwrite their existing versions.
4. Do not copy the `APPLY_INSTRUCTIONS.md` file into `backend`.

## Test in PowerShell

From `doc-ocr\backend`:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_phi_topology.py test_gdt_detect.py test_gdt_pipeline.py `
  test_page_scan.py test_text_layer_scan.py test_fcf_hairline.py
```

Expected result for this batch: `226 passed`.

The previously reported Windows-only synthetic embedded-image fixture failure
in `test_pdf_text_layer.py` is deliberately deferred. This batch neither fixes
nor depends on that test.

## Suggested commit message

`perf(detection): port safe Chitra scan optimizations`
