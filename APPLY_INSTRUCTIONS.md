# QTracX MVP2 — Chitra Port Batch 3

This ZIP adds human-readable diagnostic reports for auto-balloon scan jobs. It
is intended for branch `mvp2-chitra-new-feature-port` after commit `dd4c204`.

It does not replace MVP2's model pool, inference gate, scheduler, queue,
authentication, network handling, frontend, detection rules or exports.

## Included functionality

- One report per completed, failed or cancelled scan under
  `backend/logs/runs/`.
- Queue delay, stage timings and operation timings.
- OCR model-call duration and time waiting for model-bundle capacity.
- Result counts, observed backend CPU/RAM and overlapping scan jobs.
- A compact `backend/logs/runs/index.log` summary.
- Automatic retention, configurable with `QTRACX_RUN_REPORT_KEEP` and
  defaulting to 300 reports.
- Privacy controls that exclude drawing content, filenames, recognized values,
  coordinates, users, credentials and raw exception text.

`psutil` is added as a direct dependency. It is BSD-3-Clause licensed.

## Apply

1. Confirm you are on `mvp2-chitra-new-feature-port` and your working tree is
   clean.
2. Extract this ZIP into the repository root.
3. Allow the included files to overwrite their existing versions.
4. Keep `APPLY_INSTRUCTIONS.md` outside the application source if desired.

## Dependency check

From `doc-ocr\backend` in PowerShell:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install "psutil>=5.9,<8"
```

The package is usually already present with the Paddle environment, but it is
now declared directly so a fresh installation is reproducible.

## Tests

From `doc-ocr\backend`:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_run_report.py test_inference_gate.py test_scan_jobs.py `
  test_config_env.py test_app_logging.py
```

Expected result: `35 passed`.

Then run the model-pool and detection regression set:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_ocr_inference_gate.py test_ocr_model_pool.py test_ocr_capacity.py `
  test_ocr_runtime.py test_phi_topology.py test_gdt_detect.py `
  test_gdt_pipeline.py test_page_scan.py test_text_layer_scan.py `
  test_fcf_hairline.py
```

Expected result: `241 passed, 4 subtests passed`.

## Manual verification

1. Start the backend normally.
2. Run an auto-balloon page or selection scan.
3. Open `backend\logs\runs\index.log`.
4. Open the report named on its latest line.
5. Confirm that stage timing and model-capacity timing are present and no
   recognized drawing values appear.

## Suggested commit message

`feat(logging): add private per-scan performance reports`
