# QTracX MVP2 configuration and logging overlay

Base commit: `a8b8b01545281893a58ac02ec404de0b393c1854`

This overlay is for `mvp2-configuration-and-logging`. It adds validated
HTTP/LAN runtime settings, configurable UI/backend ports, structured rotating
backend logs, request correlation, safe frontend diagnostics, and tests. It
does not add HTTPS or change OCR, detection, filtering, or balloon placement.

## Apply

1. Check out `mvp2-configuration-and-logging` and pull its latest changes.
2. Extract this ZIP directly into the `doc-ocr` repository root.
3. Allow the extracted files to replace matching files.
4. Copy the required deployment values from `.env.local.example` to your
   `.env.local`.
5. Run the checks below and perform one manual auto-balloon scan.

```powershell
npm run typecheck
npm run lint
npm run test:frontend
npm run build

cd backend
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_config_env.py test_app_logging.py test_client_diagnostics.py `
  test_scan_jobs.py
```

Logs are written to `backend/logs/qtracx-backend.log` unless
`QTRACX_LOG_DIR` overrides the location.

Suggested commit message:

`feat(diagnostics): add validated runtime config and correlated logging`
