# QTracX Windows logging-test cleanup fix

Apply this archive over the current uncommitted configuration-and-logging changes.
It replaces only `backend/test_app_logging.py`; do not revert the other changes.

The fix closes QTracX logging handlers before each `TemporaryDirectory` exits.
This prevents Windows `WinError 32` cleanup failures caused by an open
`qtracx-backend.log` file.

From the repository root, rerun:

```powershell
cd backend
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_config_env.py test_app_logging.py test_client_diagnostics.py `
  test_scan_jobs.py
```

Expected result: `22 passed`.

No production/backend runtime behavior is changed by this hotfix.
