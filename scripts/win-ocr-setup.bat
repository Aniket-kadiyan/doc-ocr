@echo off
REM ---------------------------------------------------------------------------
REM  Windows one-shot setup + run for the PaddleOCR backend.
REM  Run from the repo root:  scripts\win-ocr-setup.bat
REM  Requires Python 3.11+ on PATH (the "py" launcher).
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0\.."

if not exist "backend\.venv\Scripts\python.exe" (
  echo [setup] Creating virtualenv in backend\.venv ...
  python -m venv backend\.venv
  if errorlevel 1 (
    echo [setup] Failed to create venv. Is Python installed? Try: py --version
    exit /b 1
  )
)

echo [setup] Installing backend requirements ^(this can take several minutes^) ...
backend\.venv\Scripts\python -m pip install --upgrade pip
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
if errorlevel 1 (
  echo [setup] Dependency install failed. See the error above.
  exit /b 1
)

echo [setup] Starting OCR API on http://127.0.0.1:8000 ...
node scripts\run-ocr.mjs
endlocal
