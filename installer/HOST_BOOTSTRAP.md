# QTracX Windows host bootstrap

The host bootstrap is the first installer stage that can run on a clean
64-bit Windows machine. It does not assume that Python, Node.js, npm, Git, a
virtual environment, Python packages, JavaScript packages, or an MSSQL ODBC
driver are already present.

The final graphical installer will call the same operations. Until that UI is
implemented, `bootstrap.ps1` provides plan, install and verification modes.

## Safely test on an existing development machine

Plan mode is read-only. It detects the host and prints every action that an
installation would take:

```powershell
Set-ExecutionPolicy -Scope Process Bypass

.\installer\bootstrap.ps1 `
    -Mode Plan `
    -Runtime gpu `
    -Database mssql `
    -EnvironmentDir backend\.venv-gpu
```

For the CPU environment:

```powershell
.\installer\bootstrap.ps1 `
    -Mode Plan `
    -Runtime cpu `
    -Database mssql `
    -EnvironmentDir backend\.venv
```

Plan mode does not install software, create a virtual environment, run pip,
run npm, build the frontend, download OCR models or change the database.

Verify mode is also read-only, but returns a failure when a required runtime,
package set or frontend dependency is unavailable:

```powershell
.\installer\bootstrap.ps1 `
    -Mode Verify `
    -Runtime gpu `
    -Database mssql `
    -EnvironmentDir backend\.venv-gpu
```

## ODBC detection and installation

For MSSQL, the bootstrap inspects the 64-bit ODBC drivers registered on the
host. It selects the newest supported installed driver:

1. `ODBC Driver 18 for SQL Server`
2. `ODBC Driver 17 for SQL Server`

The exact detected name is returned as `selectedDriver` and is passed into the
later configuration/database step. It is not permanently hard-coded to Driver
18. Legacy entries such as `SQL Server` are not selected for a new deployment.

If neither Driver 18 nor 17 is installed, plan mode returns this command:

```powershell
winget install --exact --id Microsoft.msodbcsql.18 `
    --source winget --scope machine --architecture x64 --silent `
    --accept-package-agreements --accept-source-agreements `
    --disable-interactivity
```

Install mode can execute the same command. If WinGet is unavailable, download
the x64 driver from Microsoft's official page:

<https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server>

The Microsoft ODBC Driver is a Microsoft redistributable with Microsoft
license terms; it is not an MIT/BSD/Apache open-source library. It is required
only for MSSQL. PostgreSQL installations use the BSD-licensed `pg8000` driver
and do not require Microsoft ODBC.

## Python and Node.js

The bootstrap requires 64-bit CPython 3.13 and Node.js 20.19 or newer with
npm. If they are missing, plan mode returns the commands and install mode uses:

```powershell
winget install --exact --id Python.Python.3.13 `
    --source winget --scope machine --architecture x64 --silent `
    --accept-package-agreements --accept-source-agreements `
    --disable-interactivity

winget install --exact --id OpenJS.NodeJS.LTS `
    --source winget --scope machine --architecture x64 --silent `
    --accept-package-agreements --accept-source-agreements `
    --disable-interactivity
```

The Node package intentionally follows the currently supported LTS release
instead of installing the now out-of-maintenance Node 20 release on a new
client server. The application still accepts an existing Node 20.19 or newer
runtime.

## Install application dependencies

On a clean deployment, the default environment is `backend\.venv`. The CPU
or GPU Paddle requirement set is installed into that single production
environment according to `-Runtime`.

```powershell
.\installer\bootstrap.ps1 `
    -Mode Install `
    -Runtime cpu `
    -Database mssql
```

Install mode:

1. installs missing host prerequisites through WinGet;
2. refreshes the current process `PATH`;
3. creates `backend\.venv` if it does not exist;
4. installs the selected Python runtime manifest;
5. runs `npm ci` from the checked-in lock file; and
6. verifies the installed Python and frontend dependencies.

It never deletes or silently recreates an existing environment. If the target
directory exists but is not a valid environment, setup stops for review.

## Production frontend build and OCR-model warmup

The frontend backend URL is a deployment-time value embedded in the production
build. Generate `.env.local` first, then add `-BuildFrontend`:

```powershell
.\installer\bootstrap.ps1 `
    -Mode Install `
    -Runtime gpu `
    -Database mssql `
    -BuildFrontend
```

Add `-WarmModels` to initialize PaddleOCR and download/cache its model files.
This is recommended while the deployment server has internet access:

```powershell
.\installer\bootstrap.ps1 `
    -Mode Install `
    -Runtime gpu `
    -Database mssql `
    -BuildFrontend `
    -WarmModels
```

GPU mode is not silently changed to CPU mode. It requires an NVIDIA GPU to be
detected, installs the checked-in CUDA 12.6 Paddle manifest, verifies that the
installed Paddle build has CUDA support, and can validate actual model startup
through `-WarmModels`.

## Testing on a genuinely clean host

Plan mode is the safe test on the current development machine. The complete
install path should additionally be tested in a disposable Windows 11 virtual
machine or Windows Sandbox because an already configured computer cannot prove
that prerequisite installation works.

Use a source-only QTracX ZIP, open an elevated 64-bit PowerShell terminal, run
Plan mode first, then Install mode. The graphical wizard will eventually own
these steps and present the same results without requiring PowerShell commands.
