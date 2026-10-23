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

When `-EnvironmentDir` is omitted, the selected runtime determines the target:

- CPU mode uses `backend\.venv`;
- GPU mode uses `backend\.venv-gpu`.

The application plan now reports `environment_action` as `create`, `reuse`,
or `blocked-invalid-directory`. To prove the fresh-environment path safely on
an already configured GPU development machine, point Plan mode at a path that
does not exist:

```powershell
.\installer\bootstrap.ps1 `
    -Mode Plan `
    -Runtime gpu `
    -Database mssql `
    -EnvironmentDir backend\.venv-gpu-fresh-plan
```

The application plan must report `"environment_action": "create"`, and its
first action must be `python-environment`. Because this is Plan mode, the path
must remain absent:

```powershell
Test-Path backend\.venv-gpu-fresh-plan
```

The expected result is `False`.

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

If neither Driver 18 nor 17 is installed, plan mode returns an exact-version
WinGet command and the direct official installer URL. The clean-host catalog
currently pins Microsoft ODBC Driver 18.7.1.1 x64:

```powershell
winget install --exact --id Microsoft.msodbcsql.18 --version 18.7.1.1 `
    --source winget --scope machine --architecture x64 --silent `
    --accept-package-agreements --accept-source-agreements `
    --disable-interactivity
```

Install mode can execute the same command. If WinGet is unavailable, use the
direct x64 package URL printed by Plan mode. The catalog URL is versioned and
points to Microsoft's download service; the general release page remains:

<https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server>

The Microsoft ODBC Driver is a Microsoft redistributable with Microsoft
license terms; it is not an MIT/BSD/Apache open-source library. It is required
only for MSSQL. PostgreSQL installations use the BSD-licensed `pg8000` driver
and do not require Microsoft ODBC.

## Python and Node.js

The bootstrap accepts 64-bit CPython 3.13 and Node.js 20.19 or newer with
npm 10 or newer. It probes and invokes the `npm-cli.js` bundled with the
selected Node.js installation. This prevents an obsolete user-level npm under
`%APPDATA%` from overriding the supported npm shipped with Node.js.
Fresh installations are pinned to the tested CPython 3.13.14 x64 package and
the supported Node.js 24.22.1 LTS x64 package. Plan output includes the exact
official URL, filename and version for each package, even when the current host
already satisfies the requirement.

```powershell
winget install --exact --id Python.Python.3.13 --version 3.13.14 `
    --source winget --scope machine --architecture x64 --silent `
    --accept-package-agreements --accept-source-agreements `
    --disable-interactivity

winget install --exact --id OpenJS.NodeJS.LTS --version 24.22.1 `
    --source winget --scope machine --architecture x64 --silent `
    --accept-package-agreements --accept-source-agreements `
    --disable-interactivity
```

The Node package intentionally follows the currently supported LTS release
instead of installing the now out-of-maintenance Node 20 release on a new
client server. The application still accepts an existing Node 20.19 or newer
runtime when its bundled npm is version 10 or newer.

## Optional offline prerequisite bundle

Customer installers can include Python, Node.js, Git and MSSQL ODBC setup
packages. The source repository intentionally does not store those binaries.
On the release-build machine, prepare the bundle with:

```powershell
.\installer\prepare-prerequisites.ps1 -Mode Download
```

This downloads only the exact URLs in
`config\prerequisites.windows-x64.json`, validates vendor-published SHA-256
hashes where the vendor publishes one, calculates every resulting file hash,
and writes `installer\prerequisites\prerequisites.lock.json`. Include that
folder in the release installer package.

At deployment time, `bootstrap.ps1` prefers a bundled installer only when its
SHA-256 matches the catalog or generated lock. Otherwise, it reports the
bundle problem and uses exact-version WinGet. If WinGet is unavailable, the
error includes the direct official URL.

Before redistributing third-party binaries, the release owner must review the
applicable redistribution and license terms. In particular, Microsoft ODBC is
a Microsoft redistributable rather than an MIT/BSD/Apache dependency.

## Install application dependencies

On a clean deployment, CPU mode creates `backend\.venv` and GPU mode creates
`backend\.venv-gpu`. Only the requirement set matching `-Runtime` is installed
in that environment.

```powershell
.\installer\bootstrap.ps1 `
    -Mode Install `
    -Runtime cpu `
    -Database mssql
```

Install mode:

1. installs missing host prerequisites through WinGet;
2. refreshes the current process `PATH`;
3. creates the mode-specific virtual environment if it does not exist;
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
