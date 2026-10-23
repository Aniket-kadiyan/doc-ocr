# QTracX deployment installer

This top-level folder contains the Windows deployment installer, separately
from the application backend. The installer is being built in layers so the
same validated operations can be used by the graphical wizard, unattended
installation, repair and upgrade modes.

## Implemented foundation

- approved installation defaults in `config/installer.defaults.json`;
- two application-source modes:
  - an offline, dependency-free QTracX source ZIP selected by the installer;
  - the predefined Git repository and branch compiled into installer defaults;
- validated `.env.local` generation in `setup_config.py`;
- read-only host, port, runtime and NVIDIA GPU detection in `preflight.py`;
- CPU mode when no NVIDIA GPU is detected;
- local (`localhost`) and central LAN deployment addressing;
- database defaults `QTracX` and `qtracx_app`;
- one model instance and two scan workers by default; and
- tests that prevent passwords, ports and deployment bindings from being
  rendered incorrectly.

## Clean-machine host bootstrap

`bootstrap.ps1` is the phase-zero Windows entry point. It runs before a Python
environment exists and therefore does not assume that Python, Node.js, npm,
Git, an ODBC driver or application dependencies are installed.

Its default `Plan` mode is read-only, making it safe to test on an existing
development machine:

```powershell
.\installer\bootstrap.ps1 -Mode Plan -Runtime gpu -Database mssql `
    -EnvironmentDir backend\.venv-gpu
```

`Install` mode can install missing 64-bit CPython 3.13, current Node.js LTS,
Git when requested and Microsoft ODBC Driver 18 when MSSQL has no supported
driver. It then delegates to `runtime_setup.py` to create `backend\.venv` for
CPU mode or `backend\.venv-gpu` for GPU mode, install only the matching Python
manifest and run `npm ci`. An explicit `-EnvironmentDir` may override the
mode-specific default.

Plan output includes `environment_action`: `create` for a missing target,
`reuse` for a valid existing environment, or `blocked-invalid-directory` when
the target exists but is not a virtual environment. Plan mode never creates
the directory or installs dependencies.

For MSSQL, an existing 64-bit Driver 18 is preferred and Driver 17 is accepted.
The detected driver name is returned for configuration instead of assuming a
fixed version. If neither is present, the exact WinGet command and official
manual-download page are included in the plan.

Frontend build and OCR-model warmup are explicit options. The build requires
the deployment `.env.local` to exist first because the browser-to-backend URL
is fixed at deployment time. See [HOST_BOOTSTRAP.md](HOST_BOOTSTRAP.md).

## Database bootstrap

`database_bootstrap.py` supports MSSQL through the MIT-0 `pyodbc` driver and
PostgreSQL through the BSD-3-Clause `pg8000` driver. It:

1. tests the supplied database-administrator connection;
2. creates the `QTracX` database when it is missing;
3. creates the `qtracx_app` database login/role when it is missing;
4. grants the permissions required by QTracX and Alembic;
5. verifies the application login independently;
6. applies every Alembic migration and checks the required tables; and
7. creates or repairs the initial QTracX application administrator.

Two provisioning modes are available:

- `installer`: enter `sa`, `postgres`, or another sufficiently privileged
  database account. The installer creates the database and application user.
- `preconfigured`: IT creates the `QTracX` database and a dedicated account
  beforehand. The installer receives only that account's connection details,
  verifies its schema-migration permissions, and never asks for `sa` or a
  PostgreSQL superuser password.

Database-administrator credentials are used only in process memory. The CLI
accepts them through hidden prompts or JSON stdin, never command arguments.
An existing `qtracx_app` password is preserved unless password reset is
explicitly selected. Existing QTracX application administrators are not given
a new password automatically.

Run the temporary interactive setup interface from the repository root:

```powershell
.\backend\.venv-gpu\Scripts\python.exe installer\database_bootstrap.py --interactive
```

Test the selected administrator or preconfigured application connection
without changing the database:

```powershell
.\backend\.venv-gpu\Scripts\python.exe installer\database_bootstrap.py --interactive --check-only
```

Use `.venv` instead of `.venv-gpu` for the CPU environment. The graphical
installer will call the same bootstrap function and show its progress stages.

Database-administrator credentials remain absent from persisted installer
configuration and generated application configuration.

## Run the preflight inventory

From the repository root on Windows:

```powershell
.\backend\.venv-gpu\Scripts\python.exe installer\preflight.py
```

For a machine using the CPU environment:

```powershell
.\backend\.venv\Scripts\python.exe installer\preflight.py
```

Machine-readable output for the graphical installer:

```powershell
.\backend\.venv\Scripts\python.exe installer\preflight.py --json
```

The `gpu_option_available` result is true only when `nvidia-smi` reports an
NVIDIA GPU. The final installation verification will additionally require the
selected Paddle GPU environment to pass its CUDA runtime check.

## Code-source modes

The offline mode accepts a source ZIP containing `backend/`, `package.json`
and the remaining source tree. It rejects unsafe paths and generated content
such as `node_modules`, `.next`, `.venv`, `.venv-gpu`, `.git` and Python cache
folders. It never overwrites a non-empty installation directory.

The Git mode does not accept a repository or branch from the deployment user.
It reads the predefined HTTPS repository and branch from
`config/installer.defaults.json`, performs a shallow single-branch clone and
removes Git metadata from the deployed copy. This keeps release-source control
with the installer build while still allowing a connected installation.

The current build default is branch `MVP2`. Change that build default to the
approved release branch before producing a customer installer.

## Next installer layer

The next layer will add the Windows graphical wizard over the tested host
bootstrap, source, preflight, configuration and database-bootstrap operations.
