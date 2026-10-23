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

Database-administrator credentials are intentionally absent from persisted
installer configuration. A later setup step will receive them transiently,
create the database/application user, run migrations and then discard them.

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

The next layer will add database bootstrap commands for MSSQL and PostgreSQL,
including connection testing, creation of the `QTracX` database and
`qtracx_app` login, Alembic migrations and initial QTracX administrator setup.
The graphical wizard will call those commands rather than implementing SQL in
its UI.
