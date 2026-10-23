# QTracX Windows deployment wizard

`wizard.ps1` is the first usable graphical layer over the validated installer
modules. It uses Windows PowerShell and Windows Forms already included with
Windows 11; it does not add a GUI framework or another application dependency.

## Launch

From the installer/source folder in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\installer\wizard.ps1
```

The script requests administrator approval because clean-machine installation
may install Python, Node.js, Git and the MSSQL ODBC driver at machine scope.
The form is DPI-aware, resizable and scrollable so every field remains
available on scaled or lower-resolution displays.

## Information collected

The wizard collects:

- source mode: dependency-free source ZIP or predefined Git branch;
- an empty installation destination;
- central LAN or localhost deployment;
- server address and configurable UI/backend ports;
- CPU mode, or GPU mode when `nvidia-smi` detects an NVIDIA GPU;
- model-instance and scan-worker counts;
- normal HTTP or optional HTTPS certificate/key paths;
- MSSQL or PostgreSQL and installer-managed or preconfigured provisioning;
- the dedicated QTracX database account; and
- the initial in-application QTracX administrator.

Database-administrator and initial application-administrator passwords remain
inside the wizard process and are sent to the database bootstrap through
standard input. They are not included in process arguments or the installer
log. The dedicated application database password is necessarily persisted in
the generated `.env.local` connection string so the backend can connect after
restart.

## Check Plan

`Check plan` is read-only. On an existing development computer, select
`Predefined Git branch`, complete the non-secret server/database fields, and
click `Check plan`. It checks host prerequisites without acquiring source,
creating a virtual environment, installing packages, changing the database or
writing configuration. The output always lists the pinned prerequisite
versions, exact official download URLs and optional bundle status—even when no
host action is required.

## Install sequence

After final confirmation, `Install QTracX` performs these stages:

1. install or verify Windows host prerequisites;
2. acquire the selected source into an empty destination;
3. write the validated deployment `.env.local`;
4. create the CPU/GPU virtual environment and install dependencies;
5. run `npm ci` and create the production frontend build;
6. optionally warm the OCR models;
7. create/verify the database account, apply migrations and create the first
   QTracX administrator; and
8. verify the completed application runtime and frontend build.

The source ZIP path is validated and unsafe/generated content is rejected. Git
mode is fixed by `config/installer.defaults.json`; this development build uses
the `deployment-readiness` branch.

## Current boundary

This slice installs and verifies the application but does not yet register the
backend/frontend as Windows services, add startup recovery, create firewall
rules, or add repair/uninstall behavior. Those operations belong to the next
installer slice. Until then, continue using the existing backend and frontend
start commands after installation.

Full installation should be tested in Windows Sandbox or a disposable Windows
11 VM. Use `Check plan` rather than `Install QTracX` on the development machine
unless a separate empty destination and disposable database are available.
