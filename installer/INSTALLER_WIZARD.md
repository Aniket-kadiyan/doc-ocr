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
8. verify the completed application runtime, frontend build and bundled CAD
   converter;
9. restrict the installed source/configuration to `SYSTEM` and local
   Administrators;
10. register backend/frontend startup tasks with restart recovery;
11. add private/domain inbound firewall rules only in central-LAN mode; and
12. start both components and wait for backend/frontend health checks.

After source acquisition, the wizard uses the runtime, configuration and
database installer modules from the acquired Git branch or ZIP. This keeps the
installed application and its setup logic on the same revision.

The source ZIP path is validated and unsafe/generated content is rejected. Git
mode is fixed by `config/installer.defaults.json`; this validation build uses
the isolated `mvp2-chitra-final-port` branch.

## Operations after installation

The installation is automatically available after Windows startup. The
installer uses Task Scheduler instead of a third-party service wrapper, so no
additional service-host license is introduced. Startup tasks run as `SYSTEM`,
restart after failure and keep bounded stdout/stderr logs. See
`SERVICE_OPERATIONS.md` for status, verification, start, stop, repair and
non-destructive unregister commands.

Full installation should be tested in Windows Sandbox or a disposable Windows
11 VM. Use `Check plan` rather than `Install QTracX` on the development machine
unless a separate empty destination and disposable database are available.

