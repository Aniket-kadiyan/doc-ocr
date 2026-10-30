# QTracX startup and repair operations

Run these commands from an elevated Windows PowerShell session. Replace
`C:\QTracX` when a different installation directory was selected, and use
`-Runtime gpu` for the GPU installation.

## Status and health

Task registration and last exit results:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  C:\QTracX\installer\service_manager.ps1 `
  -Mode Status -ApplicationRoot C:\QTracX -Runtime cpu
```

Verify both registered tasks and HTTP(S) health without changing them:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  C:\QTracX\installer\service_manager.ps1 `
  -Mode Verify -ApplicationRoot C:\QTracX -Runtime cpu
```

Service-host logs are stored under:

```text
C:\QTracX\backend\logs\service
```

Each backend/frontend log rotates at 25 MB and keeps five archives. The
application's structured logs and privacy-safe run reports remain under the
configured `QTRACX_LOG_DIR`.

## Start and stop

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  C:\QTracX\installer\service_manager.ps1 `
  -Mode Stop -ApplicationRoot C:\QTracX -Runtime cpu
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  C:\QTracX\installer\service_manager.ps1 `
  -Mode Start -ApplicationRoot C:\QTracX -Runtime cpu
```

## Repair startup integration

Repair re-validates the installed layout, re-applies protected directory
permissions, replaces both startup tasks, recreates the LAN firewall rules
when applicable, starts the application and waits for health checks:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  C:\QTracX\installer\service_manager.ps1 `
  -Mode Repair -ApplicationRoot C:\QTracX -Runtime cpu
```

## Non-destructive unregister

This removes the two QTracX scheduled tasks and QTracX firewall rules. It does
not delete application files, `.env.local`, checksheets, logs or database
data. `-Mode Uninstall` is an equivalent compatibility alias.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  C:\QTracX\installer\service_manager.ps1 `
  -Mode Unregister -ApplicationRoot C:\QTracX -Runtime cpu
```
