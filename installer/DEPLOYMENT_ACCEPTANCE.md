# QTracX deployment acceptance gates

The release workflow must complete before a Windows package is offered for
installation. It verifies installer contracts on Windows, frontend lint/types/
tests/build, the complete CPU backend test suite, and a self-contained Windows
CAD-converter build. The archive contains SHA-256 output plus the ACadSharp and
.NET notices required by its bundled dependencies.

## Automated release gates

- Windows installer Python and PowerShell contract tests pass.
- CPU requirements install in a dedicated Python 3.13 x64 environment.
- Backend tests pass against the CPU Paddle distribution.
- Frontend lint, type checking, tests and production build pass.
- ACadSharp restores and publishes for `win-x64` as a self-contained single
  executable.
- The dependency-clean ZIP contains no `.git`, virtual environment,
  `node_modules`, `.next`, cached Python files or deployment `.env.local`.

## Windows installation acceptance

Run the produced ZIP in Windows Sandbox or a disposable Windows 11 VM:

1. install in CPU mode from the offline archive;
2. verify `.env.local` contains CAD and bounded run-report settings;
3. verify database migrations and initial application sign-in;
4. open one DXF and one DWG and confirm the drawing renders;
5. restart Windows and confirm both QTracX scheduled tasks are running;
6. run `service_manager.ps1 -Mode Verify` and require both health checks;
7. stop one component with Task Scheduler and confirm restart recovery; and
8. run Repair, then Unregister, and confirm neither operation deletes data.

## NVIDIA hardware acceptance

CI does not claim CUDA correctness. Before approving a GPU customer package,
run this separate gate on the target GPU class:

1. `nvidia-smi` reports the expected adapter and supported driver;
2. GPU setup creates only `backend\.venv-gpu` and installs
   `paddlepaddle-gpu==3.2.2` from the CUDA 12.6 manifest;
3. installer verification reports `cuda_compiled: true` and no CPU Paddle
   distribution in that environment;
4. `/health` reports the effective OCR device as `gpu:0`;
5. one raster drawing completes OCR and one text PDF completes its queued scan;
6. two simultaneous users preserve owner isolation, queue progress,
   cancellation and interactive-priority behavior; and
7. a reboot restores both scheduled tasks and the GPU health result.

Record the GPU model, VRAM, driver, Windows build, elapsed scan time and health
output with the release acceptance record.
