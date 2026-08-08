#!/usr/bin/env node
/** Run Python tests with the repository's OS-specific backend virtualenv. */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(scriptDir, "..");
const backendDir = join(repoRoot, "backend");
const isWindows = process.platform === "win32";
const virtualenvPython = isWindows
  ? join(backendDir, ".venv", "Scripts", "python.exe")
  : join(backendDir, ".venv", "bin", "python");
const python = existsSync(virtualenvPython)
  ? virtualenvPython
  : isWindows
    ? "py"
    : "python3";

const fastTests = [
  "backend/test_angle_utils.py",
  "backend/test_checksheet_converter.py",
  "backend/test_detection_passes.py",
  "backend/test_page_layout.py",
  "backend/test_page_scan.py",
  "backend/test_page_value_filters.py",
  "backend/test_paddle_parse.py",
  "backend/test_region_cluster.py",
  "backend/test_scan_jobs.py",
  "backend/test_scan_progress.py",
  "backend/test_segment_quality.py",
  "tools/balloon_builder/test_balloon_builder.py",
];

const mode = process.argv.includes("--ocr") ? "ocr" : "fast";
const existingPythonPath = process.env.PYTHONPATH;
const pythonPath = [backendDir, repoRoot, existingPythonPath]
  .filter(Boolean)
  .join(delimiter);

if (!existsSync(virtualenvPython)) {
  console.warn(
    `[python-tests] No backend virtualenv found; falling back to ${python}.\n` +
      "[python-tests] Install backend/requirements-dev.txt for the complete test environment."
  );
}

const commands =
  mode === "ocr"
    ? [
        // OpenCV-dependent unit checks, followed by the executable synthetic
        // PaddleOCR smoke test (test_dimensions.py is intentionally not a
        // pytest module).
        ["-m", "pytest", "-q", "backend/test_region_detect.py"],
        ["backend/test_dimensions.py"],
      ]
    : [["-m", "pytest", "-q", ...fastTests]];

function runPython(arguments_) {
  return new Promise((resolve, reject) => {
    const child = spawn(python, arguments_, {
      cwd: repoRoot,
      env: { ...process.env, PYTHONPATH: pythonPath },
      stdio: "inherit",
      shell: false,
    });
    child.on("error", reject);
    child.on("exit", (code) => resolve(code ?? 1));
  });
}

for (const command of commands) {
  try {
    const code = await runPython(command);
    if (code !== 0) process.exit(code);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[python-tests] Failed to start Python: ${message}`);
    process.exit(1);
  }
}
