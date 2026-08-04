#!/usr/bin/env node
/**
 * Cross-platform launcher for the PaddleOCR backend.
 *
 * Picks the correct venv Python for the current OS:
 *   - Windows: backend\.venv\Scripts\python.exe
 *   - macOS/Linux: backend/.venv/bin/python
 * then runs `python main.py` from inside backend/ (so uvicorn's reloader can
 * import "main:app"). The backend loads .env.local / .env itself via
 * config_env.load_env_files(), so no dotenv wrapper is needed here.
 *
 * Flags (cross-platform, no shell env syntax needed):
 *   --debug   → sets DEBUG_DUMP=1        (write pipeline step dumps)
 *   --force   → sets DEBUG_DUMP_FORCE=1  (re-dump even if already saved)
 * Any variable already set in the shell is also inherited.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, "..");
const backendDir = join(repoRoot, "backend");
const isWin = process.platform === "win32";

const venvPython = isWin
  ? join(backendDir, ".venv", "Scripts", "python.exe")
  : join(backendDir, ".venv", "bin", "python");

let python = venvPython;
if (!existsSync(venvPython)) {
  // No venv yet — fall back to a system interpreter and warn loudly.
  python = isWin ? "py" : "python3";
  console.warn(
    `\n[run-ocr] No virtualenv found at ${venvPython}\n` +
      `[run-ocr] Falling back to system "${python}". Create the venv first:\n` +
      (isWin
        ? `           cd backend\n` +
          `           py -m venv .venv\n` +
          `           .venv\\Scripts\\python -m pip install -r requirements.txt\n\n`
        : `           cd backend\n` +
          `           python3 -m venv .venv\n` +
          `           .venv/bin/python -m pip install -r requirements.txt\n\n`)
  );
}

const argv = process.argv.slice(2);
const env = { ...process.env };
if (argv.includes("--debug")) env.DEBUG_DUMP = "1";
if (argv.includes("--force")) {
  env.DEBUG_DUMP = "1";
  env.DEBUG_DUMP_FORCE = "1";
}

const child = spawn(python, ["main.py"], {
  cwd: backendDir,
  stdio: "inherit",
  env,
  shell: false,
});

child.on("error", (err) => {
  console.error(`[run-ocr] Failed to start Python: ${err.message}`);
  process.exit(1);
});
child.on("exit", (code) => process.exit(code ?? 0));
