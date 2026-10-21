#!/usr/bin/env node
/** Start Next.js with a cross-platform, configurable default port of 3002. */

import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(scriptDir, "..");
const mode = process.argv[2];
if (mode !== "dev" && mode !== "start") {
  console.error("Usage: node scripts/run-next.mjs <dev|start>");
  process.exit(2);
}

function configuredPort() {
  if (process.env.PORT?.trim()) return process.env.PORT.trim();
  for (const fileName of [".env.local", ".env"]) {
    const path = join(repoRoot, fileName);
    if (!existsSync(path)) continue;
    for (const rawLine of readFileSync(path, "utf8").split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#") || !line.includes("=")) continue;
      const separator = line.indexOf("=");
      if (line.slice(0, separator).trim() !== "PORT") continue;
      let value = line.slice(separator + 1).trim();
      if (value.length >= 2 && (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      )) value = value.slice(1, -1);
      if (value) return value;
    }
  }
  return "3002";
}

const nextCli = join(repoRoot, "node_modules", "next", "dist", "bin", "next");
if (!existsSync(nextCli)) {
  console.error("[run-next] Next.js is not installed. Run npm install first.");
  process.exit(1);
}
const args = [nextCli, mode];
if (mode === "dev") args.push("--turbopack");
const child = spawn(process.execPath, args, {
  cwd: repoRoot,
  stdio: "inherit",
  env: { ...process.env, PORT: configuredPort() },
  shell: false,
});
child.on("error", (error) => {
  console.error(`[run-next] Failed to start Next.js: ${error.message}`);
  process.exit(1);
});
child.on("exit", (code) => process.exit(code ?? 0));
