#!/usr/bin/env node
/** Start Next.js with cross-platform, configurable bind host and port. */

import { spawn } from "node:child_process";
import { createServer as createHttpsServer } from "node:https";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  configuredValue,
  resolveHttpsTransport,
} from "./runtime-transport.mjs";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(scriptDir, "..");
const mode = process.argv[2];
if (mode !== "dev" && mode !== "start") {
  console.error("Usage: node scripts/run-next.mjs <dev|start>");
  process.exit(2);
}

const configuredPort = () => configuredValue(repoRoot, "PORT", "3002");
const configuredHost = () => configuredValue(repoRoot, "UI_HOST", "127.0.0.1");
const transport = resolveHttpsTransport(repoRoot);

if (transport.requested && !transport.active) {
  console.warn(
    `[run-next] HTTPS requested but unavailable: ${transport.fallbackReason}. ` +
      "Falling back to HTTP."
  );
}

async function startHttps() {
  const port = Number(configuredPort());
  const host = configuredHost();
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("PORT must be an integer between 1 and 65535");
  }

  const nextModule = await import("next");
  const createNextApp = nextModule.default;
  const app = createNextApp({
    dev: mode === "dev",
    dir: repoRoot,
    hostname: host,
    port,
  });
  const handle = app.getRequestHandler();
  await app.prepare();

  const server = createHttpsServer(
    { cert: transport.cert, key: transport.key },
    (request, response) => handle(request, response)
  );
  server.on("error", (error) => {
    console.error(`[run-next] HTTPS server failed: ${error.message}`);
    process.exitCode = 1;
  });
  server.listen(port, host, () => {
    console.log(`[run-next] QTracX UI ready at https://${host}:${port}`);
  });

  const close = () => {
    server.close(() => process.exit(0));
  };
  process.once("SIGINT", close);
  process.once("SIGTERM", close);
}

if (transport.active) {
  startHttps().catch((error) => {
    console.error(`[run-next] Failed to start HTTPS: ${error.message}`);
    process.exit(1);
  });
} else {
  const nextCli = join(repoRoot, "node_modules", "next", "dist", "bin", "next");
  if (!existsSync(nextCli)) {
    console.error("[run-next] Next.js is not installed. Run npm install first.");
    process.exit(1);
  }
  const args = [nextCli, mode, "--hostname", configuredHost()];
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
}
