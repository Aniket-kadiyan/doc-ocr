/** Shared environment loading and optional HTTPS validation for the UI. */

import { existsSync, readFileSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { createSecureContext } from "node:tls";

export function configuredValue(repoRoot, name, fallback = "") {
  if (process.env[name]?.trim()) return process.env[name].trim();
  for (const fileName of [".env.local", ".env"]) {
    const path = resolve(repoRoot, fileName);
    if (!existsSync(path)) continue;
    for (const rawLine of readFileSync(path, "utf8").split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#") || !line.includes("=")) continue;
      const separator = line.indexOf("=");
      if (line.slice(0, separator).trim() !== name) continue;
      let value = line.slice(separator + 1).trim();
      if (
        value.length >= 2 &&
        ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'")))
      ) {
        value = value.slice(1, -1);
      }
      if (value) return value;
    }
  }
  return fallback;
}

export function configuredBoolean(repoRoot, name, fallback = false) {
  const raw = configuredValue(repoRoot, name, "").toLowerCase();
  if (!raw) return fallback;
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off"].includes(raw)) return false;
  throw new Error(
    `${name} must be one of: 1, 0, true, false, yes, no, on, off`
  );
}

function configuredPath(repoRoot, raw) {
  if (!raw) return null;
  return isAbsolute(raw) ? raw : resolve(repoRoot, raw);
}

export function resolveHttpsTransport(
  repoRoot,
  dependencies = { existsSync, readFileSync, createSecureContext }
) {
  const requested = configuredBoolean(repoRoot, "HTTPS_ENABLED", false);
  if (!requested) {
    return {
      requested: false,
      active: false,
      protocol: "http",
      fallbackReason: null,
      certFile: null,
      keyFile: null,
      cert: null,
      key: null,
    };
  }

  const certFile = configuredPath(
    repoRoot,
    configuredValue(repoRoot, "TLS_CERT_FILE", "")
  );
  const keyFile = configuredPath(
    repoRoot,
    configuredValue(repoRoot, "TLS_KEY_FILE", "")
  );
  const fallback = (reason) => ({
    requested: true,
    active: false,
    protocol: "http",
    fallbackReason: reason,
    certFile: null,
    keyFile: null,
    cert: null,
    key: null,
  });

  if (!certFile || !keyFile) {
    return fallback("TLS_CERT_FILE and TLS_KEY_FILE must both be configured");
  }
  if (!dependencies.existsSync(certFile)) {
    return fallback("TLS certificate file is unavailable");
  }
  if (!dependencies.existsSync(keyFile)) {
    return fallback("TLS private-key file is unavailable");
  }

  try {
    const cert = dependencies.readFileSync(certFile);
    const key = dependencies.readFileSync(keyFile);
    dependencies.createSecureContext({ cert, key });
    return {
      requested: true,
      active: true,
      protocol: "https",
      fallbackReason: null,
      certFile,
      keyFile,
      cert,
      key,
    };
  } catch {
    return fallback("TLS certificate or private key is invalid");
  }
}
