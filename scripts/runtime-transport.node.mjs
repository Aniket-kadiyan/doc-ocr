import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import {
  configuredBoolean,
  resolveHttpsTransport,
} from "./runtime-transport.mjs";

const original = {
  HTTPS_ENABLED: process.env.HTTPS_ENABLED,
  TLS_CERT_FILE: process.env.TLS_CERT_FILE,
  TLS_KEY_FILE: process.env.TLS_KEY_FILE,
};

function restore(name, value) {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}

afterEach(() => {
  for (const [name, value] of Object.entries(original)) restore(name, value);
});

test("HTTP remains the default", () => {
  delete process.env.HTTPS_ENABLED;
  delete process.env.TLS_CERT_FILE;
  delete process.env.TLS_KEY_FILE;
  const result = resolveHttpsTransport("C:/qtracx");
  assert.equal(result.requested, false);
  assert.equal(result.active, false);
  assert.equal(result.protocol, "http");
});

test("missing certificate configuration falls back to HTTP", () => {
  process.env.HTTPS_ENABLED = "true";
  delete process.env.TLS_CERT_FILE;
  delete process.env.TLS_KEY_FILE;
  const result = resolveHttpsTransport("C:/qtracx");
  assert.equal(result.requested, true);
  assert.equal(result.active, false);
  assert.match(result.fallbackReason, /both be configured/);
});

test("a valid certificate pair activates HTTPS", () => {
  process.env.HTTPS_ENABLED = "true";
  process.env.TLS_CERT_FILE = "cert.pem";
  process.env.TLS_KEY_FILE = "key.pem";
  const result = resolveHttpsTransport("C:/qtracx", {
    existsSync: () => true,
    readFileSync: (path) => Buffer.from(path),
    createSecureContext: ({ cert, key }) => {
      assert.ok(cert.length > 0);
      assert.ok(key.length > 0);
    },
  });
  assert.equal(result.active, true);
  assert.equal(result.protocol, "https");
  assert.equal(result.fallbackReason, null);
});

test("invalid certificate content falls back to HTTP", () => {
  process.env.HTTPS_ENABLED = "true";
  process.env.TLS_CERT_FILE = "cert.pem";
  process.env.TLS_KEY_FILE = "key.pem";
  const result = resolveHttpsTransport("C:/qtracx", {
    existsSync: () => true,
    readFileSync: () => Buffer.from("invalid"),
    createSecureContext: () => {
      throw new Error("invalid PEM");
    },
  });
  assert.equal(result.active, false);
  assert.match(result.fallbackReason, /invalid/);
});

test("invalid HTTPS_ENABLED values are rejected", () => {
  process.env.HTTPS_ENABLED = "sometimes";
  assert.throws(
    () => configuredBoolean("C:/qtracx", "HTTPS_ENABLED", false),
    /must be one of/
  );
});
