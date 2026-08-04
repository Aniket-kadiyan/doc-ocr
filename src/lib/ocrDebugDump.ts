/** Client-side control for OCR step dumps (backend/debug_output/). */

import { DEBUG_DUMP_ENABLED } from "@/lib/featureFlags";

const LS_KEY = "doc_ocr_debug_dump";
const LS_FORCE_KEY = "doc_ocr_debug_dump_force";

export function isOcrDebugDumpEnabled(): boolean {
  // Hard gate: when the feature is off (prod default), no dump is ever
  // produced, regardless of any stale localStorage toggle or env value.
  if (!DEBUG_DUMP_ENABLED) return false;
  if (typeof window !== "undefined") {
    const ls = localStorage.getItem(LS_KEY);
    if (ls === "0") return false;
    if (ls === "1") return true;
  }
  const env = process.env.NEXT_PUBLIC_OCR_DEBUG_DUMP;
  if (env === "0" || env === "false") return false;
  if (env === "1" || env === "true") return true;
  // Default OFF: dumps only happen when explicitly enabled via
  // NEXT_PUBLIC_OCR_DEBUG_DUMP=1 (or the in-app localStorage toggle).
  return false;
}

export function isOcrDebugDumpForce(): boolean {
  if (!DEBUG_DUMP_ENABLED) return false;
  if (typeof window !== "undefined") {
    return localStorage.getItem(LS_FORCE_KEY) === "1";
  }
  return process.env.NEXT_PUBLIC_OCR_DEBUG_DUMP_FORCE === "1";
}

export function setOcrDebugDumpEnabled(on: boolean): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(LS_KEY, on ? "1" : "0");
  window.dispatchEvent(new CustomEvent("ocr-debug-dump-changed"));
}

export function setOcrDebugDumpForce(on: boolean): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(LS_FORCE_KEY, on ? "1" : "0");
  window.dispatchEvent(new CustomEvent("ocr-debug-dump-changed"));
}

export function readOcrDebugDumpToggles(): { enabled: boolean; force: boolean } {
  return {
    enabled: isOcrDebugDumpEnabled(),
    force: isOcrDebugDumpForce(),
  };
}
