/**
 * Per-browser scan options.
 *
 * GD&T frame reading is on by default. Turning it off skips the feature
 * control frame pass, which costs 3-20 s a page and finds nothing on a sheet
 * without GD&T; frames can still be read afterwards from the GD&T Symbols
 * tool.
 */

const GDT_KEY = "doc_ocr_scan_gdt";
export const SCAN_SETTINGS_EVENT = "scan-settings-changed";

// Where the choice lives when storage is blocked (private windows).
let gdtInMemory = true;

export function isGdtScanEnabled(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const stored = localStorage.getItem(GDT_KEY);
    return stored == null ? gdtInMemory : stored !== "0";
  } catch {
    return gdtInMemory;
  }
}

export function setGdtScanEnabled(on: boolean): void {
  if (typeof window === "undefined") return;
  gdtInMemory = on;
  try {
    localStorage.setItem(GDT_KEY, on ? "1" : "0");
  } catch {
    // Storage blocked: the option lasts for this page only.
  }
  window.dispatchEvent(new CustomEvent(SCAN_SETTINGS_EVENT));
}
