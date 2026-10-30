/** Browser-local options that alter automatic drawing scans. */

const GDT_KEY = "doc_ocr_scan_gdt";
export const SCAN_SETTINGS_EVENT = "scan-settings-changed";

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

export function setGdtScanEnabled(enabled: boolean): void {
  if (typeof window === "undefined") return;
  gdtInMemory = enabled;
  try {
    localStorage.setItem(GDT_KEY, enabled ? "1" : "0");
  } catch {
    // Private/locked-down browsers retain the setting for this page only.
  }
  window.dispatchEvent(new CustomEvent(SCAN_SETTINGS_EVENT));
}
