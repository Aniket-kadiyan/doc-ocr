/** Browser-local options that alter automatic drawing scans. */

const GDT_KEY = "doc_ocr_scan_gdt";
const SLANTED_KEY = "doc_ocr_scan_slanted";
export const SCAN_SETTINGS_EVENT = "scan-settings-changed";

let gdtInMemory = true;
let slantedInMemory = true;

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

export function isSlantedScanEnabled(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const stored = localStorage.getItem(SLANTED_KEY);
    return stored == null ? slantedInMemory : stored !== "0";
  } catch {
    return slantedInMemory;
  }
}

export function setSlantedScanEnabled(enabled: boolean): void {
  if (typeof window === "undefined") return;
  slantedInMemory = enabled;
  try {
    localStorage.setItem(SLANTED_KEY, enabled ? "1" : "0");
  } catch {
    // Private/locked-down browsers retain the setting for this page only.
  }
  window.dispatchEvent(new CustomEvent(SCAN_SETTINGS_EVENT));
}

const SECTION_NUMBERING_KEY = "doc_ocr_section_numbering";
const SECTION_OVERLAY_KEY = "doc_ocr_section_overlay";

let sectionNumberingInMemory = true;
let sectionOverlayInMemory = false;

/** Whole-page scans number balloons one drawing section at a time. */
export function isSectionNumberingEnabled(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const stored = localStorage.getItem(SECTION_NUMBERING_KEY);
    return stored == null ? sectionNumberingInMemory : stored !== "0";
  } catch {
    return sectionNumberingInMemory;
  }
}

export function setSectionNumberingEnabled(enabled: boolean): void {
  if (typeof window === "undefined") return;
  sectionNumberingInMemory = enabled;
  try {
    localStorage.setItem(SECTION_NUMBERING_KEY, enabled ? "1" : "0");
  } catch {
    // Private/locked-down browsers retain the setting for this page only.
  }
  window.dispatchEvent(new CustomEvent(SCAN_SETTINGS_EVENT));
}

/** Outline the detected drawing sections over the page. Off by default. */
export function isSectionOverlayEnabled(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const stored = localStorage.getItem(SECTION_OVERLAY_KEY);
    return stored == null ? sectionOverlayInMemory : stored === "1";
  } catch {
    return sectionOverlayInMemory;
  }
}

export function setSectionOverlayEnabled(enabled: boolean): void {
  if (typeof window === "undefined") return;
  sectionOverlayInMemory = enabled;
  try {
    localStorage.setItem(SECTION_OVERLAY_KEY, enabled ? "1" : "0");
  } catch {
    // Private/locked-down browsers retain the setting for this page only.
  }
  window.dispatchEvent(new CustomEvent(SCAN_SETTINGS_EVENT));
}

const CUSTOM_SYMBOLS_KEY = "doc_ocr_scan_custom_symbols";

let customSymbolsInMemory = true;

/** Look for the custom symbol library's enabled symbols during scans. */
export function isCustomSymbolScanEnabled(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const stored = localStorage.getItem(CUSTOM_SYMBOLS_KEY);
    return stored == null ? customSymbolsInMemory : stored !== "0";
  } catch {
    return customSymbolsInMemory;
  }
}

export function setCustomSymbolScanEnabled(enabled: boolean): void {
  if (typeof window === "undefined") return;
  customSymbolsInMemory = enabled;
  try {
    localStorage.setItem(CUSTOM_SYMBOLS_KEY, enabled ? "1" : "0");
  } catch {
    // Private/locked-down browsers retain the setting for this page only.
  }
  window.dispatchEvent(new CustomEvent(SCAN_SETTINGS_EVENT));
}
