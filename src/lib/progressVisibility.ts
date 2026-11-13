export const SCAN_PROGRESS_VISIBILITY_EVENT =
    "qtracx-scan-progress-visibility";

const STORAGE_KEY = "qtracx.scan-progress-visible";

export function isScanProgressVisible(): boolean {
    if (typeof window === "undefined") return true;

    return window.localStorage.getItem(STORAGE_KEY) !== "false";
}

export function setScanProgressVisible(visible: boolean): void {
    if (typeof window === "undefined") return;

    window.localStorage.setItem(STORAGE_KEY, visible ? "true" : "false");

    window.dispatchEvent(
        new CustomEvent<boolean>(SCAN_PROGRESS_VISIBILITY_EVENT, {
            detail: visible,
        })
    );
}