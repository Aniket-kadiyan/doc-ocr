import type { ScanLiveness, ScanProgress } from "@/types/scanJob";

export function formatScanDuration(totalSeconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = safeSeconds % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function scanWorkCounter(progress: ScanProgress): string | null {
  if (progress.batchTotal > 0) {
    return `Batch ${progress.batchCurrent}/${progress.batchTotal}`;
  }
  if (progress.objectTotal > 0) {
    return `Object ${progress.objectCurrent}/${progress.objectTotal}`;
  }
  if (progress.passTotal > 0) {
    const pass = `Pass ${progress.passCurrent}/${progress.passTotal}`;
    return progress.tileTotal > 1
      ? `${pass} · Tile ${progress.tileCurrent}/${progress.tileTotal}`
      : pass;
  }
  if (progress.total > 0) {
    return `${progress.completed}/${progress.total} work units`;
  }
  return null;
}

export const scanLivenessLabel: Record<ScanLiveness, string> = {
  queued: "Queued",
  working: "Working",
  long_running: "Long-running step",
  slow_progress: "Service responsive — slow progress",
  possibly_stalled: "Heartbeat missing — possibly stalled",
  cancelling: "Stopping after current OCR operation",
  cancelled: "Stopped",
  complete: "Complete",
  failed: "Failed",
};

export function heartbeatLabel(ageSeconds: number): string {
  return ageSeconds < 1
    ? "heartbeat now"
    : `heartbeat ${Math.floor(ageSeconds)}s ago`;
}
