import { describe, expect, it } from "vitest";
import {
  formatScanDuration,
  heartbeatLabel,
  scanLivenessLabel,
  scanWorkCounter,
} from "@/lib/scanProgress";
import type { ScanProgress } from "@/types/scanJob";

const progress = (overrides: Partial<ScanProgress> = {}): ScanProgress => ({
  jobId: "job-1",
  status: "running",
  stage: "detecting",
  message: "Running detection",
  percent: 10,
  completed: 3,
  total: 20,
  passCurrent: 2,
  passTotal: 33,
  tileCurrent: 3,
  tileTotal: 5,
  objectCurrent: 0,
  objectTotal: 0,
  candidateCount: 17,
  operationLabel: "source contrast",
  elapsedSeconds: 70,
  stepElapsedSeconds: 8,
  heartbeatAgeSeconds: 0,
  progressAgeSeconds: 8,
  estimatedRemainingSeconds: 120,
  liveness: "working",
  ...overrides,
});

describe("scan progress display helpers", () => {
  it("formats elapsed and ETA durations without fake precision", () => {
    expect(formatScanDuration(0)).toBe("0:00");
    expect(formatScanDuration(125)).toBe("2:05");
    expect(formatScanDuration(3723)).toBe("1:02:03");
  });

  it("shows pass and tile progress for area-sensitive detection", () => {
    expect(scanWorkCounter(progress())).toBe("Pass 2/33 · Tile 3/5");
    expect(scanWorkCounter(progress({ tileCurrent: 1, tileTotal: 1 }))).toBe(
      "Pass 2/33"
    );
  });

  it("switches to object counters during recognition", () => {
    expect(
      scanWorkCounter(
        progress({
          stage: "recognizing",
          objectCurrent: 7,
          objectTotal: 19,
        })
      )
    ).toBe("Object 7/19");
  });

  it("reports heartbeat recency directly", () => {
    expect(heartbeatLabel(0)).toBe("heartbeat now");
    expect(heartbeatLabel(7.8)).toBe("heartbeat 7s ago");
  });

  it("distinguishes slow progress from a missing heartbeat", () => {
    expect(scanLivenessLabel.slow_progress).toBe(
      "Service responsive — slow progress"
    );
    expect(scanLivenessLabel.possibly_stalled).toBe(
      "Heartbeat missing — possibly stalled"
    );
  });
});
