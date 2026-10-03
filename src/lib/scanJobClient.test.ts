import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cancelScanJob,
  isScanJobCancelledError,
  runScanJob,
  ScanJobCancelledError,
} from "@/lib/scanJobClient";
import type { ScanProgress } from "@/types/scanJob";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("scan job cancellation", () => {
  it("posts to the encoded cancellation endpoint and maps its state", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        job_id: "job/one",
        status: "cancelling",
        stage: "cancelling",
        message: "Stopping after the current OCR operation",
        percent: 72,
        completed: 3,
        total: 5,
        liveness: "cancelling",
        result: null,
      }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    const progress = await cancelScanJob("job/one");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/ocr\/scan-jobs\/job%2Fone\/cancel$/),
      { method: "POST", cache: "no-store" }
    );
    expect(progress).toMatchObject({
      jobId: "job/one",
      status: "cancelling",
      stage: "cancelling",
      percent: 72,
      liveness: "cancelling",
    });
  });

  it("identifies the cancellation error used by the atomic UI path", () => {
    expect(isScanJobCancelledError(new ScanJobCancelledError())).toBe(true);
    expect(isScanJobCancelledError(new Error("ordinary failure"))).toBe(false);
  });
});

describe("scan overlay coordinates", () => {
  // The scan reads a 250dpi render while the viewer shows 1.5x, so the
  // backend measures everything in those larger pixels.
  const displayScale = 250 / 72 / 1.5;

  const canvas = {
    toBlob: (callback: (blob: Blob | null) => void) =>
      callback(new Blob(["x"], { type: "image/png" })),
  } as unknown as HTMLCanvasElement;

  const snapshot = (overlay: Record<string, unknown>) => ({
    job_id: "job-1",
    status: "succeeded",
    stage: "finalizing",
    message: "done",
    percent: 100,
    completed: 1,
    total: 1,
    liveness: "working",
    overlay,
    result: {
      coordinate_space: "page",
      regions: [],
      review_candidates: [],
    },
  });

  it("brings processing geometry back to viewer pixels", async () => {
    const overlay = {
      enabled: true,
      page_width: 2924,
      page_height: 2068,
      scope_kind: "page",
      table_masks: [{ x: 100, y: 200, width: 300, height: 400 }],
      panels: [
        { x: 0, y: 0, width: 954, height: 1268, id: "P1", state: "active" },
      ],
      overlaps: [{ x: 10, y: 20, width: 30, height: 40 }],
      candidates: [
        {
          id: "C1",
          bbox: { x: 1010, y: 136, width: 291, height: 61 },
          state: "eligible",
        },
      ],
    };
    const seen: ScanProgress[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => snapshot(overlay),
      }))
    );

    await runScanJob({
      sourceCanvas: canvas,
      bbox: { x: 0, y: 0, width: 1263, height: 893 },
      page: 1,
      scopeKind: "page",
      displayScale,
      onProgress: (progress) => seen.push(progress),
    });

    const mapped = seen.at(-1)?.overlay;
    expect(mapped).toBeTruthy();
    // The page it measured, stated in the same units as the geometry, is what
    // lets the viewer tell a section crop from the drawing.
    expect(mapped?.pageWidth).toBeCloseTo(2924 / displayScale, 6);
    expect(mapped?.pageHeight).toBeCloseTo(2068 / displayScale, 6);
    expect(mapped?.candidates[0].bbox.x).toBeCloseTo(1010 / displayScale, 6);
    expect(mapped?.candidates[0].bbox.width).toBeCloseTo(291 / displayScale, 6);
    expect(mapped?.panels[0].width).toBeCloseTo(954 / displayScale, 6);
    expect(mapped?.tableMasks[0].y).toBeCloseTo(200 / displayScale, 6);
    expect(mapped?.overlaps[0].height).toBeCloseTo(40 / displayScale, 6);
    // Panel identity survives the mapping.
    expect(mapped?.panels[0]).toMatchObject({ id: "P1", state: "active" });
  });
});
