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
  // The backend reads a 250dpi page while the viewer displays a 1.5x render.
  const displayScale = 250 / 72 / 1.5;
  const sourceCanvas = {
    toBlob: (callback: BlobCallback) =>
      callback(new Blob(["png"], { type: "image/png" })),
  } as HTMLCanvasElement;

  it("maps all processing geometry back to viewer pixels", async () => {
    const overlay = {
      enabled: true,
      page_width: 2924,
      page_height: 2068,
      scope_kind: "page",
      table_masks: [{ x: 100, y: 200, width: 300, height: 400 }],
      panels: [
        {
          x: 0,
          y: 0,
          width: 954,
          height: 1268,
          id: "P1",
          state: "active",
        },
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
    const progressEvents: ScanProgress[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          job_id: "job-overlay",
          status: "succeeded",
          stage: "finalizing",
          message: "Complete",
          percent: 100,
          completed: 1,
          total: 1,
          liveness: "complete",
          overlay,
          result: {
            coordinate_space: "page",
            regions: [],
            review_candidates: [],
          },
        }),
      }))
    );

    await runScanJob({
      sourceCanvas,
      bbox: { x: 0, y: 0, width: 1263, height: 893 },
      page: 1,
      scopeKind: "page",
      displayScale,
      onProgress: (progress) => progressEvents.push(progress),
    });

    const mapped = progressEvents.at(-1)?.overlay;
    expect(mapped?.pageWidth).toBeCloseTo(2924 / displayScale, 6);
    expect(mapped?.pageHeight).toBeCloseTo(2068 / displayScale, 6);
    expect(mapped?.candidates[0].bbox.x).toBeCloseTo(
      1010 / displayScale,
      6
    );
    expect(mapped?.candidates[0].bbox.width).toBeCloseTo(
      291 / displayScale,
      6
    );
    expect(mapped?.panels[0].width).toBeCloseTo(954 / displayScale, 6);
    expect(mapped?.tableMasks[0].y).toBeCloseTo(200 / displayScale, 6);
    expect(mapped?.overlaps[0].height).toBeCloseTo(40 / displayScale, 6);
    expect(mapped?.panels[0]).toMatchObject({ id: "P1", state: "active" });
  });
});

describe("scan job source evidence", () => {
  it("attaches the original PDF and returns source accounting", async () => {
    let submittedForm: FormData | undefined;
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      submittedForm = init?.body as FormData;
      return {
        ok: true,
        status: 202,
        json: async () => ({
          job_id: "job-native",
          status: "completed",
          stage: "complete",
          message: "Complete",
          percent: 100,
          completed: 1,
          total: 1,
          result: {
            count: 0,
            coordinate_space: "page",
            regions: [],
            review_candidates: [],
            source_profile: {
              kind: "vector",
              native_text_available: true,
              native_span_count: 12,
            },
            recognition_source_counts: { native_pdf: 3, ocr: 2 },
          },
        }),
      };
    });
    vi.stubGlobal("fetch", fetchMock);
    const sourceCanvas = {
      toBlob: (callback: BlobCallback) =>
        callback(new Blob(["png"], { type: "image/png" })),
    } as HTMLCanvasElement;
    const sourceDocument = new File(["%PDF-1.7"], "drawing.pdf", {
      type: "application/pdf",
    });

    const result = await runScanJob({
      sourceCanvas,
      sourceDocument,
      bbox: { x: 0, y: 0, width: 100, height: 80 },
      page: 1,
      scopeKind: "page",
    });

    const attached = submittedForm?.get("source_document") as File;
    expect(attached.name).toBe("drawing.pdf");
    expect(attached.type).toBe("application/pdf");
    expect(await attached.text()).toBe("%PDF-1.7");
    expect(result.sourceProfile?.kind).toBe("vector");
    expect(result.recognitionSourceCounts).toEqual({ native_pdf: 3, ocr: 2 });
  });
});
