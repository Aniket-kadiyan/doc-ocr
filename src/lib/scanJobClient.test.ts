import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cancelScanJob,
  isScanJobCancelledError,
  runScanJob,
  ScanJobCancelledError,
} from "@/lib/scanJobClient";

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
