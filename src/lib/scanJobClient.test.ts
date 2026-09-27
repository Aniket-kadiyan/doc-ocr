import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cancelScanJob,
  isScanJobCancelledError,
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
