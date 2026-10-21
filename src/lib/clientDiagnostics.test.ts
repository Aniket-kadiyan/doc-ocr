import { afterEach, describe, expect, it, vi } from "vitest";
import {
  reportClientError, resetClientDiagnosticRateLimitForTests,
} from "@/lib/clientDiagnostics";
import { ApiRequestError } from "@/lib/requestDiagnostics";

afterEach(() => {
  resetClientDiagnosticRateLimitForTests();
  vi.unstubAllGlobals();
});

describe("client diagnostics", () => {
  it("sends only allow-listed metadata", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
      void _input;
      void _init;
      return new Response(null, { status: 204 });
    });
    vi.stubGlobal("fetch", fetchMock);
    await reportClientError(
      "http://localhost:8000",
      new ApiRequestError("private drawing value", {
        requestId: "request-reference-4", operation: "scan.create", status: 503,
      }),
      { kind: "unhandled_rejection", operation: "scan.create" }
    );
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.credentials).toBe("include");
    const payload = JSON.parse(String(init.body));
    expect(payload).toEqual({
      kind: "api_error", operation: "scan.create",
      error_type: "ApiRequestError", http_status: 503,
    });
    expect(JSON.stringify(payload)).not.toContain("private drawing value");
  });

  it("swallows delivery failures", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));
    await expect(reportClientError(
      "http://localhost:8000", new TypeError("private"),
      { kind: "runtime_error", operation: "window.error" }
    )).resolves.toBeTruthy();
  });
});
