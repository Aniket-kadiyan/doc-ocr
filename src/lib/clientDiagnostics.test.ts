import { afterEach, describe, expect, it, vi } from "vitest";
import {
  reportClientError,
  resetClientDiagnosticRateLimitForTests,
} from "@/lib/clientDiagnostics";
import { ApiRequestError } from "@/lib/requestDiagnostics";

afterEach(() => {
  resetClientDiagnosticRateLimitForTests();
  vi.unstubAllGlobals();
});

describe("client diagnostics", () => {
  it("sends only allow-listed metadata and reuses an API request reference", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) => {
        void _input;
        void _init;
        return new Response(null, { status: 204 });
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const error = new ApiRequestError("private drawing value 12.5", {
      requestId: "request-reference-4",
      operation: "scan.create",
      status: 503,
    });

    const reference = await reportClientError("http://localhost:8000", error, {
      kind: "unhandled_rejection",
      operation: "scan.create",
    });

    expect(reference).toBe("request-reference-4");
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const payload = JSON.parse(String(init.body)) as Record<string, unknown>;
    expect(payload).toEqual({
      kind: "api_error",
      operation: "scan.create",
      error_type: "ApiRequestError",
      http_status: 503,
    });
    expect(JSON.stringify(payload)).not.toContain("12.5");
    expect(new Headers(init.headers).get("X-Request-ID")).toBe(
      "request-reference-4"
    );
  });

  it("swallows a diagnostics delivery failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("offline");
      })
    );

    await expect(
      reportClientError("http://localhost:8000", new TypeError("private"), {
        kind: "runtime_error",
        operation: "window.error",
      })
    ).resolves.toMatch(/^[A-Za-z0-9][A-Za-z0-9._:-]+$/);
  });
});
