import { afterEach, describe, expect, it, vi } from "vitest";
import {
  apiRequestError,
  ApiRequestError,
  fetchWithRequestId,
  responseRequestId,
} from "@/lib/requestDiagnostics";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("request diagnostics", () => {
  it("adds a safe request ID and remembers it when a mock response has no headers", async () => {
    const response = { ok: true, status: 200 } as Response;
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) => {
        void _input;
        void _init;
        return response;
      }
    );
    vi.stubGlobal("fetch", fetchMock);

    const received = await fetchWithRequestId(
      "http://localhost:8000/health",
      { method: "GET" },
      { operation: "health.read" }
    );

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const requestId = new Headers(request.headers).get("X-Request-ID");
    expect(requestId).toMatch(/^[A-Za-z0-9][A-Za-z0-9._:-]+$/);
    expect(responseRequestId(received)).toBe(requestId);
  });

  it("uses the backend response ID in the user-facing error", () => {
    const response = new Response("", {
      status: 503,
      headers: { "X-Request-ID": "backend-reference-7" },
    });

    const error = apiRequestError(response, "Could not start scan.", "scan.create");

    expect(error).toBeInstanceOf(ApiRequestError);
    expect(error.requestId).toBe("backend-reference-7");
    expect(error.message).toContain("Diagnostic reference: backend-reference-7");
  });

  it("replaces network exception text with a safe correlated message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("drawing value 12.5 and secret connection details");
      })
    );

    await expect(
      fetchWithRequestId(
        "http://localhost:8000/ocr/segment",
        { method: "POST" },
        { operation: "ocr.segment" }
      )
    ).rejects.toMatchObject({
      name: "ApiRequestError",
      operation: "ocr.segment",
    });

    try {
      await fetchWithRequestId(
        "http://localhost:8000/ocr/segment",
        { method: "POST" },
        { operation: "ocr.segment" }
      );
    } catch (error) {
      expect((error as Error).message).not.toContain("12.5");
      expect((error as Error).message).toContain("Diagnostic reference:");
    }
  });
});
