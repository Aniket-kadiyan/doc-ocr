import { afterEach, describe, expect, it, vi } from "vitest";
import {
  apiRequestError, ApiRequestError, fetchWithRequestId, responseRequestId,
} from "@/lib/requestDiagnostics";

afterEach(() => vi.unstubAllGlobals());

describe("request diagnostics", () => {
  it("adds and remembers a request ID", async () => {
    const response = { ok: true, status: 200 } as Response;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
      void _input;
      void _init;
      return response;
    });
    vi.stubGlobal("fetch", fetchMock);
    const received = await fetchWithRequestId(
      "http://localhost:8000/health", { method: "GET" }, { operation: "health.read" }
    );
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const id = new Headers(init.headers).get("X-Request-ID");
    expect(id).toBeTruthy();
    expect(responseRequestId(received)).toBe(id);
  });

  it("uses the backend response ID in an error", () => {
    const response = new Response("", {
      status: 503, headers: { "X-Request-ID": "backend-reference-7" },
    });
    const error = apiRequestError(response, "Could not start scan.", "scan.create");
    expect(error).toBeInstanceOf(ApiRequestError);
    expect(error.message).toContain("backend-reference-7");
  });

  it("does not copy a network exception message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new Error("private drawing value 12.5");
    }));
    await expect(fetchWithRequestId(
      "http://localhost:8000/ocr/segment", {}, { operation: "ocr.segment" }
    )).rejects.not.toThrow("12.5");
  });
});
