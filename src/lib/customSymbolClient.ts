/** Client for the backend's shared custom symbol library. */

import { getOcrApiUrl } from "@/lib/paddleOcrClient";
import { apiRequestError, fetchWithRequestId } from "@/lib/requestDiagnostics";
import type {
  CustomSymbol,
  CustomSymbolLibrary,
  CustomSymbolUpdate,
  NewCustomSymbol,
} from "@/types/customSymbol";

const OPERATION = "custom-symbols.request";

async function responseError(response: Response): Promise<string> {
  const fallback = `Server responded ${response.status} ${response.statusText}`;
  const text = await response.text().catch(() => "");
  if (!text.trim()) return fallback;
  try {
    const body = JSON.parse(text) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.trim()) {
      return body.detail;
    }
    if (Array.isArray(body.detail) && body.detail.length > 0) {
      return body.detail
        .map((item) => {
          const entry = item as { msg?: unknown };
          return typeof entry.msg === "string"
            ? entry.msg
            : JSON.stringify(item);
        })
        .join("; ");
    }
  } catch {
    // Keep the plain-text server response below.
  }
  return text.trim();
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetchWithRequestId(`${getOcrApiUrl()}${path}`, init, {
    operation: OPERATION,
  });
  if (!response.ok) {
    throw apiRequestError(response, await responseError(response), OPERATION);
  }
  return response;
}

export async function listCustomSymbols(): Promise<CustomSymbolLibrary> {
  const response = await request("/custom-symbols", { cache: "no-store" });
  return (await response.json()) as CustomSymbolLibrary;
}

export async function createCustomSymbol(
  symbol: NewCustomSymbol
): Promise<CustomSymbol> {
  const form = new FormData();
  form.append("file", symbol.file, symbol.file.name);
  form.append("name", symbol.name);
  form.append("label", symbol.label);
  form.append("category", symbol.category);
  form.append("threshold", String(symbol.threshold));
  form.append("match_rotations", String(symbol.matchRotations));
  form.append("any_size", String(symbol.anySize));
  const response = await request("/custom-symbols", {
    method: "POST",
    body: form,
  });
  return (await response.json()) as CustomSymbol;
}

export async function updateCustomSymbol(
  id: string,
  changes: CustomSymbolUpdate
): Promise<CustomSymbol> {
  const response = await request(`/custom-symbols/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
  return (await response.json()) as CustomSymbol;
}

export async function deleteCustomSymbol(id: string): Promise<void> {
  await request(`/custom-symbols/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}
