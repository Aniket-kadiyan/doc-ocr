import { getOcrApiUrl } from "@/lib/paddleOcrClient";
import type {
  ChecksheetDetail,
  ChecksheetRunResponse,
  ChecksheetSnapshotPayload,
  ChecksheetSummary,
  ReadingPatch,
} from "@/types/checksheet";

const endpoint = (path: string) => `${getOcrApiUrl()}${path}`;

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
    // Keep the useful plain-text proxy/server response below.
  }
  return text.trim();
}

async function jsonRequest<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(endpoint(path), init);
  if (!response.ok) throw new Error(await responseError(response));
  return (await response.json()) as T;
}

function snapshotForm(
  snapshot: ChecksheetSnapshotPayload,
  document: File
): FormData {
  const form = new FormData();
  form.append("snapshot_json", JSON.stringify(snapshot));
  form.append("document", document, document.name);
  return form;
}

export async function createChecksheet(
  snapshot: ChecksheetSnapshotPayload,
  document: File
): Promise<ChecksheetRunResponse> {
  return jsonRequest<ChecksheetRunResponse>("/checksheets", {
    method: "POST",
    body: snapshotForm(snapshot, document),
  });
}

export async function createChecksheetRevision(
  checksheetId: string,
  snapshot: ChecksheetSnapshotPayload,
  document: File
): Promise<ChecksheetRunResponse> {
  return jsonRequest<ChecksheetRunResponse>(
    `/checksheets/${encodeURIComponent(checksheetId)}/revisions`,
    {
      method: "POST",
      body: snapshotForm(snapshot, document),
    }
  );
}

export async function listChecksheets(args?: {
  search?: string;
  archived?: boolean;
}): Promise<ChecksheetSummary[]> {
  const query = new URLSearchParams();
  if (args?.search?.trim()) query.set("search", args.search.trim());
  if (args?.archived) query.set("archived", "true");
  const suffix = query.size ? `?${query.toString()}` : "";
  const result = await jsonRequest<{ items: ChecksheetSummary[] }>(
    `/checksheets${suffix}`
  );
  return result.items;
}

export async function getChecksheet(
  checksheetId: string
): Promise<ChecksheetDetail> {
  return jsonRequest<ChecksheetDetail>(
    `/checksheets/${encodeURIComponent(checksheetId)}`
  );
}

export async function getChecksheetRun(
  checksheetId: string,
  runId: string
): Promise<ChecksheetRunResponse> {
  return jsonRequest<ChecksheetRunResponse>(
    `/checksheets/${encodeURIComponent(checksheetId)}/runs/${encodeURIComponent(
      runId
    )}`
  );
}

export async function saveChecksheetReadings(
  checksheetId: string,
  runId: string,
  readings: ReadingPatch[]
): Promise<{ status: "saved"; updated_at: string }> {
  return jsonRequest(
    `/checksheets/${encodeURIComponent(checksheetId)}/runs/${encodeURIComponent(
      runId
    )}/readings`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ readings }),
    }
  );
}

export async function completeChecksheetRun(
  checksheetId: string,
  runId: string
): Promise<ChecksheetRunResponse> {
  return jsonRequest<ChecksheetRunResponse>(
    `/checksheets/${encodeURIComponent(checksheetId)}/runs/${encodeURIComponent(
      runId
    )}/complete`,
    { method: "POST" }
  );
}

export async function startChecksheetRun(
  checksheetId: string,
  revisionId?: string
): Promise<ChecksheetRunResponse> {
  return jsonRequest<ChecksheetRunResponse>(
    `/checksheets/${encodeURIComponent(checksheetId)}/runs`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision_id: revisionId ?? null }),
    }
  );
}

export async function updateChecksheet(
  checksheetId: string,
  patch: { name?: string; archived?: boolean }
): Promise<ChecksheetDetail> {
  return jsonRequest<ChecksheetDetail>(
    `/checksheets/${encodeURIComponent(checksheetId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }
  );
}

export async function duplicateChecksheet(
  checksheetId: string,
  name?: string
): Promise<ChecksheetRunResponse> {
  return jsonRequest<ChecksheetRunResponse>(
    `/checksheets/${encodeURIComponent(checksheetId)}/duplicate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name?.trim() || null }),
    }
  );
}

export async function permanentlyDeleteChecksheet(
  checksheetId: string
): Promise<void> {
  const response = await fetch(
    endpoint(`/checksheets/${encodeURIComponent(checksheetId)}`),
    { method: "DELETE" }
  );
  if (!response.ok) throw new Error(await responseError(response));
}

export function checksheetDocumentUrl(documentId: string): string {
  return endpoint(`/checksheet-documents/${encodeURIComponent(documentId)}`);
}
