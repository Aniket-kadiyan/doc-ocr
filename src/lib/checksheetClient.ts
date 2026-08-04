import type { InspectionJSON } from "@/lib/export";
import { getOcrApiUrl } from "@/lib/paddleOcrClient";

/** Result returned after the backend has persisted a generated template. */
export interface SavedChecksheetTemplate {
  template_id: string;
  template_name: string;
  file_name: string;
  row_count: number;
  replaced: boolean;
  index_updated: boolean;
}

interface SaveChecksheetTemplateArgs {
  sourceJson: InspectionJSON;
  sourceFileName: string;
}

/** Extract a useful FastAPI error without assuming every error is JSON. */
async function responseError(response: Response): Promise<string> {
  const fallback = `Server responded ${response.status} ${response.statusText}`;
  const text = await response.text().catch(() => "");
  if (!text.trim()) return fallback;

  try {
    const body = JSON.parse(text) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.trim()) {
      return body.detail;
    }
  } catch {
    // A non-JSON backend/proxy response is already useful as plain text.
  }

  return text.trim();
}

/**
 * Ask the local Python backend to convert and save one Digital Checksheet
 * template. The backend owns the configured destination directory; no path is
 * accepted from browser code.
 */
export async function saveChecksheetTemplate({
  sourceJson,
  sourceFileName,
}: SaveChecksheetTemplateArgs): Promise<SavedChecksheetTemplate> {
  const response = await fetch(`${getOcrApiUrl()}/checksheet/templates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_json: sourceJson,
      source_file_name: sourceFileName,
    }),
  });

  if (!response.ok) {
    throw new Error(await responseError(response));
  }

  return (await response.json()) as SavedChecksheetTemplate;
}
