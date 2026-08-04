import type { Annotation } from "@/types/annotation";
import { buildInspectionSheet, type InspectionSheet } from "@/lib/project";

/** In-memory form of the values-only inspection JSON export. */
export interface InspectionJSON {
  data: Array<Record<string, string>>;
  extra_columns: string[];
}

/** Build the inspection JSON object for downloads and backend conversion. */
export function buildInspectionJSON(
  annotations: Annotation[],
  extraColumns: string[] = []
): InspectionJSON {
  const sheet = buildInspectionSheet(annotations, extraColumns);
  const data = sheet.rows.map((row) =>
    Object.fromEntries(sheet.headers.map((header, i) => [header, row[i]]))
  );
  return { data, extra_columns: sheet.extraColumns };
}

/**
 * Values-only JSON: one object per value under `data` with the S.no / Label /
 * Value / Tolerance / extra columns / Method / Tool fields, plus the
 * inspector-filled `extra_columns` names listed separately (rows carry whatever
 * readings were entered in the checksheet). No bbox/rotation/etc. — use Save
 * Project for a reloadable file.
 */
export function exportInspectionJSON(
  annotations: Annotation[],
  extraColumns: string[] = []
): string {
  return JSON.stringify(buildInspectionJSON(annotations, extraColumns), null, 2);
}

/** Serialize an inspection sheet (S.no, Label, Value, Tolerance, extra columns,
 * Method, Tool) to CSV. */
export function inspectionSheetCSV(sheet: InspectionSheet): string {
  const csv = (s: string) => `"${String(s).replace(/"/g, '""')}"`;
  return [sheet.headers, ...sheet.rows]
    .map((row) => row.map(csv).join(","))
    .join("\n");
}

/** Build and serialize the inspection sheet with the given extra columns. */
export function exportInspectionCSV(
  annotations: Annotation[],
  extraColumns: string[] = []
): string {
  return inspectionSheetCSV(buildInspectionSheet(annotations, extraColumns));
}

export function exportXML(annotations: Annotation[]): string {
  const items = annotations
    .map(
      (a) => `  <annotation id="${a.id}" number="${a.number}" page="${a.page}" type="${a.type}" confidence="${a.confidence}">
    <value>${escapeXml(a.value)}</value>
    <label>${escapeXml(a.label)}</label>
    <range>${escapeXml(a.range ?? "")}</range>
    <method>${escapeXml(a.method ?? "")}</method>
    <tool>${escapeXml(a.tool ?? "")}</tool>
    <bbox x="${a.bbox.x}" y="${a.bbox.y}" width="${a.bbox.width}" height="${a.bbox.height}" rotation="${a.rotation}"/>
  </annotation>`
    )
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>\n<annotations>\n${items}\n</annotations>`;
}

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function downloadFile(
  content: string,
  filename: string,
  mime: string
): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
