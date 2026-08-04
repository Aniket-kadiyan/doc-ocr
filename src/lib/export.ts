import type { Annotation } from "@/types/annotation";
import { buildInspectionSheet, type InspectionSheet } from "@/lib/project";

const TOLERANCE_NUMBER = String.raw`(?:\\d+(?:\\.\\d+)?|\\.\\d+)`;
const VALID_TOLERANCE = new RegExp(
  String.raw`^(?:${TOLERANCE_NUMBER}|±\\s*${TOLERANCE_NUMBER}|\\+\\s*${TOLERANCE_NUMBER}\\s*,?\\s*-\\s*${TOLERANCE_NUMBER})import type { Annotation } from "@/types/annotation";
import { buildInspectionSheet, type InspectionSheet } from "@/lib/project";


);
const VALID_EMBEDDED_TOLERANCE = new RegExp(
  String.raw`^[-+]?\\d+(?:\\.\\d+)?\\s*(?:±\\s*${TOLERANCE_NUMBER}|\\+\\s*${TOLERANCE_NUMBER}\\s*,?\\s*-\\s*${TOLERANCE_NUMBER})import type { Annotation } from "@/types/annotation";
import { buildInspectionSheet, type InspectionSheet } from "@/lib/project";


);

/**
 * Return dimensions containing tolerance-like text that cannot be interpreted.
 * The annotations remain untouched and editable; callers use this only to
 * prevent saving/exporting an ambiguous checksheet.
 */
export function findMalformedToleranceAnnotations(
  annotations: Annotation[]
): Annotation[] {
  return annotations.filter((annotation) => {
    if ((annotation.kind ?? "dimension") !== "dimension") return false;

    const tolerance = (annotation.range ?? "").trim();
    if (tolerance) return !VALID_TOLERANCE.test(tolerance);

    const value = annotation.value.trim();
    const nominal = value.match(/[-+]?\\d+(?:\\.\\d+)?/);
    if (!nominal || nominal.index == null) {
      return /[±+-]/.test(value);
    }

    const remainder = value.slice(nominal.index + nominal[0].length);
    const hasToleranceIntent = /[±+-]/.test(remainder);
    return hasToleranceIntent && !VALID_EMBEDDED_TOLERANCE.test(value);
  });
}

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
