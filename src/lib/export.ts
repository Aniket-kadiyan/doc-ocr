import type { Annotation } from "@/types/annotation";
import {
  buildInspectionSheet,
  type InspectionSheet,
} from "@/lib/project";

const TOLERANCE_NUMBER = String.raw`(?:\d+(?:\.\d+)?|\.\d+)`;
const VALID_TOLERANCE = new RegExp(
  String.raw`^(?:${TOLERANCE_NUMBER}|±\s*${TOLERANCE_NUMBER}|\+\s*${TOLERANCE_NUMBER}\s*,?\s*-\s*${TOLERANCE_NUMBER})$`
);
const VALID_EMBEDDED_TOLERANCE = new RegExp(
  String.raw`^[-+]?\d+(?:\.\d+)?\s*(?:±\s*${TOLERANCE_NUMBER}|\+\s*${TOLERANCE_NUMBER}\s*,?\s*-\s*${TOLERANCE_NUMBER})$`
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
    const nominal = value.match(/[-+]?\d+(?:\.\d+)?/);
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

/**
 * Annotations as XML, carrying the SAME table as the CSV and JSON exports:
 * S.no, Label, Value, Tolerance, the inspector's extra columns, Method, Tool.
 *
 * Column names are carried in a `name` attribute rather than as element names
 * because the extra columns are typed by the user and may contain spaces or
 * other characters that are not valid in an XML element name.
 */
export function exportXML(
  annotations: Annotation[],
  extraColumns: string[] = []
): string {
  const sheet = buildInspectionSheet(annotations, extraColumns);

  const columns = sheet.headers
    .map((h) => `    <column>${escapeXml(h)}</column>`)
    .join("\n");

  const rows = sheet.rows
    .map((row) => {
      const cells = row
        .map(
          (cell, i) =>
            `      <cell name="${escapeXml(sheet.headers[i])}">${escapeXml(
              cell
            )}</cell>`
        )
        .join("\n");
      return `    <row>\n${cells}\n    </row>`;
    })
    .join("\n");

  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    "<inspection>",
    "  <columns>",
    columns,
    "  </columns>",
    "  <rows>",
    rows,
    "  </rows>",
    "</inspection>",
    "",
  ].join("\n");
}

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** Excel (notably on macOS) reads a BOM-less .csv as legacy Mac Roman/ANSI,
 * which mangles Ø, ° and en dashes. A UTF-8 BOM makes it detect UTF-8.
 * Deliberately CSV-only: a BOM breaks strict JSON/XML parsers. */
export const UTF8_BOM = "\ufeff";

export function downloadFile(
  content: string,
  filename: string,
  mime: string
): void {
  const body = mime.includes("csv") ? UTF8_BOM + content : content;
  downloadBlob(new Blob([body], { type: mime }), filename);
}

/** Save an already-built Blob (e.g. a rendered PNG) to the user's downloads. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  // Give the browser a tick to start the download before dropping the URL —
  // a ballooned page PNG is large enough for an immediate revoke to race it.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
