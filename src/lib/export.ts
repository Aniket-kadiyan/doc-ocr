import type { Annotation } from "@/types/annotation";
import { isNoteAnnotation } from "@/lib/notes";
import {
  buildInspectionSheet,
  toleranceForExport,
  type InspectionSheet,
} from "@/lib/project";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
} from "@/types/documentMetadata";

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
 *
 * A drawing NOTES block is prose, not a measurement, and it reads as a
 * malformed tolerance on sight: the first number is its point marker ("1.")
 * and almost any note carries a hyphen after it ("ASTM B633-LATEST REV."),
 * which looks exactly like an unreadable minus tolerance. That blocked every
 * save and export on a sheet whose notes were ballooned, so notes are exempt —
 * they are never given a tolerance and never inspected as a value.
 */
export function findMalformedToleranceAnnotations(
  annotations: Annotation[]
): Annotation[] {
  return annotations.filter((annotation) => {
    if ((annotation.kind ?? "dimension") !== "dimension") return false;
    if (isNoteAnnotation(annotation)) return false;

    const tolerance = toleranceForExport(
      annotation.range,
      annotation.value
    ).trim();
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
/** In-memory form of the inspection JSON export. */
export interface InspectionJSON {
  metadata: DocumentMetadata;
  data: Array<Record<string, string>>;
  extra_columns: string[];
}

/** Build the inspection JSON object for downloads and backend conversion. */
export function buildInspectionJSON(
  annotations: Annotation[],
  extraColumns: string[] = [],
  metadata?: Partial<DocumentMetadata> | null
): InspectionJSON {
  const sheet = buildInspectionSheet(annotations, extraColumns);
  const data = sheet.rows.map((row) =>
    Object.fromEntries(sheet.headers.map((header, i) => [header, row[i]]))
  );
  return {
    metadata: normalizeDocumentMetadata(metadata),
    data,
    extra_columns: sheet.extraColumns,
  };
}

/**
 * Inspection JSON: document metadata plus one object per value under `data`
 * with the S.no / Label / Value / Tolerance / extra columns / Method / Tool
 * fields, plus the
 * inspector-filled `extra_columns` names listed separately (rows carry whatever
 * readings were entered in the checksheet). No bbox/rotation/etc. — use Save
 * Project for a reloadable file.
 */
export function exportInspectionJSON(
  annotations: Annotation[],
  extraColumns: string[] = [],
  metadata?: Partial<DocumentMetadata> | null
): string {
  return JSON.stringify(
    buildInspectionJSON(annotations, extraColumns, metadata),
    null,
    2
  );
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
  extraColumns: string[] = [],
  metadata?: Partial<DocumentMetadata> | null
): string {
  const normalized = normalizeDocumentMetadata(metadata);
  const csv = (s: string) => `"${String(s).replace(/"/g, '""')}"`;
  const metadataRows = [
    ["Metadata Field", "Value"],
    ["Part Name", normalized.partName],
    ["Document Number", normalized.documentNumber],
    ["Revision Number", normalized.revisionNumber],
  ];
  return [
    ...metadataRows.map((row) => row.map(csv).join(",")),
    "",
    inspectionSheetCSV(buildInspectionSheet(annotations, extraColumns)),
  ].join("\n");
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
  extraColumns: string[] = [],
  metadata?: Partial<DocumentMetadata> | null
): string {
  const sheet = buildInspectionSheet(annotations, extraColumns);
  const normalizedMetadata = normalizeDocumentMetadata(metadata);

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
    "  <metadata>",
    `    <partName>${escapeXml(normalizedMetadata.partName)}</partName>`,
    `    <documentNumber>${escapeXml(normalizedMetadata.documentNumber)}</documentNumber>`,
    `    <revisionNumber>${escapeXml(normalizedMetadata.revisionNumber)}</revisionNumber>`,
    "  </metadata>",
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
