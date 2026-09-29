import type { Annotation } from "@/types/annotation";
import { balloonLabel } from "@/lib/featureLabel";
import { collectNotePoints } from "@/lib/notes";
import { getTitleKeywords } from "@/lib/titleKeywords";
import {
  isMetadataTitleAnnotation,
  metadataFieldForTitleLabel,
} from "@/lib/titleMetadata";
import { renumberValueAnnotations } from "@/lib/annotationNumbers";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
} from "@/types/documentMetadata";

/**
 * Self-contained project bundle: the source drawing (embedded as a data URL so
 * the file is portable) plus every annotation and a verification payload that a
 * remote checking server can consume directly. Saving produces one `.docbox.json`
 * file; loading restores both the drawing and its annotations.
 */
export const PROJECT_FORMAT = "doc-ocr-box.project";
export const PROJECT_VERSION = 1;

export interface ProjectSource {
  fileName: string;
  mimeType: string;
  fileType: "pdf" | "image";
  /** data:<mime>;base64,... — the original bytes, so the bundle is portable. */
  dataUrl: string;
}

/** Minimal {value,label} view sent to a remote server for checking. */
export interface VerificationItem {
  number: number;
  label: string;
  value: string;
  type: Annotation["type"];
  kind: NonNullable<Annotation["kind"]>;
  method: string;
  tool: string;
  range: string;
  /** id of the mapped label annotation (dimensions only). */
  labelId: string;
  page: number;
  confidence: number;
  bbox: Annotation["bbox"];
}

/**
 * Inspection table laid out for the shop floor: one row per dimension with
 * columns S.no, Label, Method, tool, value+range, then one empty column per
 * physical part (part1…partN) for measured readings.
 */
export interface InspectionSheet {
  /** Names of the extra, inspector-filled columns inserted before Method/Tool. */
  extraColumns: string[];
  headers: string[];
  rows: string[][];
}

export interface VerificationPayload {
  projectName: string;
  metadata: DocumentMetadata;
  /** Optional remote endpoint baked into the file for reference. */
  endpoint: string;
  items: VerificationItem[];
  /** Present when the user supplied a part count for inspection. */
  sheet?: InspectionSheet;
}

/** Combine a dimension's value with its ± range, e.g. `5.27 +0.50, -0.50`. */
export function valueWithRange(a: Annotation): string {
  return a.range ? `${a.value} ${a.range}` : a.value;
}

const EXPORT_TOLERANCE_NUMBER = String.raw`(?:\d+(?:\.\d+)?|\.\d+)`;
const SINGLE_UPPER_TOLERANCE = new RegExp(
  String.raw`^\+\s*(${EXPORT_TOLERANCE_NUMBER})$`
);
const SINGLE_LOWER_TOLERANCE = new RegExp(
  String.raw`^-\s*(${EXPORT_TOLERANCE_NUMBER})$`
);

/**
 * Complete a one-sided tolerance for exported data without changing the
 * editable annotation. Existing zero, symmetric, and two-sided tolerances are
 * returned unchanged.
 */
export function toleranceForExport(
  tolerance?: string,
  value?: string
): string {
  const original = tolerance ?? "";
  const trimmed = original.trim();

  const upper = SINGLE_UPPER_TOLERANCE.exec(trimmed);
  if (upper) return `+${upper[1]}, -0`;

  const lower = SINGLE_LOWER_TOLERANCE.exec(trimmed);
  if (lower) return `+0, -${lower[1]}`;

  // An explicit range always wins. Only inspect Value when Range is empty.
  if (trimmed) return original;

  const embedded = new RegExp(
    String.raw`([+-])\s*(${EXPORT_TOLERANCE_NUMBER})\s*$`
  ).exec(value?.trim() ?? "");
  if (embedded && embedded.index != null) {
    const specification = (value ?? "").slice(0, embedded.index);
    // The sign must follow an existing numeric specification. This prevents a
    // negative nominal such as R-5 from being mistaken for a tolerance.
    if (/\d\s*$/.test(specification)) {
      return embedded[1] === "+"
        ? `+${embedded[2]}, -0`
        : `+0, -${embedded[2]}`;
    }
  }

  return original;
}

/** New projects contain value annotations only. Older projects may still have
 * separate label annotations, which are merged into their mapped value here. */
export interface LegacyAnnotationMigration {
  annotations: Annotation[];
  orphanLabelCount: number;
}

export function normalizeLegacyAnnotations(
  annotations: Annotation[]
): LegacyAnnotationMigration {
  const legacyLabels = new Map(
    annotations
      .filter((annotation) => annotation.kind === "label")
      .map((annotation) => [annotation.id, annotation])
  );
  const values = annotations.filter(
    (annotation) => annotation.kind !== "label"
  );
  const usedLabelIds = new Set(
    values
      .map((annotation) => annotation.labelId)
      .filter((id): id is string => Boolean(id && legacyLabels.has(id)))
  );

  const normalized = values.map((annotation) => {
    const parent = annotation.labelId
      ? legacyLabels.get(annotation.labelId)
      : undefined;
    const {
      labelId: _labelId,
      labelSource: _labelSource,
      ...valueAnnotation
    } = annotation;
    void _labelId;
    void _labelSource;

    return {
      ...valueAnnotation,
      kind: "dimension" as const,
      // The old parent was the editable source of truth for label text.
      label: parent ? parent.value : annotation.label ?? "",
      method: annotation.method || parent?.method || undefined,
      tool: annotation.tool || parent?.tool || undefined,
    };
  });

  return {
    annotations: renumberValueAnnotations(normalized),
    orphanLabelCount: [...legacyLabels.keys()].filter(
      (id) => !usedLabelIds.has(id)
    ).length,
  };
}

/** Values only, including old records where kind was omitted. */
export function valueAnnotations(annotations: Annotation[]): Annotation[] {
  return normalizeLegacyAnnotations(annotations).annotations.sort(
    (left, right) => left.number - right.number
  );
}

/** Build the inspection table. Columns: S.no, Label, Value, Tolerance, then any
 * inspector-filled `extraColumns` (left blank), then Method and Tool. One row
 * per value (dimension). */
export function buildInspectionSheet(
  annotations: Annotation[],
  extraColumns: string[] = []
): InspectionSheet {
  const headers = [
    "S.no",
    "Label",
    "Value",
    "Tolerance",
    ...extraColumns,
    "Method",
    "Tool",
  ];
  const rows = valueAnnotations(annotations)
    .filter((annotation) => !isMetadataTitleAnnotation(annotation))
    .sort((a, b) => a.number - b.number)
    .map((a) => [
      String(a.number),
      oneLine(balloonLabel(a)),
      oneLine(a.value),
      toleranceForExport(a.range, a.value),
      ...extraColumns.map((col) => a.extras?.[col] ?? ""),
      a.method ?? "",
      a.tool ?? "",
    ]);
  return {
    extraColumns,
    headers,
    rows: [
      ...rows,
      // Title-block fields first, then the notes paragraph below them.
      ...titleFieldRows(annotations, headers.length),
      ...notesRows(annotations, headers.length),
    ],
  };
}

/** Blank rows separating the dimension table from the title-block fields. */
const TITLE_GAP_ROWS = 3;

/**
 * The configured "Keywords to look for" and the value read for each, written
 * under the dimension rows: keyword in the Label column, value in the Value
 * column.
 *
 * Every configured keyword gets a row, including ones the OCR could not find —
 * a blank value is a prompt to fill the field in by hand, where dropping the
 * row would hide the miss entirely.
 */
function titleFieldRows(annotations: Annotation[], width: number): string[][] {
  const keywords = getTitleKeywords().filter(
    (keyword) => metadataFieldForTitleLabel(keyword) === null
  );
  if (keywords.length === 0) return [];

  const found = annotations.filter(
    (a) => a.type === "Title Block" && !isMetadataTitleAnnotation(a)
  );
  const norm = (s: string) => s.toUpperCase().replace(/[^A-Z0-9]+/g, " ").trim();

  const blank = () => Array<string>(width).fill("");
  const rows = keywords.map((keyword) => {
    const match = found.find((a) => norm(a.label).startsWith(norm(keyword)));
    const row = blank();
    row[1] = match ? oneLine(match.label) : keyword; // Label column
    row[2] = match ? oneLine(match.value) : ""; // Value column
    return row;
  });

  return [...Array.from({ length: TITLE_GAP_ROWS }, blank), ...rows];
}

/** Flatten a multi-line read onto one sheet line. A dimension never contains a
 * newline; a notes block kept inline does, and it renders as one very tall cell
 * in Excel if left as-is. */
function oneLine(s: string): string {
  return s.replace(/\s*\n\s*/g, " ").trim();
}

/** Blank rows separating the dimension table from the NOTES block below it. */
const NOTES_GAP_ROWS = 6;

/**
 * The NOTES block appended under the dimension rows: a gap, a "NOTES" heading,
 * then one row per numbered point with the drawing's own numbering kept in the
 * text. Returns [] when the drawing has no notes, so sheets without a notes
 * block are unchanged.
 *
 * Notes are additive: the note annotation keeps its balloon and its ordinary
 * dimension row above, and is expanded point-by-point here.
 */
function notesRows(annotations: Annotation[], width: number): string[][] {
  const points = collectNotePoints(annotations);
  if (points.length === 0) return [];

  const blank = () => Array<string>(width).fill("");
  const textRow = (text: string) => {
    const row = blank();
    row[1] = text; // Label column
    return row;
  };

  return [
    ...Array.from({ length: NOTES_GAP_ROWS }, blank),
    textRow("NOTES"),
    ...points.map(textRow),
  ];
}

export interface ProjectBundle {
  format: typeof PROJECT_FORMAT;
  version: number;
  projectName: string;
  savedAt: number;
  source: ProjectSource;
  annotations: Annotation[];
  metadata: DocumentMetadata;
  /** Optional/legacy: older project files embedded a verification payload that
   * just mirrored `annotations`. New saves omit it; "Send for Verification"
   * rebuilds it on demand. */
  verification?: VerificationPayload;
}

/** Reduce annotations to the value/label view a checking server needs. When
 * `extraColumns` is given, attach the inspection sheet with those columns. */
export function buildVerificationPayload(
  annotations: Annotation[],
  projectName: string,
  endpoint = verificationEndpoint(),
  extraColumns?: string[],
  metadata?: Partial<DocumentMetadata> | null
): VerificationPayload {
  return {
    projectName,
    metadata: normalizeDocumentMetadata(metadata),
    endpoint,
    ...(extraColumns != null
      ? { sheet: buildInspectionSheet(annotations, extraColumns) }
      : {}),
    items: valueAnnotations(annotations)
      .filter((annotation) => !isMetadataTitleAnnotation(annotation))
      .map((a) => ({
      number: a.number,
      label: a.label ?? "",
      value: a.value,
      type: a.type,
      kind: "dimension",
      method: a.method ?? "",
      tool: a.tool ?? "",
      range: toleranceForExport(a.range, a.value),
      labelId: "",
      page: a.page,
      confidence: a.confidence,
      bbox: a.bbox,
      })),
  };
}

/** Configured verification server, if any (NEXT_PUBLIC_VERIFY_API_URL). */
export function verificationEndpoint(): string {
  return process.env.NEXT_PUBLIC_VERIFY_API_URL ?? "";
}

/** Serialize a full project (drawing + annotations + verification) to JSON. */
export function buildProjectBundle(args: {
  projectName: string;
  source: ProjectSource;
  annotations: Annotation[];
  metadata?: Partial<DocumentMetadata> | null;
  savedAt: number;
}): string {
  const bundle: ProjectBundle = {
    format: PROJECT_FORMAT,
    version: PROJECT_VERSION,
    projectName: args.projectName,
    savedAt: args.savedAt,
    source: args.source,
    annotations: normalizeLegacyAnnotations(args.annotations).annotations,
    metadata: normalizeDocumentMetadata(args.metadata),
  };
  return JSON.stringify(bundle, null, 2);
}

/** Parse and validate a `.docbox.json` bundle. Throws on malformed input. */
export function parseProjectBundle(text: string): ProjectBundle {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error("Not a valid project file (invalid JSON).");
  }
  const b = raw as Partial<ProjectBundle>;
  if (b?.format !== PROJECT_FORMAT) {
    throw new Error("Not a doc-ocr-box project file.");
  }
  if (!b.source?.dataUrl || !b.source.fileType) {
    throw new Error("Project file is missing its embedded drawing.");
  }
  if (!Array.isArray(b.annotations)) {
    throw new Error("Project file is missing its annotations.");
  }
  return {
    ...(b as ProjectBundle),
    metadata: normalizeDocumentMetadata(b.metadata),
  };
}

/** POST the value/label payload to a remote server for checking. */
export async function sendForVerification(
  url: string,
  payload: VerificationPayload
): Promise<unknown> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Server responded ${res.status} ${res.statusText}`);
  }
  const ct = res.headers.get("content-type") ?? "";
  return ct.includes("application/json") ? res.json() : res.text();
}

/** Read a File as a base64 data URL (for embedding the source drawing). */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error("File read failed"));
    reader.readAsDataURL(file);
  });
}

/** Reconstruct a File from a stored data URL (for re-rendering on load). */
export function dataUrlToFile(
  dataUrl: string,
  fileName: string,
  mimeType: string
): File {
  const comma = dataUrl.indexOf(",");
  const base64 = dataUrl.slice(comma + 1);
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new File([bytes], fileName, {
    type: mimeType || dataUrl.slice(5, comma).split(";")[0],
  });
}
