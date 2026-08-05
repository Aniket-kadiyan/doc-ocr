export type DimensionType =
  | "Diameter"
  | "Radius"
  | "Angle"
  | "Tolerance"
  | "Linear"
  | "Note"
  | "Unknown";

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Legacy label metadata retained only so older saved projects can be read. */
export type LabelSource = "manual" | "ocr";

/**
 * New annotations are always dimensions/values. The "label" kind remains in
 * the type only for automatic migration of older label-first project files.
 */
export type AnnotationKind = "dimension" | "label";

/** Inspection tools offered in the dimension popup's Tool dropdown (placeholder
 * set — swap for the real shop list later). */
export const TOOL_OPTIONS = [
  "Vernier Caliper",
  "Micrometer",
  "Height Gauge",
  "Pressure Gauge",
] as const;

export type ToolOption = (typeof TOOL_OPTIONS)[number];

export interface Annotation {
  id: string;
  number: number;
  /** Optional checksheet metadata; an empty string means no label is assigned. */
  label: string;
  value: string;
  type: DimensionType;
  confidence: number;
  bbox: BBox;
  rotation: number;
  page: number;
  createdAt: number;
  /** New records use "dimension". "label" is accepted only during migration. */
  kind?: AnnotationKind;
  /** Backend flagged the OCR read as uncertain. */
  needsReview?: boolean;
  /** Legacy label-first project field. */
  labelSource?: LabelSource;
  /** Optional inspection method associated directly with this value. */
  method?: string;
  /** Optional inspection tool associated directly with this value. */
  tool?: string;
  /** Legacy parent-label id, removed when an older project is normalized. */
  labelId?: string;
  /** Editable tolerance; numeric values default to "0" and embedded ± values
   * are normalized to "+x, -x". */
  range?: string;
  /** Inspector-filled readings keyed by extra-column name (e.g. {part1: "32.01"}).
   * Populated from the checksheet web view; flows into CSV/JSON exports. */
  extras?: Record<string, string>;
}

export interface OCRWordBox {
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export type OcrEngine = "paddleocr" | "paddleocr+vision" | "paddleocr+compose";

export interface OCRResult {
  text: string;
  confidence: number;
  rotation: number;
  orientation: "horizontal" | "vertical" | "rotated";
  words: OCRWordBox[];
  engine?: OcrEngine;
  /** Cross-variant agreement ratio (0–1) from the backend consensus vote. */
  agreement?: number;
  /** Backend flagged this read as uncertain — prompt the user to verify. */
  needsReview?: boolean;
  /** Tight bbox of the detected text in source-canvas coords (snap target). */
  valueBox?: BBox;
  /** backend/debug_output/<hash>/ when step dump succeeded */
  debugDumpDir?: string;
  debugDumpSkipped?: string;
}

export interface PendingSelection {
  bbox: BBox;
  page: number;
  ocrResult: OCRResult;
}
