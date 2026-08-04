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

/** How a label's text was produced: typed by hand or OCR'd from the drawing. */
export type LabelSource = "manual" | "ocr";

/**
 * Two kinds of annotation share the canvas:
 *  - "dimension": a value read from the drawing (Draw Box / Auto-Segment).
 *  - "label": a free callout added with the Add Label tool. Labels are drawn in
 *    a different color and numbered in their own 1,2,3… sequence.
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
  label: string;
  value: string;
  type: DimensionType;
  confidence: number;
  bbox: BBox;
  rotation: number;
  page: number;
  createdAt: number;
  /** "dimension" (default) or "label". Drives color + numbering sequence. */
  kind?: AnnotationKind;
  /** Backend flagged the OCR read as uncertain. */
  needsReview?: boolean;
  /** For labels: whether the text was typed manually or OCR'd from the drawing. */
  labelSource?: LabelSource;
  /** Inspection method (free text) — dimensions only. */
  method?: string;
  /** Inspection tool chosen from TOOL_OPTIONS — dimensions only. */
  tool?: string;
  /** id of the label-kind annotation this dimension maps to. Selecting either
   * highlights both on the drawing and in the sidebar. */
  labelId?: string;
  /** Tolerance after a ± in the value, formatted "+x, -x" — dimensions only. */
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
  /** "dimension" (a value) or "label" (Add Label). Defaults to dimension. */
  kind?: AnnotationKind;
  /** For label pending: whether the user chose to type it or OCR the box. */
  labelSource?: LabelSource;
  /** When adding a value to a specific label, the id of that label. The popup
   * binds the value to it one-to-one instead of showing a label picker. */
  labelId?: string;
}

/** Which way the Add Label tool is capturing a label: typed or OCR'd. */
export type LabelInputMode = "manual" | "ocr";
