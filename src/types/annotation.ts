/**
 * Feature category assigned by the GD&T rule engine (GDTOCR spec). The string
 * values are the single source of truth shared with the backend
 * (backend/feature_dictionary.py CAT_* constants). Keep the two in sync.
 */
export const DIMENSION_TYPES = [
  "Diameter",
  "Radius",
  "Chamfer",
  "Angle",
  "Taper",
  "Linear",
  "Tolerance",
  "Thread",
  "Hole",
  "GD&T",
  "Datum",
  "Surface Finish",
  "Weld",
  "Material",
  "Heat Treatment",
  "Coating",
  "General Note",
  "Reference",
  "Basic",
  "Note",
  "Title Block",
  "Unknown",
] as const;

export type DimensionType =
  | "Diameter"
  | "Radius"
  | "Chamfer"
  | "Angle"
  | "Taper"
  | "Linear"
  | "Tolerance"
  | "Thread"
  | "Hole"
  | "GD&T"
  | "Datum"
  | "Surface Finish"
  | "Weld"
  | "Material"
  | "Heat Treatment"
  | "Coating"
  | "General Note"
  | "Reference"
  | "Basic"
  | "Note"
  | "Title Block"
  | "Unknown";

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type RecognitionSource = "native_pdf" | "ocr" | "native_pdf+ocr";

/** Independent readings retained for audit and later review. */
export interface RecognitionEvidence {
  selectedSource: RecognitionSource;
  sources: Array<"native_pdf" | "ocr">;
  nativeText: string;
  ocrText: string;
  agreement: number;
  conflict: boolean;
  nativeSpanIds: string[];
  nativeBBox?: BBox;
}

/** One primitive text detection retained inside an assembled callout. */
export interface EngineeringObjectChildEvidence {
  candidateId: string;
  text: string;
  rawText: string;
  bbox: BBox;
  polygon?: Array<[number, number]>;
  confidence: number;
  orientation: "horizontal" | "vertical" | "rotated";
  rotation: number;
  role: string;
  recognitionSource?: RecognitionSource;
  recognitionEvidence?: RecognitionEvidence;
  sourceConflict?: boolean;
}

/** Lossless evidence describing how primitive detections formed one object. */
export interface EngineeringObjectAssembly {
  objectId: string;
  assemblyId: string;
  rule: string;
  conflict: boolean;
  reviewReason?: string;
  children: EngineeringObjectChildEvidence[];
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
  /** Optional rule-engine subtype retained for downstream labeling/export. */
  subtype?: string;
  confidence: number;
  bbox: BBox;
  rotation: number;
  /** Tight rotated rectangle for slanted callouts (source coords + clockwise
   * degrees). When present the drawing renders this instead of the loose
   * axis-aligned {@link bbox}, which is oversized for diagonal text. */
  orientedBox?: BBox & { rotation: number };
  page: number;
  createdAt: number;
  /** New records use "dimension". "label" is accepted only during migration. */
  kind?: AnnotationKind;
  /** Backend flagged the OCR read as uncertain. */
  needsReview?: boolean;
  /** Native-PDF/OCR readings and the source selected for this value. */
  recognitionEvidence?: RecognitionEvidence;
  /** Primitive detections retained when this value was assembled. */
  assembly?: EngineeringObjectAssembly;
  /** Presentation-only per-balloon visibility. Hidden values remain numbered,
   * persisted, exported, and active for duplicate prevention. */
  hidden?: boolean;
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
  /** Legacy browser-based inspector readings retained for older projects and
   * values-only CSV/JSON exports. Internal checksheet runs use backend storage. */
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
  /** Native-PDF/OCR readings and the source selected for this value. */
  recognitionEvidence?: RecognitionEvidence;
  assembly?: EngineeringObjectAssembly;
  /** Rule-engine classification returned by manual region OCR. */
  category?: string;
  subtype?: string;
  label?: string;
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
  source?: "manual" | "scan_review" | "scan_candidate";
  reviewCandidateId?: string;
  reviewReason?: string;
  scanCandidateState?: "review" | "other";
  suggestedType?: DimensionType;
  suggestedSubtype?: string;
  suggestedLabel?: string;
  orientedBox?: BBox & { rotation: number };
}
