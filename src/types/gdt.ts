/**
 * GD&T recognition types, mirroring the backend payloads.
 *
 * The field names match `backend/gdt_detect.Detection.to_dict`,
 * `backend/fcf_parse.FeatureControlFrame.to_dict` and
 * `backend/gdt_pipeline.PageResult.to_dict` exactly. Keep the two in sync —
 * the backend is the source of truth, as it is for `DimensionType`.
 */

import type { BBox } from "@/types/annotation";

/** The recognizer committed to this reading. */
export const GDT_CONFIRMED = "confirmed";
/** Something is there; it is one of `alternatives`, but which is not settled. */
export const GDT_AMBIGUOUS = "ambiguous";
/** A mark that nothing in the registry explains well enough. */
export const GDT_UNKNOWN = "unknown";

export type GdtStatus =
  | typeof GDT_CONFIRMED
  | typeof GDT_AMBIGUOUS
  | typeof GDT_UNKNOWN;

export type GdtCategory =
  | "geometric_tolerance"
  | "datum"
  | "material_condition"
  | "dimensioning"
  | "manufacturing"
  | "frame_element"
  | "rule_or_principle";

export type GdtKind =
  | "standalone"
  | "modifier"
  | "composite"
  | "frame"
  | "concept"
  | "textual";

/**
 * Whether a frame field was read, is genuinely absent, could not be read, or
 * cannot apply. A `null` value alone cannot carry that distinction, and the
 * difference matters: "absent" on a material condition means RFS, while
 * "unreadable" means nobody knows yet.
 */
export type GdtFieldState =
  | "present"
  | "absent"
  | "unreadable"
  | "not_applicable";

export interface GdtField<T> {
  value: T | null;
  state: GdtFieldState;
  confidence: number;
  raw: string | null;
}

export interface GdtStandards {
  standards: string[];
  retired_in: string | null;
  current_in: string[];
  exclusive_to: string | null;
  is_legacy: boolean;
  note: string | null;
}

export interface GdtCandidate {
  symbol_id: string;
  name: string;
  confidence: number;
  method: string;
}

export interface GdtBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A value read alongside a symbol. `unit` is always the drawing's own. */
export interface GdtValue {
  kind: "length" | "angle" | "ratio" | "count" | "letter" | "tolerance";
  value: number | string | null;
  unit?: string;
  raw?: string;
  numerator?: number;
  denominator?: number;
}

export interface GdtDetection {
  symbol_id: string | null;
  name: string;
  status: GdtStatus;
  confidence: number;
  bounding_box: GdtBoundingBox;
  page: number;
  method: string;
  raw_ocr: string | null;
  normalized: string | null;
  alternatives: GdtCandidate[];
  associated_text: string | null;
  value: GdtValue | null;
  warnings: string[];
  evidence?: Record<string, unknown>;
  frame_id?: string;
  category?: GdtCategory;
  kind?: GdtKind;
  description?: string;
  standards?: GdtStandards;
}

export interface GdtDatumRef {
  label: string;
  precedence: number;
  material_condition: string | null;
  confidence: number;
  raw: string | null;
}

export interface GdtCompartment {
  index: number;
  bounding_box: GdtBoundingBox;
  text: string;
  confidence: number;
  detection: GdtDetection | null;
}

export interface GdtFrame {
  type: "feature_control_frame";
  id: string;
  page: number;
  bounding_box: GdtBoundingBox;
  symbol: GdtDetection | null;
  tolerance: GdtField<number>;
  diameter_zone: GdtField<boolean>;
  material_condition: GdtField<string>;
  modifiers: string[];
  datums: GdtDatumRef[];
  compartments: GdtCompartment[];
  segment_index: number;
  composite_id: string | null;
  confidence: number;
  warnings: string[];
  /** Always "drawing_unit" — the unit comes from the drawing, not the symbol. */
  tolerance_unit: string;
  leader_point?: { x: number; y: number };
}

export interface GdtDatumAssociation {
  frame_id: string | null;
  label: string;
  precedence: number;
  material_condition: string | null;
  resolved: boolean;
  datum_feature_bbox: GdtBoundingBox | null;
  note: string | null;
}

export interface GdtStandardsFinding {
  symbol_id: string;
  name: string;
  issue: "standard_mismatch" | "retired_control";
  detail: string;
  confidence: number;
}

export interface GdtPageResult {
  page: number;
  image_size: { width: number; height: number };
  frames: GdtFrame[];
  symbols: GdtDetection[];
  datum_associations: GdtDatumAssociation[];
  standards_findings: GdtStandardsFinding[];
  errors: Array<{ stage: string; error: string; [key: string]: unknown }>;
  counts: {
    frames: number;
    symbols: number;
    confirmed: number;
    ambiguous: number;
    unknown: number;
    errors: number;
  };
  elapsed_ms: number;
  provenance: Record<string, unknown>;
  /** Content hash of the submitted image; ties a correction to this drawing. */
  document_hash?: string;
}

// ── The registry, as served by /gdt/registry ─────────────────────────────────

export interface GdtRegistryEntry {
  id: string;
  name: string;
  category: GdtCategory;
  kind: GdtKind;
  unicode: string | null;
  aliases: string[];
  ocr_variants: string[];
  description: string;
  visual: string;
  template: string | null;
  confusable_with: string[];
  strategy: string[];
  detectable: boolean;
  frame_characteristic: boolean;
  value: {
    kind: string;
    optional: boolean;
    pattern: string | null;
    note: string | null;
  };
  standards: GdtStandards;
}

export interface GdtRegistry {
  version: number;
  categories: GdtCategory[];
  kinds: GdtKind[];
  strategies: string[];
  count: number;
  symbols: GdtRegistryEntry[];
}

// ── Corrections ──────────────────────────────────────────────────────────────

export type GdtCorrectionAction =
  | "reclassify"
  | "reject"
  | "confirm"
  | "add";

export interface GdtCorrection {
  action: GdtCorrectionAction;
  detected_symbol_id?: string | null;
  corrected_symbol_id?: string | null;
  page?: number;
  bounding_box?: GdtBoundingBox | null;
  detected_confidence?: number | null;
  detected_status?: string | null;
  document_hash?: string | null;
  note?: string | null;
}

/** A detection paired with the page-space box the viewer should paint. */
export interface GdtOverlayItem {
  detection: GdtDetection;
  bbox: BBox;
  /** The frame this symbol belongs to, when it sits in one. */
  frame: GdtFrame | null;
}
