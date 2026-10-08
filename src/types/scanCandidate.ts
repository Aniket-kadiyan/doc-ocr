import type {
  BBox,
  DimensionType,
  EngineeringDisposition,
  EngineeringObjectAssembly,
  EngineeringValueParse,
  RecognitionEvidence,
} from "@/types/annotation";

export type ScanCandidateState = "review" | "other" | "ignored";
export type RestorableScanCandidateState = Exclude<
  ScanCandidateState,
  "ignored"
>;

/**
 * A detected OCR object that has not become an accepted balloon.
 *
 * These records deliberately live beside annotations rather than inside the
 * annotation store. That keeps exports accepted-only while preserving every
 * unresolved or deliberately ignored detector outcome for later inspection.
 */
export interface ScanCandidate {
  id: string;
  /** Backend identity within the scan result (for diagnostics/audit). */
  sourceCandidateId?: string;
  page: number;
  order: number;
  state: ScanCandidateState;
  /** State restored when an ignored candidate is returned to the queue. */
  restoreState?: RestorableScanCandidateState;
  text: string;
  /** Unmodified final OCR text before display-oriented symbol normalization. */
  rawText: string;
  preliminaryText?: string;
  confidence: number;
  recognized: boolean;
  reason: string;
  rule?: string;
  type?: string;
  category?: DimensionType | string;
  subtype?: string;
  label?: string;
  orientation: "horizontal" | "vertical" | "rotated";
  rotation: number;
  recoveryAttempted?: boolean;
  authoritativeReread?: boolean;
  /** Native-PDF/OCR readings retained while this object awaits disposition. */
  recognitionEvidence?: RecognitionEvidence;
  /** Primitive detector evidence retained on the logical review object. */
  assembly?: EngineeringObjectAssembly;
  /** Lossless parse evidence retained while the object awaits disposition. */
  engineeringParse?: EngineeringValueParse;
  /** Structure-first rule that placed the object in Review or Other. */
  engineeringDisposition?: EngineeringDisposition;
  valueBox: BBox;
  orientedBox?: BBox & { rotation: number };
  /** Audit trail for geometry-equivalent reads merged during rescans. */
  duplicateSourceIds?: string[];
  duplicateCount?: number;
  createdAt: number;
  updatedAt: number;
}
