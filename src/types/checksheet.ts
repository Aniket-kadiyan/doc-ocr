import type {
  BBox,
  DimensionType,
  EngineeringDisposition,
  EngineeringObjectAssembly,
  EngineeringSymbolEvidence,
  EngineeringValueParse,
} from "@/types/annotation";
import type { DocumentMetadata } from "@/types/documentMetadata";
import type { ScanCandidateState } from "@/types/scanCandidate";

export type ChecksheetRunStatus = "draft" | "completed";

export interface ChecksheetSnapshotItem {
  annotation_id: string;
  balloon_number: number;
  page: number;
  bbox: BBox;
  oriented_box?: (BBox & { rotation: number }) | null;
  rotation: number;
  label: string;
  specification: string;
  tolerance: string;
  method: string;
  tool: string;
  dimension_type: DimensionType;
  assembly?: EngineeringObjectAssembly | null;
  engineering_symbol?: EngineeringSymbolEvidence | null;
  engineering_parse?: EngineeringValueParse | null;
  engineering_disposition?: EngineeringDisposition | null;
}

export interface ChecksheetSnapshotPayload {
  name: string;
  drawing_name: string;
  source_project_id?: string;
  source_file_type: "pdf" | "image";
  pdf_render_scale: number;
  metadata: DocumentMetadata;
  scan_candidates: ChecksheetScanCandidate[];
  reading_columns: string[];
  items: ChecksheetSnapshotItem[];
}

export interface ChecksheetScanCandidate {
  candidate_id: string;
  source_candidate_id?: string;
  page: number;
  order: number;
  state: ScanCandidateState;
  restore_state?: "review" | "other";
  text: string;
  raw_text: string;
  preliminary_text?: string;
  confidence: number;
  recognized: boolean;
  reason: string;
  rule?: string;
  type?: string;
  category?: string;
  subtype?: string;
  label?: string;
  orientation: "horizontal" | "vertical" | "rotated";
  rotation: number;
  recovery_attempted: boolean;
  authoritative_reread: boolean;
  assembly?: EngineeringObjectAssembly | null;
  engineering_symbol?: EngineeringSymbolEvidence | null;
  engineering_parse?: EngineeringValueParse | null;
  engineering_disposition?: EngineeringDisposition | null;
  bbox: BBox;
  oriented_box?: (BBox & { rotation: number }) | null;
  duplicate_source_ids: string[];
  duplicate_count: number;
  created_at: number;
  updated_at: number;
}

export interface ChecksheetColumn {
  id: string;
  name: string;
  position: number;
}

export interface ChecksheetRow {
  id: string;
  annotation_id: string;
  balloon_number: number;
  page: number;
  bbox: BBox;
  oriented_box?: (BBox & { rotation: number }) | null;
  rotation: number;
  label: string;
  specification: string;
  tolerance: string;
  method: string;
  tool: string;
  dimension_type: DimensionType;
  assembly?: EngineeringObjectAssembly | null;
  engineering_symbol?: EngineeringSymbolEvidence | null;
  engineering_parse?: EngineeringValueParse | null;
  engineering_disposition?: EngineeringDisposition | null;
  readings: Record<string, string>;
}

export interface ChecksheetRunResponse {
  checksheet: {
    id: string;
    name: string;
    drawing_name: string;
    archived: boolean;
  };
  revision: {
    id: string;
    revision_number: number;
    pdf_render_scale: number;
    metadata: DocumentMetadata;
    scan_candidates: ChecksheetScanCandidate[];
  };
  run: {
    id: string;
    run_number: number;
    status: ChecksheetRunStatus;
    created_at: string;
    updated_at: string;
    completed_at: string | null;
  };
  document: {
    id: string;
    file_name: string;
    mime_type: string;
    file_type: "pdf" | "image";
    size_bytes: number;
  };
  columns: ChecksheetColumn[];
  rows: ChecksheetRow[];
}

export interface ChecksheetSummary {
  id: string;
  name: string;
  drawing_name: string;
  source_project_id: string | null;
  archived: boolean;
  archived_at: string | null;
  created_at: string;
  updated_at: string;
  revision_id: string;
  revision_number: number;
  row_count: number;
  run_count: number;
  draft_count: number;
  completed_count: number;
  latest_run_id: string | null;
  latest_run_status: ChecksheetRunStatus | null;
  latest_draft_run_id: string | null;
}

export interface ChecksheetRunSummary {
  id: string;
  run_number: number;
  revision_id: string;
  revision_number: number;
  status: ChecksheetRunStatus;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface ChecksheetDetail {
  id: string;
  name: string;
  drawing_name: string;
  source_project_id: string | null;
  archived: boolean;
  archived_at: string | null;
  created_at: string;
  updated_at: string;
  revisions: Array<{
    id: string;
    revision_number: number;
    row_count: number;
    candidate_count: number;
    created_at: string;
    metadata: DocumentMetadata;
  }>;
  runs: ChecksheetRunSummary[];
}

export interface ReadingPatch {
  annotation_id: string;
  column_id: string;
  value: string;
}
