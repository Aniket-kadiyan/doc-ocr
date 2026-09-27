import type { BBox, DimensionType } from "@/types/annotation";

export type ChecksheetRunStatus = "draft" | "completed";

export interface ChecksheetSnapshotItem {
  annotation_id: string;
  balloon_number: number;
  page: number;
  bbox: BBox;
  rotation: number;
  label: string;
  specification: string;
  tolerance: string;
  method: string;
  tool: string;
  dimension_type: DimensionType;
}

export interface ChecksheetSnapshotPayload {
  name: string;
  drawing_name: string;
  source_project_id?: string;
  source_file_type: "pdf" | "image";
  pdf_render_scale: number;
  reading_columns: string[];
  items: ChecksheetSnapshotItem[];
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
  rotation: number;
  label: string;
  specification: string;
  tolerance: string;
  method: string;
  tool: string;
  dimension_type: DimensionType;
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
    created_at: string;
  }>;
  runs: ChecksheetRunSummary[];
}

export interface ReadingPatch {
  annotation_id: string;
  column_id: string;
  value: string;
}
