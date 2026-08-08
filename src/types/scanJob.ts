import type { BBox } from "@/types/annotation";

export type ScanScopeKind = "section" | "page";

export type ScanJobStatus = "queued" | "running" | "succeeded" | "failed";

export type ScanLiveness =
  | "queued"
  | "working"
  | "long_running"
  | "slow_progress"
  | "possibly_stalled"
  | "complete"
  | "failed";

export type ScanOverlayPanelState = "pending" | "active" | "completed";
export type ScanOverlayCandidateState =
  | "detected"
  | "eligible"
  | "excluded"
  | "unread";

export interface ScanOverlayPanel extends BBox {
  id: string;
  label: string;
  state: ScanOverlayPanelState;
}

export interface ScanOverlayCandidate {
  bbox: BBox;
  state: ScanOverlayCandidateState;
  text?: string;
  reason?: string;
  rule?: string;
}

/** Temporary page-coordinate geometry used only while diagnosing a scan. */
export interface ScanDebugOverlay {
  enabled: boolean;
  pageWidth: number;
  pageHeight: number;
  scopeKind: ScanScopeKind;
  tableMasks: BBox[];
  panels: ScanOverlayPanel[];
  overlaps: BBox[];
  candidates: ScanOverlayCandidate[];
}

/** Genuine backend progress for one atomic auto-balloon scan. */
export interface ScanProgress {
  jobId: string;
  status: ScanJobStatus;
  stage: string;
  message: string;
  percent: number;
  completed: number;
  total: number;
  passCurrent: number;
  passTotal: number;
  tileCurrent: number;
  tileTotal: number;
  objectCurrent: number;
  objectTotal: number;
  batchCurrent: number;
  batchTotal: number;
  candidateCount: number;
  operationLabel: string;
  elapsedSeconds: number;
  stepElapsedSeconds: number;
  heartbeatAgeSeconds: number;
  progressAgeSeconds: number;
  estimatedRemainingSeconds: number | null;
  liveness: ScanLiveness;
  overlay: ScanDebugOverlay | null;
}

export interface ScanCompletionSummary {
  scopeKind: ScanScopeKind;
  added: number;
  detected: number;
  recognized: number;
  eligible: number;
  excluded: number;
  unread: number;
  skippedExisting: number;
  skippedDuplicates: number;
}
