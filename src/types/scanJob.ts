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
  candidateCount: number;
  operationLabel: string;
  elapsedSeconds: number;
  stepElapsedSeconds: number;
  heartbeatAgeSeconds: number;
  progressAgeSeconds: number;
  estimatedRemainingSeconds: number | null;
  liveness: ScanLiveness;
}

export interface ScanCompletionSummary {
  added: number;
  detected: number;
  recognized: number;
  unread: number;
  skippedExisting: number;
  skippedDuplicates: number;
}
