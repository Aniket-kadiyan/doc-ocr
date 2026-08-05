export type ScanScopeKind = "section" | "page";

export type ScanJobStatus = "queued" | "running" | "succeeded" | "failed";

/** Genuine backend progress for one atomic auto-balloon scan. */
export interface ScanProgress {
  jobId: string;
  status: ScanJobStatus;
  stage: string;
  message: string;
  percent: number;
  completed: number;
  total: number;
}

export interface ScanCompletionSummary {
  added: number;
  skippedExisting: number;
  skippedDuplicates: number;
}
