import type { BBox } from "@/types/annotation";
import type {
  ScanLiveness,
  ScanProgress,
  ScanScopeKind,
} from "@/types/scanJob";
import { cropRegion } from "@/lib/canvasUtils";
import {
  getOcrApiUrl,
  mapSegmentRegions,
  type ApiSegmentResponse,
  type SegmentRegion,
} from "@/lib/paddleOcrClient";
import {
  isOcrDebugDumpEnabled,
  isOcrDebugDumpForce,
} from "@/lib/ocrDebugDump";

const POLL_INTERVAL_MS = 400;

interface ApiScanJobSnapshot {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  stage: string;
  message: string;
  percent: number;
  completed: number;
  total: number;
  pass_current?: number;
  pass_total?: number;
  tile_current?: number;
  tile_total?: number;
  object_current?: number;
  object_total?: number;
  candidate_count?: number;
  operation_label?: string;
  elapsed_seconds?: number;
  step_elapsed_seconds?: number;
  heartbeat_age_seconds?: number;
  progress_age_seconds?: number;
  estimated_remaining_seconds?: number | null;
  liveness?: ScanLiveness;
  error?: string | null;
  result?: ApiSegmentResponse | null;
}

export interface RunScanJobOptions {
  sourceCanvas: HTMLCanvasElement;
  bbox: BBox;
  page: number;
  scopeKind: ScanScopeKind;
  displayScale?: number;
  onProgress?: (progress: ScanProgress) => void;
}

export interface ScanJobResult {
  regions: SegmentRegion[];
  detected: number;
  recognized: number;
  eligible: number;
  excluded: number;
  unread: number;
  filterRuleCounts: Record<string, number>;
}

const sleep = (milliseconds: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

function toProgress(snapshot: ApiScanJobSnapshot): ScanProgress {
  return {
    jobId: snapshot.job_id,
    status: snapshot.status,
    stage: snapshot.stage,
    message: snapshot.message,
    percent: snapshot.percent,
    completed: snapshot.completed,
    total: snapshot.total,
    passCurrent: snapshot.pass_current ?? 0,
    passTotal: snapshot.pass_total ?? 0,
    tileCurrent: snapshot.tile_current ?? 0,
    tileTotal: snapshot.tile_total ?? 0,
    objectCurrent: snapshot.object_current ?? 0,
    objectTotal: snapshot.object_total ?? 0,
    candidateCount: snapshot.candidate_count ?? 0,
    operationLabel: snapshot.operation_label ?? "",
    elapsedSeconds: snapshot.elapsed_seconds ?? 0,
    stepElapsedSeconds: snapshot.step_elapsed_seconds ?? 0,
    heartbeatAgeSeconds: snapshot.heartbeat_age_seconds ?? 0,
    progressAgeSeconds: snapshot.progress_age_seconds ?? 0,
    estimatedRemainingSeconds:
      snapshot.estimated_remaining_seconds ?? null,
    liveness:
      snapshot.liveness ??
      (snapshot.status === "queued" ? "queued" : "working"),
  };
}

async function readError(response: Response, fallback: string): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail || fallback;
  } catch {
    return fallback;
  }
}

function scanRequestOptions(): {
  query: string;
  headers: Record<string, string>;
} {
  const debugDump = isOcrDebugDumpEnabled();
  const debugForce = isOcrDebugDumpForce();
  const params = new URLSearchParams();
  if (debugDump) params.set("debug_dump", "1");
  if (debugForce) params.set("debug_dump_force", "1");

  const headers: Record<string, string> = {};
  if (debugDump) headers["X-Debug-Dump"] = "1";
  if (debugForce) headers["X-Debug-Dump-Force"] = "1";
  return { query: params.toString(), headers };
}

/**
 * Run one backend scan job and return candidates only after it succeeds.
 * Polling exposes genuine stage/counter progress without publishing partial
 * annotations into the drawing.
 */
export async function runScanJob({
  sourceCanvas,
  bbox,
  page,
  scopeKind,
  displayScale = 1,
  onProgress,
}: RunScanJobOptions): Promise<ScanJobResult> {
  const crop = cropRegion(sourceCanvas, bbox, displayScale);
  const blob = await new Promise<Blob>((resolve, reject) => {
    crop.toBlob((encoded) => {
      if (encoded) resolve(encoded);
      else reject(new Error("Failed to encode scan area"));
    }, "image/png");
  });

  const form = new FormData();
  form.append("file", blob, "scan.png");
  form.append("scope_kind", scopeKind);
  form.append("page", String(page));
  form.append("scope_x", String(bbox.x));
  form.append("scope_y", String(bbox.y));
  form.append("scope_width", String(bbox.width));
  form.append("scope_height", String(bbox.height));

  const baseUrl = getOcrApiUrl();
  const { query, headers } = scanRequestOptions();
  const createUrl = query
    ? `${baseUrl}/ocr/scan-jobs?${query}`
    : `${baseUrl}/ocr/scan-jobs`;
  const created = await fetch(createUrl, {
    method: "POST",
    headers,
    body: form,
  });
  if (!created.ok) {
    throw new Error(
      await readError(created, `Could not start scan: ${created.status}`)
    );
  }

  let snapshot = (await created.json()) as ApiScanJobSnapshot;
  onProgress?.(toProgress(snapshot));

  while (snapshot.status === "queued" || snapshot.status === "running") {
    await sleep(POLL_INTERVAL_MS);
    const response = await fetch(
      `${baseUrl}/ocr/scan-jobs/${encodeURIComponent(snapshot.job_id)}`,
      { method: "GET", cache: "no-store" }
    );
    if (!response.ok) {
      throw new Error(
        await readError(response, `Could not read scan progress: ${response.status}`)
      );
    }
    snapshot = (await response.json()) as ApiScanJobSnapshot;
    onProgress?.(toProgress(snapshot));
  }

  if (snapshot.status === "failed") {
    throw new Error(snapshot.error || "Auto-balloon scan failed");
  }
  if (!snapshot.result) {
    throw new Error("Auto-balloon scan completed without a result");
  }

  const regions = mapSegmentRegions(
    snapshot.result.regions ?? [],
    bbox,
    displayScale
  );
  const detected = snapshot.result.detected_count ?? regions.length;
  const recognized =
    snapshot.result.recognized_count ??
    regions.filter((region) => region.recognized).length;
  const unread =
    snapshot.result.unread_count ?? Math.max(0, detected - recognized);
  const eligible = snapshot.result.eligible_count ?? regions.length;
  const excluded =
    snapshot.result.excluded_count ?? Math.max(0, recognized - eligible);
  const filterRuleCounts = snapshot.result.filter_rule_counts ?? {};

  return {
    regions,
    detected,
    recognized,
    eligible,
    excluded,
    unread,
    filterRuleCounts,
  };
}
