import type { BBox } from "@/types/annotation";
import type {
  ScanDebugOverlay,
  ScanLiveness,
  ScanOverlayCandidateState,
  ScanOverlayPanelState,
  ScanJobStatus,
  ScanProgress,
  ScanScopeKind,
} from "@/types/scanJob";
import {
  getOcrApiUrl,
  mapPageSegmentRegions,
  mapSegmentRegions,
  type ApiSegmentResponse,
  type SegmentRegion,
} from "@/lib/paddleOcrClient";
import {
  isOcrDebugDumpEnabled,
  isOcrDebugDumpForce,
} from "@/lib/ocrDebugDump";

const POLL_INTERVAL_MS = 400;

interface ApiScanOverlay {
  enabled?: boolean;
  page_width?: number;
  page_height?: number;
  scope_kind?: ScanScopeKind;
  table_masks?: BBox[];
  panels?: Array<
    BBox & {
      id?: string;
      label?: string;
      state?: ScanOverlayPanelState;
    }
  >;
  overlaps?: BBox[];
  candidates?: Array<{
    id?: string;
    bbox?: BBox;
    state?: ScanOverlayCandidateState;
    text?: string;
    reason?: string;
    rule?: string;
  }>;
}

interface ApiScanJobSnapshot {
  job_id: string;
  status: ScanJobStatus;
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
  batch_current?: number;
  batch_total?: number;
  candidate_count?: number;
  operation_label?: string;
  elapsed_seconds?: number;
  step_elapsed_seconds?: number;
  heartbeat_age_seconds?: number;
  progress_age_seconds?: number;
  estimated_remaining_seconds?: number | null;
  liveness?: ScanLiveness;
  overlay?: ApiScanOverlay | null;
  error?: string | null;
  result?: ApiSegmentResponse | null;
}

export interface RunScanJobOptions {
  sourceCanvas: HTMLCanvasElement;
  bbox: BBox;
  page: number;
  scopeKind: ScanScopeKind;
  displayScale?: number;
  existingValueBoxes?: BBox[];
  onProgress?: (progress: ScanProgress) => void;
}

export interface ScanJobResult {
  regions: SegmentRegion[];
  detected: number;
  recognized: number;
  eligible: number;
  excluded: number;
  reviewRequired: number;
  unread: number;
  skippedExisting: number;
  reviewCandidates: SegmentRegion[];
  filterRuleCounts: Record<string, number>;
}

export class ScanJobCancelledError extends Error {
  readonly code = "SCAN_CANCELLED";

  constructor(message = "Auto-balloon scan stopped") {
    super(message);
    this.name = "ScanJobCancelledError";
  }
}

export function isScanJobCancelledError(
  error: unknown
): error is ScanJobCancelledError {
  return (
    error instanceof ScanJobCancelledError ||
    (typeof error === "object" &&
      error !== null &&
      "code" in error &&
      error.code === "SCAN_CANCELLED")
  );
}

const sleep = (milliseconds: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

/**
 * Bring one progress overlay into viewer coordinates.
 *
 * The scan reads a higher-resolution render than the one on screen, so the
 * backend describes this geometry in those larger pixels, exactly like the
 * region boxes that mapPageSegmentRegions has always divided down. The
 * overlay was passed through untouched, so on a PDF page every processing
 * box was drawn about 2.3x too large and too far down and right.
 *
 * The page dimensions come down with it: they say which image the geometry
 * belongs to, which is what lets the viewer refuse to paint a section crop's
 * coordinates onto the whole page.
 */
function toDebugOverlay(
  overlay: ApiScanOverlay | null | undefined,
  displayScale = 1
): ScanDebugOverlay | null {
  if (!overlay?.enabled) return null;
  const scale = displayScale > 0 ? displayScale : 1;
  const toViewer = (value: number) => value / scale;
  const boxToViewer = (box: BBox): BBox => ({
    x: toViewer(box.x),
    y: toViewer(box.y),
    width: toViewer(box.width),
    height: toViewer(box.height),
  });
  return {
    enabled: true,
    pageWidth: toViewer(overlay.page_width ?? 0),
    pageHeight: toViewer(overlay.page_height ?? 0),
    scopeKind: overlay.scope_kind ?? "page",
    tableMasks: (overlay.table_masks ?? []).map(boxToViewer),
    panels: (overlay.panels ?? []).map((panel, index) => ({
      ...boxToViewer(panel),
      id: panel.id ?? `P${index + 1}`,
      label: panel.label ?? panel.id ?? `P${index + 1}`,
      state: panel.state ?? "pending",
    })),
    overlaps: (overlay.overlaps ?? []).map(boxToViewer),
    candidates: (overlay.candidates ?? [])
      .filter((candidate): candidate is typeof candidate & { bbox: BBox } =>
        Boolean(candidate.bbox)
      )
      .map((candidate) => ({
        id: candidate.id,
        bbox: boxToViewer(candidate.bbox),
        state: candidate.state ?? "detected",
        text: candidate.text,
        reason: candidate.reason,
        rule: candidate.rule,
      })),
  };
}

function toProgress(
  snapshot: ApiScanJobSnapshot,
  displayScale = 1
): ScanProgress {
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
    batchCurrent: snapshot.batch_current ?? 0,
    batchTotal: snapshot.batch_total ?? 0,
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
    overlay: toDebugOverlay(snapshot.overlay, displayScale),
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

/** Request cooperative cancellation and return the backend's new state. */
export async function cancelScanJob(jobId: string): Promise<ScanProgress> {
  if (!jobId) throw new Error("Cannot stop a scan before it has started");

  const response = await fetch(
    `${getOcrApiUrl()}/ocr/scan-jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST", cache: "no-store" }
  );
  if (!response.ok) {
    throw new Error(
      await readError(response, `Could not stop scan: ${response.status}`)
    );
  }
  return toProgress((await response.json()) as ApiScanJobSnapshot);
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
  existingValueBoxes = [],
  onProgress,
}: RunScanJobOptions): Promise<ScanJobResult> {
  const blob = await new Promise<Blob>((resolve, reject) => {
    sourceCanvas.toBlob((encoded) => {
      if (encoded) resolve(encoded);
      else reject(new Error("Failed to encode the complete drawing page"));
    }, "image/png");
  });

  const form = new FormData();
  form.append("file", blob, "scan-page.png");
  form.append("scope_kind", scopeKind);
  form.append("page", String(page));
  // The uploaded image is the crop taken at displayScale, so everything
  // describing it has to be in those same pixels. The scope and the existing
  // balloons arrive in viewer units; sending them unscaled while scanning a
  // higher-resolution render would put the scope and every overlap test in the
  // wrong coordinate space, and a re-scan would duplicate balloons it should
  // have recognised as already present.
  const toCropPixels = (value: number) => value * displayScale;
  form.append("scope_x", String(toCropPixels(bbox.x)));
  form.append("scope_y", String(toCropPixels(bbox.y)));
  form.append("scope_width", String(toCropPixels(bbox.width)));
  form.append("scope_height", String(toCropPixels(bbox.height)));
  form.append(
    "existing_value_boxes",
    JSON.stringify(
      existingValueBoxes.map((box) => ({
        x: toCropPixels(box.x),
        y: toCropPixels(box.y),
        width: toCropPixels(box.width),
        height: toCropPixels(box.height),
      }))
    )
  );

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
  onProgress?.(toProgress(snapshot, displayScale));

  while (
    snapshot.status === "queued" ||
    snapshot.status === "running" ||
    snapshot.status === "cancelling"
  ) {
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
    onProgress?.(toProgress(snapshot, displayScale));
  }

  if (snapshot.status === "failed") {
    throw new Error(snapshot.error || "Auto-balloon scan failed");
  }
  if (snapshot.status === "cancelled") {
    throw new ScanJobCancelledError(
      snapshot.message || "Auto-balloon scan stopped"
    );
  }
  if (!snapshot.result) {
    throw new Error("Auto-balloon scan completed without a result");
  }

  const regions = snapshot.result.coordinate_space === "page"
    ? mapPageSegmentRegions(
        snapshot.result.regions ?? [],
        // Viewer-space bounds, not the scanned canvas: that canvas may be a
        // higher-resolution render, and its own size would validate hi-res
        // coordinates as in-bounds and place every balloon off the drawing.
        bbox,
        displayScale
      )
    : mapSegmentRegions(
        snapshot.result.regions ?? [],
        bbox,
        displayScale
      );
  const reviewCandidates = snapshot.result.coordinate_space === "page"
    ? mapPageSegmentRegions(
        snapshot.result.review_candidates ?? [],
        bbox,
        displayScale
      )
    : mapSegmentRegions(
        snapshot.result.review_candidates ?? [],
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
  const reviewRequired =
    snapshot.result.review_count ?? reviewCandidates.length;
  const filterRuleCounts = snapshot.result.filter_rule_counts ?? {};
  const skippedExisting = snapshot.result.skipped_existing_count ?? 0;

  return {
    regions,
    detected,
    recognized,
    eligible,
    excluded,
    reviewRequired,
    unread,
    skippedExisting,
    reviewCandidates,
    filterRuleCounts,
  };
}
