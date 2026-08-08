import type { BBox, OCRResult } from "@/types/annotation";
import { cropRegion, CROP_PAD_PX } from "@/lib/canvasUtils";
import { fixEngineeringSymbols } from "@/lib/engineeringSymbols";
import { isOcrDebugDumpEnabled, isOcrDebugDumpForce } from "@/lib/ocrDebugDump";
import { mergeSymbolHints } from "@/lib/visualSymbols";

export type OcrEngine = "paddleocr" | "paddleocr+vision" | "paddleocr+compose";

export interface OcrApiHealth {
  available: boolean;
  paddleocr: boolean;
  trocr: boolean;
}

const DEFAULT_HOST = "http://localhost";
const DEFAULT_PORT = "8000";

/**
 * Resolve the OCR backend base URL. Precedence:
 *   1. NEXT_PUBLIC_OCR_API_URL — full URL override (e.g. https://ocr.example.com)
 *   2. NEXT_PUBLIC_OCR_API_HOST + NEXT_PUBLIC_OCR_API_PORT — host/port composed
 *      (host may omit the scheme; http:// is assumed)
 *   3. http://localhost:8000
 * The returned URL never has a trailing slash.
 */
export function getOcrApiUrl(): string {
  const fullUrl = process.env.NEXT_PUBLIC_OCR_API_URL?.trim();
  if (fullUrl) return fullUrl.replace(/\/+$/, "");

  let host = (process.env.NEXT_PUBLIC_OCR_API_HOST?.trim() || DEFAULT_HOST).replace(
    /\/+$/,
    ""
  );
  if (!/^https?:\/\//i.test(host)) host = `http://${host}`;

  const port = process.env.NEXT_PUBLIC_OCR_API_PORT?.trim() || DEFAULT_PORT;
  return port ? `${host}:${port}` : host;
}

let healthCache: { at: number; health: OcrApiHealth } | null = null;

export async function checkOcrApiHealth(
  force = false
): Promise<OcrApiHealth> {
  const url = getOcrApiUrl();
  const now = Date.now();
  if (!force && healthCache && now - healthCache.at < 10_000) {
    return healthCache.health;
  }

  try {
    const res = await fetch(`${url}/health`, { method: "GET" });
    if (!res.ok) throw new Error("health check failed");
    const data = (await res.json()) as {
      paddleocr?: boolean;
      trocr?: boolean;
    };
    const health: OcrApiHealth = {
      available: Boolean(data.paddleocr),
      paddleocr: Boolean(data.paddleocr),
      trocr: Boolean(data.trocr),
    };
    healthCache = { at: now, health };
    return health;
  } catch {
    const health: OcrApiHealth = {
      available: false,
      paddleocr: false,
      trocr: false,
    };
    healthCache = { at: now, health };
    return health;
  }
}

interface ApiRecognizeResponse {
  text: string;
  confidence: number;
  engine: string;
  orientation?: "horizontal" | "vertical" | "rotated";
  rotation?: number;
  words?: Array<{
    text: string;
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
  }>;
  symbols_detected?: {
    diameter?: boolean;
    diameter_score?: number;
    plus_minus?: boolean;
    degree?: boolean;
  };
  agreement?: number;
  needs_review?: boolean;
  text_bbox?: { x: number; y: number; width: number; height: number };
  type?: string;
  debug_dump_dir?: string;
  debug_dump?: {
    active?: boolean;
    skipped_reason?: string | null;
    dir?: string | null;
  };
  timings_ms?: {
    pipeline_total: number;
    paddle_total: number;
    paddle_pass_count: number;
    paddle_passes: Array<{
      stage: string;
      elapsed_ms: number;
      width: number;
      height: number;
      det: boolean;
    }>;
  };
}

export interface SegmentRegion {
  /** Stable backend identity for a deduplicated whole-page detector object. */
  candidateId?: string;
  text: string;
  confidence: number;
  type?: string;
  orientation: "horizontal" | "vertical" | "rotated";
  rotation: number;
  needsReview: boolean;
  /** False when detection succeeded but the single OCR pass was empty/invalid. */
  recognized: boolean;
  /** Whole-page acceptance rule; absent for section scans. */
  pageFilterRule?: string;
  /** Human-readable whole-page acceptance reason. */
  pageFilterReason?: string;
  /** Why this detected object needs manual confirmation. */
  reviewReason?: string;
  /** True when bounded post-detection recovery was attempted. */
  recoveryAttempted?: boolean;
  /** Region box mapped into source-canvas coordinates. */
  valueBox: BBox;
}

export interface ApiSegmentRegion {
  candidate_id?: string;
  bbox: { x: number; y: number; width: number; height: number };
  text: string;
  confidence: number;
  type?: string;
  orientation?: "horizontal" | "vertical" | "rotated";
  rotation?: number;
  needs_review?: boolean;
  recognized?: boolean;
  page_filter_rule?: string;
  page_filter_reason?: string;
  review_reason?: string;
  recovery_attempted?: boolean;
}

export interface ApiSegmentResponse {
  count: number;
  detected_count?: number;
  recognized_count?: number;
  eligible_count?: number;
  excluded_count?: number;
  review_count?: number;
  unread_count?: number;
  filter_rule_counts?: Record<string, number>;
  coordinate_space?: "scope" | "page";
  regions: ApiSegmentRegion[];
  review_candidates?: ApiSegmentRegion[];
  candidate_outcomes?: Array<{
    candidate_id?: string;
    bbox: BBox;
    state: "eligible" | "excluded" | "review";
    text?: string;
    reason?: string;
    rule?: string;
  }>;
}

function mappedSegmentRegion(r: ApiSegmentRegion, valueBox: BBox): SegmentRegion {
  return {
    candidateId: r.candidate_id,
    text: fixEngineeringSymbols(r.text ?? ""),
    confidence: r.confidence ?? 0,
    type: r.type,
    orientation: r.orientation ?? "horizontal",
    rotation: r.rotation ?? 0,
    needsReview: r.needs_review ?? false,
    recognized: r.recognized ?? Boolean((r.text ?? "").trim()),
    pageFilterRule: r.page_filter_rule,
    pageFilterReason: r.page_filter_reason,
    reviewReason: r.review_reason,
    recoveryAttempted: r.recovery_attempted,
    valueBox,
  };
}

/** Map backend crop coordinates into the drawing's base canvas coordinates. */
export function mapSegmentRegions(
  regions: ApiSegmentRegion[],
  bbox: BBox,
  displayScale = 1
): SegmentRegion[] {
  const pad = Math.max(4, CROP_PAD_PX / displayScale);

  return regions.map((r) => {
    const mapped: BBox = {
      x: bbox.x + (r.bbox.x - CROP_PAD_PX) / displayScale,
      y: bbox.y + (r.bbox.y - CROP_PAD_PX) / displayScale,
      width: r.bbox.width / displayScale,
      height: r.bbox.height / displayScale,
    };

    const outOfBounds =
      !Number.isFinite(mapped.x) ||
      !Number.isFinite(mapped.y) ||
      !Number.isFinite(mapped.width) ||
      !Number.isFinite(mapped.height) ||
      mapped.x < bbox.x - pad ||
      mapped.y < bbox.y - pad ||
      mapped.x + mapped.width > bbox.x + bbox.width + pad ||
      mapped.y + mapped.height > bbox.y + bbox.height + pad;

    const valueBox: BBox = outOfBounds
      ? {
          x: bbox.x,
          y: bbox.y,
          width: bbox.width,
          height: bbox.height,
        }
      : mapped;

    return mappedSegmentRegion(r, valueBox);
  });
}

/** Map API regions that are already expressed in full-page coordinates. */
export function mapPageSegmentRegions(
  regions: ApiSegmentRegion[],
  pageBounds: BBox
): SegmentRegion[] {
  return regions.map((r) => {
    const direct: BBox = {
      x: r.bbox.x,
      y: r.bbox.y,
      width: r.bbox.width,
      height: r.bbox.height,
    };
    const invalid =
      !Number.isFinite(direct.x) ||
      !Number.isFinite(direct.y) ||
      !Number.isFinite(direct.width) ||
      !Number.isFinite(direct.height) ||
      direct.width <= 0 ||
      direct.height <= 0 ||
      direct.x < pageBounds.x ||
      direct.y < pageBounds.y ||
      direct.x + direct.width > pageBounds.x + pageBounds.width ||
      direct.y + direct.height > pageBounds.y + pageBounds.height;

    return mappedSegmentRegion(r, invalid ? pageBounds : direct);
  });
}

/**
 * Auto-segment one selection into multiple per-value reads.
 *
 * Crops + POSTs the drawn box exactly like {@link runPaddleOcr}, then maps each
 * returned region's box (received-crop pixels) back into source-canvas coords
 * using the same arithmetic as the single-value valueBox mapping below.
 */
export async function runSegmentOcr(
  sourceCanvas: HTMLCanvasElement,
  bbox: BBox,
  displayScale = 1
): Promise<SegmentRegion[]> {
  const url = getOcrApiUrl();
  const crop = cropRegion(sourceCanvas, bbox, displayScale);

  const blob = await new Promise<Blob>((resolve, reject) => {
    crop.toBlob((b) => {
      if (b) resolve(b);
      else reject(new Error("Failed to encode crop"));
    }, "image/png");
  });

  const form = new FormData();
  form.append("file", blob, "crop.png");

  const debugDump = isOcrDebugDumpEnabled();
  const debugForce = isOcrDebugDumpForce();
  const params = new URLSearchParams();
  if (debugDump) params.set("debug_dump", "1");
  if (debugForce) params.set("debug_dump_force", "1");
  const segmentUrl = params.toString()
    ? `${url}/ocr/segment?${params}`
    : `${url}/ocr/segment`;

  const headers: Record<string, string> = {};
  if (debugDump) headers["X-Debug-Dump"] = "1";
  if (debugForce) headers["X-Debug-Dump-Force"] = "1";

  const res = await fetch(segmentUrl, {
    method: "POST",
    headers,
    body: form,
  });

  if (!res.ok) {
    throw new Error(`Segment API error: ${res.status}`);
  }

  const data = (await res.json()) as ApiSegmentResponse;

  return mapSegmentRegions(data.regions ?? [], bbox, displayScale);
}

export async function runPaddleOcr(
  sourceCanvas: HTMLCanvasElement,
  bbox: BBox,
  displayScale = 1
): Promise<OCRResult & { engine: OcrEngine }> {
  const url = getOcrApiUrl();
  const crop = cropRegion(sourceCanvas, bbox, displayScale);

  const blob = await new Promise<Blob>((resolve, reject) => {
    crop.toBlob((b) => {
      if (b) resolve(b);
      else reject(new Error("Failed to encode crop"));
    }, "image/png");
  });

  const form = new FormData();
  form.append("file", blob, "crop.png");

  const debugDump = isOcrDebugDumpEnabled();
  const debugForce = isOcrDebugDumpForce();
  const params = new URLSearchParams();
  if (debugDump) params.set("debug_dump", "1");
  if (debugForce) params.set("debug_dump_force", "1");
  const recognizeUrl = params.toString()
    ? `${url}/ocr/recognize?${params}`
    : `${url}/ocr/recognize`;

  const headers: Record<string, string> = {};
  if (debugDump) headers["X-Debug-Dump"] = "1";
  if (debugForce) headers["X-Debug-Dump-Force"] = "1";

  const res = await fetch(recognizeUrl, {
    method: "POST",
    headers,
    body: form,
  });

  if (!res.ok) {
    throw new Error(`OCR API error: ${res.status}`);
  }

  const data = (await res.json()) as ApiRecognizeResponse;
  const engine = (data.engine ?? "paddleocr") as OcrEngine;
  // O1 performance telemetry. This can be removed or feature-gated after
  // optimization work has been completed.
  if (data.timings_ms && typeof window !== "undefined") {
    console.info("[ocr timing]", data.timings_ms);
  }

  const base = data.text ?? "";
  const composed = (data.engine ?? "").includes("compose");
  const merged =
    composed || /^[\s]*Ø/.test(base)
      ? base
      : mergeSymbolHints(base, data.symbols_detected ?? {});

  // Map the backend's tight text bbox (received-crop pixels) back to
  // source-canvas coordinates so the balloon can snap to the actual value.
  let valueBox: BBox | undefined;
  const tb = data.text_bbox;
  if (tb && tb.width > 0 && tb.height > 0) {
    const mapped: BBox = {
      x: bbox.x + (tb.x - CROP_PAD_PX) / displayScale,
      y: bbox.y + (tb.y - CROP_PAD_PX) / displayScale,
      width: tb.width / displayScale,
      height: tb.height / displayScale,
    };

    // If mapping goes far outside the user-drawn bbox, ignore it.
    // This prevents annotations from rendering off-canvas when bbox mapping
    // is wrong for some crop sizes / rotations.
    const pad = Math.max(4, CROP_PAD_PX / displayScale);
    const outOfBounds =
      !Number.isFinite(mapped.x) ||
      !Number.isFinite(mapped.y) ||
      !Number.isFinite(mapped.width) ||
      !Number.isFinite(mapped.height) ||
      mapped.x < bbox.x - pad ||
      mapped.y < bbox.y - pad ||
      mapped.x + mapped.width > bbox.x + bbox.width + pad ||
      mapped.y + mapped.height > bbox.y + bbox.height + pad;

    if (!outOfBounds) {
      valueBox = mapped;
    }
  }

  if (debugDump && typeof window !== "undefined") {
    const info = data.debug_dump;
    if (info?.dir) {
      console.info("[ocr debug] wrote steps →", info.dir);
    } else if (info?.skipped_reason) {
      console.warn("[ocr debug] skipped:", info.skipped_reason);
    }
  }

  return {
    text: fixEngineeringSymbols(merged),
    confidence: data.confidence ?? 0,
    rotation: data.rotation ?? 0,
    orientation: data.orientation ?? "horizontal",
    words: data.words ?? [],
    engine,
    agreement: data.agreement,
    needsReview: data.needs_review,
    valueBox,
    debugDumpDir: data.debug_dump_dir ?? data.debug_dump?.dir ?? undefined,
    debugDumpSkipped: data.debug_dump?.skipped_reason ?? undefined,
  };
}
