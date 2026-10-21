/**
 * Client for the backend's GD&T routes.
 *
 * Reuses `getOcrApiUrl` so the GD&T pass is reached through exactly the same
 * configuration as OCR — one backend URL, one place to change it.
 *
 * Every call here fails soft. The GD&T pass is additive to a workflow that
 * already works without it, so a backend that is older than this frontend, or
 * is briefly unreachable, must leave ballooning usable rather than breaking
 * the page.
 */

import { cropRegion, CROP_PAD_PX } from "@/lib/canvasUtils";
import { getOcrApiUrl } from "@/lib/paddleOcrClient";
import { apiRequestError, fetchWithRequestId } from "@/lib/requestDiagnostics";
import type { BBox } from "@/types/annotation";
import type {
  GdtCorrection,
  GdtPageResult,
  GdtRegistry,
} from "@/types/gdt";

/** Milliseconds before a recognition request is abandoned. */
const RECOGNIZE_TIMEOUT_MS = 180_000;
const QUICK_TIMEOUT_MS = 15_000;

async function fetchJson<T>(
  path: string,
  init: RequestInit & { timeoutMs?: number } = {}
): Promise<T> {
  const { timeoutMs = QUICK_TIMEOUT_MS, ...rest } = init;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchWithRequestId(
      `${getOcrApiUrl()}${path}`,
      { ...rest, signal: controller.signal },
      { operation: "gdt.request" }
    );
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw apiRequestError(
        response,
        `${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`,
        "gdt.request"
      );
    }
    return (await response.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

let registryCache: GdtRegistry | null = null;

/**
 * The symbol registry, fetched once per session.
 *
 * Served by the backend rather than duplicated here on purpose: the names,
 * meanings and standards a reviewer reads come from the same table the
 * recognizer decides with, so the two cannot drift apart.
 */
export async function fetchGdtRegistry(force = false): Promise<GdtRegistry | null> {
  if (registryCache && !force) return registryCache;
  try {
    registryCache = await fetchJson<GdtRegistry>("/gdt/registry");
    return registryCache;
  } catch {
    return null;
  }
}

export interface RecognizeGdtOptions {
  page?: number;
  /** Read frame compartments with OCR. Needed for tolerances and datums. */
  readText?: boolean;
  /** Also look for symbols outside feature control frames. */
  detectLooseSymbols?: boolean;
  /**
   * The standard the drawing is released to. Used only to report symbols that
   * do not belong to it — never to change a reading.
   */
  declaredStandard?: string;
  signal?: AbortSignal;
}

/**
 * Run GD&T recognition over a region of the drawing.
 *
 * `region` is in page coordinates; omit it to read the whole page. The
 * returned boxes are in the *submitted image's* coordinates, so a region call
 * gets them shifted back here, which keeps every box the caller sees in page
 * space regardless of what was sent.
 */
export async function recognizeGdtSymbols(
  canvas: HTMLCanvasElement,
  region: BBox | null,
  options: RecognizeGdtOptions = {}
): Promise<GdtPageResult | null> {
  const {
    page = 1,
    readText = true,
    detectLooseSymbols = true,
    declaredStandard = "",
    signal,
  } = options;

  const blob = await regionBlob(canvas, region);
  if (!blob) return null;

  const form = new FormData();
  form.append("file", blob, "region.png");
  form.append("declared_standard", declaredStandard);

  const query = new URLSearchParams({
    page: String(page),
    read_text: String(readText),
    detect_loose_symbols: String(detectLooseSymbols),
  });

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), RECOGNIZE_TIMEOUT_MS);
  signal?.addEventListener("abort", () => controller.abort());
  try {
    const response = await fetchWithRequestId(
      `${getOcrApiUrl()}/ocr/gdt-symbols?${query.toString()}`,
      { method: "POST", body: form, signal: controller.signal },
      { operation: "gdt.recognize" }
    );
    if (!response.ok) return null;
    const result = (await response.json()) as GdtPageResult;
    // `cropRegion` lays the region onto a white canvas inset by CROP_PAD_PX on
    // every side, so a box found at the crop's origin is really at the
    // region's origin minus the pad. Subtracting it here is what keeps the
    // overlay aligned; forgetting it shifts every box by 20px, which looks
    // like a detector problem and is not one.
    return region
      ? shiftResult(
          result,
          region.x - CROP_PAD_PX,
          region.y - CROP_PAD_PX
        )
      : result;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function regionBlob(
  canvas: HTMLCanvasElement,
  region: BBox | null
): Promise<Blob | null> {
  const source = region ? cropRegion(canvas, region) : canvas;
  return new Promise((resolve) =>
    source.toBlob((blob) => resolve(blob), "image/png")
  );
}

/**
 * Translate a result from region coordinates back into page coordinates.
 *
 * Without this, every box from a region call is reported relative to the crop
 * and lands in the wrong place on the sheet — the failure is invisible when
 * the region happens to start at the origin, which is exactly how it survives
 * casual testing.
 */
export function shiftResult(
  result: GdtPageResult,
  dx: number,
  dy: number
): GdtPageResult {
  if (dx === 0 && dy === 0) return result;
  return mapResultBoxes(result, (box) => ({
    ...box,
    x: box.x + dx,
    y: box.y + dy,
  }));
}

/**
 * Divide every box by `factor`, for a result read from an upscaled render.
 *
 * Reading a page at 3× is the single most effective thing available for the
 * small modifiers — a circled Ⓜ at a typical frame height is around 20px with
 * a 10px letter inside, which is at the limit of what any method resolves —
 * but the boxes come back in the upscaled image's pixels and are three times
 * too far from the origin until this runs.
 */
export function scaleResult(
  result: GdtPageResult,
  factor: number
): GdtPageResult {
  if (factor === 1 || factor <= 0) return result;
  return mapResultBoxes(result, (box) => ({
    ...box,
    x: box.x / factor,
    y: box.y / factor,
    ...("width" in box
      ? {
          width: (box as { width: number }).width / factor,
          height: (box as { height: number }).height / factor,
        }
      : {}),
  }));
}

type AnyBox = { x: number; y: number; width?: number; height?: number };

/** Apply one transform to every box a result carries, in one place. */
function mapResultBoxes(
  result: GdtPageResult,
  move: <T extends AnyBox>(box: T) => T
): GdtPageResult {
  const shiftBox = move;
  const shiftDetection = <T extends { bounding_box: AnyBox }>(
    detection: T
  ): T => ({ ...detection, bounding_box: shiftBox(detection.bounding_box) });

  return {
    ...result,
    symbols: result.symbols.map(shiftDetection),
    frames: result.frames.map((frame) => ({
      ...frame,
      bounding_box: shiftBox(frame.bounding_box),
      symbol: frame.symbol ? shiftDetection(frame.symbol) : null,
      compartments: frame.compartments.map((cell) => ({
        ...cell,
        bounding_box: shiftBox(cell.bounding_box),
        detection: cell.detection ? shiftDetection(cell.detection) : null,
      })),
      leader_point: frame.leader_point
        ? (shiftBox(frame.leader_point) as { x: number; y: number })
        : undefined,
    })),
    datum_associations: result.datum_associations.map((association) => ({
      ...association,
      datum_feature_bbox: association.datum_feature_bbox
        ? shiftBox(association.datum_feature_bbox)
        : null,
    })),
  };
}

/**
 * Record an operator's judgement about a detection.
 *
 * Returns whether it was stored. A failure is reported to the caller rather
 * than swallowed: an operator who corrects a detection and is told nothing
 * will reasonably assume it was saved.
 */
export async function submitGdtCorrection(
  correction: GdtCorrection
): Promise<{ ok: boolean; error?: string }> {
  try {
    await fetchJson("/gdt/corrections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(correction),
    });
    return { ok: true };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "could not save",
    };
  }
}

export async function fetchGdtCorrectionSummary(): Promise<{
  total: number;
  by_action: Record<string, number>;
  confusions: Record<string, number>;
} | null> {
  try {
    return await fetchJson("/gdt/corrections/summary");
  } catch {
    return null;
  }
}
