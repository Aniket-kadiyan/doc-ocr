/**
 * OCR via local PaddleOCR API only (no Tesseract).
 * Start: uvicorn main:app --reload --port 8000
 */

import type { BBox, OCRResult } from "@/types/annotation";
import type {
  RunScanJobOptions,
  ScanJobResult,
} from "@/lib/scanJobClient";
import {
  abandonScanJob,
  cancelScanJob,
  isScanJobCancelledError,
  runScanJob,
} from "@/lib/scanJobClient";
import type { ScanProgress } from "@/types/scanJob";
import {
  checkOcrApiHealth,
  runPaddleOcr,
  runSegmentOcr,
  runReferencePoints,
  runTitleFields,
  type OcrApiHealth,
  type OcrEngine,
  type SegmentRegion,
} from "@/lib/paddleOcrClient";

export type { OcrEngine, OcrApiHealth, SegmentRegion };
export { isScanJobCancelledError };

export type OCRResultWithEngine = OCRResult & { engine?: OcrEngine };

let activeEngine: OcrEngine | "loading" | "offline" = "loading";
let apiHealth: OcrApiHealth = {
  available: false,
  paddleocr: false,
  trocr: false,
};

export function getActiveOcrEngine(): typeof activeEngine {
  return activeEngine;
}

export function getOcrApiStatus(): OcrApiHealth {
  return apiHealth;
}

export async function preloadOcr(): Promise<void> {
  apiHealth = await checkOcrApiHealth(true);
  if (apiHealth.available) {
    activeEngine = "paddleocr";
    return;
  }
  activeEngine = "offline";
}

export function isOcrReady(): boolean {
  return activeEngine === "paddleocr";
}

export async function runOCR(
  sourceCanvas: HTMLCanvasElement,
  bbox: BBox,
  displayScale = 1
): Promise<OCRResultWithEngine> {
  apiHealth = await checkOcrApiHealth();

  if (!apiHealth.available) {
    throw new Error(
      "PaddleOCR API is not running. Start it with: uvicorn main:app --reload --port 8000"
    );
  }

  const result = await runPaddleOcr(sourceCanvas, bbox, displayScale);
  activeEngine = result.engine ?? "paddleocr";
  return result;
}

export async function runSegment(
  sourceCanvas: HTMLCanvasElement,
  bbox: BBox,
  displayScale = 1
): Promise<SegmentRegion[]> {
  apiHealth = await checkOcrApiHealth();

  if (!apiHealth.available) {
    throw new Error(
      "PaddleOCR API is not running. Start it with: uvicorn main:app --reload --port 8000"
    );
  }

  activeEngine = "paddleocr";
  return runSegmentOcr(sourceCanvas, bbox, displayScale);
}

/**
 * Read the configured title-block keywords from the whole page.
 *
 * Kept beside {@link runSegment} so both share the API health check, but it is
 * a separate request: the title block's position on the sheet has nothing to do
 * with the rectangle drawn for Auto-Segment.
 */
export async function runPageTitleFields(pageCanvas: HTMLCanvasElement) {
  apiHealth = await checkOcrApiHealth();
  if (!apiHealth.available) {
    throw new Error(
      "PaddleOCR API is not running. Start it with: uvicorn main:app --reload --port 8000"
    );
  }
  return runTitleFields(pageCanvas);
}

/**
 * Read the sheet's coordinate reference table and locate its point names.
 *
 * Beside {@link runPageTitleFields} for the same reason: it is a whole-sheet
 * read whose subject sits wherever the drawing office put it, not inside any
 * rectangle the user drew.
 */
export async function runPageReferencePoints(
  pageCanvas: HTMLCanvasElement,
  displayScale = 1
) {
  apiHealth = await checkOcrApiHealth();
  if (!apiHealth.available) {
    throw new Error(
      "PaddleOCR API is not running. Start it with: uvicorn main:app --reload --port 8000"
    );
  }
  return runReferencePoints(pageCanvas, displayScale);
}

export async function runAutoBalloonScan(
  options: RunScanJobOptions
): Promise<ScanJobResult> {
  apiHealth = await checkOcrApiHealth();

  if (!apiHealth.available) {
    throw new Error(
      "PaddleOCR API is not running. Start it with: uvicorn main:app --reload --port 8000"
    );
  }

  activeEngine = "paddleocr";
  return runScanJob(options);
}

export async function stopAutoBalloonScan(
  jobId: string
): Promise<ScanProgress> {
  return cancelScanJob(jobId);
}

/** Stop a scan during page unload; see {@link abandonScanJob}. */
export function abandonAutoBalloonScan(jobId: string): void {
  abandonScanJob(jobId);
}

export async function terminateOCR(): Promise<void> {
  /* no-op — server-side OCR */
}
