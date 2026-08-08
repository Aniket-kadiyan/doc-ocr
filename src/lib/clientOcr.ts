/**
 * OCR via local PaddleOCR API only (no Tesseract).
 * Start: uvicorn main:app --reload --port 8000
 */

import type { BBox, OCRResult } from "@/types/annotation";
import type {
  RunScanJobOptions,
  ScanJobResult,
} from "@/lib/scanJobClient";
import { runScanJob } from "@/lib/scanJobClient";
import {
  checkOcrApiHealth,
  runPaddleOcr,
  runSegmentOcr,
  type OcrApiHealth,
  type OcrEngine,
  type SegmentRegion,
} from "@/lib/paddleOcrClient";

export type { OcrEngine, OcrApiHealth, SegmentRegion };

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

export async function terminateOCR(): Promise<void> {
  /* no-op — server-side OCR */
}
