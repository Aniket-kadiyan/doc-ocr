import * as pdfjs from "pdfjs-dist";
import { getOcrApiUrl } from "@/lib/paddleOcrClient";
import {
  detectSourceFileKind,
  isTiffKind,
  type SourceFileKind,
} from "@/lib/sourceFile";
import { apiRequestError, fetchWithRequestId } from "@/lib/requestDiagnostics";

/** Base PDF raster scale used by stored annotation coordinates and previews. */
export const PDF_RENDER_SCALE = 1.5;

/**
 * Resolution the on-screen page bitmap is rendered at, and the pixel budget
 * that caps it.
 *
 * Deliberately independent of {@link PDF_RENDER_SCALE}. Zoom in the viewer is
 * a display transform over a bitmap rendered once per page, so that bitmap's
 * own resolution is the ceiling on how sharp the sheet can ever look —  at
 * 1.5 the ceiling is 108dpi, and the drawing goes soft the moment it is
 * zoomed past 1:1. Rendering the displayed copy finer changes no geometry,
 * because it is drawn into the same PDF_RENDER_SCALE-sized box that every
 * stored annotation is measured in.
 *
 * The budget is what keeps a large sheet from asking the browser for a canvas
 * it will refuse. It sits well below the scan and reference-point budgets on
 * purpose: those renders are cropped, sent, and dropped, while this one is
 * held for as long as the page is open.
 */
export const DISPLAY_RENDER_SCALE = 4;
export const DISPLAY_MAX_PIXELS = 24_000_000;

/** The finest scale this page can be displayed at within the pixel budget. */
export function displayRenderScale(width: number, height: number): number {
  const budget = Math.sqrt(DISPLAY_MAX_PIXELS / (width * height));
  return Math.max(PDF_RENDER_SCALE, Math.min(DISPLAY_RENDER_SCALE, budget));
}

if (typeof window !== "undefined") {
  pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.mjs`;
}

export async function loadPdfDocument(file: File): Promise<pdfjs.PDFDocumentProxy> {
  const buffer = await file.arrayBuffer();
  return pdfjs.getDocument({ data: buffer }).promise;
}

export async function renderPdfPage(
  doc: pdfjs.PDFDocumentProxy,
  pageNum: number,
  scale: number
): Promise<{ canvas: HTMLCanvasElement; width: number; height: number }> {
  const page = await doc.getPage(pageNum);
  const viewport = page.getViewport({ scale });
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context unavailable");

  canvas.width = viewport.width;
  canvas.height = viewport.height;

  await page.render({
    canvasContext: ctx,
    viewport,
  }).promise;

  return { canvas, width: viewport.width, height: viewport.height };
}

export interface RenderedImagePage {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  totalPages: number;
  sourceFormat: SourceFileKind;
}

async function blobToCanvas(blob: Blob): Promise<{
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
}> {
  const url = URL.createObjectURL(blob);
  const img = new Image();
  try {
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () =>
        reject(new Error("The drawing image could not be decoded."));
      img.src = url;
    });

    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas context unavailable");
    ctx.drawImage(img, 0, 0);
    return { canvas, width: canvas.width, height: canvas.height };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function responseError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || `Image service returned HTTP ${response.status}.`;
  } catch {
    return `Image service returned HTTP ${response.status}.`;
  }
}

async function loadTiffPage(
  file: File,
  page: number
): Promise<RenderedImagePage> {
  const form = new FormData();
  form.append("file", file, file.name || "drawing.tiff");
  form.append("page", String(page));
  const response = await fetchWithRequestId(
    `${getOcrApiUrl()}/documents/raster-page`,
    { method: "POST", body: form },
    { operation: "document.raster-page" }
  );
  if (!response.ok) {
    throw apiRequestError(
      response, await responseError(response), "document.raster-page"
    );
  }

  const totalPages = Number(
    response.headers.get("X-Document-Page-Count") || "1"
  );
  const rendered = await blobToCanvas(await response.blob());
  return {
    ...rendered,
    totalPages:
      Number.isInteger(totalPages) && totalPages > 0 ? totalPages : 1,
    sourceFormat: "tiff",
  };
}

/** Render one 1-based page from a raster drawing. TIFF frames behave as pages. */
export async function loadImageFile(
  file: File,
  page = 1
): Promise<RenderedImagePage> {
  if (!Number.isInteger(page) || page < 1) {
    throw new Error("The image page must be a positive whole number.");
  }
  const sourceFormat = await detectSourceFileKind(file);
  if (sourceFormat === "pdf") {
    throw new Error("Use the PDF loader for PDF drawings.");
  }
  if (isTiffKind(sourceFormat)) return loadTiffPage(file, page);
  if (page !== 1) throw new Error("This image contains only one page.");
  const rendered = await blobToCanvas(file);
  return { ...rendered, totalPages: 1, sourceFormat };
}
