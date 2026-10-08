import * as pdfjs from "pdfjs-dist";

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

export async function loadImageFile(
  file: File
): Promise<{ canvas: HTMLCanvasElement; width: number; height: number }> {
  const url = URL.createObjectURL(file);
  const img = new Image();

  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = reject;
    img.src = url;
  });

  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context unavailable");
  ctx.drawImage(img, 0, 0);

  URL.revokeObjectURL(url);
  return { canvas, width: canvas.width, height: canvas.height };
}
