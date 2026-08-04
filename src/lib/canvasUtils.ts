import type { BBox } from "@/types/annotation";

/** White padding (px) added around each crop sent to the OCR API. */
export const CROP_PAD_PX = 20;
const PAD_PX = CROP_PAD_PX;
const MIN_OCR_EDGE = 96;
const MAX_UPSCALE = 6;

export function cropRegion(
  source: HTMLCanvasElement,
  bbox: BBox,
  scale = 1
): HTMLCanvasElement {
  const x = Math.max(0, Math.round(bbox.x * scale));
  const y = Math.max(0, Math.round(bbox.y * scale));
  const w = Math.max(1, Math.round(bbox.width * scale));
  const h = Math.max(1, Math.round(bbox.height * scale));

  const temp = document.createElement("canvas");
  temp.width = w + PAD_PX * 2;
  temp.height = h + PAD_PX * 2;

  const ctx = temp.getContext("2d");
  if (!ctx) return temp;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, temp.width, temp.height);
  ctx.drawImage(source, x, y, w, h, PAD_PX, PAD_PX, w, h);
  return temp;
}

export function normalizeBBox(
  start: { x: number; y: number },
  end: { x: number; y: number }
): BBox {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);
  return { x, y, width, height };
}

export function rotateCanvas(
  source: HTMLCanvasElement,
  angleDeg: number
): HTMLCanvasElement {
  const rad = (angleDeg * Math.PI) / 180;
  const sin = Math.abs(Math.sin(rad));
  const cos = Math.abs(Math.cos(rad));
  const w = source.width * cos + source.height * sin;
  const h = source.width * sin + source.height * cos;

  const temp = document.createElement("canvas");
  temp.width = Math.ceil(w);
  temp.height = Math.ceil(h);

  const ctx = temp.getContext("2d");
  if (!ctx) return source;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, temp.width, temp.height);
  ctx.translate(temp.width / 2, temp.height / 2);
  ctx.rotate(rad);
  ctx.drawImage(source, -source.width / 2, -source.height / 2);

  return temp;
}

export function canvasToImageData(canvas: HTMLCanvasElement): ImageData | null {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/** Scale small crops — engineering text is often tiny on drawings. */
export function upscaleForOcr(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const maxEdge = Math.max(canvas.width, canvas.height);
  let result = canvas;

  if (maxEdge < MIN_OCR_EDGE) {
    const factor = Math.min(
      MAX_UPSCALE,
      Math.ceil(MIN_OCR_EDGE / Math.max(maxEdge, 1))
    );
    const temp = document.createElement("canvas");
    temp.width = Math.round(canvas.width * factor);
    temp.height = Math.round(canvas.height * factor);
    const ctx = temp.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, temp.width, temp.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(canvas, 0, 0, temp.width, temp.height);
      result = temp;
    }
  }

  return enhanceContrast(result);
}

/** Grayscale + stretch contrast (keeps thin dimension strokes). */
export function enhanceContrast(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const temp = document.createElement("canvas");
  temp.width = canvas.width;
  temp.height = canvas.height;
  const ctx = temp.getContext("2d");
  if (!ctx) return canvas;

  ctx.drawImage(canvas, 0, 0);
  const imageData = ctx.getImageData(0, 0, temp.width, temp.height);
  const data = imageData.data;

  let min = 255;
  let max = 0;
  const gray: number[] = [];

  for (let i = 0; i < data.length; i += 4) {
    const g =
      0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    gray.push(g);
    min = Math.min(min, g);
    max = Math.max(max, g);
  }

  const range = Math.max(max - min, 1);

  for (let i = 0, gi = 0; i < data.length; i += 4, gi++) {
    const stretched = ((gray[gi] - min) / range) * 255;
    const v = stretched < 100 ? 0 : stretched > 210 ? 255 : stretched;
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
  }

  ctx.putImageData(imageData, 0, 0);
  return temp;
}

export function invertCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const temp = document.createElement("canvas");
  temp.width = canvas.width;
  temp.height = canvas.height;
  const ctx = temp.getContext("2d");
  if (!ctx) return canvas;
  ctx.drawImage(canvas, 0, 0);
  const imageData = ctx.getImageData(0, 0, temp.width, temp.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 255 - data[i];
    data[i + 1] = 255 - data[i + 1];
    data[i + 2] = 255 - data[i + 2];
  }
  ctx.putImageData(imageData, 0, 0);
  return temp;
}

/** @deprecated Use enhanceContrast */
export function preprocessForOCR(canvas: HTMLCanvasElement): HTMLCanvasElement {
  return enhanceContrast(canvas);
}
