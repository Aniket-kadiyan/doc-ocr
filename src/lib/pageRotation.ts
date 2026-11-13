import {
  PAGE_ROTATIONS,
  type PageRotation,
  type PageRotations,
} from "@/types/pageRotation";

export function normalizePageRotation(value: unknown): PageRotation {
  const numeric = Number(value);
  return PAGE_ROTATIONS.includes(numeric as PageRotation)
    ? (numeric as PageRotation)
    : 0;
}

export function normalizePageRotations(value: unknown): PageRotations {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};

  const rotations: PageRotations = {};
  for (const [rawPage, rawRotation] of Object.entries(value)) {
    const page = Number(rawPage);
    if (!Number.isInteger(page) || page < 1) continue;
    const rotation = normalizePageRotation(rawRotation);
    if (rotation !== 0) rotations[page] = rotation;
  }
  return rotations;
}

export function pageRotationFor(
  rotations: PageRotations,
  page: number
): PageRotation {
  return normalizePageRotation(rotations[page]);
}

export function nextClockwiseRotation(rotation: PageRotation): PageRotation {
  return ((rotation + 90) % 360) as PageRotation;
}

/** Rotate a raster page clockwise without changing the source canvas. */
export function rotateCanvasPage(
  source: HTMLCanvasElement,
  rotation: PageRotation
): HTMLCanvasElement {
  if (rotation === 0) return source;

  const rotated = document.createElement("canvas");
  const swapAxes = rotation === 90 || rotation === 270;
  rotated.width = swapAxes ? source.height : source.width;
  rotated.height = swapAxes ? source.width : source.height;

  const context = rotated.getContext("2d");
  if (!context) throw new Error("Canvas context unavailable");

  if (rotation === 90) {
    context.translate(rotated.width, 0);
    context.rotate(Math.PI / 2);
  } else if (rotation === 180) {
    context.translate(rotated.width, rotated.height);
    context.rotate(Math.PI);
  } else {
    context.translate(0, rotated.height);
    context.rotate(-Math.PI / 2);
  }

  context.drawImage(source, 0, 0);
  return rotated;
}
