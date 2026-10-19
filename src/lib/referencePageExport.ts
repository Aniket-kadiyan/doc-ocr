import type { PageCanvasProvider } from "@/lib/ballooned";
import { referenceColor } from "@/lib/referenceColors";
import { layoutReferenceTags } from "@/lib/referenceTagLayout";
import {
  normalizePageReferenceTables,
  REFERENCE_AXES,
} from "@/lib/referenceValues";
import type {
  PageReferenceTable,
  ReferencePoint,
} from "@/types/referencePoint";

export interface ReferencePageImage {
  page: number;
  dataUrl: string;
  width: number;
  height: number;
}

const TAG_HEADER_HEIGHT = 20;
const TAG_ROW_HEIGHT = 17;
const TAG_HEIGHT = TAG_HEADER_HEIGHT + REFERENCE_AXES.length * TAG_ROW_HEIGHT;
const TAG_MIN_WIDTH = 82;
const TAG_PADDING = 7;
const TAG_GAP = 8;
const RING_PADDING = 3;

function coordinateValue(point: ReferencePoint, axis: string): string {
  return (
    point.coordinates.find(
      (coordinate) => coordinate.axis.trim().toUpperCase() === axis
    )?.value.trim() ?? ""
  );
}

function tagWidth(point: ReferencePoint): number {
  const longest = Math.max(
    point.symbol.trim().length,
    ...REFERENCE_AXES.map(
      (axis) => `${axis} ${coordinateValue(point, axis)}`.length
    )
  );
  return Math.max(TAG_MIN_WIDTH, TAG_PADDING * 2 + longest * 7);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
}

function drawReferenceMarker(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  point: ReferencePoint,
  pointIndex: number,
  markerIndex: number,
  placement: { x: number; y: number },
  width: number
) {
  const marker = point.markers[markerIndex];
  const color = referenceColor(pointIndex);
  const ring = {
    x: marker.bbox.x - RING_PADDING,
    y: marker.bbox.y - RING_PADDING,
    width: marker.bbox.width + RING_PADDING * 2,
    height: marker.bbox.height + RING_PADDING * 2,
  };
  const x = clamp(placement.x, 0, canvas.width - width);
  const y = clamp(placement.y, 0, canvas.height - TAG_HEIGHT);
  const tagIsLeft = x + width / 2 < ring.x + ring.width / 2;
  const tagIsAbove = y + TAG_HEIGHT / 2 < ring.y + ring.height / 2;
  const anchorX = tagIsLeft ? x + width : x;
  const anchorY = tagIsAbove ? y + TAG_HEIGHT : y;

  context.save();
  context.strokeStyle = color.fill;
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(ring.x + ring.width / 2, ring.y + ring.height / 2);
  context.lineTo(anchorX, anchorY);
  context.stroke();
  context.strokeRect(ring.x, ring.y, ring.width, ring.height);

  context.fillStyle = "#ffffff";
  context.strokeStyle = "#1f2937";
  context.lineWidth = 1;
  context.fillRect(x, y, width, TAG_HEIGHT);
  context.strokeRect(x, y, width, TAG_HEIGHT);

  context.fillStyle = color.fill;
  context.fillRect(x, y, width, TAG_HEADER_HEIGHT);
  context.fillStyle = color.ink;
  context.font = "bold 14px Arial, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(
    point.symbol.trim(),
    x + width / 2,
    y + TAG_HEADER_HEIGHT / 2,
    width - TAG_PADDING * 2
  );

  REFERENCE_AXES.forEach((axis, row) => {
    const rowY = y + TAG_HEADER_HEIGHT + row * TAG_ROW_HEIGHT;
    context.fillStyle = "#6b7280";
    context.font = "bold 12px Consolas, monospace";
    context.textAlign = "left";
    context.fillText(axis, x + TAG_PADDING, rowY + TAG_ROW_HEIGHT / 2);
    context.fillStyle = "#111827";
    context.font = "12px Consolas, monospace";
    context.textAlign = "right";
    context.fillText(
      coordinateValue(point, axis),
      x + width - TAG_PADDING,
      rowY + TAG_ROW_HEIGHT / 2,
      width - TAG_PADDING * 3
    );
  });
  context.restore();
}

function drawReferences(canvas: HTMLCanvasElement, table: PageReferenceTable) {
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas context unavailable during reference export.");
  }

  const width = Math.max(TAG_MIN_WIDTH, ...table.points.map(tagWidth));
  const anchors = table.points.flatMap((point, pointIndex) =>
    point.markers.map((marker, markerIndex) => ({
      id: `${pointIndex}:${markerIndex}`,
      glyph: marker.bbox,
    }))
  );
  const placements = layoutReferenceTags(anchors, {
    width,
    height: TAG_HEIGHT,
    gap: TAG_GAP,
    bounds: { width: canvas.width, height: canvas.height },
  });

  table.points.forEach((point, pointIndex) => {
    point.markers.forEach((marker, markerIndex) => {
      const placement = placements.get(`${pointIndex}:${markerIndex}`) ?? {
        x: marker.bbox.x + marker.bbox.width + TAG_GAP,
        y: marker.bbox.y,
      };
      drawReferenceMarker(
        context,
        canvas,
        point,
        pointIndex,
        markerIndex,
        placement,
        width
      );
    });
  });
}

function copyCanvas(source: HTMLCanvasElement): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = source.width;
  canvas.height = source.height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas context unavailable during export.");
  context.drawImage(source, 0, 0);
  return canvas;
}

/** Render clean drawing pages with reference markers only—never balloons. */
export async function renderReferencePages(args: {
  renderPage: PageCanvasProvider;
  referenceTables: readonly PageReferenceTable[];
}): Promise<ReferencePageImage[]> {
  const output: ReferencePageImage[] = [];
  for (const table of normalizePageReferenceTables(args.referenceTables)) {
    const source = await args.renderPage(table.page);
    if (!source) continue;
    const canvas = copyCanvas(source);
    drawReferences(canvas, table);
    output.push({
      page: table.page,
      dataUrl: canvas.toDataURL("image/png"),
      width: canvas.width,
      height: canvas.height,
    });
  }
  return output;
}
