import type { Annotation } from "@/types/annotation";
import { balloonGeometry } from "@/lib/balloonGeometry";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
} from "@/types/documentMetadata";

/**
 * "Ballooned drawing" exports — the shareable view of a finished project.
 *
 * Two shapes, both showing exactly what the annotator saw:
 *  - {@link renderBalloonedCanvas}: one page with its balloons burned in (PNG).
 *  - the PDF in {@link import("@/lib/balloonedPdf")}: every annotated page plus
 *    the inspection table, built from the {@link BalloonedPage}s this module
 *    rasterizes.
 */

/** One rasterized page of the drawing plus the annotations that sit on it. */
export interface BalloonedPage {
  page: number;
  /** data:image/png;base64,… of the page with no balloons — the PDF export
   * draws them over it as vectors so they stay crisp at any zoom. */
  dataUrl: string;
  width: number;
  height: number;
  annotations: Annotation[];
}

/** Draw one annotation's box, leader and balloon onto a 2D context at zoom 1. */
function drawBalloon(
  ctx: CanvasRenderingContext2D,
  annotation: Annotation
): void {
  const g = balloonGeometry(annotation);

  // Dashed selection box — rotated about its top-left, as Konva's Rect does.
  ctx.save();
  ctx.translate(g.box.x, g.box.y);
  ctx.rotate((g.box.rotation * Math.PI) / 180);
  ctx.setLineDash(g.boxDash);
  ctx.lineWidth = 2;
  ctx.strokeStyle = g.accent;
  ctx.strokeRect(0, 0, g.box.width, g.box.height);
  ctx.restore();

  // Leader line from the bubble down to the top edge of the box.
  ctx.save();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(g.leader.x1, g.leader.y1);
  ctx.lineTo(g.leader.x2, g.leader.y2);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = g.accent;
  ctx.stroke();

  // Bubble + number.
  ctx.beginPath();
  ctx.arc(g.circle.cx, g.circle.cy, g.circle.r, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = g.strong;
  ctx.stroke();

  ctx.fillStyle = g.strong;
  ctx.font = `bold ${g.fontSize}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(g.number), g.circle.cx, g.circle.cy);
  ctx.restore();
}

/**
 * Copy a rendered page and burn its balloons in. The source canvas is left
 * untouched, so the viewer keeps rendering from a clean page.
 */
export function renderBalloonedCanvas(
  source: HTMLCanvasElement,
  annotations: Annotation[],
  header?: {
    projectName?: string;
    metadata?: Partial<DocumentMetadata> | null;
  }
): HTMLCanvasElement {
  const headerHeight = header ? 52 : 0;
  const out = document.createElement("canvas");
  out.width = source.width;
  out.height = source.height + headerHeight;
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("Canvas context unavailable");
  if (header) {
    const metadata = normalizeDocumentMetadata(header.metadata);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, out.width, headerHeight);
    ctx.fillStyle = "#1e293b";
    ctx.font = "bold 15px sans-serif";
    ctx.fillText(header.projectName || "Drawing", 14, 20);
    ctx.fillStyle = "#475569";
    ctx.font = "12px sans-serif";
    ctx.fillText(
      [
        `Part: ${metadata.partName || "—"}`,
        `Document: ${metadata.documentNumber || "—"}`,
        `Revision: ${metadata.revisionNumber || "—"}`,
      ].join(" · "),
      14,
      40
    );
    ctx.strokeStyle = "#e2e8f0";
    ctx.beginPath();
    ctx.moveTo(0, headerHeight - 0.5);
    ctx.lineTo(out.width, headerHeight - 0.5);
    ctx.stroke();
  }
  ctx.drawImage(source, 0, headerHeight);
  ctx.save();
  ctx.translate(0, headerHeight);
  // Boxes first, then balloons, matching the viewer's layer order.
  for (const a of annotations) drawBalloon(ctx, a);
  ctx.restore();
  return out;
}

/** Canvas → PNG blob, for downloading a ballooned page as an image. */
export function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not encode the drawing as PNG."));
    }, "image/png");
  });
}

/** Renders one clean page of the open drawing (registered by the viewer). */
export type PageCanvasProvider = (
  page: number
) => Promise<HTMLCanvasElement | null>;

/**
 * Which pages a shared export should carry: every page that has at least one
 * annotation, falling back to whatever page is on screen when there are none.
 */
export function pagesToExport(
  annotations: Annotation[],
  currentPage: number
): number[] {
  const pages = Array.from(new Set(annotations.map((a) => a.page))).sort(
    (a, b) => a - b
  );
  return pages.length ? pages : [currentPage];
}

/**
 * Rasterize the requested pages and bundle each with its annotations. Pages the
 * provider can't render (e.g. the drawing was closed) are skipped.
 */
export async function collectBalloonedPages(
  renderPage: PageCanvasProvider,
  annotations: Annotation[],
  pageNumbers: number[]
): Promise<BalloonedPage[]> {
  const pages: BalloonedPage[] = [];
  for (const page of pageNumbers) {
    const canvas = await renderPage(page);
    if (!canvas) continue;
    pages.push({
      page,
      dataUrl: canvas.toDataURL("image/png"),
      width: canvas.width,
      height: canvas.height,
      annotations: annotations.filter((a) => a.page === page),
    });
  }
  return pages;
}
