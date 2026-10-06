import type { ProjectRecord } from "@/lib/db";
import {
  loadImageFile,
  loadPdfDocument,
  PDF_RENDER_SCALE,
  renderPdfPage,
} from "@/lib/pdfLoader";
import {
  fitBalloonFontSize,
  getAnchoredBalloonGeometry,
} from "@/lib/balloonStyle";
import { valueAnnotations } from "@/lib/project";
import type { Annotation } from "@/types/annotation";
import type { BalloonStyle } from "@/types/balloonStyle";

export interface BalloonedPageImage {
  page: number;
  dataUrl: string;
  width: number;
  height: number;
}

interface RenderBalloonedPagesArgs {
  project: ProjectRecord;
  annotations: Annotation[];
  style: BalloonStyle;
  artworkImage: HTMLImageElement | null;
}

const loadArtworkImage = (dataUrl: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new window.Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load the balloon artwork."));
    image.src = dataUrl;
  });

function drawFallbackBalloon(
  context: CanvasRenderingContext2D,
  geometry: ReturnType<typeof getAnchoredBalloonGeometry>,
  style: BalloonStyle
) {
  const anchorX = geometry.x + style.anchor.x * geometry.width;
  const anchorY = geometry.y + style.anchor.y * geometry.height;

  context.save();
  context.beginPath();
  context.moveTo(anchorX, anchorY);
  context.lineTo(geometry.x + geometry.width * 0.18, geometry.y + geometry.height * 0.52);
  context.lineTo(geometry.x + geometry.width * 0.82, geometry.y + geometry.height * 0.52);
  context.closePath();
  context.fillStyle = "#f97316";
  context.strokeStyle = "#111827";
  context.lineWidth = 2;
  context.fill();
  context.stroke();

  context.beginPath();
  context.arc(
    geometry.x + geometry.width / 2,
    geometry.y + geometry.width / 2,
    geometry.width * 0.45,
    0,
    Math.PI * 2
  );
  context.fillStyle = "#ffffff";
  context.fill();
  context.stroke();
  context.restore();
}

function drawAnnotation(
  context: CanvasRenderingContext2D,
  annotation: Annotation,
  style: BalloonStyle,
  artworkImage: HTMLImageElement | null
) {
  const { bbox, number } = annotation;

  context.save();
  context.strokeStyle = "#dc2626";
  context.lineWidth = 2;
  context.setLineDash([6, 4]);
  context.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
  context.restore();

  const geometry = getAnchoredBalloonGeometry(
    style,
    bbox.x + bbox.width / 2,
    bbox.y,
    1
  );

  if (artworkImage) {
    context.drawImage(
      artworkImage,
      geometry.x,
      geometry.y,
      geometry.width,
      geometry.height
    );
  } else {
    drawFallbackBalloon(context, geometry, style);
  }

  const fontSize = fitBalloonFontSize(
    String(number),
    geometry.numberWidth,
    geometry.numberHeight,
    style.display.drawingWidth,
    style
  );

  context.save();
  context.fillStyle = style.text.color;
  context.font = `${style.text.fontWeight} ${fontSize}px ${style.text.fontFamily}`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(
    String(number),
    geometry.numberX + geometry.numberWidth / 2,
    geometry.numberY + geometry.numberHeight / 2,
    geometry.numberWidth
  );
  context.restore();
}

/**
 * Render each page containing at least one exported value exactly once. Hidden
 * annotations are deliberately included because hiding is presentation-only;
 * the same values remain present in the exported checksheet table.
 */
export async function renderBalloonedPages({
  project,
  annotations,
  style,
  artworkImage,
}: RenderBalloonedPagesArgs): Promise<BalloonedPageImage[]> {
  if (!project.fileBlob) {
    throw new Error(
      "The original drawing is unavailable. Reopen it before exporting to Excel."
    );
  }

  const values = valueAnnotations(annotations);
  const pageNumbers = [...new Set(values.map((annotation) => annotation.page))]
    .filter((page) => Number.isInteger(page) && page > 0)
    .sort((left, right) => left - right);
  if (pageNumbers.length === 0) return [];

  const sourceFile = new File([project.fileBlob], project.fileName, {
    type:
      project.mimeType ||
      project.fileBlob.type ||
      (project.fileType === "pdf" ? "application/pdf" : "application/octet-stream"),
  });
  const resolvedArtwork =
    artworkImage ?? (await loadArtworkImage(style.artwork.dataUrl).catch(() => null));
  const output: BalloonedPageImage[] = [];

  if (project.fileType === "pdf") {
    const document = await loadPdfDocument(sourceFile);
    try {
      for (const page of pageNumbers) {
        if (page > document.numPages) continue;
        const rendered = await renderPdfPage(document, page, PDF_RENDER_SCALE);
        const context = rendered.canvas.getContext("2d");
        if (!context) throw new Error("Canvas context unavailable during Excel export.");

        values
          .filter((annotation) => annotation.page === page)
          .forEach((annotation) =>
            drawAnnotation(context, annotation, style, resolvedArtwork)
          );

        output.push({
          page,
          dataUrl: rendered.canvas.toDataURL("image/png"),
          width: rendered.canvas.width,
          height: rendered.canvas.height,
        });
      }
    } finally {
      await document.destroy();
    }
    return output;
  }

  for (const page of pageNumbers) {
    let rendered: Awaited<ReturnType<typeof loadImageFile>>;
    try {
      rendered = await loadImageFile(sourceFile, page);
    } catch (reason) {
      if (
        page > 1 &&
        reason instanceof Error &&
        /only one page|not present/i.test(reason.message)
      ) {
        continue;
      }
      throw reason;
    }
    const context = rendered.canvas.getContext("2d");
    if (!context) throw new Error("Canvas context unavailable during Excel export.");
    values
      .filter((annotation) => annotation.page === page)
      .forEach((annotation) =>
        drawAnnotation(context, annotation, style, resolvedArtwork)
      );
    output.push({
      page,
      dataUrl: rendered.canvas.toDataURL("image/png"),
      width: rendered.canvas.width,
      height: rendered.canvas.height,
    });
  }
  return output;
}
