import type { Annotation } from "@/types/annotation";
import { buildInspectionSheet } from "@/lib/project";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
} from "@/types/documentMetadata";
import {
  BALLOON_COLORS,
  BALLOON_FONT_SIZE,
  BALLOON_RADIUS,
  balloonGeometry,
} from "@/lib/balloonGeometry";
import type { BalloonedPage } from "@/lib/ballooned";

/**
 * "Ballooned drawing" PDF — the shareable deliverable of a finished project:
 * every annotated page with its balloons over it, followed by the inspection
 * table. One file that opens the same everywhere and prints as-is.
 *
 * The page raster is embedded once per page; the balloons, boxes and numbers
 * are drawn as PDF vectors (not burned into the image), so they stay crisp at
 * any zoom — the same reason the old HTML export overlaid them as SVG.
 */

/** Points. Long edge of a drawing page, ≈ A3 — generous for a drawing sheet. */
const DRAWING_LONG_EDGE = 1190;
const MARGIN = 24;
/** Band above the drawing holding the project name and page number. */
const HEADER_HEIGHT = 34;
/** A4 landscape, for the inspection table. */
const TABLE_PAGE: [number, number] = [841.89, 595.28];
/** Balloons shrink with the drawing, but never below this radius (points),
 * or the numbers stop being readable on a printed sheet. */
const MIN_BALLOON_RADIUS = 6.5;
/** Hairlines can disappear on some printers — keep every stroke at least this
 * wide (points). */
const MIN_LINE_WIDTH = 0.3;

type Rgb = [number, number, number];

function rgb(hex: string): Rgb {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const isDimension = (a: Annotation) => (a.kind ?? "dimension") === "dimension";

/** Where a drawing page's raster sits on its PDF page, and at what scale. */
interface PageLayout {
  pageWidth: number;
  pageHeight: number;
  imageX: number;
  imageY: number;
  imageWidth: number;
  imageHeight: number;
  /** Source pixels → points. */
  scale: number;
}

/**
 * Size the PDF page to the page raster itself: the drawing fills the sheet at
 * its own aspect ratio, with room for the margins and the header band.
 */
function layoutPage(width: number, height: number): PageLayout {
  const scale = (DRAWING_LONG_EDGE - 2 * MARGIN) / Math.max(width, height, 1);
  const imageWidth = width * scale;
  const imageHeight = height * scale;
  return {
    pageWidth: imageWidth + 2 * MARGIN,
    pageHeight: imageHeight + 2 * MARGIN + HEADER_HEIGHT,
    imageX: MARGIN,
    imageY: MARGIN + HEADER_HEIGHT,
    imageWidth,
    imageHeight,
    scale,
  };
}

/** Draw one annotation's box, leader and balloon over the placed raster. */
function drawBalloon(
  doc: import("jspdf").jsPDF,
  annotation: Annotation,
  layout: PageLayout
): void {
  const g = balloonGeometry(annotation);
  const colors = g.isLabel ? BALLOON_COLORS.label : BALLOON_COLORS.dimension;
  const accent = rgb(colors.accent);
  const strong = rgb(colors.strong);
  const s = layout.scale;
  const px = (x: number) => layout.imageX + x * s;
  const py = (y: number) => layout.imageY + y * s;
  const stroke = (w: number) => Math.max(w * s, MIN_LINE_WIDTH);

  // Dashed box, drawn as its four (possibly rotated) corners so the rotation
  // needs no graphics-state transform.
  const rad = (g.box.rotation * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const corner = (dx: number, dy: number): [number, number] => [
    px(g.box.x + dx * cos - dy * sin),
    py(g.box.y + dx * sin + dy * cos),
  ];
  const corners: Array<[number, number]> = [
    corner(0, 0),
    corner(g.box.width, 0),
    corner(g.box.width, g.box.height),
    corner(0, g.box.height),
  ];
  doc.setDrawColor(...accent);
  doc.setLineWidth(stroke(2));
  doc.setLineDashPattern(
    g.boxDash.map((d) => Math.max(d * s, 0.6)),
    0
  );
  for (let i = 0; i < corners.length; i++) {
    const [x1, y1] = corners[i];
    const [x2, y2] = corners[(i + 1) % corners.length];
    doc.line(x1, y1, x2, y2);
  }
  doc.setLineDashPattern([], 0);

  // Balloons keep a floor size in points, so the bubble may be larger relative
  // to the drawing than on screen; it stays centered on the same anchor.
  const radius = Math.max(g.circle.r * s, MIN_BALLOON_RADIUS);
  const cx = px(g.circle.cx);
  const cy = py(g.circle.cy);

  // Leader line from the bubble's lower edge down to the top of the box.
  doc.setLineWidth(stroke(1.5));
  doc.line(cx, cy + radius, px(g.leader.x2), py(g.leader.y2));

  doc.setFillColor(255, 255, 255);
  doc.setDrawColor(...strong);
  doc.setLineWidth(Math.max(stroke(2), 0.5));
  doc.circle(cx, cy, radius, "FD");

  const digits = String(g.number);
  // Three digits in a bubble sized for one need the type pulled in a little.
  const fit = digits.length >= 3 ? 0.7 : digits.length === 2 ? 0.85 : 1;
  doc.setTextColor(...strong);
  doc.setFont("helvetica", "bold");
  doc.setFontSize((radius / BALLOON_RADIUS) * BALLOON_FONT_SIZE * fit);
  doc.text(digits, cx, cy, { align: "center", baseline: "middle" });
}

export interface BalloonedPdfArgs {
  projectName: string;
  metadata?: Partial<DocumentMetadata> | null;
  pages: BalloonedPage[];
  /** Every annotation in the project — the table spans all pages. */
  annotations: Annotation[];
  /** Inspector-filled columns to leave blank for the recipient to fill in. */
  extraColumns?: string[];
  savedAt?: number;
}

/**
 * Build the ballooned-drawing PDF: one sheet per annotated page, then the
 * inspection table (and any labels nobody attached a value to).
 */
export async function buildBalloonedPdf({
  projectName,
  metadata,
  pages,
  annotations,
  extraColumns = [],
  savedAt = Date.now(),
}: BalloonedPdfArgs): Promise<Blob> {
  // jsPDF and the table plugin are only needed for this one export — load them
  // on use so they stay out of the app's initial bundle.
  const [{ jsPDF }, { default: autoTable }] = await Promise.all([
    import("jspdf"),
    import("jspdf-autotable"),
  ]);

  const normalizedMetadata = normalizeDocumentMetadata(metadata);
  const metadataLine = [
    `Part: ${normalizedMetadata.partName || "—"}`,
    `Document: ${normalizedMetadata.documentNumber || "—"}`,
    `Revision: ${normalizedMetadata.revisionNumber || "—"}`,
  ].join(" · ");
  const first = pages[0] ? layoutPage(pages[0].width, pages[0].height) : null;
  const doc = new jsPDF({
    unit: "pt",
    compress: true,
    format: first ? [first.pageWidth, first.pageHeight] : TABLE_PAGE,
    orientation: first && first.pageWidth > first.pageHeight ? "landscape" : "portrait",
  });
  doc.setProperties({
    title: `${projectName} — Ballooned Drawing`,
    subject: "Ballooned drawing and inspection values",
    creator: "Doc OCR Box",
  });

  const stamp = new Date(savedAt).toLocaleString();

  pages.forEach((page, i) => {
    const layout = i === 0 && first ? first : layoutPage(page.width, page.height);
    if (i > 0) {
      doc.addPage(
        [layout.pageWidth, layout.pageHeight],
        layout.pageWidth > layout.pageHeight ? "landscape" : "portrait"
      );
    }

    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(30, 41, 59);
    doc.text(projectName, MARGIN, MARGIN + 11);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(71, 85, 105);
    doc.text(metadataLine, MARGIN, MARGIN + 24);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    doc.text(
      `Page ${page.page}${pages.length > 1 ? ` of ${pages.length} shown` : ""} · ${stamp}`,
      layout.pageWidth - MARGIN,
      MARGIN + 11,
      { align: "right" }
    );

    doc.addImage(
      page.dataUrl,
      "PNG",
      layout.imageX,
      layout.imageY,
      layout.imageWidth,
      layout.imageHeight
    );
    for (const a of page.annotations) drawBalloon(doc, a, layout);
  });

  // Inspection table, on its own landscape sheet(s) so wide checksheets fit.
  const sheet = buildInspectionSheet(annotations, extraColumns);
  doc.addPage(TABLE_PAGE, "landscape");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.setTextColor(30, 41, 59);
  doc.text("Inspection values", MARGIN, MARGIN + 12);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(100, 116, 139);
  const dimCount = sheet.rows.length;
  const labelCount = annotations.filter((a) => !isDimension(a)).length;
  doc.text(
    `${projectName} · ${dimCount} value${dimCount === 1 ? "" : "s"} · ${labelCount} label${
      labelCount === 1 ? "" : "s"
    } · ${stamp}`,
    MARGIN,
    MARGIN + 26
  );
  doc.setFontSize(8);
  doc.text(metadataLine, MARGIN, MARGIN + 38);

  const valueColumn = 2;
  autoTable(doc, {
    startY: MARGIN + 50,
    margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
    head: [sheet.headers],
    body: sheet.rows,
    styles: { font: "helvetica", fontSize: 9, cellPadding: 4, textColor: [30, 41, 59] },
    headStyles: {
      fillColor: [248, 250, 252],
      textColor: [71, 85, 105],
      fontStyle: "bold",
      lineColor: [226, 232, 240],
      lineWidth: { bottom: 0.7 },
    },
    bodyStyles: { lineColor: [241, 245, 249], lineWidth: { bottom: 0.5 } },
    columnStyles: {
      0: { cellWidth: 30, textColor: [100, 116, 139] },
      [valueColumn]: { font: "courier" },
    },
    theme: "plain",
  });

  // Labels nobody attached a value to still carry a balloon on the drawing, so
  // list them rather than letting them fall off the report.
  const orphanLabels = annotations
    .filter((a) => !isDimension(a) && !annotations.some((d) => d.labelId === a.id))
    .sort((a, b) => a.number - b.number);
  if (orphanLabels.length) {
    const lastTable = (doc as unknown as { lastAutoTable?: { finalY: number } })
      .lastAutoTable;
    const y = (lastTable?.finalY ?? MARGIN + 38) + 26;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(30, 41, 59);
    if (y > TABLE_PAGE[1] - MARGIN - 60) {
      doc.addPage(TABLE_PAGE, "landscape");
      doc.text("Labels without a value", MARGIN, MARGIN + 12);
    } else {
      doc.text("Labels without a value", MARGIN, y);
    }
    autoTable(doc, {
      startY: (y > TABLE_PAGE[1] - MARGIN - 60 ? MARGIN + 12 : y) + 8,
      margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      head: [["No.", "Label"]],
      body: orphanLabels.map((l) => [
        String(l.number),
        l.value || "(unnamed label)",
      ]),
      styles: { font: "helvetica", fontSize: 9, cellPadding: 4, textColor: [30, 41, 59] },
      headStyles: {
        fillColor: [248, 250, 252],
        textColor: [71, 85, 105],
        fontStyle: "bold",
        lineColor: [226, 232, 240],
        lineWidth: { bottom: 0.7 },
      },
      bodyStyles: { lineColor: [241, 245, 249], lineWidth: { bottom: 0.5 } },
      columnStyles: { 0: { cellWidth: 30, textColor: [100, 116, 139] } },
      theme: "plain",
      tableWidth: "wrap",
    });
  }

  return doc.output("blob");
}
