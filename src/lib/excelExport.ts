import type ExcelJS from "exceljs";
import type { Annotation } from "@/types/annotation";
import { buildInspectionSheet } from "@/lib/project";
import type { BalloonedPageImage } from "@/lib/balloonedPageExport";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
} from "@/types/documentMetadata";

export const EXCEL_MIME_TYPE =
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

export interface ExcelExportOptions {
  companyName: string;
  fileName: string;
  partCount: number;
  metadata?: Partial<DocumentMetadata>;
}

const TABLE_HEADER_ROW = 7;
const METADATA_WIDTH = 10;
const MAX_IMAGE_WIDTH = 720;
const MAX_IMAGE_HEIGHT = 960;

const thinBorder: Partial<ExcelJS.Borders> = {
  top: { style: "thin", color: { argb: "FF94A3B8" } },
  left: { style: "thin", color: { argb: "FF94A3B8" } },
  bottom: { style: "thin", color: { argb: "FF94A3B8" } },
  right: { style: "thin", color: { argb: "FF94A3B8" } },
};

export function partColumnNames(partCount: number): string[] {
  if (!Number.isInteger(partCount) || partCount < 1 || partCount > 20) {
    throw new Error("Number of parts must be a whole number from 1 to 20.");
  }
  return Array.from({ length: partCount }, (_, index) => `Part ${index + 1}`);
}

export function workbookTitle(fileName: string): string {
  const title = fileName.trim().replace(/\.xlsx$/i, "").trim();
  if (!title) throw new Error("Enter a file name.");
  return title;
}

export function excelDownloadName(fileName: string): string {
  const title = workbookTitle(fileName);
  const safe = title
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .replace(/[. ]+$/g, "")
    .trim();
  return `${safe || "inspection_checksheet"}.xlsx`;
}

function styleMergedValue(
  worksheet: ExcelJS.Worksheet,
  range: string,
  value: string,
  options: {
    fill: string;
    color: string;
    size: number;
    bold?: boolean;
  }
) {
  worksheet.mergeCells(range);
  const cell = worksheet.getCell(range.split(":")[0]);
  cell.value = value;
  cell.font = {
    name: "Arial",
    size: options.size,
    bold: options.bold ?? true,
    color: { argb: options.color },
  };
  cell.fill = {
    type: "pattern",
    pattern: "solid",
    fgColor: { argb: options.fill },
  };
  cell.alignment = { horizontal: "center", vertical: "middle" };
  cell.border = thinBorder;
}

function addMetadataPair(
  worksheet: ExcelJS.Worksheet,
  labelCell: string,
  valueRange: string,
  label: string,
  value: string
) {
  const labelTarget = worksheet.getCell(labelCell);
  labelTarget.value = label;
  labelTarget.font = { name: "Arial", bold: true, color: { argb: "FF0F172A" } };
  labelTarget.fill = {
    type: "pattern",
    pattern: "solid",
    fgColor: { argb: "FFE2E8F0" },
  };
  labelTarget.alignment = { horizontal: "left", vertical: "middle" };
  labelTarget.border = thinBorder;

  worksheet.mergeCells(valueRange);
  const valueTarget = worksheet.getCell(valueRange.split(":")[0]);
  valueTarget.value = value;
  valueTarget.font = { name: "Arial", color: { argb: "FF0F172A" } };
  valueTarget.alignment = { horizontal: "left", vertical: "middle" };
  valueTarget.border = thinBorder;
}

function setTableColumnWidths(
  worksheet: ExcelJS.Worksheet,
  partCount: number
) {
  const widths = [8, 22, 18, 16, ...Array(partCount).fill(12), 18, 20];
  widths.forEach((width, index) => {
    worksheet.getColumn(index + 1).width = width;
  });
}

function addBalloonedPageImages(
  workbook: ExcelJS.Workbook,
  worksheet: ExcelJS.Worksheet,
  pages: BalloonedPageImage[],
  imageStartColumn: number
) {
  const headingCell = worksheet.getCell(1, imageStartColumn);
  headingCell.value = "Ballooned Pages";
  headingCell.font = {
    name: "Arial",
    size: 14,
    bold: true,
    color: { argb: "FFFFFFFF" },
  };
  headingCell.fill = {
    type: "pattern",
    pattern: "solid",
    fgColor: { argb: "FF1E3A5F" },
  };
  headingCell.alignment = { horizontal: "center", vertical: "middle" };
  headingCell.border = thinBorder;
  worksheet.getColumn(imageStartColumn).width = 18;

  let captionRow = 2;
  for (const page of pages) {
    const scale = Math.min(
      1,
      MAX_IMAGE_WIDTH / Math.max(1, page.width),
      MAX_IMAGE_HEIGHT / Math.max(1, page.height)
    );
    const width = Math.max(1, Math.round(page.width * scale));
    const height = Math.max(1, Math.round(page.height * scale));

    const caption = worksheet.getCell(captionRow, imageStartColumn);
    caption.value = `Page ${page.page}`;
    caption.font = { name: "Arial", bold: true, color: { argb: "FF334155" } };
    caption.alignment = { horizontal: "left", vertical: "middle" };

    const imageId = workbook.addImage({
      base64: page.dataUrl,
      extension: "png",
    });
    worksheet.addImage(imageId, {
      tl: { col: imageStartColumn - 1, row: captionRow },
      ext: { width, height },
      editAs: "oneCell",
    });

    captionRow += Math.ceil(height / 20) + 3;
  }
}

/** Build a workbook without touching browser download APIs, enabling unit tests. */
export async function buildInspectionWorkbook(args: {
  annotations: Annotation[];
  pages: BalloonedPageImage[];
  options: ExcelExportOptions;
}): Promise<ExcelJS.Workbook> {
  const ExcelJSImport = await import("exceljs");
  const ExcelJSRuntime = ExcelJSImport.default ?? ExcelJSImport;
  const workbook = new ExcelJSRuntime.Workbook();
  workbook.creator = "Doc OCR Box";
  workbook.created = new Date();
  workbook.modified = new Date();

  const title = workbookTitle(args.options.fileName);
  const companyName = args.options.companyName.trim();
  if (!companyName) throw new Error("Enter a company name.");

  const partColumns = partColumnNames(args.options.partCount);
  const inspection = buildInspectionSheet(args.annotations, partColumns);
  const metadata = normalizeDocumentMetadata(args.options.metadata);
  const worksheet = workbook.addWorksheet("Checksheet", {
    views: [{ state: "frozen", ySplit: TABLE_HEADER_ROW }],
    pageSetup: {
      orientation: "landscape",
      fitToPage: true,
      fitToWidth: 1,
      fitToHeight: 0,
      margins: {
        left: 0.25,
        right: 0.25,
        top: 0.5,
        bottom: 0.5,
        header: 0.2,
        footer: 0.2,
      },
    },
  });
  worksheet.properties.defaultRowHeight = 20;

  const leftColumnCount = Math.max(METADATA_WIDTH, inspection.headers.length);
  const leftEndColumn = worksheet.getColumn(leftColumnCount).letter;
  styleMergedValue(worksheet, `A1:${leftEndColumn}1`, companyName, {
    fill: "FF1E3A5F",
    color: "FFFFFFFF",
    size: 18,
  });
  styleMergedValue(worksheet, `A2:${leftEndColumn}2`, title, {
    fill: "FFDCE6F1",
    color: "FF0F172A",
    size: 14,
  });
  worksheet.getRow(1).height = 28;
  worksheet.getRow(2).height = 24;

  addMetadataPair(worksheet, "A4", "B4:D4", "Part Name", metadata.partName);
  addMetadataPair(
    worksheet,
    "E4",
    "F4:G4",
    "Document No.",
    metadata.documentNumber
  );
  addMetadataPair(
    worksheet,
    "H4",
    "I4:J4",
    "Revision No.",
    metadata.revisionNumber
  );
  worksheet.getRow(4).height = 24;

  const headerRow = worksheet.getRow(TABLE_HEADER_ROW);
  headerRow.values = inspection.headers;
  headerRow.height = 28;
  headerRow.eachCell({ includeEmpty: true }, (cell, columnNumber) => {
    if (columnNumber > inspection.headers.length) return;
    cell.font = { name: "Arial", bold: true, color: { argb: "FFFFFFFF" } };
    cell.fill = {
      type: "pattern",
      pattern: "solid",
      fgColor: { argb: "FF334155" },
    };
    cell.alignment = {
      horizontal: "center",
      vertical: "middle",
      wrapText: true,
    };
    cell.border = thinBorder;
  });

  inspection.rows.forEach((values, index) => {
    const row = worksheet.getRow(TABLE_HEADER_ROW + index + 1);
    row.values = values;
    row.height = 22;
    row.eachCell({ includeEmpty: true }, (cell, columnNumber) => {
      if (columnNumber > inspection.headers.length) return;
      cell.font = { name: "Arial", size: 10, color: { argb: "FF0F172A" } };
      cell.alignment = {
        horizontal: columnNumber === 1 ? "center" : "left",
        vertical: "middle",
        wrapText: true,
      };
      cell.border = thinBorder;
      if (index % 2 === 1) {
        cell.fill = {
          type: "pattern",
          pattern: "solid",
          fgColor: { argb: "FFF8FAFC" },
        };
      }
    });
  });

  setTableColumnWidths(worksheet, args.options.partCount);
  worksheet.autoFilter = {
    from: { row: TABLE_HEADER_ROW, column: 1 },
    to: { row: TABLE_HEADER_ROW, column: inspection.headers.length },
  };

  const imageStartColumn = leftColumnCount + 2;
  addBalloonedPageImages(workbook, worksheet, args.pages, imageStartColumn);
  return workbook;
}

export async function downloadInspectionWorkbook(args: {
  annotations: Annotation[];
  pages: BalloonedPageImage[];
  options: ExcelExportOptions;
}): Promise<string> {
  const workbook = await buildInspectionWorkbook(args);
  const buffer = await workbook.xlsx.writeBuffer();
  const bytes = new Uint8Array(buffer);
  const blob = new Blob([bytes], { type: EXCEL_MIME_TYPE });
  const fileName = excelDownloadName(args.options.fileName);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
  return fileName;
}
