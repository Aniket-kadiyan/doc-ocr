import { describe, expect, it } from "vitest";
import {
  buildInspectionWorkbook,
  excelDownloadName,
  partColumnNames,
  workbookTitle,
} from "@/lib/excelExport";
import { makeAnnotation } from "@/test/annotationFixture";
import type { PageReferenceTable } from "@/types/referencePoint";

const ONE_PIXEL_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6ZgAAAABJRU5ErkJggg==";

const box = { x: 10, y: 20, width: 30, height: 12 };
const referenceTables: PageReferenceTable[] = [
  {
    page: 1,
    bbox: box,
    headerBox: box,
    axes: ["X", "Y", "Z"],
    nameHeader: "POINT",
    points: [
      {
        symbol: "A",
        coordinates: [
          { axis: "X", value: "1" },
          { axis: "Z", value: "3" },
        ],
        nameBox: box,
        rowBox: box,
        markers: [{ bbox: box, score: 0.9, margin: 0.2 }],
      },
    ],
  },
];

describe("Excel export options", () => {
  it("creates numbered part columns", () => {
    expect(partColumnNames(3)).toEqual(["Part 1", "Part 2", "Part 3"]);
  });

  it("rejects invalid part counts", () => {
    expect(() => partColumnNames(0)).toThrow(/1 to 20/);
    expect(() => partColumnNames(21)).toThrow(/1 to 20/);
    expect(() => partColumnNames(1.5)).toThrow(/whole number/);
  });

  it("uses the entered name as the heading and a safe xlsx file name", () => {
    expect(workbookTitle("Punch Inspection.xlsx")).toBe("Punch Inspection");
    expect(excelDownloadName("Punch: Inspection.xlsx")).toBe(
      "Punch_ Inspection.xlsx"
    );
  });
});

describe("inspection workbook", () => {
  it("builds branded metadata, table and ballooned page content", async () => {
    const workbook = await buildInspectionWorkbook({
      annotations: [
        makeAnnotation({
          id: "first",
          number: 1,
          label: "Outside diameter",
          value: "Ø25",
          range: "+0.1",
          method: "Measure",
          tool: "Micrometer",
          page: 1,
        }),
      ],
      pages: [
        { page: 1, dataUrl: ONE_PIXEL_PNG, width: 1000, height: 1400 },
      ],
      options: {
        fileName: "Punch Inspection",
        partCount: 2,
        companyLogoDataUrl: ONE_PIXEL_PNG,
        metadata: {
          partName: "Drive Bracket",
          documentNumber: "DB-1042",
          revisionNumber: "C",
        },
      },
    });
    const worksheet = workbook.getWorksheet("Checksheet");
    expect(worksheet).toBeDefined();
    if (!worksheet) throw new Error("Checksheet worksheet was not created.");

    expect(worksheet.getCell("C1").value).toBe("Senior Flexonics");
    expect(worksheet.getCell("A2").value).toBe("Punch Inspection");
    expect(worksheet.getCell("B4").value).toBe("Drive Bracket");
    expect(worksheet.getCell("F4").value).toBe("DB-1042");
    expect(worksheet.getCell("I4").value).toBe("C");
    expect(worksheet.getRow(7).values).toEqual([
      undefined,
      "S.no",
      "Label",
      "Value",
      "Tolerance",
      "Part 1",
      "Part 2",
      "Method",
      "Tool",
    ]);
    expect(worksheet.getCell("A8").value).toBe("1");
    expect(worksheet.getCell("B8").value).toBe("Outside diameter");
    expect(worksheet.getCell("D8").value).toBe("+0.1, -0");
    expect(worksheet.getImages()).toHaveLength(2);

    const buffer = await workbook.xlsx.writeBuffer();
    expect(buffer.byteLength).toBeGreaterThan(1000);
  });

  it("allows blank document metadata", async () => {
    const workbook = await buildInspectionWorkbook({
      annotations: [makeAnnotation()],
      pages: [],
      options: { fileName: "Blank Metadata", partCount: 1 },
    });
    const worksheet = workbook.getWorksheet("Checksheet");
    expect(worksheet?.getCell("B4").value).toBe("");
    expect(worksheet?.getCell("F4").value).toBe("");
    expect(worksheet?.getCell("I4").value).toBe("");
  });

  it("creates a separate References worksheet with X/Y/Z rows and images", async () => {
    const workbook = await buildInspectionWorkbook({
      annotations: [makeAnnotation({ number: 1 })],
      pages: [],
      referenceTables,
      referencePages: [
        { page: 1, dataUrl: ONE_PIXEL_PNG, width: 1000, height: 1400 },
      ],
      options: {
        fileName: "Reference Export",
        partCount: 1,
        companyLogoDataUrl: ONE_PIXEL_PNG,
      },
    });

    const checksheet = workbook.getWorksheet("Checksheet");
    const references = workbook.getWorksheet("References");
    expect(checksheet).toBeDefined();
    expect(references).toBeDefined();
    expect(checksheet?.getCell("B8").value).not.toBe("A_X");
    expect(references?.getCell("A8").value).toBe("2");
    expect(references?.getCell("B8").value).toBe("A_X");
    expect(references?.getCell("C8").value).toBe("1");
    expect(references?.getCell("B9").value).toBe("A_Y");
    expect(references?.getCell("C9").value).toBe("");
    expect(references?.getCell("B10").value).toBe("A_Z");
    expect(references?.getCell("C10").value).toBe("3");
    expect(references?.getCell("L2").value).toBe("Page 1 — References");
    expect(references?.getImages()).toHaveLength(2);
  });
});
