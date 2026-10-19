import { describe, expect, it } from "vitest";
import {
  buildInspectionWorkbook,
  excelDownloadName,
  partColumnNames,
  workbookTitle,
} from "@/lib/excelExport";
import { makeAnnotation } from "@/test/annotationFixture";

const ONE_PIXEL_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6ZgAAAABJRU5ErkJggg==";

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
});
