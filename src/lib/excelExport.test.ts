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
  it("builds the header, empty metadata, checksheet table and page image", async () => {
    const annotations = [
      makeAnnotation({
        id: "first",
        number: 1,
        label: "Outside diameter",
        value: "Ø25",
        range: "+0.1, -0.1",
        method: "Measure",
        tool: "Micrometer",
        page: 1,
      }),
      makeAnnotation({
        id: "second",
        number: 2,
        label: "Length",
        value: "50",
        range: "0",
        page: 1,
      }),
    ];

    const workbook = await buildInspectionWorkbook({
      annotations,
      pages: [
        { page: 1, dataUrl: ONE_PIXEL_PNG, width: 1000, height: 1400 },
      ],
      options: {
        companyName: "Organization Name",
        fileName: "Punch Inspection",
        partCount: 2,
      },
    });
    const worksheet = workbook.getWorksheet("Checksheet");
    expect(worksheet).toBeDefined();
    if (!worksheet) throw new Error("Checksheet worksheet was not created.");

    expect(worksheet.getCell("A1").value).toBe("Organization Name");
    expect(worksheet.getCell("A2").value).toBe("Punch Inspection");
    expect(worksheet.getCell("A4").value).toBe("Part Name");
    expect(worksheet.getCell("B4").value).toBe("");
    expect(worksheet.getCell("E4").value).toBe("Drawing No.");
    expect(worksheet.getCell("F4").value).toBe("");
    expect(worksheet.getCell("H4").value).toBe("Revision No.");
    expect(worksheet.getCell("I4").value).toBe("");

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
    expect(worksheet.getCell("E8").value).toBe("");
    expect(worksheet.getCell("F8").value).toBe("");
    expect(worksheet.getImages()).toHaveLength(1);

    const buffer = await workbook.xlsx.writeBuffer();
    expect(buffer.byteLength).toBeGreaterThan(1000);
  });
});
