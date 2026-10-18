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

const referenceTable: PageReferenceTable = {
  page: 1,
  bbox: { x: 10, y: 20, width: 300, height: 120 },
  headerBox: { x: 10, y: 20, width: 300, height: 20 },
  axes: ["X", "Y"],
  nameHeader: "POINT",
  points: [
    {
      symbol: "A",
      coordinates: [
        { axis: "X", value: "10.25" },
        { axis: "Y", value: "-4.50" },
      ],
      nameBox: { x: 10, y: 40, width: 30, height: 20 },
      rowBox: { x: 10, y: 40, width: 300, height: 20 },
      markers: [
        {
          bbox: { x: 400, y: 200, width: 12, height: 14 },
          score: 0.94,
          margin: 0.21,
        },
      ],
    },
  ],
};

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
        fileName: "Punch Inspection",
        partCount: 2,
        companyLogoDataUrl: ONE_PIXEL_PNG,
      },
    });
    const worksheet = workbook.getWorksheet("Checksheet");
    expect(worksheet).toBeDefined();
    if (!worksheet) throw new Error("Checksheet worksheet was not created.");
    expect(workbook.getWorksheet("References")).toBeUndefined();

    expect(worksheet.getCell("C1").value).toBe("Senior Flexonics");
    expect(worksheet.getCell("A2").value).toBe("Punch Inspection");
    expect(worksheet.getCell("A4").value).toBe("Part Name");
    expect(worksheet.getCell("B4").value).toBe("");
    expect(worksheet.getCell("E4").value).toBe("Document No.");
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
    expect(worksheet.getImages()).toHaveLength(2);

    const buffer = await workbook.xlsx.writeBuffer();
    expect(buffer.byteLength).toBeGreaterThan(1000);
  });

  it("writes document metadata into the workbook header", async () => {
    const workbook = await buildInspectionWorkbook({
      annotations: [makeAnnotation()],
      pages: [],
      options: {
        fileName: "Bracket Inspection",
        partCount: 1,
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

    expect(worksheet.getCell("B4").value).toBe("Drive Bracket");
    expect(worksheet.getCell("F4").value).toBe("DB-1042");
    expect(worksheet.getCell("I4").value).toBe("C");
  });

  it("writes reference values and images to a separate worksheet", async () => {
    const workbook = await buildInspectionWorkbook({
      annotations: [
        makeAnnotation({
          id: "balloon-1",
          number: 1,
          label: "Diameter",
          value: "25",
        }),
      ],
      pages: [
        { page: 1, dataUrl: ONE_PIXEL_PNG, width: 1000, height: 1400 },
      ],
      referenceTables: [referenceTable],
      referencePages: [
        { page: 1, dataUrl: ONE_PIXEL_PNG, width: 1000, height: 1400 },
      ],
      options: {
        fileName: "Reference export",
        partCount: 2,
      },
    });
    const worksheet = workbook.getWorksheet("Checksheet");
    const referenceWorksheet = workbook.getWorksheet("References");
    expect(worksheet).toBeDefined();
    expect(referenceWorksheet).toBeDefined();
    if (!worksheet) throw new Error("Checksheet worksheet was not created.");
    if (!referenceWorksheet) {
      throw new Error("References worksheet was not created.");
    }

    expect(workbook.worksheets.map((sheet) => sheet.name)).toEqual([
      "Checksheet",
      "References",
    ]);
    expect(worksheet.getRow(8).values).toEqual([
      undefined,
      "1",
      "Diameter",
      "25",
      "0",
      "",
      "",
      "",
      "",
    ]);
    expect(worksheet.getColumn(2).values).not.toContain("A_X");
    expect(worksheet.getImages()).toHaveLength(1);

    expect(referenceWorksheet.getCell("A2").value).toBe(
      "Reference export — References"
    );
    expect(referenceWorksheet.getRow(8).values).toEqual([
      undefined,
      "2",
      "A_X",
      "10.25",
      "",
      "",
      "",
      "",
      "",
    ]);
    expect(referenceWorksheet.getRow(9).values).toEqual([
      undefined,
      "3",
      "A_Y",
      "-4.50",
      "",
      "",
      "",
      "",
      "",
    ]);
    expect(referenceWorksheet.getRow(10).values).toEqual([
      undefined,
      "4",
      "A_Z",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
    expect(referenceWorksheet.getImages()).toHaveLength(1);

    const captions = (sheet: typeof worksheet): string[] => {
      const values: string[] = [];
      sheet.eachRow((row) => {
        row.eachCell((cell) => {
          if (typeof cell.value === "string" && cell.value.startsWith("Page ")) {
            values.push(cell.value);
          }
        });
      });
      return values;
    };
    expect(captions(worksheet)).toEqual(["Page 1 — Balloons"]);
    expect(captions(referenceWorksheet)).toEqual(["Page 1 — References"]);

    const buffer = await workbook.xlsx.writeBuffer();
    expect(buffer.byteLength).toBeGreaterThan(1000);
  });
});
