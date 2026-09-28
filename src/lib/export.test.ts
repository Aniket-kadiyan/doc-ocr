import { describe, expect, it } from "vitest";
import {
  exportInspectionCSV,
  exportInspectionJSON,
  exportXML,
  findMalformedToleranceAnnotations,
} from "@/lib/export";
import { makeAnnotation } from "@/test/annotationFixture";

describe("tolerance export validation", () => {
  it("accepts supported default, symmetric, and asymmetric tolerances", () => {
    const annotations = [
      makeAnnotation({ id: "zero", value: "0.02", range: "0" }),
      makeAnnotation({
        id: "symmetric",
        value: "58.21±0.05",
        range: "+0.05, -0.05",
      }),
      makeAnnotation({ id: "asymmetric", value: "25", range: "+0.1, -0.2" }),
    ];

    expect(findMalformedToleranceAnnotations(annotations)).toEqual([]);
  });

  it("identifies both field and embedded malformed expressions", () => {
    const annotations = [
      makeAnnotation({ id: "bad-field", number: 1, range: "+0.1 +0.2" }),
      makeAnnotation({ id: "bad-value", number: 2, value: "25 ±", range: undefined }),
      makeAnnotation({ id: "good", number: 3, value: "25", range: "0" }),
    ];

    expect(
      findMalformedToleranceAnnotations(annotations).map(({ id }) => id)
    ).toEqual(["bad-field", "bad-value"]);
  });

  // 47630.pdf: the ballooned NOTES block. Its first number is the point
  // marker "1." and "B633-LATEST" puts a hyphen after it, which read as an
  // unreadable minus tolerance and blocked every save and export.
  const NOTES_BLOCK = [
    "1. MATERIAL: GRADE 5 EQUIV; 85KPSI YIELD MIN.;120KPSI TENSILE MIN.",
    `2. PIN TO BE STRAIGHT WITHIN .004" OF ENTIRE LENGTH`,
    "3. PIN TO ASSEMBLED WITH CABLE AND HAIR PIN COTTER AS SHOWN",
    "4. FINISH: ZINC YELLOW PER ASTM B633-LATEST REV. TYPE II",
    "5. LANYARD RING MUST WITHSTAND 50 LBS MIN. WITHOUT OPENING.",
  ].join("\n");

  it("never blocks an export on a drawing notes block", () => {
    const annotations = [
      makeAnnotation({
        id: "notes",
        number: 21,
        value: NOTES_BLOCK,
        type: "General Note",
        range: undefined,
      }),
      // The same prose classified as Material by the keyword rules.
      makeAnnotation({
        id: "notes-as-material",
        number: 22,
        value: NOTES_BLOCK,
        type: "Material",
        range: undefined,
      }),
      makeAnnotation({ id: "good", number: 23, value: "25", range: "0" }),
    ];

    expect(findMalformedToleranceAnnotations(annotations)).toEqual([]);
  });

  it("still reports a real dimension that merely mentions a note", () => {
    const annotations = [
      makeAnnotation({ id: "bad", number: 1, value: "25 ±", range: undefined }),
    ];

    expect(
      findMalformedToleranceAnnotations(annotations).map(({ id }) => id)
    ).toEqual(["bad"]);
  });
});

describe("values-only exports", () => {
  const annotations = [
    makeAnnotation({ id: "plain", number: 1, value: "25", range: "0" }),
    makeAnnotation({
      id: "detailed",
      number: 2,
      label: "Diameter & finish",
      value: "Ø58.21",
      range: "+0.05, -0.05",
      method: "Measure",
      tool: "Micrometer",
    }),
  ];

  // A sheet is the value rows, then the title-block section: a gap, then one
  // row per configured "Keywords to look for" entry. Every keyword gets a row
  // whether or not the drawing carried a value for it, so a failed read is
  // visible to the inspector instead of silently missing.
  const valueRowCount = annotations.length;

  it("exports JSON with exactly one row per value", () => {
    const result = JSON.parse(exportInspectionJSON(annotations, ["part1"]));

    expect(result.extra_columns).toEqual(["part1"]);
    expect(result.data.filter((row: Record<string, string>) => row["S.no"]))
      .toHaveLength(valueRowCount);
    expect(result.data[0]).toEqual({
      "S.no": "1",
      // No label was stored, so the row shows the one its category earns.
      Label: "Linear Dimension",
      Value: "25",
      Tolerance: "0",
      part1: "",
      Method: "",
      Tool: "",
    });
  });

  it("appends the title-block keywords under the value rows", () => {
    const result = JSON.parse(exportInspectionJSON(annotations, ["part1"]));
    const labels = result.data
      .slice(valueRowCount)
      .map((row: Record<string, string>) => row.Label)
      .filter(Boolean);

    expect(labels).toEqual(["DWG", "REV"]);
    // No drawing was scanned here, so the values are blank to fill in by hand.
    expect(
      result.data.slice(valueRowCount).every((row: Record<string, string>) => !row.Value)
    ).toBe(true);
  });

  it("always appends the drawing notes, one row per numbered point", () => {
    const notes = makeAnnotation({
      id: "notes",
      number: 3,
      label: "General Notes",
      type: "General Note",
      range: undefined,
      value: [
        "1. MATERIAL: GRADE 5 EQUIV; 85KPSI YIELD MIN.",
        "4. FINISH: ZINC YELLOW PER ASTM B633-LATEST REV. TYPE II",
      ].join("\n"),
    });
    const result = JSON.parse(
      exportInspectionJSON([...annotations, notes], ["part1"])
    );
    const labels = result.data.map((row: Record<string, string>) => row.Label);

    // The block keeps its own numbered row among the values...
    expect(
      result.data.some(
        (row: Record<string, string>) =>
          row["S.no"] === "3" && row.Value.startsWith("1. MATERIAL")
      )
    ).toBe(true);
    // ...and is expanded point by point under a NOTES heading below.
    expect(labels).toContain("NOTES");
    expect(labels).toContain("1. MATERIAL: GRADE 5 EQUIV; 85KPSI YIELD MIN.");
    expect(labels).toContain(
      "4. FINISH: ZINC YELLOW PER ASTM B633-LATEST REV. TYPE II"
    );
  });

  it("exports stable CSV headers and optional blank fields", () => {
    const csv = exportInspectionCSV(annotations, ["part1"]);
    const lines = csv.split("\n");

    expect(lines[0]).toBe(
      '"S.no","Label","Value","Tolerance","part1","Method","Tool"'
    );
    // Header + one line per value, before the title-block section.
    expect(lines.slice(0, valueRowCount + 1)).toHaveLength(valueRowCount + 1);
    // The Label cell carries the category label, not a blank.
    expect(lines[1]).toBe('"1","Linear Dimension","25","0","","",""');
  });

  it("exports one escaped XML row per value", () => {
    const xml = exportXML(annotations, ["part1"]);

    // XML carries the same table as CSV and JSON: one <row> per sheet row.
    expect(xml.match(/<row>/g)?.length).toBeGreaterThanOrEqual(valueRowCount);
    expect(xml).toContain("Diameter &amp; finish");
    expect(xml).toContain('<cell name="Value">25</cell>');
    expect(xml).toContain("<columns>");
  });
});
