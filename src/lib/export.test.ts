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
      Label: "",
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

  it("exports stable CSV headers and optional blank fields", () => {
    const csv = exportInspectionCSV(annotations, ["part1"]);
    const lines = csv.split("\n");

    expect(lines[0]).toBe(
      '"S.no","Label","Value","Tolerance","part1","Method","Tool"'
    );
    // Header + one line per value, before the title-block section.
    expect(lines.slice(0, valueRowCount + 1)).toHaveLength(valueRowCount + 1);
    expect(lines[1]).toBe('"1","","25","0","","",""');
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
