import { describe, expect, it } from "vitest";
import {
  exportInspectionCSV,
  exportInspectionJSON,
  exportXML,
  findMalformedToleranceAnnotations,
} from "@/lib/export";
import { makeAnnotation } from "@/test/annotationFixture";

describe("tolerance export validation", () => {
  it("accepts supported default, symmetric, asymmetric, and one-sided tolerances", () => {
    const annotations = [
      makeAnnotation({ id: "zero", value: "0.02", range: "0" }),
      makeAnnotation({
        id: "symmetric",
        value: "58.21±0.05",
        range: "+0.05, -0.05",
      }),
      makeAnnotation({ id: "asymmetric", value: "25", range: "+0.1, -0.2" }),
      makeAnnotation({ id: "upper-only", value: "25", range: "+0.1" }),
      makeAnnotation({ id: "lower-only", value: "25", range: "-.5" }),
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

  it("accepts embedded one-sided tolerances and fit designations", () => {
    const annotations = [
      makeAnnotation({ id: "radius-lower", value: "R0.6-0", range: undefined }),
      makeAnnotation({ id: "linear-upper", value: "0.1+0.1", range: undefined }),
      makeAnnotation({ id: "radius-upper", value: "R0.3+0.05", range: undefined }),
      makeAnnotation({ id: "linear-lower", value: "65-0.01", range: undefined }),
      makeAnnotation({ id: "fit", value: "2N9", range: "0" }),
    ];

    expect(findMalformedToleranceAnnotations(annotations)).toEqual([]);
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

  const valueRowCount = annotations.length;

  it("exports JSON with exactly one row per value", () => {
    const result = JSON.parse(exportInspectionJSON(annotations, ["part1"]));

    expect(result.metadata).toEqual({
      partName: "",
      documentNumber: "",
      revisionNumber: "",
    });
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

  it("does not duplicate known metadata as title-block rows", () => {
    const result = JSON.parse(exportInspectionJSON(annotations, ["part1"]));
    const labels = result.data
      .slice(valueRowCount)
      .map((row: Record<string, string>) => row.Label)
      .filter(Boolean);

    expect(labels).toEqual([]);
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

  it("exports metadata before stable CSV headers and optional blank fields", () => {
    const csv = exportInspectionCSV(annotations, ["part1"]);
    const lines = csv.split("\n");

    expect(lines[0]).toBe('"Metadata Field","Value"');
    expect(lines[1]).toBe('"Part Name",""');
    expect(lines[4]).toBe("");
    expect(lines[5]).toBe(
      '"S.no","Label","Value","Tolerance","part1","Method","Tool"'
    );
    // The Label cell carries the category label, not a blank.
    expect(lines[6]).toBe('"1","Linear Dimension","25","0","","",""');
  });

  it("exports one escaped XML row per value", () => {
    const xml = exportXML(annotations, ["part1"]);

    // XML carries the same table as CSV and JSON: one <row> per sheet row.
    expect(xml.match(/<row>/g)?.length).toBeGreaterThanOrEqual(valueRowCount);
    expect(xml).toContain("<metadata>");
    expect(xml).toContain("Diameter &amp; finish");
    expect(xml).toContain('<cell name="Value">25</cell>');
    expect(xml).toContain("<columns>");
  });

  it("includes metadata in JSON and XML exports", () => {
    const metadata = {
      partName: "Drive & Bracket",
      documentNumber: "DOC<1042>",
      revisionNumber: 'C"1',
    };

    const json = JSON.parse(
      exportInspectionJSON(annotations, [], metadata)
    );
    expect(json.metadata).toEqual(metadata);

    const csv = exportInspectionCSV(annotations, [], metadata);
    expect(csv).toContain('"Part Name","Drive & Bracket"');

    const xml = exportXML(annotations, [], metadata);
    expect(xml).toContain("<partName>Drive &amp; Bracket</partName>");
    expect(xml).toContain(
      "<documentNumber>DOC&lt;1042&gt;</documentNumber>"
    );
    expect(xml).toContain("<revisionNumber>C&quot;1</revisionNumber>");
  });

  it("adds the missing zero bound to exported JSON and XML", () => {
    const oneSided = [
      makeAnnotation({ id: "upper", number: 1, value: "25", range: "+0.1" }),
      makeAnnotation({ id: "lower", number: 2, value: "30", range: "-.5" }),
    ];

    const json = JSON.parse(exportInspectionJSON(oneSided));
    expect(json.data[0].Tolerance).toBe("+0.1, -0");
    expect(json.data[1].Tolerance).toBe("+0, -.5");

    const xml = exportXML(oneSided);
    expect(xml).toContain('<cell name="Tolerance">+0.1, -0</cell>');
    expect(xml).toContain('<cell name="Tolerance">+0, -.5</cell>');
  });

  it("derives export tolerances from embedded one-sided values", () => {
    const embedded = [
      makeAnnotation({ id: "radius-lower", number: 1, value: "R0.6-0", range: undefined }),
      makeAnnotation({ id: "linear-upper", number: 2, value: "0.1+0.1", range: undefined }),
      makeAnnotation({ id: "radius-upper", number: 3, value: "R0.3+0.05", range: undefined }),
      makeAnnotation({ id: "linear-lower", number: 4, value: "65-0.01", range: undefined }),
    ];

    const json = JSON.parse(exportInspectionJSON(embedded));
    expect(json.data.map((row: { Value: string }) => row.Value)).toEqual([
      "R0.6-0",
      "0.1+0.1",
      "R0.3+0.05",
      "65-0.01",
    ]);
    expect(json.data.map((row: { Tolerance: string }) => row.Tolerance)).toEqual([
      "+0, -0",
      "+0.1, -0",
      "+0.05, -0",
      "+0, -0.01",
    ]);
  });
});
