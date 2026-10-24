import { describe, expect, it } from "vitest";
import {
  exportInspectionCSV,
  exportInspectionJSON,
  exportXML,
  findMalformedToleranceAnnotations,
} from "@/lib/export";
import { makeAnnotation } from "@/test/annotationFixture";
import type { PageReferenceTable } from "@/types/referencePoint";

const referenceBox = { x: 10, y: 20, width: 30, height: 12 };
const referenceTables: PageReferenceTable[] = [
  {
    page: 1,
    bbox: referenceBox,
    headerBox: referenceBox,
    axes: ["X", "Y", "Z"],
    nameHeader: "POINT",
    points: [
      {
        symbol: "A",
        coordinates: [
          { axis: "X", value: "1" },
          { axis: "Z", value: "3" },
        ],
        nameBox: referenceBox,
        rowBox: referenceBox,
        markers: [],
      },
    ],
  },
];

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

  it("accepts a stacked deviation pair carried only in Value", () => {
    const annotations = [
      makeAnnotation({ id: "zero-lower", value: "Ø8+0.2/0", range: undefined }),
      makeAnnotation({ id: "unsigned-lower", value: "Ø8+0.2/0.2", range: undefined }),
      makeAnnotation({ id: "signed-lower", value: "21+1/-0.5", range: undefined }),
      makeAnnotation({ id: "both-below", value: "Ø8-0.1/-0.2", range: undefined }),
    ];

    expect(
      findMalformedToleranceAnnotations(annotations).map(({ id }) => id)
    ).toEqual(["both-below"]);
  });
});

describe("inspection exports", () => {
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
  const metadata = {
    partName: "Drive Bracket",
    documentNumber: "DB-1042 & ALT",
    revisionNumber: "C",
  };

  it("exports JSON with exactly one row per value", () => {
    const result = JSON.parse(
      exportInspectionJSON(annotations, ["part1"], metadata)
    );

    expect(result.metadata).toEqual(metadata);
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
    const csv = exportInspectionCSV(annotations, ["part1"], metadata);
    const lines = csv.split("\n");

    expect(lines.slice(0, 4)).toEqual([
      '"Metadata Field","Value"',
      '"Part Name","Drive Bracket"',
      '"Document Number","DB-1042 & ALT"',
      '"Revision Number","C"',
    ]);
    expect(lines[5]).toBe(
      '"S.no","Label","Value","Tolerance","part1","Method","Tool"'
    );
    // Header + one line per value, before the title-block section.
    expect(lines.slice(5, 5 + valueRowCount + 1)).toHaveLength(
      valueRowCount + 1
    );
    expect(lines[6]).toBe('"1","","25","0","","",""');
  });

  it("exports one escaped XML row per value", () => {
    const xml = exportXML(annotations, ["part1"], metadata);

    // XML carries the same table as CSV and JSON: one <row> per sheet row.
    expect(xml.match(/<row>/g)?.length).toBeGreaterThanOrEqual(valueRowCount);
    expect(xml).toContain("Diameter &amp; finish");
    expect(xml).toContain('<cell name="Value">25</cell>');
    expect(xml).toContain("<columns>");
    expect(xml).toContain("<partName>Drive Bracket</partName>");
    expect(xml).toContain("<documentNumber>DB-1042 &amp; ALT</documentNumber>");
  });

  it("adds the missing zero bound to exported table formats", () => {
    const oneSided = [
      makeAnnotation({ id: "upper", number: 1, value: "25", range: "+0.1" }),
      makeAnnotation({ id: "lower", number: 2, value: "30", range: "-.5" }),
      makeAnnotation({ id: "embedded", number: 3, value: "R0.3+0.05", range: undefined }),
    ];

    const json = JSON.parse(exportInspectionJSON(oneSided));
    expect(json.data.slice(0, 3).map((row: { Tolerance: string }) => row.Tolerance))
      .toEqual(["+0.1, -0", "+0, -.5", "+0.05, -0"]);

    const xml = exportXML(oneSided);
    expect(xml).toContain('<cell name="Tolerance">+0.1, -0</cell>');
    expect(xml).toContain('<cell name="Tolerance">+0, -.5</cell>');
    expect(xml).toContain('<cell name="Tolerance">+0.05, -0</cell>');
  });

  it("includes X/Y/Z reference rows after balloons in JSON, CSV and XML", () => {
    const json = JSON.parse(
      exportInspectionJSON(annotations, [], undefined, referenceTables)
    );
    expect(json.data.slice(2, 5).map((row: Record<string, string>) => [
      row["S.no"],
      row.Label,
      row.Value,
    ])).toEqual([
      ["3", "A_X", "1"],
      ["4", "A_Y", ""],
      ["5", "A_Z", "3"],
    ]);

    const csv = exportInspectionCSV(annotations, [], undefined, referenceTables);
    expect(csv).toContain('"3","A_X","1","","",""');
    expect(csv).toContain('"4","A_Y","","","",""');

    const xml = exportXML(annotations, [], undefined, referenceTables);
    expect(xml).toContain('<cell name="Label">A_Z</cell>');
    expect(xml).toContain('<cell name="Value">3</cell>');
  });
});
