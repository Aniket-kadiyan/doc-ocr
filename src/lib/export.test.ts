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

  it("exports JSON with exactly one row per value", () => {
    const result = JSON.parse(exportInspectionJSON(annotations, ["part1"]));

    expect(result.extra_columns).toEqual(["part1"]);
    expect(result.data).toHaveLength(2);
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

  it("exports stable CSV headers and optional blank fields", () => {
    const csv = exportInspectionCSV(annotations, ["part1"]);

    expect(csv.split("\n")[0]).toBe(
      '"S.no","Label","Value","Tolerance","part1","Method","Tool"'
    );
    expect(csv.split("\n")).toHaveLength(3);
  });

  it("exports one escaped XML annotation per value", () => {
    const xml = exportXML(annotations);

    expect(xml.match(/<annotation /g)).toHaveLength(2);
    expect(xml).toContain("Diameter &amp; finish");
    expect(xml).not.toContain('<annotation id="label');
  });
});
