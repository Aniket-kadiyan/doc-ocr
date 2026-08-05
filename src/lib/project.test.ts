import { describe, expect, it } from "vitest";
import {
  buildInspectionSheet,
  buildProjectBundle,
  buildVerificationPayload,
  normalizeLegacyAnnotations,
  parseProjectBundle,
} from "@/lib/project";
import { makeAnnotation } from "@/test/annotationFixture";

describe("legacy annotation migration", () => {
  it("merges a mapped label into its value and reports orphan labels", () => {
    const mappedLabel = makeAnnotation({
      id: "label-mapped",
      number: 1,
      kind: "label",
      value: "Shaft diameter",
      method: "Inherited method",
      tool: "Micrometer",
    });
    const orphanLabel = makeAnnotation({
      id: "label-orphan",
      number: 2,
      kind: "label",
      value: "Unused label",
    });
    const value = makeAnnotation({
      id: "value",
      number: 7,
      label: "Stale child label",
      labelId: mappedLabel.id,
      labelSource: "ocr",
      method: "Value method",
      tool: undefined,
    });

    const migration = normalizeLegacyAnnotations([
      mappedLabel,
      orphanLabel,
      value,
    ]);

    expect(migration.orphanLabelCount).toBe(1);
    expect(migration.annotations).toHaveLength(1);
    expect(migration.annotations[0]).toMatchObject({
      id: "value",
      number: 1,
      kind: "dimension",
      label: "Shaft diameter",
      method: "Value method",
      tool: "Micrometer",
    });
    expect(migration.annotations[0]).not.toHaveProperty("labelId");
    expect(migration.annotations[0]).not.toHaveProperty("labelSource");
  });
});

describe("project and integration exports", () => {
  const values = [
    makeAnnotation({ id: "a", number: 1, value: "25", range: "0" }),
    makeAnnotation({
      id: "b",
      number: 2,
      value: "58.21±0.05",
      range: "+0.05, -0.05",
      label: "Outer diameter",
      method: "Measure",
      tool: "Micrometer",
      extras: { part1: "58.2" },
    }),
  ];

  it("builds one row per value with optional metadata and extra columns", () => {
    const sheet = buildInspectionSheet(values, ["part1", "part2"]);

    expect(sheet.headers).toEqual([
      "S.no",
      "Label",
      "Value",
      "Tolerance",
      "part1",
      "part2",
      "Method",
      "Tool",
    ]);
    expect(sheet.rows).toEqual([
      ["1", "", "25", "0", "", "", "", ""],
      [
        "2",
        "Outer diameter",
        "58.21±0.05",
        "+0.05, -0.05",
        "58.2",
        "",
        "Measure",
        "Micrometer",
      ],
    ]);
  });

  it("saves a values-only project without embedding balloon artwork", () => {
    const serialized = buildProjectBundle({
      projectName: "Test drawing",
      source: {
        fileName: "drawing.png",
        mimeType: "image/png",
        fileType: "image",
        dataUrl: "data:image/png;base64,AA==",
      },
      annotations: values,
      savedAt: 123,
    });
    const bundle = parseProjectBundle(serialized);

    expect(bundle.annotations).toHaveLength(2);
    expect(serialized).not.toContain("balloon-style");
    expect(serialized).not.toContain("artwork");
  });

  it("builds a verification payload with no separate label items", () => {
    const payload = buildVerificationPayload(
      values,
      "Test drawing",
      "https://example.invalid/verify",
      ["part1"]
    );

    expect(payload.items).toHaveLength(2);
    expect(payload.items.every(({ kind }) => kind === "dimension")).toBe(true);
    expect(payload.items[1]).toMatchObject({
      number: 2,
      label: "Outer diameter",
      value: "58.21±0.05",
      range: "+0.05, -0.05",
    });
    expect(payload.sheet?.extraColumns).toEqual(["part1"]);
  });
});
