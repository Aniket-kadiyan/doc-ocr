import { describe, expect, it } from "vitest";
import { buildChecksheetCreationSnapshot } from "@/lib/checksheetSnapshot";
import type { ProjectRecord } from "@/lib/db";
import { makeAnnotation } from "@/test/annotationFixture";
import type { PageReferenceTable } from "@/types/referencePoint";

const project: ProjectRecord = {
  id: "project-1",
  name: "bracket.png",
  fileName: "bracket.png",
  fileType: "image",
  fileBlob: new Blob(["drawing"], { type: "image/png" }),
  mimeType: "image/png",
  updatedAt: 1,
};

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

describe("checksheet creation snapshot", () => {
  it("includes the current document metadata", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "Bracket inspection",
      readingColumns: ["Part 1"],
      annotations: [makeAnnotation()],
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: {
        partName: "Drive Bracket",
        documentNumber: "DB-1042",
        revisionNumber: "C",
      },
    });

    expect(creation.payload.metadata).toEqual({
      partName: "Drive Bracket",
      documentNumber: "DB-1042",
      revisionNumber: "C",
    });
  });

  it("allows a checksheet snapshot with blank metadata", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "Blank metadata",
      readingColumns: ["Part 1"],
      annotations: [makeAnnotation()],
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: {
        partName: "",
        documentNumber: "",
        revisionNumber: "",
      },
    });

    expect(Object.values(creation.payload.metadata)).toEqual(["", "", ""]);
  });

  it("completes a one-sided tolerance in the exported snapshot", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "Tolerance export",
      readingColumns: ["Part 1"],
      annotations: [
        makeAnnotation({ value: "25", range: "+0.1" }),
        makeAnnotation({ id: "embedded", number: 2, value: "R0.3+0.05", range: undefined }),
      ],
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
    });

    expect(creation.payload.items.map((item) => item.tolerance)).toEqual([
      "+0.1, -0",
      "+0.05, -0",
    ]);
  });

  it("appends reference axes with their geometry and keeps blank axes", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "References",
      readingColumns: ["Part 1"],
      annotations: [makeAnnotation({ number: 1 })],
      referenceTables,
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
    });

    expect(creation.payload.items.slice(1).map((item) => ({
      number: item.balloon_number,
      kind: item.item_kind,
      label: item.label,
      value: item.specification,
      axis: item.reference?.axis,
    }))).toEqual([
      { number: 2, kind: "reference", label: "A_X", value: "1", axis: "X" },
      { number: 3, kind: "reference", label: "A_Y", value: "", axis: "Y" },
      { number: 4, kind: "reference", label: "A_Z", value: "3", axis: "Z" },
    ]);
  });

  it("allows a references-only checksheet", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "References only",
      readingColumns: ["Part 1"],
      annotations: [],
      referenceTables,
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
    });

    expect(creation.payload.items).toHaveLength(3);
    expect(creation.payload.items[0].balloon_number).toBe(1);
  });
});
