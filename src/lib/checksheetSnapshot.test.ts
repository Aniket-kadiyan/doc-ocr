import { describe, expect, it } from "vitest";
import { buildChecksheetCreationSnapshot } from "@/lib/checksheetSnapshot";
import type { ProjectRecord } from "@/lib/db";
import { makeAnnotation } from "@/test/annotationFixture";

const project: ProjectRecord = {
  id: "project-1",
  name: "bracket.png",
  fileName: "bracket.png",
  fileType: "image",
  fileBlob: new Blob(["drawing"], { type: "image/png" }),
  mimeType: "image/png",
  updatedAt: 1,
};

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
});
