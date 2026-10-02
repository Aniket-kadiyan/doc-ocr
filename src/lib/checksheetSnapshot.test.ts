import { describe, expect, it } from "vitest";
import { buildChecksheetCreationSnapshot } from "@/lib/checksheetSnapshot";
import type { ProjectRecord } from "@/lib/db";
import { makeAnnotation } from "@/test/annotationFixture";
import type { ScanCandidate } from "@/types/scanCandidate";

const project: ProjectRecord = {
  id: "project-1",
  name: "bracket.png",
  fileName: "bracket.png",
  fileType: "image",
  fileBlob: new Blob(["drawing"], { type: "image/png" }),
  mimeType: "image/png",
  updatedAt: 1,
};

const otherCandidate: ScanCandidate = {
  id: "candidate-1",
  sourceCandidateId: "C0012",
  page: 1,
  order: 2,
  state: "other",
  text: "REV B",
  rawText: "REV B",
  preliminaryText: "REV 8",
  confidence: 0.72,
  recognized: true,
  reason: "Title-block text is not a dimension",
  rule: "title_block",
  orientation: "horizontal",
  rotation: 0,
  valueBox: { x: 90, y: 120, width: 40, height: 10 },
  duplicateCount: 1,
  duplicateSourceIds: ["C0012", "C0013"],
  createdAt: 100,
  updatedAt: 200,
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

  it("sends candidates separately from accepted checksheet rows", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "Candidate snapshot",
      readingColumns: ["Part 1"],
      annotations: [makeAnnotation()],
      scanCandidates: [otherCandidate],
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
    });

    expect(creation.payload.items).toHaveLength(1);
    expect(creation.payload.scan_candidates).toHaveLength(1);
    expect(creation.payload.scan_candidates[0]).toMatchObject({
      candidate_id: "candidate-1",
      source_candidate_id: "C0012",
      state: "other",
      raw_text: "REV B",
      bbox: otherCandidate.valueBox,
      duplicate_count: 1,
    });
    expect(JSON.stringify(creation.payload.items)).not.toContain("candidate-1");
  });

  it("keeps oriented geometry and completes one-sided tolerance bounds", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "Rotated inspection",
      readingColumns: ["Part 1"],
      annotations: [
        makeAnnotation({
          range: "+0.1",
          orientedBox: {
            x: 12,
            y: 24,
            width: 40,
            height: 10,
            rotation: 32,
          },
        }),
      ],
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
    });

    expect(creation.payload.items[0].oriented_box).toEqual({
      x: 12,
      y: 24,
      width: 40,
      height: 10,
      rotation: 32,
    });
    expect(creation.payload.items[0].tolerance).toBe("+0.1, -0");
  });
});
