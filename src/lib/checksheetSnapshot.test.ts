import { describe, expect, it } from "vitest";
import { buildChecksheetCreationSnapshot } from "@/lib/checksheetSnapshot";
import type { ProjectRecord } from "@/lib/db";
import { makeAnnotation } from "@/test/annotationFixture";
import type {
  EngineeringDisposition,
  EngineeringValueParse,
} from "@/types/annotation";
import type { PageReferenceTable } from "@/types/referencePoint";
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

const engineeringParse: EngineeringValueParse = {
  schemaVersion: 1,
  rawText: "25+0.1",
  normalizedText: "25+0.1",
  status: "complete",
  kind: "linear",
  complete: true,
  components: { nominal: "25" },
  tokens: [],
  unparsedFragments: [],
  warnings: ["single_bound_tolerance"],
  normalizationSteps: [],
};

const engineeringDisposition: EngineeringDisposition = {
  schemaVersion: 1,
  state: "other",
  rule: "revision_history",
  reason: "Revision history content",
  parseStatus: "complete",
  parseKind: "identifier",
  hardContext: true,
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
  assembly: {
    objectId: "C0012",
    assemblyId: "C0012",
    rule: "inline_suffix",
    conflict: false,
    children: [
      {
        candidateId: "C0012",
        text: "REV",
        rawText: "REV",
        bbox: { x: 90, y: 120, width: 20, height: 10 },
        confidence: 0.72,
        orientation: "horizontal",
        rotation: 0,
        role: "other",
      },
    ],
  },
  engineeringParse,
  engineeringDisposition,
  valueBox: { x: 90, y: 120, width: 40, height: 10 },
  duplicateCount: 1,
  duplicateSourceIds: ["C0012", "C0013"],
  createdAt: 100,
  updatedAt: 200,
};

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
      assembly: otherCandidate.assembly,
      engineering_parse: engineeringParse,
      engineering_disposition: engineeringDisposition,
    });
    expect(JSON.stringify(creation.payload.items)).not.toContain("candidate-1");
  });

  it("preserves parse evidence on accepted checksheet rows", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "Parsed values",
      readingColumns: ["Part 1"],
      annotations: [makeAnnotation({ engineeringParse })],
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
    });

    expect(creation.payload.items[0].engineering_parse).toEqual(
      engineeringParse
    );
  });

  it("preserves disposition evidence on accepted checksheet rows", () => {
    const acceptedDisposition: EngineeringDisposition = {
      ...engineeringDisposition,
      state: "eligible",
      rule: "complete_engineering_object",
      reason: "Complete assembled engineering value",
      parseKind: "linear",
      hardContext: false,
    };
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "Disposed values",
      readingColumns: ["Part 1"],
      annotations: [
        makeAnnotation({ engineeringDisposition: acceptedDisposition }),
      ],
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
    });

    expect(creation.payload.items[0].engineering_disposition).toEqual(
      acceptedDisposition
    );
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

  it("appends X, Y, and Z reference rows with their review geometry", () => {
    const creation = buildChecksheetCreationSnapshot({
      checksheetName: "References",
      readingColumns: ["Part 1"],
      annotations: [makeAnnotation({ number: 1 })],
      referenceTables: [referenceTable],
      projectId: project.id,
      projectName: "Bracket drawing",
      project,
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
    });

    expect(creation.payload.items.map((item) => item.balloon_number)).toEqual([
      1, 2, 3, 4,
    ]);
    expect(creation.payload.items.map((item) => item.item_kind)).toEqual([
      "balloon",
      "reference",
      "reference",
      "reference",
    ]);
    expect(
      creation.payload.items.map(({ label, specification }) => [
        label,
        specification,
      ])
    ).toEqual([
      [creation.payload.items[0].label, creation.payload.items[0].specification],
      ["A_X", "10.25"],
      ["A_Y", "-4.50"],
      ["A_Z", ""],
    ]);
    expect(creation.payload.items[1]).toMatchObject({
      annotation_id: "reference:p1:r0:x",
      bbox: referenceTable.points[0].markers[0].bbox,
      tolerance: "",
      method: "",
      tool: "",
      dimension_type: "Reference",
      reference: {
        symbol: "A",
        axis: "X",
        point_index: 0,
        name_box: referenceTable.points[0].nameBox,
        row_box: referenceTable.points[0].rowBox,
        coordinates: referenceTable.points[0].coordinates,
        markers: referenceTable.points[0].markers,
      },
    });
  });
});
