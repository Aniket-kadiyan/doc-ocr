import { describe, expect, it } from "vitest";
import {
  mergeTitleAnnotationsIntoMetadata,
  metadataFieldForTitleLabel,
} from "@/lib/titleMetadata";
import { makeAnnotation } from "@/test/annotationFixture";

describe("title-block metadata migration", () => {
  it("maps common part, drawing, and revision aliases", () => {
    expect(metadataFieldForTitleLabel("Part Name")).toBe("partName");
    expect(metadataFieldForTitleLabel("DWG NO.")).toBe("documentNumber");
    expect(metadataFieldForTitleLabel("Revision No.")).toBe("revisionNumber");
    expect(metadataFieldForTitleLabel("DATE")).toBeNull();
  });

  it("fills blank metadata, preserves manual values, and removes duplicates", () => {
    const migration = mergeTitleAnnotationsIntoMetadata(
      [
        makeAnnotation({
          id: "part",
          number: 1,
          type: "Title Block",
          label: "PART NAME",
          value: "Drive Bracket",
        }),
        makeAnnotation({
          id: "drawing",
          number: 2,
          type: "Title Block",
          label: "DWG",
          value: "OCR-100",
        }),
        makeAnnotation({ id: "dimension", number: 3, value: "25" }),
      ],
      { documentNumber: "MANUAL-200" }
    );

    expect(migration.metadata).toEqual({
      partName: "Drive Bracket",
      documentNumber: "MANUAL-200",
      revisionNumber: "",
    });
    expect(migration.annotations.map((annotation) => annotation.id)).toEqual([
      "dimension",
    ]);
    expect(migration.annotations[0].number).toBe(1);
  });
});
