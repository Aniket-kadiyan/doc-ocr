import { beforeEach, describe, expect, it } from "vitest";
import { useDocumentMetadataStore } from "@/store/documentMetadataStore";

describe("documentMetadataStore", () => {
  beforeEach(() => {
    useDocumentMetadataStore.getState().resetMetadata();
  });

  it("allows all metadata fields to remain blank", () => {
    expect(useDocumentMetadataStore.getState().metadata).toEqual({
      partName: "",
      documentNumber: "",
      revisionNumber: "",
    });
  });

  it("updates one field without changing the other fields", () => {
    useDocumentMetadataStore.getState().setMetadata({
      partName: "Bracket",
      documentNumber: "DOC-17",
      revisionNumber: "B",
    });

    useDocumentMetadataStore
      .getState()
      .updateField("revisionNumber", "C");

    expect(useDocumentMetadataStore.getState().metadata).toEqual({
      partName: "Bracket",
      documentNumber: "DOC-17",
      revisionNumber: "C",
    });
  });

  it("normalizes missing fields when loading old data", () => {
    useDocumentMetadataStore
      .getState()
      .setMetadata({ documentNumber: "DOC-17" });

    expect(useDocumentMetadataStore.getState().metadata).toEqual({
      partName: "",
      documentNumber: "DOC-17",
      revisionNumber: "",
    });
  });
});
