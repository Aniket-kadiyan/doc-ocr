export interface DocumentMetadata {
  partName: string;
  documentNumber: string;
  revisionNumber: string;
}

export type DocumentMetadataField = keyof DocumentMetadata;

export const EMPTY_DOCUMENT_METADATA: Readonly<DocumentMetadata> =
  Object.freeze({
    partName: "",
    documentNumber: "",
    revisionNumber: "",
  });

export const DOCUMENT_METADATA_FIELDS: ReadonlyArray<{
  key: DocumentMetadataField;
  label: string;
}> = [
  { key: "partName", label: "Part Name" },
  { key: "documentNumber", label: "Document No." },
  { key: "revisionNumber", label: "Revision No." },
];

/** Keep persisted/imported metadata predictable while accepting older projects. */
export function normalizeDocumentMetadata(
  metadata?: Partial<DocumentMetadata> | null
): DocumentMetadata {
  return {
    partName:
      typeof metadata?.partName === "string" ? metadata.partName : "",
    documentNumber:
      typeof metadata?.documentNumber === "string"
        ? metadata.documentNumber
        : "",
    revisionNumber:
      typeof metadata?.revisionNumber === "string"
        ? metadata.revisionNumber
        : "",
  };
}
