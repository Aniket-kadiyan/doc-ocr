import { renumberValueAnnotations } from "@/lib/annotationNumbers";
import type { Annotation } from "@/types/annotation";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
  type DocumentMetadataField,
} from "@/types/documentMetadata";

const normalizeLabel = (label: string) =>
  label.toUpperCase().replace(/[^A-Z0-9]+/g, " ").trim();

const FIELD_ALIASES: Record<DocumentMetadataField, ReadonlySet<string>> = {
  partName: new Set(["PART", "PART NAME", "PARTNAME", "NAME OF PART"]),
  documentNumber: new Set([
    "DOC",
    "DOC NO",
    "DOC NUMBER",
    "DOCUMENT NO",
    "DOCUMENT NUMBER",
    "DWG",
    "DWG NO",
    "DWG NUMBER",
    "DRAWING",
    "DRAWING NO",
    "DRAWING NUMBER",
  ]),
  revisionNumber: new Set([
    "REV",
    "REV NO",
    "REV NUMBER",
    "REVISION",
    "REVISION NO",
    "REVISION NUMBER",
  ]),
};

/** Map common title-block labels onto the three persisted metadata fields. */
export function metadataFieldForTitleLabel(
  label: string
): DocumentMetadataField | null {
  const normalized = normalizeLabel(label);
  for (const [field, aliases] of Object.entries(FIELD_ALIASES) as Array<
    [DocumentMetadataField, ReadonlySet<string>]
  >) {
    if (aliases.has(normalized)) return field;
  }
  return null;
}

/** True for legacy auto-label rows now represented by document metadata. */
export function isMetadataTitleAnnotation(annotation: Annotation): boolean {
  return (
    annotation.type === "Title Block" &&
    metadataFieldForTitleLabel(annotation.label ?? "") !== null
  );
}

export interface TitleMetadataMigration {
  metadata: DocumentMetadata;
  annotations: Annotation[];
  migratedCount: number;
}

/**
 * Move recognized Part/Drawing/Revision title annotations into metadata.
 * Existing manual metadata wins; a title read only fills a blank field.
 */
export function mergeTitleAnnotationsIntoMetadata(
  annotations: Annotation[],
  metadata?: Partial<DocumentMetadata> | null
): TitleMetadataMigration {
  const merged = normalizeDocumentMetadata(metadata);
  let migratedCount = 0;
  const retained: Annotation[] = [];

  for (const annotation of annotations) {
    const field =
      annotation.type === "Title Block"
        ? metadataFieldForTitleLabel(annotation.label ?? "")
        : null;
    if (!field) {
      retained.push(annotation);
      continue;
    }
    if (!merged[field].trim() && annotation.value.trim()) {
      merged[field] = annotation.value.trim();
    }
    migratedCount += 1;
  }

  return {
    metadata: merged,
    annotations: renumberValueAnnotations(retained),
    migratedCount,
  };
}
