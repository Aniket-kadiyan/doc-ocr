import type { ProjectRecord } from "@/lib/db";
import { PDF_RENDER_SCALE } from "@/lib/pdfLoader";
import { toleranceForExport, valueAnnotations } from "@/lib/project";
import { buildReferenceValueRows } from "@/lib/referenceValues";
import type { Annotation } from "@/types/annotation";
import type {
  ChecksheetSnapshotItem,
  ChecksheetSnapshotPayload,
} from "@/types/checksheet";
import type { PageReferenceTable } from "@/types/referencePoint";
import {
  normalizeDocumentMetadata,
  type DocumentMetadata,
} from "@/types/documentMetadata";

interface BuildChecksheetSnapshotArgs {
  checksheetName: string;
  readingColumns: string[];
  annotations: Annotation[];
  projectId: string;
  projectName: string;
  project: ProjectRecord;
  metadata: DocumentMetadata;
  referenceTables?: readonly PageReferenceTable[];
}

export interface ChecksheetCreationSnapshot {
  payload: ChecksheetSnapshotPayload;
  document: File;
}

/** Build the immutable specification/geometry snapshot sent to the backend. */
export function buildChecksheetCreationSnapshot({
  checksheetName,
  readingColumns,
  annotations,
  projectId,
  projectName,
  project,
  metadata,
  referenceTables = [],
}: BuildChecksheetSnapshotArgs): ChecksheetCreationSnapshot {
  if (!project.fileBlob) {
    throw new Error(
      "The original drawing is unavailable. Reopen the drawing before creating a checksheet."
    );
  }
  const name = checksheetName.trim();
  if (!name) throw new Error("Enter a checksheet name.");
  const columns = readingColumns.map((column) => column.trim());
  if (columns.length === 0 || columns.some((column) => !column)) {
    throw new Error("Add at least one measured-part column.");
  }
  const folded = columns.map((column) => column.toLocaleLowerCase());
  if (new Set(folded).size !== folded.length) {
    throw new Error("Measured-part column names must be unique.");
  }
  const values = valueAnnotations(annotations);
  const references = buildReferenceValueRows(referenceTables);
  if (values.length === 0 && references.length === 0) {
    throw new Error(
      "Add at least one ballooned or reference value before creating a checksheet."
    );
  }
  const nextNumber =
    values.reduce(
      (largest, annotation) => Math.max(largest, annotation.number),
      0
    ) + 1;
  const mimeType =
    project.mimeType ||
    project.fileBlob.type ||
    (project.fileType === "pdf" ? "application/pdf" : "application/octet-stream");
  const balloonItems: ChecksheetSnapshotItem[] = values.map((annotation) => ({
    annotation_id: annotation.id,
    balloon_number: annotation.number,
    item_kind: "balloon",
    reference: null,
    page: annotation.page,
    bbox: { ...annotation.bbox },
    rotation: annotation.rotation,
    label: annotation.label ?? "",
    specification: annotation.value,
    tolerance: toleranceForExport(annotation.range, annotation.value),
    method: annotation.method ?? "",
    tool: annotation.tool ?? "",
    dimension_type: annotation.type,
  }));
  const referenceItems: ChecksheetSnapshotItem[] = references.map(
    (reference, index) => ({
      annotation_id: reference.id,
      balloon_number: nextNumber + index,
      item_kind: "reference",
      reference: {
        symbol: reference.symbol,
        axis: reference.axis,
        point_index: reference.pointIndex,
        name_box: { ...reference.nameBox },
        row_box: { ...reference.rowBox },
        coordinates: reference.coordinates.map((coordinate) => ({
          ...coordinate,
        })),
        markers: reference.markers.map((marker) => ({
          ...marker,
          bbox: { ...marker.bbox },
        })),
      },
      page: reference.page,
      bbox: { ...(reference.markers[0]?.bbox ?? reference.nameBox) },
      rotation: 0,
      label: reference.label,
      specification: reference.value,
      tolerance: "",
      method: "",
      tool: "",
      dimension_type: "Reference",
    })
  );
  return {
    payload: {
      name,
      drawing_name: projectName.trim() || project.name,
      source_project_id: projectId,
      source_file_type: project.fileType,
      pdf_render_scale: PDF_RENDER_SCALE,
      metadata: normalizeDocumentMetadata(metadata),
      reading_columns: columns,
      items: [...balloonItems, ...referenceItems],
    },
    document: new File([project.fileBlob], project.fileName, {
      type: mimeType,
    }),
  };
}
