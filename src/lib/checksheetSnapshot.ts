import type { ProjectRecord } from "@/lib/db";
import { PDF_RENDER_SCALE } from "@/lib/pdfLoader";
import { balloonLabel } from "@/lib/featureLabel";
import { toleranceForExport, valueAnnotations } from "@/lib/project";
import { isMetadataTitleAnnotation } from "@/lib/titleMetadata";
import type { Annotation } from "@/types/annotation";
import type { ChecksheetSnapshotPayload } from "@/types/checksheet";
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
  const values = valueAnnotations(annotations).filter(
    (annotation) => !isMetadataTitleAnnotation(annotation)
  );
  if (values.length === 0) {
    throw new Error("Add at least one ballooned value before creating a checksheet.");
  }
  const mimeType =
    project.mimeType ||
    project.fileBlob.type ||
    (project.fileType === "pdf" ? "application/pdf" : "application/octet-stream");
  return {
    payload: {
      name,
      drawing_name: projectName.trim() || project.name,
      source_project_id: projectId,
      source_file_type: project.fileType,
      pdf_render_scale: PDF_RENDER_SCALE,
      metadata: normalizeDocumentMetadata(metadata),
      reading_columns: columns,
      items: values.map((annotation) => ({
        annotation_id: annotation.id,
        balloon_number: annotation.number,
        page: annotation.page,
        bbox: { ...annotation.bbox },
        oriented_box: annotation.orientedBox
          ? { ...annotation.orientedBox }
          : null,
        rotation: annotation.rotation,
        label: balloonLabel(annotation),
        specification: annotation.value,
        tolerance: toleranceForExport(annotation.range, annotation.value),
        method: annotation.method ?? "",
        tool: annotation.tool ?? "",
        dimension_type: annotation.type,
      })),
    },
    document: new File([project.fileBlob], project.fileName, {
      type: mimeType,
    }),
  };
}
