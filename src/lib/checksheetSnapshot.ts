import type { ProjectRecord } from "@/lib/db";
import { PDF_RENDER_SCALE } from "@/lib/pdfLoader";
import { valueAnnotations } from "@/lib/project";
import type { Annotation } from "@/types/annotation";
import type { ChecksheetSnapshotPayload } from "@/types/checksheet";

interface BuildChecksheetSnapshotArgs {
  checksheetName: string;
  readingColumns: string[];
  annotations: Annotation[];
  projectId: string;
  projectName: string;
  project: ProjectRecord;
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
      reading_columns: columns,
      items: values.map((annotation) => ({
        annotation_id: annotation.id,
        balloon_number: annotation.number,
        page: annotation.page,
        bbox: { ...annotation.bbox },
        rotation: annotation.rotation,
        label: annotation.label ?? "",
        specification: annotation.value,
        tolerance: annotation.range ?? "",
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
