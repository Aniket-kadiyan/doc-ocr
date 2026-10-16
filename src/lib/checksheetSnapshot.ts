import type { ProjectRecord } from "@/lib/db";
import { PDF_RENDER_SCALE } from "@/lib/pdfLoader";
import { balloonLabel } from "@/lib/featureLabel";
import { toleranceForExport, valueAnnotations } from "@/lib/project";
import { buildReferenceValueRows } from "@/lib/referenceValues";
import { isMetadataTitleAnnotation } from "@/lib/titleMetadata";
import type { Annotation } from "@/types/annotation";
import type { ScanCandidate } from "@/types/scanCandidate";
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
  scanCandidates?: ScanCandidate[];
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
  scanCandidates = [],
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
  const values = valueAnnotations(annotations).filter(
    (annotation) => !isMetadataTitleAnnotation(annotation)
  );
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
    assembly: annotation.assembly ?? null,
    engineering_symbol: annotation.engineeringSymbol ?? null,
    engineering_parse: annotation.engineeringParse ?? null,
    engineering_disposition: annotation.engineeringDisposition ?? null,
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
      bbox: {
        ...(reference.markers[0]?.bbox ?? reference.nameBox),
      },
      oriented_box: null,
      rotation: 0,
      label: reference.label,
      specification: reference.value,
      tolerance: "",
      method: "",
      tool: "",
      dimension_type: "Reference",
      assembly: null,
      engineering_symbol: null,
      engineering_parse: null,
      engineering_disposition: null,
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
      scan_candidates: scanCandidates.map((candidate) => ({
        candidate_id: candidate.id,
        source_candidate_id: candidate.sourceCandidateId,
        page: candidate.page,
        order: candidate.order,
        state: candidate.state,
        restore_state: candidate.restoreState,
        text: candidate.text,
        raw_text: candidate.rawText,
        preliminary_text: candidate.preliminaryText,
        confidence: candidate.confidence,
        recognized: candidate.recognized,
        reason: candidate.reason,
        rule: candidate.rule,
        type: candidate.type,
        category: candidate.category,
        subtype: candidate.subtype,
        label: candidate.label,
        orientation: candidate.orientation,
        rotation: candidate.rotation,
        recovery_attempted: candidate.recoveryAttempted ?? false,
        authoritative_reread: candidate.authoritativeReread ?? false,
        assembly: candidate.assembly ?? null,
        engineering_symbol: candidate.engineeringSymbol ?? null,
        engineering_parse: candidate.engineeringParse ?? null,
        engineering_disposition: candidate.engineeringDisposition ?? null,
        bbox: { ...candidate.valueBox },
        oriented_box: candidate.orientedBox
          ? { ...candidate.orientedBox }
          : null,
        duplicate_source_ids: candidate.duplicateSourceIds ?? [],
        duplicate_count: candidate.duplicateCount ?? 0,
        created_at: candidate.createdAt,
        updated_at: candidate.updatedAt,
      })),
      reading_columns: columns,
      items: [...balloonItems, ...referenceItems],
    },
    document: new File([project.fileBlob], project.fileName, {
      type: mimeType,
    }),
  };
}
