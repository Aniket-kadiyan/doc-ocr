import { REFERENCE_AXES } from "@/lib/referenceValues";
import type { BBox } from "@/types/annotation";
import type { ChecksheetRow } from "@/types/checksheet";
import type { ReferenceCoordinate } from "@/types/referencePoint";

export function isChecksheetReferenceRow(row: ChecksheetRow): boolean {
  return row.item_kind === "reference";
}

export function checksheetItemCaption(row: ChecksheetRow): string {
  if (!isChecksheetReferenceRow(row)) {
    return `Balloon ${row.balloon_number}`;
  }
  const fallback = row.reference
    ? `${row.reference.symbol}_${row.reference.axis}`
    : "value";
  return `Reference ${row.label.trim() || fallback}`;
}

export function checksheetSpecificationText(row: ChecksheetRow): string {
  if (isChecksheetReferenceRow(row) && !row.specification.trim()) {
    return "Not detected";
  }
  return row.specification;
}

/** Always expose X/Y/Z in the selected reference tag, including missed axes. */
export function referenceDisplayCoordinates(
  row: ChecksheetRow
): ReferenceCoordinate[] {
  if (!isChecksheetReferenceRow(row) || !row.reference) return [];

  return REFERENCE_AXES.map((axis) => {
    const coordinate = row.reference?.coordinates.find(
      (candidate) => candidate.axis.trim().toUpperCase() === axis
    );
    return {
      axis,
      value: coordinate?.value.trim() || "Not detected",
    };
  });
}

function unionBoxes(boxes: BBox[]): BBox {
  const left = Math.min(...boxes.map((box) => box.x));
  const top = Math.min(...boxes.map((box) => box.y));
  const right = Math.max(...boxes.map((box) => box.x + box.width));
  const bottom = Math.max(...boxes.map((box) => box.y + box.height));
  return { x: left, y: top, width: right - left, height: bottom - top };
}

/** Include both the source table row and every located drawing marker. */
export function checksheetFocusBox(row: ChecksheetRow): BBox {
  if (!isChecksheetReferenceRow(row) || !row.reference) return row.bbox;
  return unionBoxes([
    row.reference.row_box,
    row.reference.name_box,
    ...row.reference.markers.map((marker) => marker.bbox),
  ]);
}

