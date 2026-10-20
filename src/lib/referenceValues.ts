import type { BBox } from "@/types/annotation";
import type {
  PageReferenceTable,
  ReferenceCoordinate,
  ReferenceMarker,
} from "@/types/referencePoint";

export const REFERENCE_AXES = ["X", "Y", "Z"] as const;
export type ReferenceAxis = (typeof REFERENCE_AXES)[number];

/** Export-neutral representation of one coordinate value. */
export interface ReferenceValueRow {
  id: string;
  page: number;
  pointIndex: number;
  symbol: string;
  axis: ReferenceAxis;
  label: string;
  value: string;
  nameBox: BBox;
  rowBox: BBox;
  coordinates: ReferenceCoordinate[];
  markers: ReferenceMarker[];
}

/** Keep at most one table per page. The latest duplicate page result wins. */
export function normalizePageReferenceTables(
  tables: readonly PageReferenceTable[]
): PageReferenceTable[] {
  const byPage = new Map<number, PageReferenceTable>();
  for (const table of tables) {
    if (
      !table ||
      !Number.isInteger(table.page) ||
      table.page < 1 ||
      !Array.isArray(table.points)
    ) {
      continue;
    }
    byPage.set(table.page, table);
  }
  return [...byPage.values()].sort((left, right) => left.page - right.page);
}

/**
 * Convert every point into exactly three rows: X, Y, then Z. Missing axes are
 * retained with blank values so no detected point is silently incomplete.
 */
export function buildReferenceValueRows(
  tables: readonly PageReferenceTable[]
): ReferenceValueRow[] {
  return normalizePageReferenceTables(tables).flatMap((table) =>
    table.points.flatMap((point, pointIndex) =>
      REFERENCE_AXES.map((axis) => {
        const coordinate = point.coordinates.find(
          (candidate) => candidate.axis.trim().toUpperCase() === axis
        );
        const symbol = point.symbol.trim();
        return {
          id: `reference:p${table.page}:r${pointIndex}:${axis.toLowerCase()}`,
          page: table.page,
          pointIndex,
          symbol,
          axis,
          label: `${symbol}_${axis}`,
          value: coordinate?.value.trim() ?? "",
          nameBox: point.nameBox,
          rowBox: point.rowBox,
          coordinates: point.coordinates,
          markers: point.markers,
        };
      })
    )
  );
}
