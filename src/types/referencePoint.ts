import type { BBox } from "@/types/annotation";

/** One X, Y, or Z value from a coordinate reference-table row. */
export interface ReferenceCoordinate {
  axis: string;
  value: string;
}

/** One place where a point name was located in a drawing view. */
export interface ReferenceMarker {
  bbox: BBox;
  /** Template-match score, from 0 to 1. */
  score: number;
  /** Lead over the next-best symbol identity. */
  margin: number;
}

/** One named row reconstructed from the drawing's coordinate table. */
export interface ReferencePoint {
  symbol: string;
  coordinates: ReferenceCoordinate[];
  /** The symbol's own table cell. */
  nameBox: BBox;
  /** The full row, used for selection highlighting. */
  rowBox: BBox;
  /** Empty when the row was read but its marker was not located in a view. */
  markers: ReferenceMarker[];
}

export interface ReferenceTable {
  bbox: BBox;
  headerBox: BBox;
  axes: string[];
  nameHeader: string;
  points: ReferencePoint[];
}

export interface ReferencePointsResult {
  found: boolean;
  /** Ruled tables found, including tables that were not coordinate tables. */
  tableCount: number;
  table: ReferenceTable | null;
  /** Rows read from the table whose names were not found in any drawing view. */
  unmatchedSymbols: string[];
}

/** Temporary reference data bound to the page from which it was read. */
export interface PageReferenceTable extends ReferenceTable {
  page: number;
}
