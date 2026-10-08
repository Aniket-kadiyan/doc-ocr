import type { BBox } from "@/types/annotation";

/**
 * A sheet's coordinate reference table and the places its point names are
 * printed in the views.
 *
 * A reference table names points — `a`, `b`, `c` … — and gives each one its
 * X/Y/Z coordinates. The views then print the bare name beside the feature it
 * belongs to, with nothing to say what the letter means. Pairing the two is
 * what this record is for.
 *
 * These are not ballooned values. They carry no tolerance, no inspection
 * method, and no balloon number, so they live beside {@link Annotation}
 * rather than becoming one.
 */

export interface ReferenceCoordinate {
  /** "X", "Y" or "Z". */
  axis: string;
  value: string;
}

/** One place in the drawing where a point name is printed. */
export interface ReferenceMarker {
  /** The printed glyph, in the drawing's own annotation coordinates. */
  bbox: BBox;
  /** Template-match score, 0–1. */
  score: number;
  /** How far that score beat the next-best reading. */
  margin: number;
}

export interface ReferencePoint {
  /** The name as the table prints it, e.g. "a". */
  symbol: string;
  coordinates: ReferenceCoordinate[];
  /** The point-name cell in the table. */
  nameBox: BBox;
  /** The whole table row, for highlighting it on the sheet. */
  rowBox: BBox;
  /** Empty when the name was never located in the views. */
  markers: ReferenceMarker[];
}

export interface ReferenceTable {
  bbox: BBox;
  headerBox: BBox;
  /** The axes the table actually has, in X/Y/Z order. */
  axes: string[];
  /** Whatever headed the name column — "POINT", "SYM", or "" for none. */
  nameHeader: string;
  points: ReferencePoint[];
}

export interface ReferencePointsResult {
  found: boolean;
  /** Ruled tables seen on the sheet, whether or not one was a point table. */
  tableCount: number;
  table: ReferenceTable | null;
  /** Named in the table but not found in any view. */
  unmatchedSymbols: string[];
}

/** A located table bound to the page it was read from. */
export interface PageReferenceTable extends ReferenceTable {
  page: number;
}
