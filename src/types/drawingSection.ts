/** One sub-drawing (view, section, detail or notes block) found on a page. */
export interface DrawingSection {
  id: string;
  kind: "view" | "text";
  /** Closed polygons in page coordinates, as flat [x0, y0, x1, y1, …]. */
  outlines: number[][];
}

/** The sections of one page, from its latest whole-page scan. */
export interface PageDrawingSections {
  page: number;
  sections: DrawingSection[];
}
