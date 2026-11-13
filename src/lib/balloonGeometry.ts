import type { Annotation, BBox } from "@/types/annotation";

/**
 * Balloon + box geometry and colors, in source-canvas coordinates at 100% zoom.
 *
 * `Balloon.tsx` and the annotation rectangles in `DrawingViewer.tsx` divide
 * every on-screen size by the live zoom so strokes keep a constant pixel size.
 * Exports render at zoom 1, so the same constants appear here undivided — which
 * makes an exported drawing look exactly like the viewer at 100%. Keep the two
 * in sync: this module is the single description both export renderers consume.
 */
export const BALLOON_RADIUS = 14;
export const BALLOON_FONT_SIZE = 12;
/** Vertical gap from the box's top edge up to the balloon's center. */
export const BALLOON_OFFSET = 28;

/** Colors, mirroring Balloon.tsx: labels indigo/violet, dimensions red. */
export const BALLOON_COLORS = {
  label: {
    accent: "#7c3aed",
    strong: "#4f46e5",
    selected: "#4f46e5",
    selectedStroke: "#4338ca",
  },
  dimension: {
    accent: "#dc2626",
    strong: "#dc2626",
    selected: "#2563eb",
    selectedStroke: "#1d4ed8",
  },
} as const;

/**
 * Anchor every marker to the displayed value box's top-left origin.
 *
 * This is deliberately independent of marker size and viewer zoom. The SVG
 * artwork owns its own connector geometry; its configured anchor point is
 * placed at this source-coordinate position by every renderer.
 */
export function balloonTopLeftAnchor(
  box: Pick<BBox, "x" | "y">
): { x: number; y: number } {
  return { x: box.x, y: box.y };
}

export interface BalloonGeometry {
  id: string;
  number: number;
  isLabel: boolean;
  /** The rectangle actually drawn: the tight oriented box when present. */
  box: { x: number; y: number; width: number; height: number; rotation: number };
  /** Dash pattern for the rectangle — labels are finely dashed. */
  boxDash: [number, number];
  /** Balloon bubble. */
  circle: { cx: number; cy: number; r: number };
  /** Leader line from the bubble down to the top edge of the box. */
  leader: { x1: number; y1: number; x2: number; y2: number };
  fontSize: number;
  /** Outline color for the box and leader. */
  accent: string;
  /** Bubble outline + number color. */
  strong: string;
}

/** Lay out one annotation's box, leader and balloon at 100% zoom. */
export function balloonGeometry(a: Annotation): BalloonGeometry {
  const isLabel = a.kind === "label";
  const colors = isLabel ? BALLOON_COLORS.label : BALLOON_COLORS.dimension;
  const src = a.orientedBox ?? a.bbox;
  const { x: anchorX, y: anchorY } = balloonTopLeftAnchor(src);
  const cy = anchorY - BALLOON_OFFSET;
  return {
    id: a.id,
    number: a.number,
    isLabel,
    box: {
      x: src.x,
      y: src.y,
      width: src.width,
      height: src.height,
      rotation: a.orientedBox?.rotation ?? 0,
    },
    boxDash: isLabel ? [3, 3] : [6, 4],
    circle: { cx: anchorX, cy, r: BALLOON_RADIUS },
    leader: {
      x1: anchorX,
      y1: cy + BALLOON_RADIUS,
      x2: anchorX,
      y2: anchorY,
    },
    fontSize: BALLOON_FONT_SIZE,
    accent: colors.accent,
    strong: colors.strong,
  };
}
