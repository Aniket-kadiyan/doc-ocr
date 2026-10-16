import type { BBox } from "@/types/annotation";

/** A located point-name glyph that needs a coordinate tag beside it. */
export interface TagAnchor {
  id: string;
  glyph: BBox;
}

export interface TagPlacement {
  x: number;
  y: number;
}

export interface TagLayoutOptions {
  width: number;
  height: number;
  /** Clearance between glyphs and tags, and between neighbouring tags. */
  gap: number;
  bounds?: { width: number; height: number };
}

const CORNERS = [
  [1, -1],
  [-1, -1],
  [1, 1],
  [-1, 1],
] as const;

function candidates(
  glyph: BBox,
  { width, height, gap }: TagLayoutOptions
): TagPlacement[] {
  const places: TagPlacement[] = [];
  for (const [horizontal, vertical] of CORNERS) {
    places.push({
      x:
        horizontal > 0
          ? glyph.x + glyph.width + gap
          : glyph.x - gap - width,
      y: vertical > 0 ? glyph.y + glyph.height + gap : glyph.y - gap - height,
    });
  }
  const centred = glyph.x + glyph.width / 2 - width / 2;
  places.push({ x: centred, y: glyph.y - gap - height });
  places.push({ x: centred, y: glyph.y + glyph.height + gap });
  return places;
}

const overlaps = (left: BBox, right: BBox) =>
  left.x < right.x + right.width &&
  left.x + left.width > right.x &&
  left.y < right.y + right.height &&
  left.y + left.height > right.y;

/**
 * Place tags deterministically without covering another tag or point glyph.
 * When every offered position is occupied, the normal above-right placement
 * is retained; zooming then separates the fixed-screen-size tags.
 */
export function layoutReferenceTags(
  anchors: TagAnchor[],
  options: TagLayoutOptions
): Map<string, TagPlacement> {
  const { width, height, bounds } = options;
  const placed: BBox[] = [];
  const glyphs = anchors.map((anchor) => anchor.glyph);
  const result = new Map<string, TagPlacement>();

  const ordered = [...anchors].sort(
    (left, right) =>
      left.glyph.y - right.glyph.y || left.glyph.x - right.glyph.x
  );

  for (const anchor of ordered) {
    const offered = candidates(anchor.glyph, options);
    const clear = offered.find((place) => {
      const box = { ...place, width, height };
      if (
        bounds &&
        (box.x < 0 ||
          box.y < 0 ||
          box.x + width > bounds.width ||
          box.y + height > bounds.height)
      ) {
        return false;
      }
      return (
        !placed.some((taken) => overlaps(box, taken)) &&
        !glyphs.some((glyph) => overlaps(box, glyph))
      );
    });
    const chosen = clear ?? offered[0];
    placed.push({ ...chosen, width, height });
    result.set(anchor.id, chosen);
  }

  return result;
}
