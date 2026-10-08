import type { BBox } from "@/types/annotation";

/**
 * Place a coordinate tag beside every located point name without stacking the
 * tags on top of each other.
 *
 * The same part is usually drawn in several views, and a view calls out most
 * of its points within a few millimetres of each other, so tags left at a
 * fixed offset overlap and hide the values they exist to show. Each tag is
 * therefore offered a ring of positions around its glyph and takes the first
 * that is still clear.
 *
 * Positions are tried in drafting order — above-right first, then the other
 * three corners, then straight above and below — so that the common case
 * looks deliberate rather than scattered, and only a crowded neighbourhood
 * produces unusual placements.
 */

export interface TagAnchor {
  id: string;
  /** The printed glyph the tag belongs to. */
  glyph: BBox;
}

export interface TagPlacement {
  x: number;
  y: number;
}

export interface TagLayoutOptions {
  width: number;
  height: number;
  /** Clearance between the glyph and its tag, and between two tags. */
  gap: number;
  /** Nothing is placed outside this, when supplied. */
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

export function layoutReferenceTags(
  anchors: TagAnchor[],
  options: TagLayoutOptions
): Map<string, TagPlacement> {
  const { width, height, bounds } = options;
  const placed: BBox[] = [];
  // A tag must also keep off every glyph, not only off the tags already
  // placed, or it covers the very letter the reader is checking it against.
  const glyphs = anchors.map((anchor) => anchor.glyph);
  const result = new Map<string, TagPlacement>();

  // Left to right, top to bottom, so the same drawing always lays out the
  // same way however the markers arrived.
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
    // Every position was taken. The first is still the most readable place to
    // put it, and overlapping tags separate as soon as the reader zooms in,
    // because a tag holds its screen size while the drawing grows.
    const chosen = clear ?? offered[0];
    placed.push({ ...chosen, width, height });
    result.set(anchor.id, chosen);
  }

  return result;
}
