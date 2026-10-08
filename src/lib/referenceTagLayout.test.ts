import { describe, expect, it } from "vitest";
import { layoutReferenceTags, type TagAnchor } from "@/lib/referenceTagLayout";

const SIZE = { width: 40, height: 30, gap: 5 };

const glyph = (x: number, y: number) => ({ x, y, width: 8, height: 10 });

const boxes = (
  anchors: TagAnchor[],
  placements: Map<string, { x: number; y: number }>
) =>
  anchors.map((anchor) => ({
    ...placements.get(anchor.id)!,
    width: SIZE.width,
    height: SIZE.height,
  }));

const overlap = (
  left: { x: number; y: number; width: number; height: number },
  right: { x: number; y: number; width: number; height: number }
) =>
  left.x < right.x + right.width &&
  left.x + left.width > right.x &&
  left.y < right.y + right.height &&
  left.y + left.height > right.y;

describe("layoutReferenceTags", () => {
  it("places a lone tag above and to the right of its glyph", () => {
    const anchors = [{ id: "a", glyph: glyph(100, 100) }];

    const placed = layoutReferenceTags(anchors, SIZE).get("a")!;

    expect(placed.x).toBe(113);
    expect(placed.y).toBe(65);
  });

  it("gives every marker a placement", () => {
    const anchors = Array.from({ length: 12 }, (_, index) => ({
      id: `m${index}`,
      glyph: glyph(40 + index * 11, 40 + (index % 3) * 9),
    }));

    const placements = layoutReferenceTags(anchors, SIZE);

    expect(placements.size).toBe(anchors.length);
  });

  it("moves a tag aside rather than stacking it on another", () => {
    const anchors = [
      { id: "a", glyph: glyph(100, 100) },
      { id: "b", glyph: glyph(120, 100) },
    ];

    const [first, second] = boxes(anchors, layoutReferenceTags(anchors, SIZE));

    expect(overlap(first, second)).toBe(false);
  });

  it("never covers a glyph, including one belonging to another tag", () => {
    const anchors = [
      { id: "a", glyph: glyph(100, 100) },
      { id: "b", glyph: glyph(118, 72) },
      { id: "c", glyph: glyph(140, 104) },
    ];

    const placements = layoutReferenceTags(anchors, SIZE);

    for (const box of boxes(anchors, placements)) {
      for (const anchor of anchors) {
        expect(overlap(box, anchor.glyph)).toBe(false);
      }
    }
  });

  it("keeps tags inside the sheet when bounds are given", () => {
    // A glyph in the top-left corner cannot take the default placement.
    const anchors = [{ id: "a", glyph: glyph(4, 4) }];

    const placed = layoutReferenceTags(anchors, {
      ...SIZE,
      bounds: { width: 400, height: 400 },
    }).get("a")!;

    expect(placed.x).toBeGreaterThanOrEqual(0);
    expect(placed.y).toBeGreaterThanOrEqual(0);
  });

  it("lays the same markers out the same way whatever order they arrive in", () => {
    const anchors = [
      { id: "a", glyph: glyph(100, 100) },
      { id: "b", glyph: glyph(120, 100) },
      { id: "c", glyph: glyph(140, 100) },
    ];

    const forward = layoutReferenceTags(anchors, SIZE);
    const backward = layoutReferenceTags([...anchors].reverse(), SIZE);

    for (const anchor of anchors) {
      expect(backward.get(anchor.id)).toEqual(forward.get(anchor.id));
    }
  });
});
