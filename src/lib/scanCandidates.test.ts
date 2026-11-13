import { describe, expect, it } from "vitest";
import {
  bboxOverlapFraction,
  filterNewScanRegions,
  isSamePageObject,
} from "@/lib/scanCandidates";
import type { SegmentRegion } from "@/lib/paddleOcrClient";
import type { BBox } from "@/types/annotation";
import { makeAnnotation } from "@/test/annotationFixture";

function makeRegion(text: string, valueBox: BBox): SegmentRegion {
  return {
    text,
    confidence: 0.9,
    orientation: "horizontal",
    rotation: 0,
    needsReview: false,
    recognized: true,
    valueBox,
  };
}

describe("scan candidate duplicate protection", () => {
  it("treats a tight candidate contained by an existing manual box as a match", () => {
    expect(
      bboxOverlapFraction(
        { x: 20, y: 20, width: 10, height: 10 },
        { x: 10, y: 10, width: 40, height: 30 }
      )
    ).toBe(1);
  });

  it("skips overlapping candidates on the scanned page only", () => {
    const duplicate = makeRegion("25", {
      x: 12,
      y: 21,
      width: 25,
      height: 10,
    });
    const otherPage = makeRegion("30", {
      x: 110,
      y: 20,
      width: 30,
      height: 12,
    });
    const result = filterNewScanRegions(
      [duplicate, otherPage],
      [
        makeAnnotation(),
        makeAnnotation({
          id: "page-2",
          page: 2,
          bbox: { x: 110, y: 20, width: 30, height: 12 },
        }),
      ],
      1
    );

    expect(result.accepted.map((region) => region.text)).toEqual(["30"]);
    expect(result.skippedExisting).toBe(1);
    expect(result.skippedDuplicates).toBe(0);
  });

  it("keeps individually hidden balloons active for duplicate prevention", () => {
    const result = filterNewScanRegions(
      [
        makeRegion("25", {
          x: 10,
          y: 20,
          width: 30,
          height: 12,
        }),
      ],
      [makeAnnotation({ hidden: true })],
      1
    );

    expect(result.accepted).toEqual([]);
    expect(result.skippedExisting).toBe(1);
  });

  it("deduplicates one batch and returns survivors in stable page order", () => {
    const result = filterNewScanRegions(
      [
        makeRegion("third", { x: 80, y: 80, width: 20, height: 10 }),
        makeRegion("first", { x: 40, y: 10, width: 20, height: 10 }),
        makeRegion("duplicate first", {
          x: 41,
          y: 10,
          width: 18,
          height: 10,
        }),
        makeRegion("second", { x: 10, y: 80, width: 20, height: 10 }),
      ],
      [],
      1
    );

    expect(result.accepted.map((region) => region.text)).toEqual([
      "first",
      "second",
      "third",
    ]);
    expect(result.skippedExisting).toBe(0);
    expect(result.skippedDuplicates).toBe(1);
  });

  it("orders survivors one drawing section at a time", () => {
    const inSection = (text: string, rank: number, valueBox: BBox) => ({
      ...makeRegion(text, valueBox),
      sectionRank: rank,
    });
    const result = filterNewScanRegions(
      [
        inSection("right view top", 1, { x: 500, y: 10, width: 20, height: 10 }),
        inSection("left view low", 0, { x: 10, y: 300, width: 20, height: 10 }),
        inSection("left view top", 0, { x: 10, y: 50, width: 20, height: 10 }),
      ],
      [],
      1,
      undefined,
      "page"
    );

    expect(result.accepted.map((region) => region.text)).toEqual([
      "left view top",
      "left view low",
      "right view top",
    ]);
  });

  it("numbers top to bottom when section numbering is off", () => {
    const result = filterNewScanRegions(
      [
        { ...makeRegion("low", { x: 10, y: 300, width: 20, height: 10 }), sectionRank: 0 },
        { ...makeRegion("high", { x: 500, y: 10, width: 20, height: 10 }), sectionRank: 1 },
      ],
      [],
      1,
      undefined,
      "page",
      false
    );

    expect(result.accepted.map((region) => region.text)).toEqual(["high", "low"]);
  });

  it("keeps contained page objects when their sizes differ", () => {
    const parent = makeRegion("parent", {
      x: 10,
      y: 10,
      width: 160,
      height: 100,
    });
    const child = makeRegion("child", {
      x: 30,
      y: 30,
      width: 30,
      height: 12,
    });

    expect(isSamePageObject(parent.valueBox, child.valueBox)).toBe(false);
    const result = filterNewScanRegions(
      [parent, child],
      [],
      1,
      undefined,
      "page"
    );

    expect(result.accepted.map((region) => region.text)).toEqual([
      "parent",
      "child",
    ]);
    expect(result.skippedDuplicates).toBe(0);
  });

  it("removes only similar-size repeated page candidates", () => {
    const result = filterNewScanRegions(
      [
        makeRegion("first view", {
          x: 40,
          y: 10,
          width: 30,
          height: 12,
        }),
        makeRegion("overlap view", {
          x: 41,
          y: 10,
          width: 29,
          height: 12,
        }),
      ],
      [],
      1,
      undefined,
      "page"
    );

    expect(result.accepted).toHaveLength(1);
    expect(result.skippedDuplicates).toBe(1);
  });

  it("does not merge parallel neighbouring page objects", () => {
    expect(
      isSamePageObject(
        { x: 100, y: 100, width: 16, height: 100 },
        { x: 125, y: 100, width: 16, height: 100 }
      )
    ).toBe(false);
  });

  describe("feature control frames over an earlier tolerance balloon", () => {
    const frame = {
      ...makeRegion("◎ Ø0.02 A", { x: 100, y: 50, width: 115, height: 23 }),
      category: "GD&T",
      type: "GD&T",
    };
    const tolerance = makeAnnotation({
      id: "tol",
      number: 23,
      value: "Ø0.02",
      type: "Diameter",
      bbox: { x: 133, y: 50, width: 57, height: 23 },
    });

    it("takes over the untouched balloon in place", () => {
      const result = filterNewScanRegions([frame], [tolerance], 1);

      expect(result.accepted).toEqual([]);
      expect(result.upgrades).toEqual([
        { annotationId: "tol", region: frame },
      ]);
      expect(result.skippedExisting).toBe(0);
    });

    it("leaves a balloon the user retyped alone", () => {
      const edited = { ...tolerance, value: "Ø0.05" };
      const result = filterNewScanRegions([frame], [edited], 1);

      expect(result.upgrades).toEqual([]);
      expect(result.skippedExisting).toBe(1);
    });

    it("leaves an existing GD&T balloon alone", () => {
      const existingFrame = {
        ...tolerance,
        value: "◎ Ø0.02 A",
        type: "GD&T" as const,
        bbox: frame.valueBox,
      };
      const result = filterNewScanRegions([frame], [existingFrame], 1);

      expect(result.upgrades).toEqual([]);
      expect(result.skippedExisting).toBe(1);
    });

    it("never upgrades from a plain value", () => {
      const plain = makeRegion("Ø0.02 A", frame.valueBox);
      const result = filterNewScanRegions([plain], [tolerance], 1);

      expect(result.upgrades).toEqual([]);
      expect(result.skippedExisting).toBe(1);
    });
  });
});
