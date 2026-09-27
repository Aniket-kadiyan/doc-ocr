import { describe, expect, it } from "vitest";
import { mergePageReviewCandidates } from "@/lib/scanReviewCandidates";
import { makeAnnotation } from "@/test/annotationFixture";
import type { BBox } from "@/types/annotation";

interface ReviewCandidate {
  id: string;
  page: number;
  valueBox: BBox;
}

const review = (
  id: string,
  x: number,
  page = 1
): ReviewCandidate => ({
  id,
  page,
  valueBox: { x, y: 10, width: 30, height: 12 },
});

describe("scan review candidate accumulation", () => {
  it("appends section reviews in scan order and removes overlap duplicates", () => {
    const result = mergePageReviewCandidates({
      existing: [review("first", 10)],
      incoming: [review("duplicate", 11), review("second", 80)],
      newAnnotations: [],
      page: 1,
      scopeKind: "section",
    });

    expect(result.candidates.map((candidate) => candidate.id)).toEqual([
      "first",
      "second",
    ]);
    expect(result.skippedDuplicates).toBe(1);
  });

  it("lets a newly created balloon replace an older review box", () => {
    const result = mergePageReviewCandidates({
      existing: [review("resolved", 10), review("remaining", 80)],
      incoming: [],
      newAnnotations: [
        makeAnnotation({
          id: "accepted",
          bbox: { x: 10, y: 10, width: 30, height: 12 },
        }),
      ],
      page: 1,
      scopeKind: "section",
    });

    expect(result.candidates.map((candidate) => candidate.id)).toEqual([
      "remaining",
    ]);
  });

  it("replaces same-page reviews on whole-page scans but preserves other pages", () => {
    const result = mergePageReviewCandidates({
      existing: [review("old-page-one", 10), review("page-two", 10, 2)],
      incoming: [review("new-page-one", 80)],
      newAnnotations: [],
      page: 1,
      scopeKind: "page",
    });

    expect(result.candidates.map((candidate) => candidate.id)).toEqual([
      "page-two",
      "new-page-one",
    ]);
  });
});
