import { describe, expect, it } from "vitest";
import {
  ignoreScanCandidate,
  mergeScanCandidates,
  normalizeReviewCandidateNumbers,
  restoreScanCandidate,
} from "@/lib/scanCandidateLifecycle";
import { makeAnnotation } from "@/test/annotationFixture";
import type { ScanCandidate } from "@/types/scanCandidate";

const candidate = (
  id: string,
  x: number,
  state: ScanCandidate["state"] = "other"
): ScanCandidate => ({
  id,
  sourceCandidateId: id,
  page: 1,
  order: x,
  state,
  text: id,
  rawText: id,
  confidence: 0.5,
  recognized: true,
  reason: "filtered",
  orientation: "horizontal",
  rotation: 0,
  valueBox: { x, y: 10, width: 30, height: 12 },
  createdAt: 1,
  updatedAt: 1,
});

describe("durable scan candidate lifecycle", () => {
  it("merges a rescan into the existing record with an audit count", () => {
    const result = mergeScanCandidates({
      existing: [candidate("first", 10, "review")],
      incoming: [candidate("second", 11, "other")],
      acceptedAnnotations: [],
    });

    expect(result.candidates).toHaveLength(1);
    expect(result.candidates[0]).toMatchObject({
      id: "first",
      state: "review",
      reviewNumber: 1,
      duplicateCount: 1,
      duplicateSourceIds: ["first", "second"],
    });
    expect(result.mergedDuplicates).toBe(1);
  });

  it("keeps an ignored decision across a geometry-equivalent rescan", () => {
    const ignored = ignoreScanCandidate(
      [candidate("first", 10, "review")],
      "first"
    );
    const result = mergeScanCandidates({
      existing: ignored,
      incoming: [candidate("second", 11, "review")],
      acceptedAnnotations: [],
    });

    expect(result.candidates[0].state).toBe("ignored");
    expect(restoreScanCandidate(result.candidates, "first")[0].state).toBe(
      "review"
    );
  });

  it("removes candidates represented by an accepted balloon", () => {
    const result = mergeScanCandidates({
      existing: [candidate("old", 10)],
      incoming: [candidate("new", 11)],
      acceptedAnnotations: [
        makeAnnotation({ bbox: { x: 10, y: 10, width: 30, height: 12 } }),
      ],
    });

    expect(result.candidates).toEqual([]);
    expect(result.skippedAccepted).toBe(2);
  });

  it("numbers review candidates independently and preserves existing numbers", () => {
    const normalized = normalizeReviewCandidateNumbers([
      candidate("other", 5, "other"),
      { ...candidate("second", 20, "review"), reviewNumber: 4 },
      candidate("first", 10, "review"),
    ]);

    expect(normalized.find((item) => item.id === "other")?.reviewNumber).toBeUndefined();
    expect(normalized.find((item) => item.id === "first")?.reviewNumber).toBe(1);
    expect(normalized.find((item) => item.id === "second")?.reviewNumber).toBe(4);
  });

  it("keeps a review number while ignored and after restoration", () => {
    const [numbered] = normalizeReviewCandidateNumbers([
      candidate("review", 10, "review"),
    ]);
    const ignored = ignoreScanCandidate([numbered], "review");

    expect(ignored[0]).toMatchObject({
      state: "ignored",
      restoreState: "review",
      reviewNumber: 1,
    });
    expect(restoreScanCandidate(ignored, "review")[0]).toMatchObject({
      state: "review",
      reviewNumber: 1,
    });
  });
});
