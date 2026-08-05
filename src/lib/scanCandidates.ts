import type { Annotation, BBox } from "@/types/annotation";
import type { SegmentRegion } from "@/lib/paddleOcrClient";

const DEFAULT_OVERLAP_THRESHOLD = 0.55;

/** Intersection as a fraction of the smaller box (so containment is a match). */
export function bboxOverlapFraction(left: BBox, right: BBox): number {
  const x0 = Math.max(left.x, right.x);
  const y0 = Math.max(left.y, right.y);
  const x1 = Math.min(left.x + left.width, right.x + right.width);
  const y1 = Math.min(left.y + left.height, right.y + right.height);
  if (x1 <= x0 || y1 <= y0) return 0;

  const intersection = (x1 - x0) * (y1 - y0);
  const leftArea = Math.max(left.width * left.height, 1);
  const rightArea = Math.max(right.width * right.height, 1);
  return intersection / Math.min(leftArea, rightArea);
}

export interface FilteredScanRegions {
  accepted: SegmentRegion[];
  skippedExisting: number;
  skippedDuplicates: number;
}

/**
 * Apply Milestone 1 duplicate protection immediately before atomic insertion.
 * Existing manual/edited annotations always win, and accepted candidates also
 * protect against duplicates within the same backend result.
 */
export function filterNewScanRegions(
  candidates: SegmentRegion[],
  existing: Annotation[],
  page: number,
  overlapThreshold = DEFAULT_OVERLAP_THRESHOLD
): FilteredScanRegions {
  const existingBoxes = existing
    .filter(
      (annotation) =>
        annotation.kind !== "label" && annotation.page === page
    )
    .map((annotation) => annotation.bbox);

  const ordered = [...candidates].sort(
    (left, right) =>
      left.valueBox.y - right.valueBox.y ||
      left.valueBox.x - right.valueBox.x
  );
  const accepted: SegmentRegion[] = [];
  let skippedExisting = 0;
  let skippedDuplicates = 0;

  for (const candidate of ordered) {
    const overlapsExisting = existingBoxes.some(
      (box) =>
        bboxOverlapFraction(candidate.valueBox, box) >= overlapThreshold
    );
    if (overlapsExisting) {
      skippedExisting += 1;
      continue;
    }
    const duplicatesCandidate = accepted.some(
      (acceptedRegion) =>
        bboxOverlapFraction(candidate.valueBox, acceptedRegion.valueBox) >=
        overlapThreshold
    );
    if (duplicatesCandidate) {
      skippedDuplicates += 1;
      continue;
    }
    accepted.push(candidate);
  }

  return { accepted, skippedExisting, skippedDuplicates };
}
