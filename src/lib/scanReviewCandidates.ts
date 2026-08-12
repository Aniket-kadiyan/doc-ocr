import type { Annotation, BBox } from "@/types/annotation";
import type { ScanScopeKind } from "@/types/scanJob";
import { bboxOverlapFraction } from "@/lib/scanCandidates";

export interface PageReviewCandidateLike {
  page: number;
  valueBox: BBox;
}

export interface MergedReviewCandidates<T extends PageReviewCandidateLike> {
  candidates: T[];
  skippedDuplicates: number;
}

const overlapsAnnotation = (
  candidate: PageReviewCandidateLike,
  annotations: readonly Annotation[],
  overlapThreshold: number
) =>
  annotations.some(
    (annotation) =>
      annotation.kind !== "label" &&
      annotation.page === candidate.page &&
      bboxOverlapFraction(candidate.valueBox, annotation.bbox) >=
        overlapThreshold
  );

const overlapsCandidate = <T extends PageReviewCandidateLike>(
  candidate: T,
  existing: readonly T[],
  overlapThreshold: number
) =>
  existing.some(
    (other) =>
      other.page === candidate.page &&
      bboxOverlapFraction(candidate.valueBox, other.valueBox) >=
        overlapThreshold
  );

/**
 * Section scans append unresolved review objects, while a new whole-page scan
 * replaces the unresolved objects for that page. A newly created balloon always
 * wins over an older review box at the same position.
 */
export function mergePageReviewCandidates<
  T extends PageReviewCandidateLike,
>(args: {
  existing: readonly T[];
  incoming: readonly T[];
  newAnnotations: readonly Annotation[];
  page: number;
  scopeKind: ScanScopeKind;
  overlapThreshold?: number;
}): MergedReviewCandidates<T> {
  const overlapThreshold = args.overlapThreshold ?? 0.55;
  const uncovered = args.existing.filter(
    (candidate) =>
      !overlapsAnnotation(candidate, args.newAnnotations, overlapThreshold)
  );
  const retained =
    args.scopeKind === "page"
      ? uncovered.filter((candidate) => candidate.page !== args.page)
      : [...uncovered];

  const accepted: T[] = [];
  let skippedDuplicates = 0;
  for (const candidate of args.incoming) {
    if (
      overlapsAnnotation(candidate, args.newAnnotations, overlapThreshold) ||
      overlapsCandidate(candidate, retained, overlapThreshold) ||
      overlapsCandidate(candidate, accepted, overlapThreshold)
    ) {
      skippedDuplicates += 1;
      continue;
    }
    accepted.push(candidate);
  }

  return {
    candidates: [...retained, ...accepted],
    skippedDuplicates,
  };
}
