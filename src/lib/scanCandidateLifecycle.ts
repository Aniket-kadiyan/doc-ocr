import type { Annotation } from "@/types/annotation";
import type {
  RestorableScanCandidateState,
  ScanCandidate,
} from "@/types/scanCandidate";
import { bboxOverlapFraction } from "@/lib/scanCandidates";

export interface MergedScanCandidates {
  candidates: ScanCandidate[];
  mergedDuplicates: number;
  skippedAccepted: number;
}

const overlapsAnnotation = (
  candidate: ScanCandidate,
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

const matchingCandidateIndex = (
  candidate: ScanCandidate,
  candidates: readonly ScanCandidate[],
  overlapThreshold: number
) =>
  candidates.findIndex(
    (other) =>
      other.page === candidate.page &&
      bboxOverlapFraction(candidate.valueBox, other.valueBox) >=
        overlapThreshold
  );

const mergedSourceIds = (
  existing: ScanCandidate,
  incoming: ScanCandidate
): string[] =>
  Array.from(
    new Set(
      [
        ...(existing.duplicateSourceIds ?? []),
        existing.sourceCandidateId,
        ...(incoming.duplicateSourceIds ?? []),
        incoming.sourceCandidateId,
      ].filter((value): value is string => Boolean(value))
    )
  );

/**
 * Add a scan's unresolved outcomes without silently deleting older evidence.
 * Geometry-equivalent rescans are folded into the existing record and leave a
 * duplicate audit count. An explicit ignored decision always survives a rescan.
 */
export function mergeScanCandidates(args: {
  existing: readonly ScanCandidate[];
  incoming: readonly ScanCandidate[];
  acceptedAnnotations: readonly Annotation[];
  overlapThreshold?: number;
}): MergedScanCandidates {
  const overlapThreshold = args.overlapThreshold ?? 0.55;
  const candidates = args.existing.filter(
    (candidate) =>
      !overlapsAnnotation(candidate, args.acceptedAnnotations, overlapThreshold)
  );
  let skippedAccepted = args.existing.length - candidates.length;
  let mergedDuplicates = 0;

  for (const incoming of args.incoming) {
    if (
      overlapsAnnotation(incoming, args.acceptedAnnotations, overlapThreshold)
    ) {
      skippedAccepted += 1;
      continue;
    }

    const matchIndex = matchingCandidateIndex(
      incoming,
      candidates,
      overlapThreshold
    );
    if (matchIndex < 0) {
      candidates.push(incoming);
      continue;
    }

    const existing = candidates[matchIndex];
    const ignored = existing.state === "ignored";
    const sourceIds = mergedSourceIds(existing, incoming);
    candidates[matchIndex] = {
      ...existing,
      ...incoming,
      id: existing.id,
      order: existing.order,
      state: ignored
        ? "ignored"
        : existing.state === "review" || incoming.state === "review"
          ? "review"
          : incoming.state,
      restoreState: ignored
        ? existing.restoreState ?? "other"
        : undefined,
      createdAt: existing.createdAt,
      updatedAt: Math.max(existing.updatedAt, incoming.updatedAt),
      duplicateSourceIds: sourceIds,
      duplicateCount: (existing.duplicateCount ?? 0) + 1,
    };
    mergedDuplicates += 1;
  }

  return {
    candidates: candidates.sort((left, right) => left.order - right.order),
    mergedDuplicates,
    skippedAccepted,
  };
}

export function ignoreScanCandidate(
  candidates: readonly ScanCandidate[],
  id: string,
  updatedAt = Date.now()
): ScanCandidate[] {
  return candidates.map((candidate) =>
    candidate.id === id && candidate.state !== "ignored"
      ? {
          ...candidate,
          state: "ignored",
          restoreState: candidate.state as RestorableScanCandidateState,
          updatedAt,
        }
      : candidate
  );
}

export function restoreScanCandidate(
  candidates: readonly ScanCandidate[],
  id: string,
  updatedAt = Date.now()
): ScanCandidate[] {
  return candidates.map((candidate) =>
    candidate.id === id && candidate.state === "ignored"
      ? {
          ...candidate,
          state: candidate.restoreState ?? "other",
          restoreState: undefined,
          updatedAt,
        }
      : candidate
  );
}
