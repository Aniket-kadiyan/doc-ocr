import type { Annotation, BBox } from "@/types/annotation";
import type { SegmentRegion } from "@/lib/paddleOcrClient";
import type { ScanScopeKind } from "@/types/scanJob";

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

/** Strict page-duplicate match: similar size and the same centre/footprint. */
export function isSamePageObject(left: BBox, right: BBox): boolean {
  const leftWidth = Math.max(left.width, 1);
  const leftHeight = Math.max(left.height, 1);
  const rightWidth = Math.max(right.width, 1);
  const rightHeight = Math.max(right.height, 1);
  const widthRatio =
    Math.min(leftWidth, rightWidth) / Math.max(leftWidth, rightWidth);
  const heightRatio =
    Math.min(leftHeight, rightHeight) / Math.max(leftHeight, rightHeight);
  if (widthRatio < 0.65 || heightRatio < 0.65) return false;

  const x0 = Math.max(left.x, right.x);
  const y0 = Math.max(left.y, right.y);
  const x1 = Math.min(left.x + leftWidth, right.x + rightWidth);
  const y1 = Math.min(left.y + leftHeight, right.y + rightHeight);
  const intersection =
    x1 > x0 && y1 > y0 ? (x1 - x0) * (y1 - y0) : 0;
  const union =
    leftWidth * leftHeight + rightWidth * rightHeight - intersection;
  const iou = intersection / Math.max(union, 1);

  const centersAreClose =
    Math.abs(left.x + leftWidth / 2 - (right.x + rightWidth / 2)) <=
      0.35 * Math.max(leftWidth, rightWidth) &&
    Math.abs(left.y + leftHeight / 2 - (right.y + rightHeight / 2)) <=
      0.35 * Math.max(leftHeight, rightHeight);
  return iou >= 0.45 || centersAreClose;
}

export interface FilteredScanRegions {
  accepted: SegmentRegion[];
  /** Existing balloons a feature control frame replaces in place. */
  upgrades: { annotationId: string; region: SegmentRegion }[];
  skippedExisting: number;
  skippedDuplicates: number;
}

function isGdtRegion(region: { type?: string; category?: string }): boolean {
  return region.category === "GD&T" || region.type === "GD&T";
}

const compact = (text: string) => text.replace(/\s+/g, "");

/**
 * The existing balloon a feature control frame should take over, if any.
 *
 * A scan that ran before frames were read (or with them turned off) balloons
 * only the frame's tolerance, "Ø0.02", and that balloon then blocked every
 * later "◎ Ø0.02 A" as a duplicate. The frame takes the balloon over only
 * when every balloon it covers is a plain value lying inside it whose text
 * the frame still contains: anything retyped by the user, or already GD&T,
 * keeps winning.
 */
function frameUpgradeTarget(
  candidate: SegmentRegion,
  overlapping: Annotation[]
): Annotation | null {
  if (!isGdtRegion(candidate) || overlapping.length === 0) return null;
  const frameText = compact(candidate.text);
  const upgradable = overlapping.every((annotation) => {
    const value = compact(annotation.value);
    const box = annotation.bbox;
    const area = Math.max(box.width * box.height, 1);
    const frameArea = Math.max(
      candidate.valueBox.width * candidate.valueBox.height,
      1
    );
    return (
      annotation.type !== "GD&T" &&
      value.length > 0 &&
      frameText.includes(value) &&
      area < frameArea
    );
  });
  if (!upgradable) return null;
  return [...overlapping].sort((left, right) => left.number - right.number)[0];
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
  overlapThreshold = DEFAULT_OVERLAP_THRESHOLD,
  scopeKind: ScanScopeKind = "section"
): FilteredScanRegions {
  const existingValues = existing.filter(
    (annotation) => annotation.kind !== "label" && annotation.page === page
  );

  const ordered = [...candidates].sort(
    (left, right) =>
      left.valueBox.y - right.valueBox.y ||
      left.valueBox.x - right.valueBox.x
  );
  const accepted: SegmentRegion[] = [];
  const upgrades: FilteredScanRegions["upgrades"] = [];
  const upgraded = new Set<string>();
  let skippedExisting = 0;
  let skippedDuplicates = 0;

  for (const candidate of ordered) {
    const overlapping = existingValues.filter(
      (annotation) =>
        bboxOverlapFraction(candidate.valueBox, annotation.bbox) >=
        overlapThreshold
    );
    if (overlapping.length > 0) {
      const target = frameUpgradeTarget(candidate, overlapping);
      if (target && !upgraded.has(target.id)) {
        upgraded.add(target.id);
        upgrades.push({ annotationId: target.id, region: candidate });
      } else {
        skippedExisting += 1;
      }
      continue;
    }
    const duplicatesCandidate = accepted.some(
      (acceptedRegion) =>
        scopeKind === "page"
          ? isSamePageObject(candidate.valueBox, acceptedRegion.valueBox)
          : bboxOverlapFraction(candidate.valueBox, acceptedRegion.valueBox) >=
            overlapThreshold
    );
    if (duplicatesCandidate) {
      skippedDuplicates += 1;
      continue;
    }
    accepted.push(candidate);
  }

  return { accepted, upgrades, skippedExisting, skippedDuplicates };
}
