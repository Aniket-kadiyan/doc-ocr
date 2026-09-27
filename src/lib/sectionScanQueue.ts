import type { BBox } from "@/types/annotation";
import type { ScanCompletionSummary } from "@/types/scanJob";

export interface QueuedScanSection {
  id: string;
  bbox: BBox;
  page: number;
}

export interface SectionQueuePosition {
  current: number;
  total: number;
}

export type ScanRunOutcome =
  | { status: "succeeded"; summary: ScanCompletionSummary }
  | { status: "cancelled" | "failed" };

export interface SectionQueueResult {
  status: ScanRunOutcome["status"];
  completedSections: number;
  summary: ScanCompletionSummary;
}

export function emptySectionScanSummary(): ScanCompletionSummary {
  return {
    scopeKind: "section",
    added: 0,
    detected: 0,
    recognized: 0,
    eligible: 0,
    excluded: 0,
    reviewRequired: 0,
    unread: 0,
    skippedExisting: 0,
    skippedDuplicates: 0,
  };
}

export function mergeSectionScanSummaries(
  current: ScanCompletionSummary,
  incoming: ScanCompletionSummary
): ScanCompletionSummary {
  return {
    scopeKind: "section",
    added: current.added + incoming.added,
    detected: current.detected + incoming.detected,
    recognized: current.recognized + incoming.recognized,
    eligible: current.eligible + incoming.eligible,
    excluded: current.excluded + incoming.excluded,
    // Section scans accumulate unresolved reviews in one page-level list, so
    // the latest section summary is already the authoritative current count.
    reviewRequired: incoming.reviewRequired,
    unread: current.unread + incoming.unread,
    skippedExisting: current.skippedExisting + incoming.skippedExisting,
    skippedDuplicates:
      current.skippedDuplicates + incoming.skippedDuplicates,
  };
}

/**
 * Run selected sections strictly in drawing order. Each successful section is
 * handed to `afterSection` before the next backend job starts, which lets the
 * caller save and paint its balloons without publishing partial current-job
 * results.
 */
export async function runSectionScanQueue(args: {
  sections: readonly QueuedScanSection[];
  runSection: (
    section: QueuedScanSection,
    position: SectionQueuePosition
  ) => Promise<ScanRunOutcome>;
  afterSection?: (
    section: QueuedScanSection,
    position: SectionQueuePosition,
    aggregate: ScanCompletionSummary
  ) => void | Promise<void>;
}): Promise<SectionQueueResult> {
  let aggregate = emptySectionScanSummary();
  let completedSections = 0;

  for (let index = 0; index < args.sections.length; index += 1) {
    const section = args.sections[index];
    const position = { current: index + 1, total: args.sections.length };
    const outcome = await args.runSection(section, position);

    if (outcome.status !== "succeeded") {
      return {
        status: outcome.status,
        completedSections,
        summary: aggregate,
      };
    }

    completedSections += 1;
    aggregate = mergeSectionScanSummaries(aggregate, outcome.summary);
    await args.afterSection?.(section, position, aggregate);
  }

  return {
    status: "succeeded",
    completedSections,
    summary: aggregate,
  };
}
