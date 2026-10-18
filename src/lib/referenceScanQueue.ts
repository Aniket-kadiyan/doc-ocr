import type { ReferencePointsResult } from "@/types/referencePoint";

export type ReferenceScanOutcome =
  | {
      page: number;
      status: "found";
      pointCount: number;
      unmatchedSymbols: string[];
    }
  | { page: number; status: "not_found"; tableCount: number }
  | { page: number; status: "failed"; error: string };

/**
 * Read each page once and in order. A failed reference pass is recorded and
 * does not reject the queue, because its parent Auto Balloon pass has already
 * committed the page's balloons.
 */
export async function runReferenceScanQueue(args: {
  pages: readonly number[];
  runPage: (page: number) => Promise<ReferencePointsResult>;
  applyResult: (
    page: number,
    result: ReferencePointsResult
  ) => void | Promise<void>;
}): Promise<ReferenceScanOutcome[]> {
  const pages = [...new Set(args.pages)].filter(
    (page) => Number.isInteger(page) && page > 0
  );
  const outcomes: ReferenceScanOutcome[] = [];

  for (const page of pages) {
    try {
      const result = await args.runPage(page);
      await args.applyResult(page, result);
      if (result.found && result.table) {
        outcomes.push({
          page,
          status: "found",
          pointCount: result.table.points.length,
          unmatchedSymbols: [...result.unmatchedSymbols],
        });
      } else {
        outcomes.push({
          page,
          status: "not_found",
          tableCount: result.tableCount,
        });
      }
    } catch (error) {
      outcomes.push({
        page,
        status: "failed",
        error:
          error instanceof Error
            ? error.message
            : "Reference detection failed unexpectedly.",
      });
    }
  }

  return outcomes;
}

/** Only surface actionable reference results; a sheet without a reference table is normal. */
export function referenceScanNotices(
  outcomes: readonly ReferenceScanOutcome[]
): string[] {
  return outcomes.flatMap((outcome) => {
    if (outcome.status === "failed") {
      return [
        `Balloons were saved, but reference detection failed on page ${outcome.page}: ${outcome.error}`,
      ];
    }
    if (outcome.status !== "found" || outcome.unmatchedSymbols.length === 0) {
      return [];
    }
    if (
      outcome.pointCount > 0 &&
      outcome.unmatchedSymbols.length === outcome.pointCount
    ) {
      return [
        `The reference table on page ${outcome.page} was read, but none of its point names were located in the drawing views.`,
      ];
    }
    return [
      `Reference points not located on page ${outcome.page}: ${outcome.unmatchedSymbols.join(", ")}.`,
    ];
  });
}
