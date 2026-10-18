import { describe, expect, it } from "vitest";
import {
  completedReferencePages,
  referenceScanNotices,
  referenceTablesAfterPageResult,
  runReferenceScanQueue,
} from "@/lib/referenceScanQueue";
import type {
  PageReferenceTable,
  ReferencePointsResult,
} from "@/types/referencePoint";

const notFound: ReferencePointsResult = {
  found: false,
  tableCount: 0,
  table: null,
  unmatchedSymbols: [],
};

const found = (symbols: string[] = []): ReferencePointsResult => ({
  found: true,
  tableCount: 1,
  table: {
    bbox: { x: 0, y: 0, width: 100, height: 80 },
    headerBox: { x: 0, y: 0, width: 100, height: 20 },
    axes: ["X", "Y", "Z"],
    nameHeader: "POINT",
    points: ["A", "B"].map((symbol) => ({
      symbol,
      coordinates: [],
      nameBox: { x: 0, y: 20, width: 20, height: 20 },
      rowBox: { x: 0, y: 20, width: 100, height: 20 },
      markers: [],
    })),
  },
  unmatchedSymbols: symbols,
});

const savedTable = (page: number, symbol: string): PageReferenceTable => ({
  page,
  bbox: { x: 0, y: 0, width: 100, height: 80 },
  headerBox: { x: 0, y: 0, width: 100, height: 20 },
  axes: ["X", "Y", "Z"],
  nameHeader: "POINT",
  points: [
    {
      symbol,
      coordinates: [],
      nameBox: { x: 0, y: 20, width: 20, height: 20 },
      rowBox: { x: 0, y: 20, width: 100, height: 20 },
      markers: [],
    },
  ],
});

describe("reference scan queue", () => {
  it("selects only completed section pages and deduplicates them", () => {
    expect(
      completedReferencePages(
        [{ page: 2 }, { page: 2 }, { page: 3 }, { page: 4 }],
        3
      )
    ).toEqual([2, 3]);
    expect(completedReferencePages([{ page: 1 }], 0)).toEqual([]);
  });

  it("runs each page once, sequentially, in first-seen order", async () => {
    const events: string[] = [];
    const outcomes = await runReferenceScanQueue({
      pages: [2, 1, 2, 0],
      runPage: async (page) => {
        events.push(`run:${page}`);
        return page === 2 ? found() : notFound;
      },
      applyResult: async (page) => {
        events.push(`apply:${page}`);
      },
    });

    expect(events).toEqual(["run:2", "apply:2", "run:1", "apply:1"]);
    expect(outcomes.map(({ page, status }) => ({ page, status }))).toEqual([
      { page: 2, status: "found" },
      { page: 1, status: "not_found" },
    ]);
  });

  it("isolates one page failure and continues with later pages", async () => {
    const applied: number[] = [];
    const outcomes = await runReferenceScanQueue({
      pages: [1, 2],
      runPage: async (page) => {
        if (page === 1) throw new Error("service unavailable");
        return found(["B"]);
      },
      applyResult: async (page) => {
        applied.push(page);
      },
    });

    expect(applied).toEqual([2]);
    expect(outcomes).toEqual([
      { page: 1, status: "failed", error: "service unavailable" },
      {
        page: 2,
        status: "found",
        pointCount: 2,
        unmatchedSymbols: ["B"],
      },
    ]);
    expect(referenceScanNotices(outcomes)).toEqual([
      "Balloons were saved, but reference detection failed on page 1: service unavailable",
      "Reference points not located on page 2: B.",
    ]);
  });

  it("also isolates a local save failure", async () => {
    const run: number[] = [];
    const outcomes = await runReferenceScanQueue({
      pages: [1, 2],
      runPage: async (page) => {
        run.push(page);
        return found();
      },
      applyResult: async (page) => {
        if (page === 1) throw new Error("local save failed");
      },
    });

    expect(run).toEqual([1, 2]);
    expect(outcomes.map(({ page, status }) => ({ page, status }))).toEqual([
      { page: 1, status: "failed" },
      { page: 2, status: "found" },
    ]);
  });

  it("replaces one page on rerun without changing other pages", () => {
    const updated = referenceTablesAfterPageResult(
      [savedTable(1, "OLD"), savedTable(2, "B")],
      1,
      found()
    );

    expect(updated).toHaveLength(2);
    expect(updated.map(({ page }) => page)).toEqual([1, 2]);
    expect(updated.find(({ page }) => page === 1)?.points[0].symbol).toBe("A");
    expect(updated.find(({ page }) => page === 2)?.points[0].symbol).toBe("B");
  });

  it("clears only the scanned page when no reference table is found", () => {
    expect(
      referenceTablesAfterPageResult(
        [savedTable(1, "A"), savedTable(2, "B")],
        1,
        notFound
      )
    ).toEqual([savedTable(2, "B")]);
  });

  it("does not warn when a page simply has no reference table", () => {
    expect(
      referenceScanNotices([
        { page: 1, status: "not_found", tableCount: 0 },
        { page: 2, status: "not_found", tableCount: 4 },
      ])
    ).toEqual([]);
  });

  it("reports when every point name from a table is unlocated", () => {
    expect(
      referenceScanNotices([
        {
          page: 3,
          status: "found",
          pointCount: 2,
          unmatchedSymbols: ["A", "B"],
        },
      ])
    ).toEqual([
      "The reference table on page 3 was read, but none of its point names were located in the drawing views.",
    ]);
  });
});
