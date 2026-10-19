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
  it("selects completed section pages and deduplicates them", () => {
    expect(
      completedReferencePages(
        [{ page: 2 }, { page: 2 }, { page: 3 }, { page: 4 }],
        3
      )
    ).toEqual([2, 3]);
  });

  it("runs each page once and isolates a page failure", async () => {
    const applied: number[] = [];
    const outcomes = await runReferenceScanQueue({
      pages: [1, 2, 2],
      runPage: async (page) => {
        if (page === 1) throw new Error("service unavailable");
        return found(["B"]);
      },
      applyResult: async (page) => {
        applied.push(page);
      },
    });
    expect(applied).toEqual([2]);
    expect(outcomes.map(({ page, status }) => ({ page, status }))).toEqual([
      { page: 1, status: "failed" },
      { page: 2, status: "found" },
    ]);
  });

  it("replaces or clears only the scanned page", () => {
    const current = [savedTable(1, "OLD"), savedTable(2, "B")];
    const updated = referenceTablesAfterPageResult(current, 1, found());
    expect(updated.map(({ page }) => page)).toEqual([1, 2]);
    expect(updated[0].points[0].symbol).toBe("A");
    expect(referenceTablesAfterPageResult(current, 1, notFound)).toEqual([
      savedTable(2, "B"),
    ]);
  });

  it("does not warn for an ordinary page without a reference table", () => {
    expect(
      referenceScanNotices([
        { page: 1, status: "not_found", tableCount: 0 },
        { page: 2, status: "not_found", tableCount: 4 },
      ])
    ).toEqual([]);
  });
});
