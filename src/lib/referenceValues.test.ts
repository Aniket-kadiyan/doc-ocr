import { describe, expect, it } from "vitest";
import {
  buildReferenceValueRows,
  normalizePageReferenceTables,
} from "@/lib/referenceValues";
import type { PageReferenceTable } from "@/types/referencePoint";

const box = { x: 10, y: 20, width: 30, height: 12 };

function table(
  page: number,
  points: PageReferenceTable["points"]
): PageReferenceTable {
  return {
    page,
    bbox: box,
    headerBox: box,
    axes: ["X", "Y", "Z"],
    nameHeader: "POINT",
    points,
  };
}

describe("reference value rows", () => {
  it("orders pages and emits exactly X, Y, Z for every point", () => {
    const rows = buildReferenceValueRows([
      table(2, [
        {
          symbol: "B",
          coordinates: [
            { axis: "X", value: "4" },
            { axis: "Y", value: "5" },
            { axis: "Z", value: "6" },
          ],
          nameBox: box,
          rowBox: box,
          markers: [],
        },
      ]),
      table(1, [
        {
          symbol: " A ",
          coordinates: [
            { axis: "z", value: " 3 " },
            { axis: "X", value: " 1 " },
          ],
          nameBox: box,
          rowBox: box,
          markers: [{ bbox: box, score: 0.92, margin: 0.18 }],
        },
      ]),
    ]);

    expect(rows.map(({ label, value }) => [label, value])).toEqual([
      ["A_X", "1"],
      ["A_Y", ""],
      ["A_Z", "3"],
      ["B_X", "4"],
      ["B_Y", "5"],
      ["B_Z", "6"],
    ]);
    expect(rows.map(({ id }) => id)).toEqual([
      "reference:p1:r0:x",
      "reference:p1:r0:y",
      "reference:p1:r0:z",
      "reference:p2:r0:x",
      "reference:p2:r0:y",
      "reference:p2:r0:z",
    ]);
  });

  it("keeps the latest result when a page appears twice", () => {
    const first = table(3, []);
    const latest = table(3, [
      {
        symbol: "C",
        coordinates: [],
        nameBox: box,
        rowBox: box,
        markers: [],
      },
    ]);
    expect(normalizePageReferenceTables([first, latest])).toEqual([latest]);
  });
});
