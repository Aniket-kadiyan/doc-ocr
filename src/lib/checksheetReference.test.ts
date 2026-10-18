import { describe, expect, it } from "vitest";
import {
  checksheetFocusBox,
  checksheetItemCaption,
  checksheetSpecificationText,
  referenceDisplayCoordinates,
} from "@/lib/checksheetReference";
import type { ChecksheetRow } from "@/types/checksheet";

const balloonRow = (): ChecksheetRow => ({
  id: "row-1",
  annotation_id: "annotation-1",
  balloon_number: 1,
  item_kind: "balloon",
  reference: null,
  page: 1,
  bbox: { x: 50, y: 60, width: 20, height: 10 },
  oriented_box: null,
  rotation: 0,
  label: "Diameter",
  specification: "25",
  tolerance: "+0.1, -0.1",
  method: "Measure",
  tool: "Micrometer",
  dimension_type: "Linear",
  readings: {},
});

const referenceRow = (): ChecksheetRow => ({
  ...balloonRow(),
  id: "reference-row",
  annotation_id: "reference:p1:r0:z",
  balloon_number: 4,
  item_kind: "reference",
  label: "A_Z",
  specification: "",
  tolerance: "",
  method: "",
  tool: "",
  dimension_type: "Reference",
  bbox: { x: 400, y: 200, width: 12, height: 14 },
  reference: {
    symbol: "A",
    axis: "Z",
    point_index: 0,
    name_box: { x: 10, y: 40, width: 30, height: 20 },
    row_box: { x: 10, y: 40, width: 300, height: 20 },
    coordinates: [
      { axis: "X", value: "10.25" },
      { axis: "Y", value: "-4.50" },
    ],
    markers: [
      {
        bbox: { x: 400, y: 200, width: 12, height: 14 },
        score: 0.94,
        margin: 0.21,
      },
    ],
  },
});

describe("checksheet reference presentation", () => {
  it("keeps balloon presentation unchanged", () => {
    const row = balloonRow();
    expect(checksheetItemCaption(row)).toBe("Balloon 1");
    expect(checksheetSpecificationText(row)).toBe("25");
    expect(referenceDisplayCoordinates(row)).toEqual([]);
    expect(checksheetFocusBox(row)).toEqual(row.bbox);
  });

  it("labels a missing reference axis without inventing a value", () => {
    const row = referenceRow();
    expect(checksheetItemCaption(row)).toBe("Reference A_Z");
    expect(checksheetSpecificationText(row)).toBe("Not detected");
    expect(referenceDisplayCoordinates(row)).toEqual([
      { axis: "X", value: "10.25" },
      { axis: "Y", value: "-4.50" },
      { axis: "Z", value: "Not detected" },
    ]);
  });

  it("focuses both the table source and all drawing markers", () => {
    expect(checksheetFocusBox(referenceRow())).toEqual({
      x: 10,
      y: 40,
      width: 402,
      height: 174,
    });
  });
});
