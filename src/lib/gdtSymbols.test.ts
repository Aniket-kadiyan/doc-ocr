import { describe, expect, it } from "vitest";
import {
  categoryLabel,
  describeFrame,
  detectionKey,
  formatConfidence,
  framesForPage,
  groupByCategory,
  needsReview,
  overlayItems,
  reviewReason,
  statusColor,
  toleranceUnitLabel,
  withinBox,
} from "@/lib/gdtSymbols";
import { shiftResult } from "@/lib/gdtClient";
import {
  GDT_AMBIGUOUS,
  GDT_CONFIRMED,
  GDT_UNKNOWN,
  type GdtDetection,
  type GdtFrame,
  type GdtPageResult,
} from "@/types/gdt";

function detection(overrides: Partial<GdtDetection> = {}): GdtDetection {
  return {
    symbol_id: "position",
    name: "Position",
    status: GDT_CONFIRMED,
    confidence: 0.91,
    bounding_box: { x: 120, y: 240, width: 20, height: 20 },
    page: 1,
    method: "template_match",
    raw_ocr: null,
    normalized: "⌖",
    alternatives: [],
    associated_text: null,
    value: null,
    warnings: [],
    category: "geometric_tolerance",
    ...overrides,
  };
}

function frame(overrides: Partial<GdtFrame> = {}): GdtFrame {
  return {
    type: "feature_control_frame",
    id: "frame-1",
    page: 1,
    bounding_box: { x: 100, y: 230, width: 180, height: 35 },
    symbol: detection(),
    tolerance: { value: 0.05, state: "present", confidence: 0.9, raw: "0.05" },
    diameter_zone: { value: true, state: "present", confidence: 0.9, raw: "" },
    material_condition: { value: "mmc", state: "present", confidence: 0.9, raw: "" },
    modifiers: [],
    datums: [
      { label: "A", precedence: 1, material_condition: null, confidence: 0.9, raw: "A" },
      { label: "B", precedence: 2, material_condition: null, confidence: 0.9, raw: "B" },
    ],
    compartments: [],
    segment_index: 0,
    composite_id: null,
    confidence: 0.9,
    warnings: [],
    tolerance_unit: "drawing_unit",
    ...overrides,
  };
}

function pageResult(overrides: Partial<GdtPageResult> = {}): GdtPageResult {
  return {
    page: 1,
    image_size: { width: 1000, height: 800 },
    frames: [],
    symbols: [],
    datum_associations: [],
    standards_findings: [],
    errors: [],
    counts: {
      frames: 0,
      symbols: 0,
      confirmed: 0,
      ambiguous: 0,
      unknown: 0,
      errors: 0,
    },
    elapsed_ms: 10,
    provenance: {},
    ...overrides,
  };
}

describe("status presentation", () => {
  it("gives review-needed and settled readings different colours", () => {
    expect(statusColor(GDT_CONFIRMED)).not.toBe(statusColor(GDT_AMBIGUOUS));
    expect(statusColor(GDT_AMBIGUOUS)).not.toBe(statusColor(GDT_UNKNOWN));
  });

  it("formats confidence as a percentage and handles absence", () => {
    expect(formatConfidence(0.913)).toBe("91%");
    expect(formatConfidence(0)).toBe("0%");
    expect(formatConfidence(null)).toBe("—");
    expect(formatConfidence(undefined)).toBe("—");
    expect(formatConfidence(Number.NaN)).toBe("—");
  });

  it("turns a category id into a readable heading", () => {
    expect(categoryLabel("geometric_tolerance")).toBe("Geometric tolerance");
    expect(categoryLabel("datum")).toBe("Datum");
  });
});

describe("overlay items", () => {
  it("returns nothing when the result is for a different page", () => {
    const result = pageResult({ page: 2, symbols: [detection({ page: 2 })] });
    expect(overlayItems(result, 1)).toEqual([]);
  });

  it("returns nothing when there is no result", () => {
    expect(overlayItems(null, 1)).toEqual([]);
  });

  it("maps detections to page-space boxes", () => {
    const result = pageResult({ symbols: [detection()] });
    const items = overlayItems(result, 1);
    expect(items).toHaveLength(1);
    expect(items[0].bbox).toEqual({ x: 120, y: 240, width: 20, height: 20 });
  });

  it("links a detection to the frame it sits in", () => {
    const parent = frame();
    const result = pageResult({
      frames: [parent],
      symbols: [detection({ frame_id: "frame-1" })],
    });
    expect(overlayItems(result, 1)[0].frame?.id).toBe("frame-1");
  });

  it("leaves a loose detection with no frame", () => {
    const result = pageResult({ symbols: [detection()] });
    expect(overlayItems(result, 1)[0].frame).toBeNull();
  });

  it("filters frames by page", () => {
    const result = pageResult({ frames: [frame()] });
    expect(framesForPage(result, 1)).toHaveLength(1);
    expect(framesForPage(result, 2)).toHaveLength(0);
  });

  it("gives each detection a stable distinct key", () => {
    const a = detection();
    const b = detection({ bounding_box: { x: 500, y: 20, width: 10, height: 10 } });
    expect(detectionKey(a, 0)).toBe(detectionKey(a, 0));
    expect(detectionKey(a, 0)).not.toBe(detectionKey(b, 1));
  });
});

describe("describing a frame", () => {
  it("reads symbol, zone, tolerance, modifier and datums in order", () => {
    expect(describeFrame(frame())).toBe("Position | ⌀0.05 | MMC | A B");
  });

  it("orders datums by precedence, not by array order", () => {
    const scrambled = frame({
      datums: [
        { label: "C", precedence: 3, material_condition: null, confidence: 1, raw: "C" },
        { label: "A", precedence: 1, material_condition: null, confidence: 1, raw: "A" },
        { label: "B", precedence: 2, material_condition: null, confidence: 1, raw: "B" },
      ],
    });
    expect(describeFrame(scrambled)).toContain("A B C");
  });

  it("shows a datum's own material condition", () => {
    const withModifier = frame({
      datums: [
        { label: "A", precedence: 1, material_condition: "mmc", confidence: 1, raw: "A" },
      ],
    });
    expect(describeFrame(withModifier)).toContain("A(MMC)");
  });

  it("says a tolerance is unreadable rather than omitting it", () => {
    // Omitting it would read as "this frame has no tolerance", which is a
    // different and much stronger claim than "we could not read it".
    const unreadable = frame({
      tolerance: { value: null, state: "unreadable", confidence: 0, raw: null },
    });
    expect(describeFrame(unreadable)).toContain("tolerance unreadable");
  });

  it("says a modifier is unreadable rather than implying RFS", () => {
    const unreadable = frame({
      material_condition: { value: null, state: "unreadable", confidence: 0, raw: null },
    });
    expect(describeFrame(unreadable)).toContain("modifier unreadable");
  });

  it("omits an absent modifier, which legitimately means RFS", () => {
    const absent = frame({
      material_condition: { value: null, state: "absent", confidence: 0, raw: null },
    });
    expect(describeFrame(absent)).not.toContain("unreadable");
    expect(describeFrame(absent)).toBe("Position | ⌀0.05 | A B");
  });

  it("names an unrecognised control rather than leaving a blank", () => {
    expect(describeFrame(frame({ symbol: null }))).toContain(
      "Unrecognised control"
    );
  });

  it("never claims a unit the drawing did not state", () => {
    expect(toleranceUnitLabel(frame())).toBe("in the drawing's units");
    expect(toleranceUnitLabel(frame())).not.toContain("mm");
  });
});

describe("review triage", () => {
  it("surfaces ambiguous detections and ones carrying warnings", () => {
    const result = pageResult({
      symbols: [
        detection({ status: GDT_CONFIRMED }),
        detection({ status: GDT_AMBIGUOUS, confidence: 0.5 }),
        detection({ status: GDT_CONFIRMED, warnings: ["retired in 2018"] }),
      ],
    });
    expect(needsReview(result)).toHaveLength(2);
  });

  it("orders the review list by confidence", () => {
    const result = pageResult({
      symbols: [
        detection({ status: GDT_AMBIGUOUS, confidence: 0.4 }),
        detection({ status: GDT_AMBIGUOUS, confidence: 0.8 }),
      ],
    });
    expect(needsReview(result).map((d) => d.confidence)).toEqual([0.8, 0.4]);
  });

  it("prefers the recognizer's own warning as the reason", () => {
    const reason = reviewReason(
      detection({ status: GDT_AMBIGUOUS, warnings: ["same glyph as Spotface"] })
    );
    expect(reason).toBe("same glyph as Spotface");
  });

  it("names the alternatives when there is no warning", () => {
    const reason = reviewReason(
      detection({
        status: GDT_AMBIGUOUS,
        alternatives: [
          { symbol_id: "circularity", name: "Circularity", confidence: 0.6, method: "t" },
        ],
      })
    );
    expect(reason).toContain("Circularity");
  });

  it("gives a settled reading no review reason", () => {
    expect(reviewReason(detection())).toBeNull();
  });

  it("groups detections by category", () => {
    const groups = groupByCategory([
      detection({ category: "datum" }),
      detection({ category: "geometric_tolerance" }),
      detection({ category: "datum" }),
    ]);
    expect(groups.map((g) => g.category)).toEqual([
      "datum",
      "geometric_tolerance",
    ]);
    expect(groups[0].items).toHaveLength(2);
  });
});

describe("hit testing", () => {
  it("matches a detection whose centre is inside the box", () => {
    expect(
      withinBox(detection(), { x: 100, y: 200, width: 100, height: 100 })
    ).toBe(true);
  });

  it("rejects one outside", () => {
    expect(
      withinBox(detection(), { x: 0, y: 0, width: 50, height: 50 })
    ).toBe(false);
  });
});

describe("shifting a region result into page coordinates", () => {
  it("moves every kind of box by the same offset", () => {
    const result = pageResult({
      frames: [frame()],
      symbols: [detection()],
      datum_associations: [
        {
          frame_id: "frame-1",
          label: "A",
          precedence: 1,
          material_condition: null,
          resolved: true,
          datum_feature_bbox: { x: 10, y: 10, width: 5, height: 5 },
          note: null,
        },
      ],
    });
    const shifted = shiftResult(result, 1000, 500);
    expect(shifted.symbols[0].bounding_box.x).toBe(1120);
    expect(shifted.symbols[0].bounding_box.y).toBe(740);
    expect(shifted.frames[0].bounding_box.x).toBe(1100);
    expect(shifted.frames[0].symbol?.bounding_box.x).toBe(1120);
    expect(shifted.datum_associations[0].datum_feature_bbox?.x).toBe(1010);
  });

  it("leaves sizes alone", () => {
    const result = pageResult({ symbols: [detection()] });
    const shifted = shiftResult(result, 50, 50);
    expect(shifted.symbols[0].bounding_box.width).toBe(20);
    expect(shifted.symbols[0].bounding_box.height).toBe(20);
  });

  it("is a no-op at the origin", () => {
    const result = pageResult({ symbols: [detection()] });
    expect(shiftResult(result, 0, 0)).toBe(result);
  });

  it("does not mutate the input", () => {
    const result = pageResult({ symbols: [detection()] });
    shiftResult(result, 100, 100);
    expect(result.symbols[0].bounding_box.x).toBe(120);
  });
});
