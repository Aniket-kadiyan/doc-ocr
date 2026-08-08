import { describe, expect, it } from "vitest";
import {
  mapPageSegmentRegions,
  mapSegmentRegions,
} from "@/lib/paddleOcrClient";

describe("scan result coordinate mapping", () => {
  it("maps padded crop coordinates back to the base drawing", () => {
    const [mapped] = mapSegmentRegions(
      [
        {
          bbox: { x: 30, y: 40, width: 50, height: 12 },
          text: "25",
          confidence: 0.9,
        },
      ],
      { x: 100, y: 200, width: 300, height: 180 }
    );

    expect(mapped.valueBox).toEqual({
      x: 110,
      y: 220,
      width: 50,
      height: 12,
    });
  });

  it("falls back to the selected scope when backend coordinates are invalid", () => {
    const scope = { x: 100, y: 200, width: 300, height: 180 };
    const [mapped] = mapSegmentRegions(
      [
        {
          bbox: { x: 9999, y: 40, width: 50, height: 12 },
          text: "25",
          confidence: 0.9,
        },
      ],
      scope
    );

    expect(mapped.valueBox).toEqual(scope);
  });

  it("retains a detected object whose single OCR pass was unreadable", () => {
    const [mapped] = mapSegmentRegions(
      [
        {
          bbox: { x: 30, y: 40, width: 50, height: 12 },
          text: "",
          confidence: 0,
          recognized: false,
          needs_review: true,
        },
      ],
      { x: 100, y: 200, width: 300, height: 180 }
    );

    expect(mapped.text).toBe("");
    expect(mapped.recognized).toBe(false);
    expect(mapped.needsReview).toBe(true);
  });

  it("preserves the whole-page eligibility reason on accepted values", () => {
    const [mapped] = mapSegmentRegions(
      [
        {
          bbox: { x: 30, y: 40, width: 50, height: 12 },
          text: "M8",
          confidence: 0.95,
          page_filter_rule: "numeric_component",
          page_filter_reason:
            "Contains a numeric component and matches no exclusion",
        },
      ],
      { x: 100, y: 200, width: 300, height: 180 }
    );

    expect(mapped.pageFilterRule).toBe("numeric_component");
    expect(mapped.pageFilterReason).toContain("numeric component");
  });

  it("keeps full-page response coordinates without subtracting crop padding", () => {
    const [mapped] = mapPageSegmentRegions(
      [
        {
          bbox: { x: 130, y: 240, width: 50, height: 12 },
          text: "25",
          confidence: 0.9,
        },
      ],
      { x: 0, y: 0, width: 1000, height: 700 }
    );

    expect(mapped.valueBox).toEqual({
      x: 130,
      y: 240,
      width: 50,
      height: 12,
    });
  });
});
