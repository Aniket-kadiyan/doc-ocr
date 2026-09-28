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

  it("brings a higher-resolution page scan back to viewer coordinates", () => {
    // The scan may read a 250dpi render while the viewer shows 1.5x, so the
    // backend returns page coordinates in those larger pixels. Placing them
    // unscaled put every balloon about 2.3x too far right and down, off the
    // drawing entirely.
    const scale = 250 / 72 / 1.5;
    const [mapped] = mapPageSegmentRegions(
      [
        {
          bbox: {
            x: 130 * scale,
            y: 240 * scale,
            width: 50 * scale,
            height: 12 * scale,
          },
          text: "25",
          confidence: 0.9,
        },
      ],
      { x: 0, y: 0, width: 1000, height: 700 },
      scale
    );

    expect(mapped.valueBox.x).toBeCloseTo(130, 6);
    expect(mapped.valueBox.y).toBeCloseTo(240, 6);
    expect(mapped.valueBox.width).toBeCloseTo(50, 6);
    expect(mapped.valueBox.height).toBeCloseTo(12, 6);
  });

  it("judges bounds in viewer space, not scanned-image space", () => {
    // A region beyond the page must still fall back to the page rectangle
    // once scaled; checking hi-res coordinates against hi-res bounds would
    // wrongly accept it.
    const [mapped] = mapPageSegmentRegions(
      [
        {
          // 2400/2 = 1200, past the 1000-wide page once scaled.
          bbox: { x: 2400, y: 100, width: 50, height: 12 },
          text: "25",
          confidence: 0.9,
        },
      ],
      { x: 0, y: 0, width: 1000, height: 700 },
      2
    );

    expect(mapped.valueBox).toEqual({ x: 0, y: 0, width: 1000, height: 700 });
  });

  it("preserves post-scan review identity and reason", () => {
    const [mapped] = mapPageSegmentRegions(
      [
        {
          candidate_id: "C0042",
          bbox: { x: 130, y: 240, width: 50, height: 12 },
          text: "R5.0",
          confidence: 0.61,
          needs_review: true,
          review_reason: "Recovery reads did not reach a stable consensus",
          recovery_attempted: true,
        },
      ],
      { x: 0, y: 0, width: 1000, height: 700 }
    );

    expect(mapped.candidateId).toBe("C0042");
    expect(mapped.needsReview).toBe(true);
    expect(mapped.reviewReason).toContain("stable consensus");
    expect(mapped.recoveryAttempted).toBe(true);
  });
});
