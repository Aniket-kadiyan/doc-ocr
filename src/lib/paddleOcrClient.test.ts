import { describe, expect, it } from "vitest";
import {
  mapPageSegmentRegions,
  mapSegmentCandidateOutcomes,
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

  it("maps excluded outcomes with raw OCR and filter evidence intact", () => {
    const [mapped] = mapSegmentCandidateOutcomes(
      [
        {
          candidate_id: "C0043",
          bbox: { x: 260, y: 480, width: 100, height: 24 },
          state: "excluded",
          text: "REV A",
          raw_text: "REV 4",
          preliminary_text: "REY 4",
          confidence: 0.74,
          recognized: true,
          reason: "Title-block text is not a drawing value",
          rule: "title_block",
          category: "Title Block",
          orientation: "horizontal",
          rotation: 0,
          authoritative_reread: true,
        },
      ],
      { x: 0, y: 0, width: 1000, height: 700 },
      2,
      "page"
    );

    expect(mapped).toMatchObject({
      candidateId: "C0043",
      outcomeState: "excluded",
      text: "REV A",
      rawText: "REV 4",
      preliminaryText: "REY 4",
      outcomeRule: "title_block",
      category: "Title Block",
      authoritativeReread: true,
      valueBox: { x: 130, y: 240, width: 50, height: 12 },
    });
  });

  it("maps native PDF and OCR provenance on candidate outcomes", () => {
    const [mapped] = mapSegmentCandidateOutcomes(
      [
        {
          candidate_id: "C0044",
          bbox: { x: 260, y: 480, width: 140, height: 28 },
          state: "review",
          text: "R1±0.26",
          confidence: 0.93,
          recognition_source: "ocr",
          source_conflict: true,
          recognition_evidence: {
            selected_source: "ocr",
            sources: ["native_pdf", "ocr"],
            native_text: "R1±0.25",
            ocr_text: "R1±0.26",
            agreement: 0.875,
            conflict: true,
            native_span_ids: ["P1:S00017"],
            native_bbox: { x: 260, y: 480, width: 140, height: 28 },
          },
        },
      ],
      { x: 0, y: 0, width: 1000, height: 700 },
      2,
      "page"
    );

    expect(mapped.recognitionSource).toBe("ocr");
    expect(mapped.sourceConflict).toBe(true);
    expect(mapped.recognitionEvidence).toEqual({
      selectedSource: "ocr",
      sources: ["native_pdf", "ocr"],
      nativeText: "R1±0.25",
      ocrText: "R1±0.26",
      agreement: 0.875,
      conflict: true,
      nativeSpanIds: ["P1:S00017"],
      nativeBBox: { x: 260, y: 480, width: 140, height: 28 },
    });
  });

  it("maps every child detection retained by an assembled object", () => {
    const [mapped] = mapSegmentCandidateOutcomes(
      [
        {
          candidate_id: "C0002",
          object_id: "C0002",
          bbox: { x: 20, y: 40, width: 180, height: 40 },
          state: "review",
          text: "4X Ø10 THRU",
          reason: "Confirm assembled callout",
          assembly_id: "C0002",
          assembly_rule: "inline_prefix+inline_suffix",
          assembly_conflict: false,
          assembly_children: [
            {
              candidate_id: "C0001",
              text: "4X",
              bbox: { x: 20, y: 40, width: 30, height: 24 },
              confidence: 0.91,
              role: "multiplier",
            },
            {
              candidate_id: "C0002",
              text: "Ø10",
              bbox: { x: 60, y: 40, width: 50, height: 24 },
              confidence: 0.96,
              role: "base",
            },
          ],
        },
      ],
      { x: 0, y: 0, width: 1000, height: 700 },
      2,
      "page"
    );

    expect(mapped.assembly).toMatchObject({
      objectId: "C0002",
      rule: "inline_prefix+inline_suffix",
      conflict: false,
    });
    expect(mapped.assembly?.children).toHaveLength(2);
    expect(mapped.assembly?.children[0].bbox).toEqual({
      x: 10,
      y: 20,
      width: 15,
      height: 12,
    });
  });
});
