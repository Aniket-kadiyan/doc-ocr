import { describe, expect, it } from "vitest";
import { isNoteText, splitNotePoints } from "@/lib/notes";

// 47630.pdf, as the scan publishes it: one region, the drawing's own line
// breaks kept. A checksheet row carries this same string.
const NOTES_LINES = [
  "1. MATERIAL: GRADE 5 EQUIV; 85KPSI YIELD MIN.;120KPSI TENSILE MIN.",
  `2. PIN TO BE STRAIGHT WITHIN .004" OF ENTIRE LENGTH`,
  "3. PIN TO ASSEMBLED WITH CABLE AND HAIR PIN COTTER AS SHOWN",
  "4. FINISH: ZINC YELLOW PER ASTM B633-LATEST REV. TYPE II",
  "5. LANYARD RING MUST WITHSTAND 50 LBS MIN. WITHOUT OPENING.",
];

describe("recognising a notes block", () => {
  it("accepts the block by its assigned type", () => {
    expect(isNoteText("General Note", NOTES_LINES.join("\n"))).toBe(true);
    expect(isNoteText("Note", NOTES_LINES.join("\n"))).toBe(true);
  });

  it("accepts prose the classifier mislabelled, by its numbered points", () => {
    // The keyword rules call a block opening "1. MATERIAL:…" a Material.
    expect(isNoteText("Material", NOTES_LINES.join("\n"))).toBe(true);
  });

  it("rejects dimensions, including one that reads like a marker", () => {
    expect(isNoteText("Linear", "1.13[28.70]REF.")).toBe(false);
    expect(isNoteText("Diameter", "Ø0.620/0.612[15.75/15.54]")).toBe(false);
    expect(isNoteText("Linear", "1. PIN TO BE STRAIGHT")).toBe(false);
  });
});

describe("splitting a notes block into points", () => {
  it("returns one entry per numbered point, numbering kept", () => {
    expect(splitNotePoints(NOTES_LINES.join("\n"))).toEqual(NOTES_LINES);
  });

  it("splits the same block when it arrives as a single line", () => {
    expect(splitNotePoints(NOTES_LINES.join(" "))).toEqual(NOTES_LINES);
  });

  it("joins a wrapped continuation onto the point above it", () => {
    const wrapped = [
      "1. MAT'L GRADE 65-45-12 DUCTILE IRON PER ASTM A536 (LATEST",
      "REVISION). MATERIAL CERTIFICATION IS REQUIRED",
      "2. HOT DIP GALVANIZE PER ASTM A153",
    ].join("\n");

    expect(splitNotePoints(wrapped)).toEqual([
      "1. MAT'L GRADE 65-45-12 DUCTILE IRON PER ASTM A536 (LATEST REVISION). " +
        "MATERIAL CERTIFICATION IS REQUIRED",
      "2. HOT DIP GALVANIZE PER ASTM A153",
    ]);
  });
});
