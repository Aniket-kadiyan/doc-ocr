import { describe, expect, it } from "vitest";
import { overlayDescribesPage } from "@/lib/scanOverlay";

// A 250dpi A3 scan mapped back to the 1.5x viewer render.
const PAGE = { width: 1263, height: 893 };

describe("placing scan overlay geometry", () => {
  it("accepts geometry measured on the drawing itself", () => {
    expect(
      overlayDescribesPage({ pageWidth: 1263, pageHeight: 893 }, PAGE)
    ).toBe(true);
    expect(
      overlayDescribesPage({ pageWidth: 1261.4, pageHeight: 894.2 }, PAGE)
    ).toBe(true);
  });

  it("rejects a section crop reported as page geometry", () => {
    expect(
      overlayDescribesPage({ pageWidth: 452, pageHeight: 318 }, PAGE)
    ).toBe(false);
    expect(
      overlayDescribesPage({ pageWidth: 1200, pageHeight: 850 }, PAGE)
    ).toBe(false);
  });

  it("rejects missing overlay or page measurements", () => {
    expect(
      overlayDescribesPage({ pageWidth: 0, pageHeight: 0 }, PAGE)
    ).toBe(false);
    expect(
      overlayDescribesPage(
        { pageWidth: 1263, pageHeight: 893 },
        { width: 0, height: 0 }
      )
    ).toBe(false);
  });
});
