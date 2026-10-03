import { describe, expect, it } from "vitest";
import { overlayDescribesPage } from "@/lib/scanOverlay";

// A 250dpi scan of an A3 sheet, brought back to the 1.5x viewer render.
const PAGE = { width: 1263, height: 893 };

describe("placing scan overlay geometry", () => {
  it("accepts an overlay measured on the drawing itself", () => {
    expect(
      overlayDescribesPage({ pageWidth: 1263, pageHeight: 893 }, PAGE)
    ).toBe(true);
    // Rounding through the scan render and back moves an edge a little.
    expect(
      overlayDescribesPage({ pageWidth: 1261.4, pageHeight: 894.2 }, PAGE)
    ).toBe(true);
  });

  it("rejects a section crop reported as if it were the page", () => {
    // The selection the person drew, cropped and padded by the backend. Its
    // coordinates start at zero, so drawing them on the page would put every
    // processing box in the top-left corner instead of on the selection.
    expect(
      overlayDescribesPage({ pageWidth: 452, pageHeight: 318 }, PAGE)
    ).toBe(false);
    // Even a crop that covers most of the sheet is not the sheet.
    expect(
      overlayDescribesPage({ pageWidth: 1200, pageHeight: 850 }, PAGE)
    ).toBe(false);
  });

  it("rejects an overlay with no measurements at all", () => {
    expect(overlayDescribesPage({ pageWidth: 0, pageHeight: 0 }, PAGE)).toBe(
      false
    );
    expect(
      overlayDescribesPage(
        { pageWidth: 1263, pageHeight: 893 },
        { width: 0, height: 0 }
      )
    ).toBe(false);
  });
});
