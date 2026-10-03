import type { ScanDebugOverlay } from "@/types/scanJob";

/**
 * Whether a scan overlay describes the drawing rather than some other image.
 *
 * A section scan crops the selection and runs the whole-page pipeline over
 * that crop, so its progress events describe the CROP: the same field names,
 * the same "page" scope, and coordinates starting at zero in the crop's own
 * top-left corner. Painted as page coordinates they pile up in the corner of
 * the drawing, nowhere near the section the person selected.
 *
 * The overlay carries the size of the image it measured, so the mismatch is
 * detectable: geometry is drawn only once an event arrives that is genuinely
 * about the page. The final event of a section scan is one of those, and it
 * carries the section's own candidates in page coordinates.
 */
export function overlayDescribesPage(
  overlay: Pick<ScanDebugOverlay, "pageWidth" | "pageHeight">,
  pageSize: { width: number; height: number }
): boolean {
  if (overlay.pageWidth <= 0 || overlay.pageHeight <= 0) return false;
  if (pageSize.width <= 0 || pageSize.height <= 0) return false;
  const off = (measured: number, actual: number) =>
    Math.abs(measured - actual) / actual;
  // Rounding through two coordinate systems moves an edge by a pixel or two.
  // A crop is smaller than the page by far more than this.
  return (
    off(overlay.pageWidth, pageSize.width) < 0.02 &&
    off(overlay.pageHeight, pageSize.height) < 0.02
  );
}
