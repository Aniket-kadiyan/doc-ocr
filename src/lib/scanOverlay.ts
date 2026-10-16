import type { ScanDebugOverlay } from "@/types/scanJob";

/**
 * Whether a scan overlay describes the drawing rather than an internal crop.
 *
 * Section OCR runs the page pipeline over the selected crop. Intermediate
 * progress events therefore use crop-local coordinates starting at (0, 0),
 * even though their fields look like page geometry. Painting those events on
 * the drawing moves the boxes into its upper-left corner.
 *
 * The backend includes the measured image size with every overlay. A valid
 * page-space event closely matches the viewer page size; a crop-local event
 * does not. The small tolerance only accommodates coordinate rounding.
 */
export function overlayDescribesPage(
  overlay: Pick<ScanDebugOverlay, "pageWidth" | "pageHeight">,
  pageSize: { width: number; height: number }
): boolean {
  if (overlay.pageWidth <= 0 || overlay.pageHeight <= 0) return false;
  if (pageSize.width <= 0 || pageSize.height <= 0) return false;

  const relativeDifference = (measured: number, actual: number) =>
    Math.abs(measured - actual) / actual;

  return (
    relativeDifference(overlay.pageWidth, pageSize.width) < 0.02 &&
    relativeDifference(overlay.pageHeight, pageSize.height) < 0.02
  );
}
