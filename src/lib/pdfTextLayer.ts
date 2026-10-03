import {
  getOcrApiUrl,
  mapPageSegmentRegions,
  type SegmentRegion,
} from "@/lib/paddleOcrClient";
import { PDF_RENDER_SCALE } from "@/lib/pdfLoader";
import type { BBox } from "@/types/annotation";

/**
 * Reading a drawing's callouts from the PDF's own text layer.
 *
 * A vector PDF already carries every value as text with exact glyph boxes,
 * so there is nothing to recognise: the values come back with no reading
 * error at all, and in milliseconds rather than the tens of seconds a scan
 * takes. Most drawings are scans with no text at all, so this reports
 * `usable: false` and the caller runs the ordinary scan instead.
 */
export interface PdfTextLayerResult {
  usable: boolean;
  regions: SegmentRegion[];
  /** Font names found on the page, for diagnosing an unexpected result. */
  fonts: string[];
}

const UNUSABLE: PdfTextLayerResult = { usable: false, regions: [], fonts: [] };

/**
 * Read one page's callouts, or report that the page has no usable text.
 *
 * Boxes come back in the viewer's own pixels, because the backend is told
 * which raster scale the annotations use. Never throws: a backend without
 * this route, or a file it cannot parse, is just a page to scan instead.
 */
export async function readPdfTextLayer(
  file: Blob,
  page: number,
  pageBounds: BBox
): Promise<PdfTextLayerResult> {
  const form = new FormData();
  form.append("file", file, "drawing.pdf");
  form.append("page", String(page));
  form.append("render_scale", String(PDF_RENDER_SCALE));

  let response: Response;
  try {
    response = await fetch(`${getOcrApiUrl()}/ocr/text-layer`, {
      method: "POST",
      body: form,
    });
  } catch {
    return UNUSABLE;
  }
  if (!response.ok) return UNUSABLE;

  const payload = (await response.json()) as {
    usable?: boolean;
    regions?: Parameters<typeof mapPageSegmentRegions>[0];
    fonts?: string[];
  };
  if (!payload.usable || !payload.regions?.length) return UNUSABLE;

  return {
    usable: true,
    // Already in viewer pixels, so the scale is 1; this reuses the same
    // bounds check the scan route's regions go through.
    regions: mapPageSegmentRegions(payload.regions, pageBounds, 1),
    fonts: payload.fonts ?? [],
  };
}

/** Keep only the callouts whose box lies inside a drawn selection. */
export function regionsWithin(
  regions: SegmentRegion[],
  scope: BBox
): SegmentRegion[] {
  return regions.filter((region) => {
    const box = region.valueBox;
    const centreX = box.x + box.width / 2;
    const centreY = box.y + box.height / 2;
    return (
      centreX >= scope.x &&
      centreY >= scope.y &&
      centreX <= scope.x + scope.width &&
      centreY <= scope.y + scope.height
    );
  });
}
