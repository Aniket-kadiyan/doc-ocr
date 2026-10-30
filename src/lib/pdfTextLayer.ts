import {
  getOcrApiUrl,
  mapPageSegmentRegions,
  type SegmentRegion,
  type TitleField,
} from "@/lib/paddleOcrClient";
import { PDF_RENDER_SCALE } from "@/lib/pdfLoader";
import { isGdtScanEnabled } from "@/lib/scanSettings";
import { getTitleKeywords } from "@/lib/titleKeywords";
import { fetchWithRequestId } from "@/lib/requestDiagnostics";
import type { Annotation, BBox } from "@/types/annotation";

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
  /** The configured title-block fields, read from the same text layer. */
  titleFields: TitleField[];
}

const UNUSABLE: PdfTextLayerResult = {
  usable: false,
  regions: [],
  fonts: [],
  titleFields: [],
};

/**
 * Read one page's callouts, or report that the page has no usable text.
 *
 * Boxes come back in the viewer's own pixels, because the backend is told
 * which raster scale the annotations use. Never throws: a backend without
 * this route, or a file it cannot parse, is just a page to scan instead.
 *
 * `refine: false` is the fast read, well under a second: the text itself,
 * without the per-item OCR that recovers drawn marks and reads feature
 * control frames. It also carries the title-block fields. The refined read
 * is the complete one, and {@link planTextLayerRefinement} folds it into
 * what the fast read ballooned.
 */
export async function readPdfTextLayer(
  file: Blob,
  page: number,
  pageBounds: BBox,
  { refine = true }: { refine?: boolean } = {}
): Promise<PdfTextLayerResult> {
  const form = new FormData();
  form.append("file", file, "drawing.pdf");
  form.append("page", String(page));
  form.append("render_scale", String(PDF_RENDER_SCALE));
  form.append("refine", String(refine));
  form.append("gdt", String(isGdtScanEnabled()));
  if (!refine) form.append("keywords", getTitleKeywords().join(","));

  let response: Response;
  try {
    response = await fetchWithRequestId(
      `${getOcrApiUrl()}/ocr/text-layer`,
      { method: "POST", body: form },
      { operation: "pdf.text-layer" }
    );
  } catch {
    return UNUSABLE;
  }
  if (!response.ok) return UNUSABLE;

  const payload = (await response.json()) as {
    usable?: boolean;
    regions?: Parameters<typeof mapPageSegmentRegions>[0];
    fonts?: string[];
    title_fields?: TitleField[];
  };
  if (!payload.usable || !payload.regions?.length) return UNUSABLE;

  return {
    usable: true,
    // Already in viewer pixels, so the scale is 1; this reuses the same
    // bounds check the scan route's regions go through.
    regions: mapPageSegmentRegions(payload.regions, pageBounds, 1),
    fonts: payload.fonts ?? [],
    titleFields: payload.title_fields ?? [],
  };
}

/** A balloon the fast read placed, as it was placed. */
export interface PlacedTextRegion {
  annotationId: string;
  value: string;
  bbox: BBox;
}

export interface TextLayerRefinement {
  /** Balloons whose value the refined read folded into something else. */
  remove: string[];
  /** Balloons the refined read reads differently ("45" → "1×45°"). */
  update: { id: string; region: SegmentRegion }[];
  /** What only the refined read found: frames and pasted-picture notes. */
  add: SegmentRegion[];
}

const sameBox = (left: BBox, right: BBox) =>
  left.x === right.x &&
  left.y === right.y &&
  left.width === right.width &&
  left.height === right.height;

/**
 * How the refined read changes the balloons the fast read placed.
 *
 * Regions are matched by `candidateId`, which the backend keeps the same in
 * both reads. A balloon the inspector has since edited, moved or deleted is
 * theirs: the refined read never touches it.
 */
export function planTextLayerRefinement(
  placed: ReadonlyMap<string, PlacedTextRegion>,
  fastIds: ReadonlySet<string>,
  refined: SegmentRegion[],
  live: Annotation[]
): TextLayerRefinement {
  const byId = new Map(
    refined.flatMap((region) =>
      region.candidateId ? [[region.candidateId, region] as const] : []
    )
  );
  const plan: TextLayerRefinement = { remove: [], update: [], add: [] };
  for (const [candidateId, balloon] of placed) {
    const annotation = live.find((item) => item.id === balloon.annotationId);
    if (
      !annotation ||
      annotation.value !== balloon.value ||
      !sameBox(annotation.bbox, balloon.bbox)
    ) {
      continue;
    }
    const region = byId.get(candidateId);
    if (!region) {
      plan.remove.push(annotation.id);
    } else if (
      region.text.trim() !== balloon.value ||
      !sameBox(region.valueBox, balloon.bbox)
    ) {
      plan.update.push({ id: annotation.id, region });
    }
  }
  plan.add = refined.filter(
    (region) => !region.candidateId || !fastIds.has(region.candidateId)
  );
  return plan;
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

