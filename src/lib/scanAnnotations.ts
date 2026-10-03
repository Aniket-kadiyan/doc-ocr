import { v4 as uuidv4 } from "uuid";
import { classifyDimension } from "@/lib/dimensionClassifier";
import type { SegmentRegion } from "@/lib/paddleOcrClient";
import { deriveRange } from "@/lib/valueFields";
import type { Annotation, DimensionType } from "@/types/annotation";

/**
 * Turn accepted scan regions into balloons.
 *
 * Shared by every route that produces callouts — the whole-page scan, a
 * section scan, and reading a vector PDF's text layer — so a balloon carries
 * the same fields however it was found.
 */
export function annotationsFromRegions(
  regions: SegmentRegion[],
  page: number,
  createdAt: number = Date.now()
): Annotation[] {
  return regions.map((region) => {
    const value = region.text.trim();
    // Prefer the backend rule engine, which also sees the detected symbols
    // and geometry; fall back to the local text-only rules.
    const type = ((region.category as DimensionType) ||
      classifyDimension(value)) as DimensionType;
    return {
      id: uuidv4(),
      // number is assigned sequentially by addAnnotations
      number: 0,
      label: region.label?.trim() || "",
      value,
      type,
      confidence: region.confidence,
      bbox: region.valueBox,
      rotation: region.rotation,
      page,
      createdAt,
      kind: "dimension" as const,
      needsReview: region.needsReview || !region.recognized,
      range: deriveRange(value) || undefined,
    };
  });
}
