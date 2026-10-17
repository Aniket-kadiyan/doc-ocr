/**
 * Pure helpers for presenting GD&T detections.
 *
 * Kept out of the components so the rules that decide what a reviewer is told
 * — which detections are worth painting, what a frame reads as in words, what
 * colour a confidence earns — are unit-testable without rendering anything.
 *
 * The guiding rule here is the same one the backend follows: never present an
 * uncertain reading as a settled one. An ambiguous detection is drawn and
 * described differently from a confirmed one, and a frame field that could not
 * be read says so rather than appearing blank.
 */

import type { BBox } from "@/types/annotation";
import {
  GDT_AMBIGUOUS,
  GDT_CONFIRMED,
  GDT_UNKNOWN,
  type GdtBoundingBox,
  type GdtDetection,
  type GdtFrame,
  type GdtOverlayItem,
  type GdtPageResult,
  type GdtStatus,
} from "@/types/gdt";

/** Backend boxes are `{x,y,width,height}` already, but typed separately. */
export function toBBox(box: GdtBoundingBox): BBox {
  return { x: box.x, y: box.y, width: box.width, height: box.height };
}

/**
 * Colour per status. Ambiguous is amber rather than a paler green: a reviewer
 * scanning the sheet must be able to tell "this needs you" from "this is
 * settled" without reading anything.
 */
export const STATUS_COLOR: Record<GdtStatus, string> = {
  [GDT_CONFIRMED]: "#16a34a",
  [GDT_AMBIGUOUS]: "#f59e0b",
  [GDT_UNKNOWN]: "#94a3b8",
};

export const STATUS_LABEL: Record<GdtStatus, string> = {
  [GDT_CONFIRMED]: "Recognised",
  [GDT_AMBIGUOUS]: "Needs review",
  [GDT_UNKNOWN]: "Unidentified",
};

/** Frames get their own colour so they read as containers, not as symbols. */
export const FRAME_COLOR = "#2563eb";

export function statusColor(status: GdtStatus): string {
  return STATUS_COLOR[status] ?? STATUS_COLOR[GDT_UNKNOWN];
}

/** Confidence as a percentage string, or "—" when there is none. */
export function formatConfidence(confidence: number | null | undefined): string {
  if (confidence === null || confidence === undefined || Number.isNaN(confidence)) {
    return "—";
  }
  return `${Math.round(confidence * 100)}%`;
}

/**
 * Items the overlay should paint for one page.
 *
 * Detections from other pages are dropped rather than painted at the same
 * coordinates on this one — a symbol at (120, 240) exists on every page.
 */
export function overlayItems(
  result: GdtPageResult | null,
  page: number
): GdtOverlayItem[] {
  if (!result || result.page !== page) return [];
  const framesById = new Map(result.frames.map((frame) => [frame.id, frame]));
  return result.symbols
    .filter((detection) => detection.page === page)
    .map((detection) => ({
      detection,
      bbox: toBBox(detection.bounding_box),
      frame: detection.frame_id ? framesById.get(detection.frame_id) ?? null : null,
    }));
}

export function framesForPage(
  result: GdtPageResult | null,
  page: number
): GdtFrame[] {
  if (!result || result.page !== page) return [];
  return result.frames.filter((frame) => frame.page === page);
}

/**
 * A frame as an inspector would write it: symbol, tolerance, datums.
 *
 * Unreadable fields are spelled out rather than omitted. A frame rendered as
 * "Position | ⌀ | A B" when the tolerance could not be read would be read as
 * a frame with no tolerance, which is a different and much weaker statement
 * than "the tolerance is there and we could not read it".
 */
export function describeFrame(frame: GdtFrame): string {
  const parts: string[] = [];
  parts.push(frame.symbol?.name ?? "Unrecognised control");

  if (frame.tolerance.state === "present") {
    const zone = frame.diameter_zone.value ? "⌀" : "";
    parts.push(`${zone}${frame.tolerance.value}`);
  } else if (frame.tolerance.state === "unreadable") {
    parts.push("tolerance unreadable");
  }

  if (frame.material_condition.state === "present") {
    parts.push(String(frame.material_condition.value).toUpperCase());
  } else if (frame.material_condition.state === "unreadable") {
    parts.push("modifier unreadable");
  }

  if (frame.datums.length > 0) {
    const datums = [...frame.datums]
      .sort((a, b) => a.precedence - b.precedence)
      .map((datum) =>
        datum.material_condition
          ? `${datum.label}(${datum.material_condition.toUpperCase()})`
          : datum.label
      )
      .join(" ");
    parts.push(datums);
  }

  return parts.join(" | ");
}

/**
 * The unit a tolerance is in, in words.
 *
 * Always the drawing's own. Nothing in OCR establishes whether a sheet is in
 * millimetres or inches, and printing a unit that was never read would be an
 * invention an inspector might act on.
 */
export function toleranceUnitLabel(frame: GdtFrame): string {
  return frame.tolerance_unit === "drawing_unit"
    ? "in the drawing's units"
    : frame.tolerance_unit;
}

/** Detections a reviewer should look at, strongest claim first. */
export function needsReview(result: GdtPageResult | null): GdtDetection[] {
  if (!result) return [];
  return result.symbols
    .filter(
      (detection) =>
        detection.status === GDT_AMBIGUOUS || detection.warnings.length > 0
    )
    .sort((a, b) => b.confidence - a.confidence);
}

/** Group detections by category for a sectioned list. */
export function groupByCategory(
  detections: GdtDetection[]
): Array<{ category: string; items: GdtDetection[] }> {
  const groups = new Map<string, GdtDetection[]>();
  for (const detection of detections) {
    const key = detection.category ?? "unknown";
    const bucket = groups.get(key);
    if (bucket) bucket.push(detection);
    else groups.set(key, [detection]);
  }
  return [...groups.entries()]
    .map(([category, items]) => ({ category, items }))
    .sort((a, b) => a.category.localeCompare(b.category));
}

/** "geometric_tolerance" → "Geometric tolerance". */
export function categoryLabel(category: string): string {
  const words = category.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * The one-line reason a detection is not settled, or null when it is.
 *
 * Prefers the recognizer's own warning over a generic phrasing, because the
 * warnings say something specific and useful — that two symbols share a glyph,
 * that a control was retired in a revision, that a modifier is too small to
 * resolve.
 */
export function reviewReason(detection: GdtDetection): string | null {
  if (detection.warnings.length > 0) return detection.warnings[0];
  if (detection.status === GDT_AMBIGUOUS) {
    const names = detection.alternatives.map((a) => a.name).join(", ");
    return names
      ? `Could also be ${names}`
      : "The evidence does not settle this reading";
  }
  if (detection.status === GDT_UNKNOWN) {
    return "No known symbol explains this mark";
  }
  return null;
}

/**
 * Does this detection sit inside the given box?
 *
 * Used to tie a detection to an annotation a user already drew, so a symbol
 * found inside a ballooned value can be shown against it.
 */
export function withinBox(detection: GdtDetection, box: BBox): boolean {
  const { x, y, width, height } = detection.bounding_box;
  const cx = x + width / 2;
  const cy = y + height / 2;
  return (
    cx >= box.x &&
    cx <= box.x + box.width &&
    cy >= box.y &&
    cy <= box.y + box.height
  );
}

/** A short, stable key for React lists. */
export function detectionKey(detection: GdtDetection, index: number): string {
  const { x, y } = detection.bounding_box;
  return `${detection.symbol_id ?? "unknown"}-${Math.round(x)}-${Math.round(
    y
  )}-${index}`;
}
