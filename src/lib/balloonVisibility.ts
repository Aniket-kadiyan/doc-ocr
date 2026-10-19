import type { Annotation } from "@/types/annotation";

type BalloonVisibilityRecord = Pick<Annotation, "hidden" | "kind">;

const valueBalloons = (annotations: BalloonVisibilityRecord[]) =>
  annotations.filter((annotation) => annotation.kind !== "label");

/** True only when at least one value balloon exists and every value is shown. */
export function areAllValueBalloonsVisible(
  annotations: BalloonVisibilityRecord[]
): boolean {
  const values = valueBalloons(annotations);
  return values.length > 0 && values.every((annotation) => !annotation.hidden);
}

/**
 * True only when every value balloon is hidden. The viewer uses this state for
 * the declutter mode that covers accepted value regions with white rectangles.
 */
export function areAllValueBalloonsHidden(
  annotations: BalloonVisibilityRecord[]
): boolean {
  const values = valueBalloons(annotations);
  return values.length > 0 && values.every((annotation) => annotation.hidden);
}
