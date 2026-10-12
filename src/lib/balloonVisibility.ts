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
 * True only for the fully-hidden state produced by the global visibility
 * control. A mixed set means one or more balloons were hidden individually and
 * must not white out every accepted value on the drawing.
 */
export function areAllValueBalloonsHidden(
  annotations: BalloonVisibilityRecord[]
): boolean {
  const values = valueBalloons(annotations);
  return values.length > 0 && values.every((annotation) => annotation.hidden);
}
