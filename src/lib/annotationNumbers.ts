import type { Annotation } from "@/types/annotation";

/**
 * Keep visible value balloons numbered contiguously without changing their
 * stable ids or the array's rendering order. Existing balloon numbers define
 * the order; the original array position breaks ties in malformed/legacy data.
 */
export function renumberValueAnnotations(
  annotations: Annotation[]
): Annotation[] {
  const orderedValues = annotations
    .map((annotation, index) => ({ annotation, index }))
    .filter(({ annotation }) => annotation.kind !== "label")
    .sort((left, right) => {
      const leftNumber = Number.isFinite(left.annotation.number)
        ? left.annotation.number
        : Number.MAX_SAFE_INTEGER;
      const rightNumber = Number.isFinite(right.annotation.number)
        ? right.annotation.number
        : Number.MAX_SAFE_INTEGER;
      return leftNumber - rightNumber || left.index - right.index;
    });

  const numberById = new Map(
    orderedValues.map(({ annotation }, index) => [annotation.id, index + 1])
  );

  let changed = false;
  const renumbered = annotations.map((annotation) => {
    if (annotation.kind === "label") return annotation;
    const number = numberById.get(annotation.id);
    if (number == null || number === annotation.number) return annotation;
    changed = true;
    return { ...annotation, number };
  });

  return changed ? renumbered : annotations;
}
