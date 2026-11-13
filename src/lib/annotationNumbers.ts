import type { Annotation } from "@/types/annotation";
import { isNoteAnnotation } from "@/lib/notes";

/** Stable partition: drawing notes take the last balloon numbers, so they come
 * after every dimension on the drawing and in each export. */
function notesLast<T extends { annotation: Annotation }>(ordered: T[]): T[] {
  const notes = new Set(
    ordered.filter(({ annotation }) => isNoteAnnotation(annotation))
  );
  if (notes.size === 0) return ordered;
  return [
    ...ordered.filter((entry) => !notes.has(entry)),
    ...ordered.filter((entry) => notes.has(entry)),
  ];
}

/**
 * Keep visible value balloons numbered contiguously without changing their
 * stable ids or the array's rendering order. Existing balloon numbers define
 * the order; the original array position breaks ties in malformed/legacy data.
 * Notes always follow the dimensions.
 */
export function renumberValueAnnotations(
  annotations: Annotation[]
): Annotation[] {
  const orderedValues = notesLast(
    annotations
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
      })
  );

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

/**
 * Move one value balloon to a new 1-based number and shift only the values in
 * between. Stable annotation ids, array order, and all non-number metadata are
 * preserved. Notes stay after the dimensions, so a move across that boundary
 * lands on the nearest number on its own side. Legacy label records do not participate in balloon numbering.
 */
export function moveValueAnnotationToNumber(
  annotations: Annotation[],
  id: string,
  targetNumber: number
): Annotation[] {
  const orderedValues = annotations
    .map((annotation, index) => ({ annotation, index }))
    .filter(({ annotation }) => annotation.kind !== "label")
    .sort((left, right) => {
      const numberDifference =
        left.annotation.number - right.annotation.number;
      return numberDifference || left.index - right.index;
    });

  if (
    !Number.isInteger(targetNumber) ||
    targetNumber < 1 ||
    targetNumber > orderedValues.length
  ) {
    return annotations;
  }

  const currentIndex = orderedValues.findIndex(
    ({ annotation }) => annotation.id === id
  );
  const targetIndex = targetNumber - 1;
  if (currentIndex < 0 || currentIndex === targetIndex) return annotations;

  const spliced = [...orderedValues];
  const [moved] = spliced.splice(currentIndex, 1);
  spliced.splice(targetIndex, 0, moved);
  const reordered = notesLast(spliced);

  const numberById = new Map(
    reordered.map(({ annotation }, index) => [annotation.id, index + 1])
  );

  return annotations.map((annotation) => {
    if (annotation.kind === "label") return annotation;
    const number = numberById.get(annotation.id);
    return number == null || number === annotation.number
      ? annotation
      : { ...annotation, number };
  });
}
