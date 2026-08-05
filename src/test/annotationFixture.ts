import type { Annotation } from "@/types/annotation";

/** Small valid value annotation for pure frontend regression tests. */
export function makeAnnotation(
  overrides: Partial<Annotation> = {}
): Annotation {
  const base: Annotation = {
    id: "value-1",
    number: 1,
    label: "",
    value: "25",
    type: "Linear",
    confidence: 0.95,
    bbox: { x: 10, y: 20, width: 30, height: 12 },
    rotation: 0,
    page: 1,
    createdAt: 1,
    kind: "dimension",
    range: "0",
  };

  return {
    ...base,
    ...overrides,
    bbox: overrides.bbox ?? base.bbox,
  };
}
