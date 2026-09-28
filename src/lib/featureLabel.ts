import type { Annotation, DimensionType } from "@/types/annotation";

/**
 * The balloon label for a feature category.
 *
 * Mirrors the backend rule engine (`feature_classifier._label_for`), which
 * derives the label from the category it assigned, so the value here is the
 * one a scan would write. Only the parts that need the detected symbols —
 * a hole's subtype, a thread's designation, a "4X" quantity — cannot be
 * reproduced from the category alone, and those arrive with the scan.
 */
const LABEL_BY_TYPE: Partial<Record<DimensionType, string>> = {
  Diameter: "Diameter",
  Radius: "Radius",
  Chamfer: "Chamfer",
  Angle: "Angle",
  Taper: "Taper",
  Linear: "Linear Dimension",
  Tolerance: "Toleranced Dimension",
  Thread: "Thread",
  Hole: "Hole",
  "GD&T": "Feature Control Frame",
  Datum: "Datum",
  "Surface Finish": "Surface Finish",
  Weld: "Weld",
  Material: "Material",
  "Heat Treatment": "Heat Treatment",
  Coating: "Coating",
  "General Note": "General Note",
  Note: "General Note",
  Reference: "Reference Dimension",
  Basic: "Basic Dimension",
};

/** The rule engine's catch-all for a category it could not name. */
const UNCLASSIFIED_LABEL = "Feature";

export function labelForDimensionType(type: string): string {
  return LABEL_BY_TYPE[type as DimensionType] ?? UNCLASSIFIED_LABEL;
}

/**
 * The label to show for one balloon on the inspection sheet or checksheet.
 *
 * A stored label wins: it is either the rule engine's own, written by the
 * scan, or one a person typed. Balloons drawn by hand never had one, and so
 * did every balloon scanned before the page route began publishing the rule
 * engine's label — those would otherwise show an empty Label column for a
 * whole drawing. They fall back to the label for their category, which is
 * what a rescan would write.
 */
export function balloonLabel(
  annotation: Pick<Annotation, "label" | "type">
): string {
  return (annotation.label ?? "").trim() || labelForDimensionType(annotation.type);
}
