import type { DimensionType } from "@/types/annotation";

/**
 * First-guess balloon labels per feature category. The backend rule engine
 * usually supplies a more specific label (e.g. "4X Through Hole"); these are
 * the fallback options offered in the label editor when it doesn't.
 */
const SUGGESTIONS: Record<DimensionType, string[]> = {
  Diameter: ["Diameter", "Outer Diameter", "Inner Diameter", "Bore Diameter", "Hole Diameter"],
  Radius: ["Radius", "Corner Radius", "Fillet Radius", "Blend Radius"],
  Chamfer: ["Chamfer", "Edge Chamfer"],
  Angle: ["Angle", "Included Angle", "Bevel Angle"],
  Taper: ["Taper"],
  Tolerance: ["Toleranced Dimension", "Dimensional Tolerance", "Fit Tolerance"],
  Linear: ["Linear Dimension", "Length", "Width", "Depth", "Height", "Distance"],
  Thread: ["Thread", "Tapped Hole"],
  Hole: ["Hole", "Through Hole", "Counterbore", "Countersink", "Spotface", "Tapped Hole"],
  "GD&T": ["Feature Control Frame", "Position", "Flatness", "Perpendicularity", "Runout"],
  Datum: ["Datum"],
  "Surface Finish": ["Surface Finish", "Roughness"],
  Weld: ["Weld", "Fillet Weld"],
  Material: ["Material"],
  "Heat Treatment": ["Heat Treatment", "Hardness"],
  Coating: ["Coating", "Plating"],
  "General Note": ["General Note"],
  Reference: ["Reference Dimension"],
  Basic: ["Basic Dimension"],
  Note: ["General Note", "Surface Finish", "Material Spec"],
  // A title-block field carries its own label from the drawing ("DWG NO.",
  // "REV"), so these are only a fallback for a hand-retyped one.
  "Title Block": ["Drawing Number", "Revision", "Title Block Field"],
  Unknown: ["Dimension", "Feature"],
};

export function suggestLabel(type: DimensionType, value: string): string {
  const options = SUGGESTIONS[type] ?? SUGGESTIONS.Unknown;
  const base = options[0];

  if (type === "Diameter" && /inner|bore/i.test(value)) return "Inner Diameter";
  if (type === "Radius" && /fillet/i.test(value)) return "Fillet Radius";
  if (type === "Hole" && /cbore|counterbore/i.test(value)) return "Counterbore";
  if (type === "Hole" && /csk|countersink/i.test(value)) return "Countersink";
  if (type === "Hole" && /thru|through/i.test(value)) return "Through Hole";

  return base;
}

/** Full option list for a category, used to populate the label dropdown. */
export function labelOptions(type: DimensionType): string[] {
  return SUGGESTIONS[type] ?? SUGGESTIONS.Unknown;
}
