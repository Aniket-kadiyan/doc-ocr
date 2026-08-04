import type { DimensionType } from "@/types/annotation";

const SUGGESTIONS: Record<DimensionType, string[]> = {
  Diameter: ["Outer Diameter", "Inner Diameter", "Bore Diameter", "Hole Diameter"],
  Radius: ["Corner Radius", "Fillet Radius", "Blend Radius"],
  Angle: ["Chamfer Angle", "Bevel Angle", "Included Angle"],
  Tolerance: ["Dimensional Tolerance", "Fit Tolerance"],
  Linear: ["Length", "Width", "Depth", "Height", "Distance"],
  Note: ["General Note", "Surface Finish", "Material Spec"],
  Unknown: ["Dimension", "Feature"],
};

export function suggestLabel(type: DimensionType, value: string): string {
  const options = SUGGESTIONS[type];
  const base = options[0];

  if (type === "Diameter" && /inner|bore/i.test(value)) {
    return "Inner Diameter";
  }
  if (type === "Radius" && /fillet/i.test(value)) {
    return "Fillet Radius";
  }

  return base;
}
