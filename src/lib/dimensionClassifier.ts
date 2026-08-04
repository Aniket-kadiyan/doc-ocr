import type { DimensionType } from "@/types/annotation";

export function classifyDimension(text: string): DimensionType {
  const t = text.trim();

  if (/[Øø]/.test(t)) return "Diameter";
  if (/\bR\d/.test(t) || /^R[\d.]/.test(t)) return "Radius";
  if (/°|['"]/.test(t) && /\d/.test(t)) return "Angle";
  if (/±/.test(t)) return "Tolerance";
  if (/^\d+(\.\d+)?/.test(t)) return "Linear";
  if (t.length > 0) return "Note";

  return "Unknown";
}
