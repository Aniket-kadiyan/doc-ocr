/**
 * Shared helpers for editing a dimension value's fields — used by the
 * create-time popup (AnnotationPopup) and the value metadata editor.
 */

/** Engineering/GD&T symbols often missed by OCR — click to insert into the value. */
export const SPECIAL_SYMBOLS: { ch: string; title: string }[] = [
  { ch: "Ø", title: "Diameter" },
  { ch: "⌀", title: "Diameter (GD&T)" },
  { ch: "φ", title: "Phi" },
  { ch: "±", title: "Plus / minus tolerance" },
  { ch: "°", title: "Degree" },
  { ch: "×", title: "Times / by" },
  { ch: "□", title: "Square" },
  { ch: "R", title: "Radius" },
  { ch: "∠", title: "Angle" },
  { ch: "⊥", title: "Perpendicular" },
  { ch: "∥", title: "Parallel" },
  { ch: "⌖", title: "Position" },
  { ch: "√", title: "Square root" },
  { ch: "′", title: "Minute / foot" },
  { ch: "″", title: "Second / inch" },
];

/** Pull the tolerance after a ± and format it as "+x, -x" (symmetric). Empty
 * when the value has no ± tolerance. */
export function deriveRange(value: string): string {
  const idx = value.indexOf("±");
  if (idx === -1) return "";
  const after = value.slice(idx + 1);
  const m = after.match(/\d*\.?\d+/);
  return m ? `+${m[0]}, -${m[0]}` : "";
}
