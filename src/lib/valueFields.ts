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

const TOLERANCE_NUMBER = String.raw`(?:\d+(?:\.\d+)?|\.\d+)`;

function formatTolerance(value: number): string {
  const rounded = Math.round((value + Number.EPSILON) * 1_000_000) / 1_000_000;
  return rounded.toFixed(6).replace(/\.?0+$/, "");
}

function formatSymmetricTolerance(value: number): string {
  const formatted = formatTolerance(value);
  return `+${formatted}, -${formatted}`;
}

/** Convert a complete decimal/DMS tolerance to decimal degrees. */
function angularToleranceDegrees(text: string): number | null {
  const normalized = text
    .trim()
    .replace(/[˚⁰]/g, "°")
    .replace(/[’´]/g, "′")
    .replace(/[”ʺ]/g, "″");
  if (!normalized) return null;

  const decimalMatch = new RegExp(`^(${TOLERANCE_NUMBER})\\s*°?$`).exec(
    normalized
  );
  if (decimalMatch) return Number(decimalMatch[1]);

  const dmsMatch = new RegExp(
    `^(?:(${TOLERANCE_NUMBER})\\s*°\\s*)?` +
      `(?:(${TOLERANCE_NUMBER})\\s*[′'])?\\s*` +
      `(?:(${TOLERANCE_NUMBER})\\s*[″"])?$`
  ).exec(normalized);
  if (!dmsMatch || (!dmsMatch[2] && !dmsMatch[3])) return null;

  const degrees = Number(dmsMatch[1] ?? 0);
  const minutes = Number(dmsMatch[2] ?? 0);
  const seconds = Number(dmsMatch[3] ?? 0);
  if (minutes >= 60 || seconds >= 60) return null;
  return degrees + minutes / 60 + seconds / 3600;
}

/**
 * Derive the editable Tolerance field from Value.
 *
 * - complete ± decimal/DMS tolerances are normalized to decimal values;
 * - complete asymmetric tolerances are copied into the shared field;
 * - a numeric value without an explicit tolerance defaults to zero;
 * - malformed/incomplete tolerance intent stays empty so export validation can
 *   block it rather than silently treating it as zero.
 */
export function deriveRange(value: string): string {
  const idx = value.indexOf("±");
  if (idx !== -1) {
    const tolerance = angularToleranceDegrees(value.slice(idx + 1));
    return tolerance == null ? "" : formatSymmetricTolerance(tolerance);
  }

  const asymmetricMatch = new RegExp(
    `\\+\\s*(${TOLERANCE_NUMBER})\\s*°?\\s*,?\\s*` +
      `-\\s*(${TOLERANCE_NUMBER})\\s*°?\\s*$`
  ).exec(value);
  if (asymmetricMatch) {
    return `+${formatTolerance(Number(asymmetricMatch[1]))}, ` +
      `-${formatTolerance(Number(asymmetricMatch[2]))}`;
  }

  const nominal = new RegExp(`[-+]?${TOLERANCE_NUMBER}`).exec(value);
  if (!nominal || nominal.index == null) return "";

  const remainder = value.slice(nominal.index + nominal[0].length);
  if (/[±+-]/.test(remainder)) return "";
  if (/~|\b(?:max(?:imum)?|min(?:imum)?)\b/i.test(remainder)) return "";
  return "0";
}
