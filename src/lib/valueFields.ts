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
 * The tolerance of a stacked deviation pair, "+0.2/0" or "+0.2/-0.1", as
 * "+0.2, -0" / "+0.2, -0.1"; "" when `value` does not end in one.
 *
 * Both routes publish a drawing's upper and lower deviation, set one over
 * the other, as "upper/lower" after the nominal ("Ø8+0.2/0", "21+1/0").
 * A lower deviation printed without a sign is taken as negative: a drawn
 * minus is lost from the text layer, and a positive lower limit equal to or
 * above a "+" upper one is no tolerance zone at all. A pair the "+a, -b"
 * field cannot hold (both limits below the nominal) is left unparsed, so
 * export validation still blocks it.
 */
export function stackedDeviationRange(value: string): string {
  const pair = new RegExp(
    `([+-]?)\\s*(${TOLERANCE_NUMBER})\\s*°?\\s*/\\s*` +
      `([+-]?)\\s*(${TOLERANCE_NUMBER})\\s*°?\\s*$`
  ).exec(value);
  if (!pair || pair.index == null) return "";
  // The pair qualifies a number: "1/1" on a sheet count is not a tolerance.
  if (!/\d\s*°?\s*$/.test(value.slice(0, pair.index))) return "";

  const upper = Number(pair[2]);
  const lower = Number(pair[4]);
  // An unsigned upper limit is a fraction ("3/8"), not a deviation.
  if (pair[1] !== "+" && upper !== 0) return "";
  if (pair[3] === "+" && lower !== 0) return "";
  return `+${formatTolerance(upper)}, -${formatTolerance(lower)}`;
}

/**
 * A single deviation after the nominal, "Ø37.65+0.01" or "R0.6-0.1", as
 * "+0.01, -0" / "+0, -0.1"; "" when `value` carries no such deviation.
 *
 * The other limit is the nominal itself, the reading project.ts already
 * gives a balloon whose Value carries one. A count printed after it
 * ("R0.3+0.05 (34x)") is not part of the tolerance.
 */
export function singleDeviationRange(value: string): string {
  const text = value.replace(/\s*\(\s*\d+\s*[x×]\s*\)\s*$/i, "");
  const deviation = new RegExp(
    `([+-])\\s*(${TOLERANCE_NUMBER})\\s*°?\\s*$`
  ).exec(text);
  if (!deviation || deviation.index == null) return "";
  const specification = text.slice(0, deviation.index);
  // It qualifies a number, and is the only sign after it: "25 +0.1 +0.2"
  // is malformed, and a negative nominal ("R-5") is not a deviation.
  if (!/\d\s*°?\s*$/.test(specification)) return "";
  const nominal = new RegExp(TOLERANCE_NUMBER).exec(specification);
  if (!nominal || nominal.index == null) return "";
  if (/[±+-]/.test(specification.slice(nominal.index + nominal[0].length))) {
    return "";
  }
  const amount = formatTolerance(Number(deviation[2]));
  return deviation[1] === "+" ? `+${amount}, -0` : `+0, -${amount}`;
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

  const stacked = stackedDeviationRange(value);
  if (stacked) return stacked;

  const single = singleDeviationRange(value);
  if (single) return single;

  const nominal = new RegExp(`[-+]?${TOLERANCE_NUMBER}`).exec(value);
  if (!nominal || nominal.index == null) return "";

  const remainder = value.slice(nominal.index + nominal[0].length);
  if (/[±+-]/.test(remainder)) return "";
  if (/~|\b(?:max(?:imum)?|min(?:imum)?)\b/i.test(remainder)) return "";
  return "0";
}
