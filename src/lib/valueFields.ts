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

  // Some PDF text layers drop the minute mark. After a degree mark, the
  // following whole number can only be minutes (0.5° would remain decimal).
  const minutesMatch = new RegExp(
    `^(${TOLERANCE_NUMBER})\\s*°\\s*(${TOLERANCE_NUMBER})$`
  ).exec(normalized);
  if (minutesMatch) {
    const minutes = Number(minutesMatch[2]);
    return minutes < 60 ? Number(minutesMatch[1]) + minutes / 60 : null;
  }

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

/** Convert a stacked deviation pair at the end of a value into Range form. */
export function stackedDeviationRange(value: string): string {
  const pair = new RegExp(
    `([+-]?)\\s*(${TOLERANCE_NUMBER})\\s*°?\\s*/\\s*` +
      `([+-]?)\\s*(${TOLERANCE_NUMBER})\\s*°?\\s*$`
  ).exec(value);
  if (!pair || pair.index == null) return "";
  if (!/\d\s*°?\s*$/.test(value.slice(0, pair.index))) return "";

  const upper = Number(pair[2]);
  const lower = Number(pair[4]);
  // An unsigned non-zero upper value is a fraction, not a deviation.
  if (pair[1] !== "+" && upper !== 0) return "";
  if (pair[3] === "+" && lower !== 0) return "";
  return `+${formatTolerance(upper)}, -${formatTolerance(lower)}`;
}

/** Convert one deviation after a nominal into a zero-sided Range value. */
export function singleDeviationRange(value: string): string {
  const text = value.replace(/\s*\(\s*\d+\s*[x×]\s*\)\s*$/i, "");
  const deviation = new RegExp(
    `([+-])\\s*(${TOLERANCE_NUMBER})\\s*°?\\s*$`
  ).exec(text);
  if (!deviation || deviation.index == null) return "";
  const specification = text.slice(0, deviation.index);
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
 * Remove counts, dual units, TYP text and thread classes that can look like
 * tolerance signs before interpreting the dimension.
 */
export function toleranceSpecification(value: string): string {
  let text = value.trim();
  for (;;) {
    const next = text
      .replace(/\s*\[[^\]]*\]$/, "")
      .replace(/\s*\(\s*\d+\s*[x×]\s*\)$/i, "")
      .replace(/\s+\d+\s*(?:[x×]|plcs?\.?|places)$/i, "")
      .replace(/\s+typ\.?$/i, "")
      .trim();
    if (next === text) break;
    text = next;
  }
  return text
    .replace(/^\d+\s*(?:[x×]|-|plcs?\.?)\s*(?=[^\d\s.+-])/i, "")
    .replace(
      /(M\s*\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)?)\s*-\s*\d[a-h](?:\d[a-h])?\b/gi,
      "$1"
    )
    .replace(/\s*\(\s*([+-][^()]*?)\s*\)$/, " $1");
}

/** Whether the cleaned value contains a sign intended as a tolerance. */
export function hasToleranceIntent(value: string): boolean {
  const text = toleranceSpecification(value);
  const nominal = /[-+]?\d+(?:\.\d+)?/.exec(text);
  if (!nominal || nominal.index == null) return /[±+-]/.test(text);
  return /[±+-]/.test(text.slice(nominal.index + nominal[0].length));
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
export function deriveRange(rawValue: string): string {
  const value = toleranceSpecification(rawValue);
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
  // A signed nominal following a radius/diameter prefix (for example R-5)
  // is not a one-sided tolerance and must remain unresolved for export.
  if (nominal.index > 0 && /^[+-]/.test(nominal[0])) return "";

  const remainder = value.slice(nominal.index + nominal[0].length);
  if (/[±+-]/.test(remainder)) return "";
  if (/~|\b(?:max(?:imum)?|min(?:imum)?)\b/i.test(remainder)) return "";
  return "0";
}
