const UNSIGNED_NUMBER = String.raw`(?:\d+(?:\.\d+)?|\.\d+)`;
const SIGNED_NUMBER = String.raw`[-+]?${UNSIGNED_NUMBER}`;

export interface ChecksheetNumericRange {
  min?: number;
  max?: number;
}

export type ChecksheetReadingState =
  | "empty"
  | "in_range"
  | "out_of_range"
  | "invalid"
  | "unrestricted";

export interface ChecksheetReadingValidation {
  state: ChecksheetReadingState;
  range: ChecksheetNumericRange | null;
  message: string;
}

function normalizeEngineeringText(value: string): string {
  return value
    .trim()
    .replace(/[−–—]/g, "-")
    .replace(/[˚⁰]/g, "°")
    .replace(/[’´]/g, "′")
    .replace(/[”ʺ]/g, "″")
    .replace(/[⌀øφ]/gi, "Ø");
}

function parseAngleNominal(text: string): number | null {
  const angle = new RegExp(`(${SIGNED_NUMBER})\\s*°`).exec(text);
  if (!angle || angle.index == null) return null;

  const degrees = Number(angle[1]);
  const remainder = text.slice(angle.index + angle[0].length).trim();
  const dms = new RegExp(
    `^(${UNSIGNED_NUMBER})\\s*[′']` +
      `(?:\\s*(${UNSIGNED_NUMBER})\\s*[″\"])?`
  ).exec(remainder);
  if (!dms) return degrees;

  const minutes = Number(dms[1]);
  const seconds = Number(dms[2] ?? 0);
  if (minutes >= 60 || seconds >= 60) return degrees;
  const fraction = minutes / 60 + seconds / 3600;
  return degrees < 0 ? degrees - fraction : degrees + fraction;
}

export function checksheetNominalValue(specification: string): number | null {
  const text = normalizeEngineeringText(specification);
  if (!text) return null;

  const angle = parseAngleNominal(text);
  if (angle != null) return angle;

  // Prefer the measured diameter/radius over a leading quantity such as
  // "6 x Ø5".
  const feature = new RegExp(`(?:Ø|\\bR)\\s*(${SIGNED_NUMBER})`, "i").exec(
    text
  );
  if (feature) return Number(feature[1]);

  const first = new RegExp(SIGNED_NUMBER).exec(text);
  return first ? Number(first[0]) : null;
}

function parseTolerance(
  tolerance: string
): { lower: number; upper: number } | null {
  const text = normalizeEngineeringText(tolerance);
  if (!text) return null;

  const plain = new RegExp(`^(${UNSIGNED_NUMBER})\\s*°?$`).exec(text);
  if (plain) {
    const value = Number(plain[1]);
    return { lower: value, upper: value };
  }

  const symmetric = new RegExp(
    `^±\\s*(${UNSIGNED_NUMBER})\\s*°?$`
  ).exec(text);
  if (symmetric) {
    const value = Number(symmetric[1]);
    return { lower: value, upper: value };
  }

  const asymmetric = new RegExp(
    `^\\+\\s*(${UNSIGNED_NUMBER})\\s*°?\\s*,?\\s*` +
      `-\\s*(${UNSIGNED_NUMBER})\\s*°?$`
  ).exec(text);
  if (asymmetric) {
    return {
      lower: Number(asymmetric[2]),
      upper: Number(asymmetric[1]),
    };
  }

  return null;
}

function embeddedRange(specification: string): ChecksheetNumericRange | null {
  const text = normalizeEngineeringText(specification);
  if (!text) return null;

  const explicit = new RegExp(
    `(${SIGNED_NUMBER})\\s*~\\s*(${SIGNED_NUMBER})`,
    "i"
  ).exec(text);
  if (explicit) {
    return { min: Number(explicit[1]), max: Number(explicit[2]) };
  }

  const symmetric = new RegExp(
    `(${SIGNED_NUMBER})\\s*°?\\s*±\\s*(${UNSIGNED_NUMBER})\\s*°?`,
    "i"
  ).exec(text);
  if (symmetric) {
    const nominal = Number(symmetric[1]);
    const tolerance = Number(symmetric[2]);
    return { min: nominal - tolerance, max: nominal + tolerance };
  }

  const asymmetric = new RegExp(
    `(${SIGNED_NUMBER})\\s*°?\\s*\\+\\s*(${UNSIGNED_NUMBER})\\s*°?\\s*,?\\s*` +
      `-\\s*(${UNSIGNED_NUMBER})\\s*°?`,
    "i"
  ).exec(text);
  if (asymmetric) {
    const nominal = Number(asymmetric[1]);
    return {
      min: nominal - Number(asymmetric[3]),
      max: nominal + Number(asymmetric[2]),
    };
  }

  const maximum = new RegExp(`(${SIGNED_NUMBER})\\s*(?:max|maximum)\\b`, "i").exec(
    text
  );
  if (maximum) return { max: Number(maximum[1]) };

  const minimum = new RegExp(`(${SIGNED_NUMBER})\\s*(?:min|minimum)\\b`, "i").exec(
    text
  );
  if (minimum) return { min: Number(minimum[1]) };

  return null;
}

export function checksheetNumericRange(
  specification: string,
  tolerance: string
): ChecksheetNumericRange | null {
  const nominal = checksheetNominalValue(specification);
  const parsedTolerance = parseTolerance(tolerance);
  if (nominal != null && parsedTolerance) {
    return {
      min: nominal - parsedTolerance.lower,
      max: nominal + parsedTolerance.upper,
    };
  }

  const embedded = embeddedRange(specification);
  if (embedded) return embedded;

  // Do not manufacture an exact range when a non-empty tolerance exists but
  // cannot be interpreted.
  if (tolerance.trim()) return null;

  // A numeric specification without an explicit tolerance is exact.
  return nominal == null ? null : { min: nominal, max: nominal };
}

export function parseChecksheetReading(reading: string): number | null {
  const normalized = normalizeEngineeringText(reading);
  if (!new RegExp(`^${SIGNED_NUMBER}$`).test(normalized)) return null;
  const value = Number(normalized);
  return Number.isFinite(value) ? value : null;
}

function formatLimit(value: number): string {
  const rounded = Math.round((value + Number.EPSILON) * 1_000_000) / 1_000_000;
  return String(rounded);
}

function rangeMessage(range: ChecksheetNumericRange): string {
  if (range.min != null && range.max != null) {
    if (range.min === range.max) return `Required value: ${formatLimit(range.min)}`;
    return `Allowed: ${formatLimit(range.min)}–${formatLimit(range.max)}`;
  }
  if (range.min != null) return `Minimum: ${formatLimit(range.min)}`;
  if (range.max != null) return `Maximum: ${formatLimit(range.max)}`;
  return "";
}

export function validateChecksheetReading(
  reading: string,
  specification: string,
  tolerance: string
): ChecksheetReadingValidation {
  const trimmed = reading.trim();
  const range = checksheetNumericRange(specification, tolerance);
  if (!trimmed) return { state: "empty", range, message: "" };

  const numericValue = parseChecksheetReading(trimmed);
  if (numericValue == null) {
    return {
      state: "invalid",
      range,
      message: "Enter a numeric reading.",
    };
  }

  if (!range) return { state: "unrestricted", range: null, message: "" };

  const value = numericValue;
  const magnitude = Math.max(
    1,
    Math.abs(value),
    Math.abs(range.min ?? 0),
    Math.abs(range.max ?? 0)
  );
  const epsilon = magnitude * 1e-9;
  const below = range.min != null && value < range.min - epsilon;
  const above = range.max != null && value > range.max + epsilon;
  if (below || above) {
    return {
      state: "out_of_range",
      range,
      message: rangeMessage(range),
    };
  }

  return { state: "in_range", range, message: "" };
}
