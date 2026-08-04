/**
 * Apply symbol hints from API only — never guess Ø/± on the client.
 */

export interface SymbolHints {
  diameter?: boolean;
  /** Backend prefix vision score; avoid adding Ø when low */
  diameter_score?: number;
  plus_minus?: boolean;
  degree?: boolean;
}

const PHI_SCORE_MIN = 0.32;

function isTolerancePair(main: string, tol: string): boolean {
  const a = parseFloat(main);
  const b = parseFloat(tol);
  if (Number.isNaN(a) || Number.isNaN(b) || b <= 0 || a <= 0) return false;
  if (b >= a * 0.5) return false;
  return b <= 2 || b < a;
}

export function mergeSymbolHints(text: string, hints: SymbolHints): string {
  let t = text.trim();
  if (!t) return t;

  const phiOk = (hints.diameter_score ?? 0) >= PHI_SCORE_MIN;

  if (phiOk && !t.startsWith("Ø") && !t.startsWith("R")) {
    t = t.replace(/^[\s[\({]*[φΦ⌀ø]\s*/, "Ø");
    if (!t.startsWith("Ø")) t = "Ø" + t;
  }

  if (
    hints.plus_minus &&
    !t.includes("±") &&
    !t.includes("+/-")
  ) {
    const m = t.match(/^(\d+\.?\d*)\s+(\d+\.?\d*)\s*$/);
    if (m && isTolerancePair(m[1], m[2])) {
      t = `${m[1]}±${m[2]}`;
    }
  }

  if (hints.degree && !t.includes("°")) {
    t = t.replace(/(\d)\s*['′]/, "$1°");
  }

  return t;
}
