/**
 * Light cleanup only — does NOT add Ø or ± that were not in OCR/API hints.
 */

export function countEngineeringSymbols(text: string): number {
  return (text.match(/[±°Ø'"]/g) ?? []).length;
}

export function engineeringQualityScore(text: string, confidence: number): number {
  if (!text.trim()) return 0;
  let score = confidence * 10;
  score += countEngineeringSymbols(text) * 3;
  score += (text.match(/\d/g) ?? []).length * 0.5;
  return score;
}

export function fixEngineeringSymbols(text: string): string {
  if (!text) return "";

  return text
    .replace(/\+\/−|\+\/-/g, "±")
    .replace(/^[\s]*(?:⊕|∅|⌀|φ|Φ|ø)/, "Ø")
    .replace(/\s*([±°Ø'"])\s*/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

export function pickBestOcrCandidate(
  candidates: Array<{ text: string; confidence: number }>
): { text: string; confidence: number } {
  const fixed = candidates
    .map((c) => ({
      text: fixEngineeringSymbols(c.text),
      confidence: c.confidence,
    }))
    .filter((c) => c.text.length > 0);

  if (fixed.length === 0) return { text: "", confidence: 0 };

  return fixed.reduce((best, cur) =>
    engineeringQualityScore(cur.text, cur.confidence) >
    engineeringQualityScore(best.text, best.confidence)
      ? cur
      : best
  );
}
