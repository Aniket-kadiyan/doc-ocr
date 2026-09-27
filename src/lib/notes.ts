import type { Annotation } from "@/types/annotation";

/**
 * Splitting a drawing's NOTES block into one row per numbered point.
 *
 * A notes block reaches us as one OCR read covering the whole paragraph, so the
 * text arrives with the drawing's own line wrapping still in it:
 *
 *     NOTES
 *     1.MAT'L GRADE 65-45-12 DUCTILE IRON PER ASTM A536(LASTEST
 *     REVISION). MATERIAL CERTIFICATON OF ANALYSIS IS REQUIRED
 *     2. HOT DIP GALVANIZE PER ASTM A153 (LATEST REVISION)
 *
 * A point owns every following line until the next point marker, so the wrapped
 * remainder is joined back onto the point it belongs to.
 */

/** Types the classifier assigns to a notes block (both sides use these names). */
const NOTE_TYPES = new Set(["General Note", "Note"]);

/** A line that opens a numbered point: "1.", "2)", "3 ." at the line start.
 * The separator is required — making it optional turns ordinary lines into
 * false markers and splits a note mid-sentence. */
const POINT_START = /^\s*(\d{1,2})\s*[.)]\s*/;

/** A standalone "NOTES" / "NOTE:" heading line above the first point. */
const NOTES_HEADING = /^\s*NOTES?\s*[:.-]?\s*$/i;

/** Inline fallback marker, used only when the block arrived as a single line. */
const INLINE_POINT = /(?:^|\s)(\d{1,2})\s*[.)]\s*(?=\D)/g;

/**
 * True when this annotation is a drawing notes block rather than a dimension.
 *
 * The assigned type alone is not enough: the classifier keys off keywords, so a
 * block opening "1.MAT'L GRADE 65-45-12…" is labelled Material, not General
 * Note. Any text that resolves into two or more numbered points is a notes
 * block whatever it was called — no dimension callout has that shape.
 */
export function isNoteAnnotation(a: Annotation): boolean {
  return NOTE_TYPES.has(a.type) || splitNotePoints(a.value).length >= 2;
}

/**
 * Split one notes block into its numbered points, in drawing order.
 *
 * The leading number is deliberately kept in the returned text: the sheet shows
 * the points exactly as the drawing numbers them.
 */
export function splitNotePoints(text: string): string[] {
  const lines = (text ?? "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !NOTES_HEADING.test(l));

  const points: string[] = [];
  for (const line of lines) {
    if (POINT_START.test(line) || points.length === 0) {
      points.push(line);
    } else {
      // A wrapped continuation of the point above.
      points[points.length - 1] += ` ${line}`;
    }
  }

  // The recogniser sometimes returns the whole block as one line. Only then
  // fall back to splitting inside the line, and only when the markers actually
  // run 1, 2, 3… — a lone "1." inside prose must not split a real sentence,
  // and decimals like ".12R" never match, since a digit cannot follow the dot.
  if (points.length === 1) {
    const inline = splitInline(points[0]);
    if (inline.length > 1) return inline;
  }

  return points.map(collapseSpaces).filter((p) => p.length > 0);
}

/** Split a single-line block on its "1. … 2. …" markers when they're sequential. */
function splitInline(line: string): string[] {
  const marks = [...line.matchAll(INLINE_POINT)];
  if (marks.length < 2) return [];

  // The markers must start at the top of a list and climb. Requiring an exact
  // 1,2,3… run would collapse the whole block when the recogniser drops a
  // single marker ("5." read as "S."), so increasing is enough.
  const nums = marks.map((m) => Number(m[1]));
  const ordered =
    nums[0] <= 2 && nums.every((n, i) => i === 0 || n > nums[i - 1]);
  if (!ordered) return [];

  return marks
    .map((m, i) => {
      const from = m.index! + (m[0].startsWith(" ") ? 1 : 0);
      const to = i + 1 < marks.length ? marks[i + 1].index! : line.length;
      return collapseSpaces(line.slice(from, to));
    })
    .filter((p) => p.length > 0);
}

function collapseSpaces(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

/**
 * Every note point across all note annotations on the drawing, in balloon order.
 * Returns [] when the drawing carries no notes, which is the signal to leave the
 * sheet exactly as it was.
 */
export function collectNotePoints(annotations: Annotation[]): string[] {
  return annotations
    .filter((a) => (a.kind ?? "dimension") === "dimension")
    .filter(isNoteAnnotation)
    .sort((a, b) => a.number - b.number)
    .flatMap((a) => splitNotePoints(a.value));
}
