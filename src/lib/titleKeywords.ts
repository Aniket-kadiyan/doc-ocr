/**
 * "Keywords to look for": the title-block fields the user wants pulled off the
 * drawing (DWG NO., REV, DATE …).
 *
 * A drawing's title block is a table of label/value pairs, not dimensions, so
 * nothing in the dimension pipeline captures it. The configured keywords are
 * sent to the backend segmenter, which finds each label in the bottom-right
 * table and reads the value beside or below it. The results are written to the
 * export as their own section.
 *
 * Stored per browser in localStorage, following the OCR debug-dump toggles.
 */

const LS_KEY = "doc_ocr_title_keywords";

/** Fired after a change so open panels re-read the list. */
export const KEYWORDS_CHANGED_EVENT = "title-keywords-changed";

/** Shipped default: the two fields nearly every drawing carries. */
export const DEFAULT_KEYWORDS = ["DWG", "REV"];

/** At most this many keywords — the list is a lookup per keyword per drawing. */
const MAX_KEYWORDS = 20;

/** Parse a comma/newline separated list into clean, de-duplicated keywords. */
export function parseKeywords(raw: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of raw.split(/[,\n]/)) {
    const kw = part.trim().replace(/\s+/g, " ");
    if (!kw) continue;
    // Case-insensitive de-dupe: the backend match is case-insensitive too, so
    // "Rev" and "REV" would otherwise produce two identical rows.
    const key = kw.toUpperCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(kw);
    if (out.length >= MAX_KEYWORDS) break;
  }
  return out;
}

/** The configured keywords, or the defaults when nothing has been saved yet. */
export function getTitleKeywords(): string[] {
  if (typeof window === "undefined") return DEFAULT_KEYWORDS;
  try {
    const raw = localStorage.getItem(LS_KEY);
    // An explicitly saved empty list means "look for nothing" and must be
    // honoured; only a missing key falls back to the defaults.
    if (raw === null) return DEFAULT_KEYWORDS;
    return parseKeywords(raw);
  } catch {
    // Private mode / blocked storage: fall back rather than break the export.
    return DEFAULT_KEYWORDS;
  }
}

export function setTitleKeywords(keywords: string[]): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(LS_KEY, keywords.join(", "));
  } catch {
    // Nothing persists, but the in-memory session still works.
  }
  window.dispatchEvent(new CustomEvent(KEYWORDS_CHANGED_EVENT));
}
