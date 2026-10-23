import { classifyValues, type TitleField } from "@/lib/paddleOcrClient";
import type { Annotation, DimensionType } from "@/types/annotation";

/**
 * The balloon label and category the labelling rules give a value.
 *
 * The rules live in the backend rule engine (feature_classifier). Scans get
 * them with every region; a value that is typed, corrected or accepted from
 * the review queue has to ask for them, or it is saved with no label at all —
 * which is how some drawings ended up with every scanned balloon labelled and
 * every reviewed one blank.
 */
export interface RuleVerdict {
  category: string;
  label: string;
}

/** Text compared the way the rules see it: case and spacing do not matter. */
function sameText(left: string, right: string): boolean {
  const squash = (s: string) => s.replace(/\s+/g, "").toUpperCase();
  return squash(left) === squash(right);
}

/**
 * The rules' verdict for `value`.
 *
 * `known` is a verdict already in hand (from recognition, or from the scan
 * that produced a review item) and `knownFor` the text it was computed for.
 * It is reused only while the value still says the same thing; any edit is
 * classified again. Returns null when the backend cannot be reached, so the
 * caller saves the value unlabelled rather than with a guessed label.
 */
export async function ruleForValue(
  value: string,
  known?: RuleVerdict | null,
  knownFor?: string
): Promise<RuleVerdict | null> {
  const text = value.trim();
  if (!text) return null;
  if (known && (known.label || known.category) && knownFor !== undefined) {
    if (sameText(text, knownFor)) return known;
  }
  try {
    const [result] = await classifyValues([text]);
    return result ? { category: result.category, label: result.label } : null;
  } catch {
    return null;
  }
}

/**
 * A dimension type for a rule category. The rule engine's categories are the
 * app's annotation types (DIMENSION_TYPES mirrors CAT_*), so the category is
 * used as it is — the same as a scanned region's (annotationsFromRegions).
 * The client classifier answers only when the rules could not be asked.
 */
export function typeForVerdict(
  verdict: RuleVerdict | null,
  fallback: DimensionType
): DimensionType {
  return (verdict?.category as DimensionType) || fallback;
}

/**
 * The label and type after the user edits a value.
 *
 * An unchanged value keeps everything (`type` undefined means "as it was").
 * An edited value is classified again by the rules. Its label follows the
 * rules only while it is still the rules' own: empty, or the label the rules
 * gave the old value, and not changed in the same edit. A label the user
 * typed is theirs and is never replaced.
 */
export async function relabelForEdit(
  annotation: Pick<Annotation, "value" | "label">,
  nextValue: string,
  nextLabel: string
): Promise<{ label: string; type?: DimensionType }> {
  const previous = annotation.value.trim();
  if (sameText(nextValue, previous)) return { label: nextLabel };
  let verdicts: { category: string; label: string }[] = [];
  try {
    verdicts = await classifyValues(previous ? [previous, nextValue] : [nextValue]);
  } catch {
    return { label: nextLabel };
  }
  const now = verdicts[verdicts.length - 1];
  const before = previous ? verdicts[0] : undefined;
  const current = (annotation.label ?? "").trim();
  const userEdited = nextLabel !== current;
  const rulesOwn = current === "" || (before !== undefined && current === before.label);
  return {
    label: !userEdited && rulesOwn && now ? now.label : nextLabel,
    type: now ? typeForVerdict(now, "Unknown") : undefined,
  };
}

/** One label the rules required that is not on the drawing's values. */
export interface LabelGap {
  kind: "unlabelled_value" | "title_keyword_unread";
  /** Balloon number for a value; the configured keyword for a title field. */
  ref: string;
}

/**
 * What the labelling rules require that the drawing does not carry.
 *
 * Two rules, both existing ones: every value with text gets the rule
 * engine's balloon label (the engine labels every non-empty text), and every
 * configured title-block keyword gets a value. A legitimate exclusion is not
 * reported: an empty value has no label to give, and a keyword nobody
 * configured is not looked for. Used to warn instead of silently exporting a
 * sheet with holes in it.
 */
export function findLabelGaps(
  annotations: Annotation[],
  keywords: string[]
): LabelGap[] {
  const values = annotations.filter((a) => (a.kind ?? "dimension") === "dimension");
  const gaps: LabelGap[] = values
    .filter((a) => a.value.trim() !== "" && !(a.label ?? "").trim())
    .sort((a, b) => a.number - b.number)
    .map((a) => ({ kind: "unlabelled_value", ref: String(a.number) }));
  for (const keyword of keywords) {
    if (!titleFieldFor(values, keyword)?.value.trim()) {
      gaps.push({ kind: "title_keyword_unread", ref: keyword });
    }
  }
  return gaps;
}

const normalizeLabel = (s: string) =>
  s.toUpperCase().replace(/[^A-Z0-9]+/g, " ").trim();
// Field-name chrome the backend strips before matching ("DWG NO." -> "DWG").
const LABEL_CHROME = /\b(NO|NUM|NUMBER|BY)\b\.?/gi;
const normalizeKeyword = (s: string) =>
  normalizeLabel(s.replace(LABEL_CHROME, " "));

/**
 * The title-block annotation read for a configured keyword.
 *
 * By the keyword it was read for, when the annotation records one: the read
 * label differs between sheets ("DWG NO.", "DWG", "DWG NO") and matching on
 * it lost values — keyword "DWG NO." never matched a field read as "DWG:
 * 47630", so the export showed the field empty although the drawing view had
 * it. Older projects carry no keyword, and fall back to the backend's own
 * matching rule: the label starts with the keyword, field-name chrome aside.
 */
export function titleFieldFor(
  annotations: Annotation[],
  keyword: string
): Annotation | undefined {
  const fields = annotations.filter((a) => a.type === "Title Block");
  const byKeyword = fields.find(
    (a) => a.titleKeyword !== undefined && normalizeLabel(a.titleKeyword) === normalizeLabel(keyword)
  );
  if (byKeyword) return byKeyword;
  const needle = normalizeKeyword(keyword);
  if (!needle) return undefined;
  return fields.find(
    (a) => a.titleKeyword === undefined && normalizeKeyword(a.label ?? "").startsWith(needle)
  );
}

/**
 * Title-block annotations for the fields read off a page.
 *
 * One per configured keyword that was found with a value, tagged with that
 * keyword. A keyword that already has a value on the drawing is skipped, so
 * re-scanning never duplicates a field, and the check is by keyword rather
 * than by read label, which varies.
 */
export function titleFieldAnnotations(
  fields: TitleField[],
  existing: Annotation[],
  page: number,
  createdAt: number,
  makeId: () => string
): Annotation[] {
  const out: Annotation[] = [];
  for (const field of fields) {
    const value = field.value.trim();
    if (!value || !field.bbox) continue;
    const keyword = field.keyword || field.label;
    if (titleFieldFor([...existing, ...out], keyword)?.value.trim()) continue;
    out.push({
      id: makeId(),
      number: 0,
      label: field.label,
      value,
      type: "Title Block",
      confidence: field.confidence,
      bbox: field.bbox,
      rotation: 0,
      page,
      createdAt,
      kind: "dimension",
      titleKeyword: keyword,
    });
  }
  return out;
}

/**
 * One sentence naming the gaps, for the export menu; "" when there are none.
 * Balloon numbers and keywords only — never the drawing's text.
 */
export function describeLabelGaps(gaps: LabelGap[]): string {
  const values = gaps.filter((gap) => gap.kind === "unlabelled_value").map((gap) => gap.ref);
  const keywords = gaps.filter((gap) => gap.kind === "title_keyword_unread").map((gap) => gap.ref);
  const parts: string[] = [];
  if (values.length > 0) {
    const shown = values.slice(0, 8).join(", ") + (values.length > 8 ? ", …" : "");
    parts.push(
      `${values.length} value${values.length === 1 ? " has" : "s have"} no label (balloon${values.length === 1 ? "" : "s"} ${shown})`
    );
  }
  if (keywords.length > 0) {
    parts.push(
      `title field${keywords.length === 1 ? "" : "s"} not read: ${keywords.join(", ")}`
    );
  }
  return parts.length ? `${parts.join("; ")}.` : "";
}
