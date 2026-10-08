/**
 * Colours that tie a reference-table row to its markers on the drawing.
 *
 * Colour here is wayfinding, not identity: every marker prints its own point
 * name and every sidebar row is labelled with it, so a reader never has to
 * tell two hues apart to know which point they are looking at. That is what
 * makes it safe to reuse the sequence past its eighth hue on a table with
 * more rows than that — the letter still says which point it is.
 *
 * The eight hues are the validated categorical order, checked against a white
 * sheet: every hue clears the lightness band, the chroma floor, the adjacent
 * colour-vision-deficiency separation and the normal-vision floor. Three of
 * them fall below 3:1 against white, which obliges visible labels — markers
 * are drawn as filled chips carrying their name in ink chosen for contrast,
 * and the sidebar repeats every row as text.
 */

/** Fixed order. A row keeps its hue when other rows are filtered away. */
const SERIES = [
  "#2a78d6",
  "#eb6834",
  "#1baf7a",
  "#eda100",
  "#e87ba4",
  "#008300",
  "#4a3aa7",
  "#e34948",
] as const;

const INK_DARK = "#111827";
const INK_LIGHT = "#ffffff";

export interface ReferenceColor {
  /** Chip fill and the ring drawn around the glyph on the drawing. */
  fill: string;
  /** Text drawn on top of {@link fill}. */
  ink: string;
  /** Faint wash for the selected sidebar row. */
  wash: string;
}

const channel = (hex: string, offset: number) =>
  parseInt(hex.slice(offset, offset + 2), 16) / 255;

/** sRGB relative luminance, for picking readable text over a fill. */
function luminance(hex: string): number {
  const linear = (value: number) =>
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  return (
    0.2126 * linear(channel(hex, 1)) +
    0.7152 * linear(channel(hex, 3)) +
    0.0722 * linear(channel(hex, 5))
  );
}

/**
 * The colour for the point at `index` in the table, counted from the top.
 *
 * Indexed by position rather than by name so the same table always paints the
 * same way, and so a point whose marker was not found still holds its slot.
 */
export function referenceColor(index: number): ReferenceColor {
  const fill = SERIES[((index % SERIES.length) + SERIES.length) % SERIES.length];
  return {
    fill,
    ink: luminance(fill) > 0.45 ? INK_DARK : INK_LIGHT,
    wash: `${fill}1f`,
  };
}

export const REFERENCE_SERIES_COUNT = SERIES.length;
