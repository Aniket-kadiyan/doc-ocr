/**
 * Stable categorical colours linking reference-table rows with their drawing
 * markers. Identity never depends on colour alone: every marker and row also
 * carries the printed point symbol.
 */

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
  fill: string;
  ink: string;
  /** Faint selection wash for the reconstructed table row. */
  wash: string;
}

const channel = (hex: string, offset: number) =>
  parseInt(hex.slice(offset, offset + 2), 16) / 255;

function luminance(hex: string): number {
  const linear = (value: number) =>
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  return (
    0.2126 * linear(channel(hex, 1)) +
    0.7152 * linear(channel(hex, 3)) +
    0.0722 * linear(channel(hex, 5))
  );
}

/** A point retains its colour by row position even when other rows are absent. */
export function referenceColor(index: number): ReferenceColor {
  const fill = SERIES[((index % SERIES.length) + SERIES.length) % SERIES.length];
  return {
    fill,
    ink: luminance(fill) > 0.45 ? INK_DARK : INK_LIGHT,
    wash: `${fill}1f`,
  };
}

export const REFERENCE_SERIES_COUNT = SERIES.length;
