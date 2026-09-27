import {
  checksheetNominalValue,
  checksheetNumericRange,
  parseChecksheetReading,
  type ChecksheetNumericRange,
} from "@/lib/checksheetRange";
import type { ChecksheetColumn, ChecksheetRow } from "@/types/checksheet";

export interface ChecksheetChartReading {
  columnId: string;
  columnName: string;
  columnIndex: number;
  value: number;
  inRange: boolean | null;
}

export interface ChecksheetHistogramBin {
  start: number;
  end: number;
  total: number;
  inRange: number;
  outOfRange: number;
  unrestricted: number;
}

export interface ChecksheetRowChartData {
  readings: ChecksheetChartReading[];
  invalidReadingCount: number;
  nominal: number | null;
  range: ChecksheetNumericRange | null;
  mean: number | null;
  domain: { min: number; max: number };
  histogramBins: ChecksheetHistogramBin[];
}

function readingIsInRange(
  value: number,
  range: ChecksheetNumericRange | null
): boolean | null {
  if (!range) return null;
  const magnitude = Math.max(
    1,
    Math.abs(value),
    Math.abs(range.min ?? 0),
    Math.abs(range.max ?? 0)
  );
  const epsilon = magnitude * 1e-9;
  if (range.min != null && value < range.min - epsilon) return false;
  if (range.max != null && value > range.max + epsilon) return false;
  return true;
}

function paddedDomain(values: number[]): { min: number; max: number } {
  if (values.length === 0) return { min: 0, max: 1 };
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const span = rawMax - rawMin;
  if (span > 0) {
    const padding = span * 0.12;
    return { min: rawMin - padding, max: rawMax + padding };
  }

  const padding =
    rawMin === 0 ? 0.1 : Math.max(Math.abs(rawMin) * 0.03, 0.001);
  return { min: rawMin - padding, max: rawMax + padding };
}

function histogramBins(
  readings: ChecksheetChartReading[],
  domain: { min: number; max: number }
): ChecksheetHistogramBin[] {
  if (readings.length === 0) return [];
  const binCount = Math.min(
    8,
    Math.max(4, Math.ceil(Math.sqrt(readings.length) * 2))
  );
  const width = (domain.max - domain.min) / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    start: domain.min + index * width,
    end: domain.min + (index + 1) * width,
    total: 0,
    inRange: 0,
    outOfRange: 0,
    unrestricted: 0,
  }));

  for (const reading of readings) {
    const rawIndex = Math.floor((reading.value - domain.min) / width);
    const index = Math.max(0, Math.min(binCount - 1, rawIndex));
    const bin = bins[index];
    bin.total += 1;
    if (reading.inRange === true) bin.inRange += 1;
    else if (reading.inRange === false) bin.outOfRange += 1;
    else bin.unrestricted += 1;
  }

  return bins;
}

export function buildChecksheetRowChartData(
  row: ChecksheetRow,
  columns: ChecksheetColumn[]
): ChecksheetRowChartData {
  const range = checksheetNumericRange(row.specification, row.tolerance);
  const nominal = checksheetNominalValue(row.specification);
  let invalidReadingCount = 0;
  const readings: ChecksheetChartReading[] = [];

  columns.forEach((column, columnIndex) => {
    const raw = row.readings[column.id] ?? "";
    if (!raw.trim()) return;
    const value = parseChecksheetReading(raw);
    if (value == null) {
      invalidReadingCount += 1;
      return;
    }
    readings.push({
      columnId: column.id,
      columnName: column.name,
      columnIndex,
      value,
      inRange: readingIsInRange(value, range),
    });
  });

  const mean =
    readings.length > 0
      ? readings.reduce((sum, reading) => sum + reading.value, 0) /
        readings.length
      : null;
  const domainValues = readings.map((reading) => reading.value);
  if (nominal != null) domainValues.push(nominal);
  if (range?.min != null) domainValues.push(range.min);
  if (range?.max != null) domainValues.push(range.max);
  const domain = paddedDomain(domainValues);

  return {
    readings,
    invalidReadingCount,
    nominal,
    range,
    mean,
    domain,
    histogramBins: histogramBins(readings, domain),
  };
}
