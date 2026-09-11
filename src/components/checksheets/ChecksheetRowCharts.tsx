import { buildChecksheetRowChartData } from "@/lib/checksheetChartData";
import type {
  ChecksheetColumn,
  ChecksheetRow,
} from "@/types/checksheet";

interface ChecksheetRowChartsProps {
  columns: ChecksheetColumn[];
  row: ChecksheetRow;
}

const WIDTH = 560;
const HEIGHT = 240;
const MARGIN = { top: 14, right: 16, bottom: 48, left: 54 };
const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;

function formatNumber(value: number): string {
  const rounded = Math.round((value + Number.EPSILON) * 1_000_000) / 1_000_000;
  return String(rounded);
}

function yPosition(value: number, min: number, max: number): number {
  return MARGIN.top + ((max - value) / (max - min)) * PLOT_HEIGHT;
}

function xPosition(index: number, count: number): number {
  if (count <= 1) return MARGIN.left + PLOT_WIDTH / 2;
  return MARGIN.left + (index / (count - 1)) * PLOT_WIDTH;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-md bg-slate-100 px-2 py-1 text-[11px] text-slate-600">
      {label} <strong className="font-mono text-slate-800">{value}</strong>
    </span>
  );
}

function EmptyChart({ text }: { text: string }) {
  return (
    <div className="flex h-44 items-center justify-center px-6 text-center text-sm text-slate-400">
      {text}
    </div>
  );
}

export function ChecksheetRowCharts({
  columns,
  row,
}: ChecksheetRowChartsProps) {
  const chart = buildChecksheetRowChartData(row, columns);
  const { min, max } = chart.domain;
  const yTicks = Array.from(
    { length: 5 },
    (_, index) => min + ((max - min) * index) / 4
  );
  const linePoints = chart.readings.map((reading) => ({
    ...reading,
    x: xPosition(reading.columnIndex, columns.length),
    y: yPosition(reading.value, min, max),
  }));
  const linePath = linePoints
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`)
    .join(" ");

  const histogramMax = Math.max(
    1,
    ...chart.histogramBins.map((bin) => bin.total)
  );
  const histogramY = (count: number) =>
    MARGIN.top + PLOT_HEIGHT - (count / histogramMax) * PLOT_HEIGHT;
  const histogramXTicks = Array.from(
    { length: 5 },
    (_, index) => min + ((max - min) * index) / 4
  );
  const histogramYTicks = Array.from(
    new Set([0, Math.ceil(histogramMax / 2), histogramMax])
  );

  const acceptableBand = (orientation: "horizontal" | "vertical") => {
    if (!chart.range) return null;
    const lower = chart.range.min ?? min;
    const upper = chart.range.max ?? max;
    if (upper < min || lower > max) return null;
    const clampedLower = Math.max(min, lower);
    const clampedUpper = Math.min(max, upper);
    if (orientation === "horizontal") {
      const top = yPosition(clampedUpper, min, max);
      const bottom = yPosition(clampedLower, min, max);
      return (
        <rect
          x={MARGIN.left}
          y={top}
          width={PLOT_WIDTH}
          height={Math.max(0, bottom - top)}
          fill="#dcfce7"
          opacity="0.7"
        />
      );
    }
    const left =
      MARGIN.left + ((clampedLower - min) / (max - min)) * PLOT_WIDTH;
    const right =
      MARGIN.left + ((clampedUpper - min) / (max - min)) * PLOT_WIDTH;
    return (
      <rect
        x={left}
        y={MARGIN.top}
        width={Math.max(0, right - left)}
        height={PLOT_HEIGHT}
        fill="#dcfce7"
        opacity="0.7"
      />
    );
  };

  return (
    <div className="mt-3 space-y-3">
      <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-200 px-3 py-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-slate-800">X̄ chart</h2>
              <p className="text-xs text-slate-400">
                Balloon {row.balloon_number} · {chart.readings.length} numeric reading
                {chart.readings.length === 1 ? "" : "s"}
              </p>
            </div>
            <div className="flex flex-wrap justify-end gap-1.5">
              {chart.mean != null && (
                <Metric label="X̄" value={formatNumber(chart.mean)} />
              )}
              {chart.nominal != null && (
                <Metric label="Nominal" value={formatNumber(chart.nominal)} />
              )}
              {chart.range?.min != null && (
                <Metric label="LSL" value={formatNumber(chart.range.min)} />
              )}
              {chart.range?.max != null && (
                <Metric label="USL" value={formatNumber(chart.range.max)} />
              )}
            </div>
          </div>
        </div>

        {chart.readings.length === 0 ? (
          <EmptyChart text="Enter a numeric reading in the active row to display the X̄ chart." />
        ) : (
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            className="block h-auto w-full"
            role="img"
            aria-label={`X-bar chart for balloon ${row.balloon_number}`}
          >
            {acceptableBand("horizontal")}
            {yTicks.map((tick) => {
              const y = yPosition(tick, min, max);
              return (
                <g key={tick}>
                  <line
                    x1={MARGIN.left}
                    x2={WIDTH - MARGIN.right}
                    y1={y}
                    y2={y}
                    stroke="#e2e8f0"
                    strokeWidth="1"
                  />
                  <text
                    x={MARGIN.left - 8}
                    y={y + 3}
                    textAnchor="end"
                    fontSize="9"
                    fill="#64748b"
                  >
                    {formatNumber(tick)}
                  </text>
                </g>
              );
            })}

            {chart.range?.min != null && (
              <line
                x1={MARGIN.left}
                x2={WIDTH - MARGIN.right}
                y1={yPosition(chart.range.min, min, max)}
                y2={yPosition(chart.range.min, min, max)}
                stroke="#dc2626"
                strokeDasharray="6 4"
                strokeWidth="1.5"
              />
            )}
            {chart.range?.max != null && (
              <line
                x1={MARGIN.left}
                x2={WIDTH - MARGIN.right}
                y1={yPosition(chart.range.max, min, max)}
                y2={yPosition(chart.range.max, min, max)}
                stroke="#dc2626"
                strokeDasharray="6 4"
                strokeWidth="1.5"
              />
            )}
            {chart.nominal != null && (
              <line
                x1={MARGIN.left}
                x2={WIDTH - MARGIN.right}
                y1={yPosition(chart.nominal, min, max)}
                y2={yPosition(chart.nominal, min, max)}
                stroke="#475569"
                strokeDasharray="3 3"
                strokeWidth="1.25"
              />
            )}
            {chart.mean != null && (
              <line
                x1={MARGIN.left}
                x2={WIDTH - MARGIN.right}
                y1={yPosition(chart.mean, min, max)}
                y2={yPosition(chart.mean, min, max)}
                stroke="#2563eb"
                strokeWidth="1.75"
              />
            )}
            {linePoints.length > 1 && (
              <path
                d={linePath}
                fill="none"
                stroke="#64748b"
                strokeWidth="1.5"
              />
            )}
            {linePoints.map((point) => (
              <circle
                key={point.columnId}
                cx={point.x}
                cy={point.y}
                r="5"
                fill={
                  point.inRange === true
                    ? "#16a34a"
                    : point.inRange === false
                      ? "#dc2626"
                      : "#2563eb"
                }
                stroke="white"
                strokeWidth="2"
              >
                <title>
                  {point.columnName}: {formatNumber(point.value)}
                  {point.inRange === true
                    ? " (in range)"
                    : point.inRange === false
                      ? " (out of range)"
                      : ""}
                </title>
              </circle>
            ))}
            {columns.map((column, index) => {
              const x = xPosition(index, columns.length);
              return (
                <text
                  key={column.id}
                  x={x}
                  y={HEIGHT - 9}
                  textAnchor="end"
                  transform={`rotate(-32 ${x} ${HEIGHT - 9})`}
                  fontSize="9"
                  fill="#64748b"
                >
                  {column.name}
                </text>
              );
            })}
          </svg>
        )}
      </section>

      <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-200 px-3 py-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-slate-800">Histogram</h2>
              <p className="text-xs text-slate-400">
                Distribution of the active row&apos;s numeric readings
              </p>
            </div>
            <div className="flex items-center gap-2 text-[11px] text-slate-500">
              <span className="inline-flex items-center gap-1">
                <span className="h-2.5 w-2.5 rounded-sm bg-emerald-600" /> In range
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="h-2.5 w-2.5 rounded-sm bg-red-600" /> Out
              </span>
            </div>
          </div>
          {chart.invalidReadingCount > 0 && (
            <p className="mt-1 text-[11px] text-red-600">
              {chart.invalidReadingCount} nonnumeric reading
              {chart.invalidReadingCount === 1 ? " was" : "s were"} excluded.
            </p>
          )}
        </div>

        {chart.readings.length === 0 ? (
          <EmptyChart text="Enter numeric readings in the active row to display their distribution." />
        ) : (
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            className="block h-auto w-full"
            role="img"
            aria-label={`Histogram for balloon ${row.balloon_number}`}
          >
            {acceptableBand("vertical")}
            {histogramYTicks.map((tick) => {
              const y = histogramY(tick);
              return (
                <g key={tick}>
                  <line
                    x1={MARGIN.left}
                    x2={WIDTH - MARGIN.right}
                    y1={y}
                    y2={y}
                    stroke="#e2e8f0"
                    strokeWidth="1"
                  />
                  <text
                    x={MARGIN.left - 8}
                    y={y + 3}
                    textAnchor="end"
                    fontSize="9"
                    fill="#64748b"
                  >
                    {tick}
                  </text>
                </g>
              );
            })}
            {chart.range?.min != null && (
              <line
                x1={MARGIN.left + ((chart.range.min - min) / (max - min)) * PLOT_WIDTH}
                x2={MARGIN.left + ((chart.range.min - min) / (max - min)) * PLOT_WIDTH}
                y1={MARGIN.top}
                y2={MARGIN.top + PLOT_HEIGHT}
                stroke="#dc2626"
                strokeDasharray="6 4"
                strokeWidth="1.5"
              />
            )}
            {chart.range?.max != null && (
              <line
                x1={MARGIN.left + ((chart.range.max - min) / (max - min)) * PLOT_WIDTH}
                x2={MARGIN.left + ((chart.range.max - min) / (max - min)) * PLOT_WIDTH}
                y1={MARGIN.top}
                y2={MARGIN.top + PLOT_HEIGHT}
                stroke="#dc2626"
                strokeDasharray="6 4"
                strokeWidth="1.5"
              />
            )}
            {chart.histogramBins.map((bin, index) => {
              const slotWidth = PLOT_WIDTH / chart.histogramBins.length;
              const x = MARGIN.left + index * slotWidth + 2;
              const width = Math.max(1, slotWidth - 4);
              let cursor = MARGIN.top + PLOT_HEIGHT;
              const segments = [
                { count: bin.inRange, fill: "#16a34a" },
                { count: bin.unrestricted, fill: "#2563eb" },
                { count: bin.outOfRange, fill: "#dc2626" },
              ];
              return (
                <g key={`${bin.start}-${bin.end}`}>
                  <title>
                    {formatNumber(bin.start)}–{formatNumber(bin.end)}: {bin.total}
                  </title>
                  {segments.map((segment) => {
                    if (segment.count === 0) return null;
                    const segmentHeight =
                      (segment.count / histogramMax) * PLOT_HEIGHT;
                    cursor -= segmentHeight;
                    return (
                      <rect
                        key={segment.fill}
                        x={x}
                        y={cursor}
                        width={width}
                        height={segmentHeight}
                        rx="2"
                        fill={segment.fill}
                      />
                    );
                  })}
                </g>
              );
            })}
            {histogramXTicks.map((tick) => {
              const x = MARGIN.left + ((tick - min) / (max - min)) * PLOT_WIDTH;
              return (
                <text
                  key={tick}
                  x={x}
                  y={HEIGHT - 17}
                  textAnchor="middle"
                  fontSize="9"
                  fill="#64748b"
                >
                  {formatNumber(tick)}
                </text>
              );
            })}
            <text
              x={MARGIN.left + PLOT_WIDTH / 2}
              y={HEIGHT - 3}
              textAnchor="middle"
              fontSize="9"
              fill="#64748b"
            >
              Reading value
            </text>
          </svg>
        )}
      </section>
    </div>
  );
}
