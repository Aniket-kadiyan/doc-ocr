"use client";

import { referenceColor } from "@/lib/referenceColors";
import type { PageReferenceTable } from "@/types/referencePoint";

interface ReferencePanelProps {
  table: PageReferenceTable;
  selectedSymbol: string | null;
  disabled?: boolean;
  onSelect: (symbol: string | null) => void;
  onClear: () => void;
}

/** Reconstruct the detected coordinate table as selectable sidebar rows. */
export function ReferencePanel({
  table,
  selectedSymbol,
  disabled = false,
  onSelect,
  onClear,
}: ReferencePanelProps) {
  const located = table.points.filter((point) => point.markers.length > 0);
  const missing = table.points.length - located.length;

  return (
    <section>
      <div className="flex items-start justify-between gap-2 px-2 py-2">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-slate-900">
            Reference points
          </h2>
          <p className="text-xs text-slate-500">
            Page {table.page} · {table.points.length} point
            {table.points.length === 1 ? "" : "s"} · {table.axes.join("/")}
            {missing > 0 && (
              <>
                {" · "}
                <span className="text-amber-700">{missing} not located</span>
              </>
            )}
          </p>
        </div>
        <button
          type="button"
          onClick={onClear}
          disabled={disabled}
          title="Remove temporary reference-point markers"
          className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-200 hover:text-slate-800 disabled:opacity-50"
        >
          Clear
        </button>
      </div>

      <div className="pb-3">
        <table className="w-full border-separate border-spacing-0 text-left">
          <thead>
            <tr className="text-[10px] uppercase tracking-wide text-slate-500">
              <th scope="col" className="px-2 py-1 font-semibold">
                {table.nameHeader || "Point"}
              </th>
              {table.axes.map((axis) => (
                <th
                  key={axis}
                  scope="col"
                  className="px-1 py-1 text-right font-semibold"
                >
                  {axis}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.points.map((point, index) => {
              const color = referenceColor(index);
              const selected = selectedSymbol === point.symbol;
              const unlocated = point.markers.length === 0;
              return (
                <tr
                  key={`${point.symbol}-${index}`}
                  onClick={() =>
                    !disabled && onSelect(selected ? null : point.symbol)
                  }
                  title={
                    unlocated
                      ? `${point.symbol} was read from the table but not located in a drawing view`
                      : `Highlight ${point.symbol} in ${point.markers.length} drawing location${
                          point.markers.length === 1 ? "" : "s"
                        }`
                  }
                  style={selected ? { backgroundColor: color.wash } : undefined}
                  className={`text-xs transition ${
                    disabled
                      ? "cursor-not-allowed opacity-60"
                      : "cursor-pointer hover:bg-slate-100"
                  }`}
                >
                  <th
                    scope="row"
                    className="px-2 py-1 text-left font-medium text-slate-800"
                  >
                    <span className="flex items-center gap-1.5">
                      <span
                        aria-hidden
                        style={{ backgroundColor: color.fill }}
                        className={`inline-block h-3 w-3 shrink-0 rounded-sm ring-1 ring-slate-900/30 ${
                          selected ? "ring-2 ring-slate-900/60" : ""
                        }`}
                      />
                      <span className="font-mono">{point.symbol}</span>
                      {unlocated ? (
                        <span
                          title="Named in the table but not found in the views"
                          className="rounded bg-amber-100 px-1 text-[9px] font-medium text-amber-800"
                        >
                          ?
                        </span>
                      ) : (
                        point.markers.length > 1 && (
                          <span className="text-[9px] text-slate-400">
                            ×{point.markers.length}
                          </span>
                        )
                      )}
                    </span>
                  </th>
                  {table.axes.map((axis) => (
                    <td
                      key={axis}
                      className="px-1 py-1 text-right font-mono text-[11px] text-slate-700"
                    >
                      {point.coordinates.find((item) => item.axis === axis)
                        ?.value ?? "—"}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
