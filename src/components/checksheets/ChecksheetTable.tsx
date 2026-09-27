"use client";

import { validateChecksheetReading } from "@/lib/checksheetRange";
import type { ChecksheetColumn, ChecksheetRow } from "@/types/checksheet";

interface ChecksheetTableProps {
  columns: ChecksheetColumn[];
  rows: ChecksheetRow[];
  activeAnnotationId: string | null;
  readOnly: boolean;
  onActivate: (annotationId: string) => void;
  onReadingChange: (
    annotationId: string,
    columnId: string,
    value: string
  ) => void;
  onNavigate: (offset: number) => void;
}

export function ChecksheetTable({
  columns,
  rows,
  activeAnnotationId,
  readOnly,
  onActivate,
  onReadingChange,
  onNavigate,
}: ChecksheetTableProps) {
  return (
    <div className="overflow-auto rounded-xl border border-slate-200 bg-white shadow-sm">
      <table className="w-full min-w-[900px] border-collapse text-left">
        <thead className="sticky top-0 z-10 bg-slate-50 shadow-[0_1px_0_0_#e2e8f0]">
          <tr className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            <th className="w-16 px-3 py-3">Balloon</th>
            <th className="px-3 py-3">Label</th>
            <th className="px-3 py-3">Specification</th>
            <th className="px-3 py-3">Tolerance</th>
            {columns.map((column) => (
              <th key={column.id} className="min-w-36 px-3 py-3 text-blue-700">
                {column.name}
              </th>
            ))}
            <th className="px-3 py-3">Method</th>
            <th className="px-3 py-3">Tool</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const active = row.annotation_id === activeAnnotationId;
            return (
              <tr
                key={row.annotation_id}
                data-checksheet-row={row.annotation_id}
                onClick={() => onActivate(row.annotation_id)}
                className={`border-b border-slate-100 transition-colors ${
                  active
                    ? "bg-blue-100/80 shadow-[inset_4px_0_0_#2563eb]"
                    : "hover:bg-blue-50/50"
                }`}
              >
                <td className="px-3 py-2 text-center">
                  <span
                    className={`inline-flex h-8 min-w-8 items-center justify-center rounded-full px-2 text-sm font-bold ${
                      active
                        ? "bg-blue-600 text-white"
                        : "border border-slate-300 bg-white text-slate-700"
                    }`}
                  >
                    {row.balloon_number}
                  </span>
                </td>
                <td className="max-w-48 px-3 py-2 text-sm text-slate-700">
                  {row.label || <span className="text-slate-300">—</span>}
                </td>
                <td className="px-3 py-2 font-mono text-sm font-medium text-slate-900">
                  {row.specification}
                </td>
                <td className="px-3 py-2 font-mono text-sm text-slate-700">
                  {row.tolerance || <span className="text-slate-300">—</span>}
                </td>
                {columns.map((column) => {
                  const reading = row.readings[column.id] ?? "";
                  const validation = validateChecksheetReading(
                    reading,
                    row.specification,
                    row.tolerance
                  );
                  const failed =
                    validation.state === "invalid" ||
                    validation.state === "out_of_range";
                  const passed = validation.state === "in_range";
                  const tone = failed
                    ? "border-red-400 bg-red-50 text-red-900 focus:border-red-500 focus:ring-red-500/20"
                    : passed
                      ? "border-emerald-400 bg-emerald-50 text-emerald-950 focus:border-emerald-500 focus:ring-emerald-500/20"
                      : readOnly
                        ? "border-transparent bg-slate-100 text-slate-600"
                        : "border-blue-200 bg-white text-slate-900 focus:border-blue-500 focus:ring-blue-500/20";
                  return (
                    <td key={column.id} className="px-2 py-1.5 align-top">
                      <input
                        value={reading}
                        readOnly={readOnly}
                        inputMode="decimal"
                        aria-invalid={failed || undefined}
                        title={validation.message || undefined}
                        onFocus={() => onActivate(row.annotation_id)}
                        onClick={(event) => event.stopPropagation()}
                        onChange={(event) =>
                          onReadingChange(
                            row.annotation_id,
                            column.id,
                            event.target.value
                          )
                        }
                        onKeyDown={(event) => {
                          if (
                            event.key === "Enter" ||
                            event.key === "ArrowDown"
                          ) {
                            event.preventDefault();
                            onNavigate(1);
                          } else if (event.key === "ArrowUp") {
                            event.preventDefault();
                            onNavigate(-1);
                          }
                        }}
                        className={`w-full rounded-lg border px-2.5 py-2 font-mono text-sm outline-none focus:ring-2 ${tone}`}
                      />
                      {failed && validation.message && (
                        <p className="mt-1 text-[11px] font-medium leading-tight text-red-600">
                          {validation.message}
                        </p>
                      )}
                    </td>
                  );
                })}
                <td className="max-w-48 px-3 py-2 text-sm text-slate-600">
                  {row.method || <span className="text-slate-300">—</span>}
                </td>
                <td className="max-w-48 px-3 py-2 text-sm text-slate-600">
                  {row.tool || <span className="text-slate-300">—</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
