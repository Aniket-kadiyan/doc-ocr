"use client";

import type { Annotation } from "@/types/annotation";
import { BalloonThumbnail } from "@/components/BalloonThumbnail";

interface SidebarProps {
  annotations: Annotation[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onEdit: (id: string) => void;
  onDelete: (id: string) => void;
}

/** One sidebar record per ballooned value. Label, method, and tool are optional
 * metadata on that same record rather than separate annotations. */
export function Sidebar({
  annotations,
  selectedId,
  onSelect,
  onEdit,
  onDelete,
}: SidebarProps) {
  const values = annotations
    .filter((annotation) => annotation.kind !== "label")
    .sort((left, right) => left.number - right.number);

  return (
    <aside className="flex w-72 shrink-0 flex-col border-l border-slate-200 bg-slate-50">
      <div className="border-b border-slate-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-900">Values</h2>
        <p className="text-xs text-slate-500">
          {values.length} ballooned value{values.length === 1 ? "" : "s"}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {values.length === 0 ? (
          <p className="px-2 py-6 text-center text-sm text-slate-500">
            Choose Draw Value, then draw a box around a value on the drawing.
          </p>
        ) : (
          <ul>
            {values.map((annotation) => {
              const selected = selectedId === annotation.id;
              const details = [
                annotation.type,
                annotation.range && `Tolerance ${annotation.range}`,
                annotation.method,
                annotation.tool,
              ]
                .filter(Boolean)
                .join(" · ");

              return (
                <li key={annotation.id} className="mb-2">
                  <button
                    type="button"
                    onClick={() => {
                      onSelect(annotation.id);
                      onEdit(annotation.id);
                    }}
                    title="Edit this value and its optional inspection metadata"
                    className={`w-full rounded-lg px-3 py-2 text-left transition ${
                      selected
                        ? "bg-blue-100 ring-1 ring-blue-300"
                        : "bg-white hover:bg-slate-100"
                    }`}
                  >
                    <span className="flex items-start gap-2">
                      <BalloonThumbnail
                        number={annotation.number}
                        selected={selected}
                      />
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-1">
                          {annotation.needsReview && (
                            <span
                              title="Low-confidence read — verify"
                              className="inline-block h-2 w-2 shrink-0 rounded-full bg-amber-500"
                            />
                          )}
                          <span className="truncate font-mono text-xs text-slate-800">
                            {annotation.value}
                          </span>
                        </span>
                        {annotation.label && (
                          <span className="mt-1 block truncate text-xs font-medium text-slate-600">
                            {annotation.label}
                          </span>
                        )}
                        {details && (
                          <span className="mt-1 block text-[10px] text-slate-400">
                            {details}
                          </span>
                        )}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onDelete(annotation.id)}
                    className="mt-0.5 w-full text-center text-[10px] text-red-500 hover:underline"
                  >
                    Remove
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}
