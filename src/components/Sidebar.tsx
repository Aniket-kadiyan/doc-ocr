"use client";

import type { Annotation } from "@/types/annotation";
import { classifyDimension } from "@/lib/dimensionClassifier";

interface SidebarProps {
  annotations: Annotation[];
  selectedId: string | null;
  /** Selected annotation plus any mapped partner(s) — all rendered highlighted. */
  highlightedIds?: Set<string>;
  onSelect: (id: string) => void;
  onUpdate: (id: string, patch: Partial<Annotation>) => void;
  onDelete: (id: string) => void;
  /** Start drawing a value box that binds one-to-one to this label. */
  onAddValue: (labelId: string) => void;
  /** Open the edit modal for a label and all the values associated with it. */
  onEditLabel: (labelId: string) => void;
}

const isDimension = (a: Annotation) => (a.kind ?? "dimension") === "dimension";

export function Sidebar({
  annotations,
  selectedId,
  highlightedIds,
  onSelect,
  onUpdate,
  onDelete,
  onAddValue,
  onEditLabel,
}: SidebarProps) {
  const labels = annotations
    .filter((a) => a.kind === "label")
    .sort((a, b) => a.number - b.number);
  const labelIds = new Set(labels.map((l) => l.id));

  const dimensions = annotations.filter(isDimension);
  // Each label owns at most one value (one-to-one). Unmapped values — e.g. from
  // Auto-Segment — keep their own section until mapped to a label.
  const valueForLabel = (labelId: string) =>
    dimensions.find((d) => d.labelId === labelId);
  const unmappedValues = dimensions
    .filter((d) => !d.labelId || !labelIds.has(d.labelId))
    .sort((a, b) => a.number - b.number);

  const isHighlighted = (id: string) =>
    highlightedIds?.has(id) ?? selectedId === id;

  // Editable text + detail line shared by mapped values and unmapped values.
  const valueDetail = (ann: Annotation) =>
    [ann.type, ann.tool, ann.range && `± ${ann.range}`]
      .filter(Boolean)
      .join(" · ");

  const valueRow = (ann: Annotation) => (
    <div className="space-y-1">
      <div className="flex items-center gap-1">
        {ann.needsReview && (
          <span
            title="Low-confidence read — verify"
            className="inline-block h-2 w-2 shrink-0 rounded-full bg-amber-500"
          />
        )}
        <input
          type="text"
          value={ann.value}
          onClick={(e) => e.stopPropagation()}
          onChange={(e) =>
            onUpdate(ann.id, {
              value: e.target.value,
              type: classifyDimension(e.target.value),
            })
          }
          className="w-full rounded border border-transparent bg-transparent px-1 py-0.5 font-mono text-xs text-slate-800 hover:border-slate-200 focus:border-blue-400 focus:bg-white focus:outline-none"
        />
      </div>
      {valueDetail(ann) && (
        <p className="px-1 text-[10px] text-slate-400">{valueDetail(ann)}</p>
      )}
    </div>
  );

  // A label and its one-to-one value. Clicking the row opens the edit modal
  // where the label and all its values can be edited.
  const renderLabel = (label: Annotation) => {
    const value = valueForLabel(label.id);
    const highlighted = isHighlighted(label.id);
    return (
      <li key={label.id} className="mb-2">
        <div
          onClick={() => onEditLabel(label.id)}
          title="Click to edit this label and its values"
          className={`w-full cursor-pointer rounded-lg px-3 py-2 text-left transition ${
            highlighted
              ? "bg-indigo-100 ring-1 ring-indigo-300"
              : "bg-white hover:bg-slate-100"
          }`}
        >
          <div className="flex items-start gap-2">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border-2 border-indigo-500 text-xs font-bold text-indigo-600">
              {label.number}
            </span>
            <div className="min-w-0 flex-1 space-y-1">
              <p className="truncate px-1 py-0.5 text-xs font-medium text-slate-800">
                {label.value || (
                  <span className="text-slate-400">(unnamed label)</span>
                )}
              </p>

              {/* The value belonging to this label, nested one-to-one. */}
              {value ? (
                <div className="ml-1 space-y-0.5 border-l-2 border-blue-200 pl-2">
                  <p className="flex items-center gap-1 px-1 font-mono text-xs text-slate-800">
                    {value.needsReview && (
                      <span
                        title="Low-confidence read — verify"
                        className="inline-block h-2 w-2 shrink-0 rounded-full bg-amber-500"
                      />
                    )}
                    {value.value}
                  </p>
                  {valueDetail(value) && (
                    <p className="px-1 text-[10px] text-slate-400">
                      {valueDetail(value)}
                    </p>
                  )}
                </div>
              ) : (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onAddValue(label.id);
                  }}
                  className="ml-1 rounded border border-blue-200 bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-700 hover:bg-blue-100"
                >
                  + Add value
                </button>
              )}
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(label.id);
          }}
          className="mt-0.5 w-full text-center text-[10px] text-red-500 hover:underline"
        >
          Remove label
        </button>
      </li>
    );
  };

  // An unmapped value (e.g. from Auto-Segment) with a dropdown to attach it to
  // a label. Only labels without a value are offered — values are one-to-one.
  const labelOptions = labels
    .filter((a) => a.value.trim() && !valueForLabel(a.id))
    .map((a) => ({ id: a.id, text: a.value.trim() }));
  const labelText = (id?: string) =>
    id ? annotations.find((a) => a.id === id)?.value : undefined;

  const renderUnmapped = (ann: Annotation) => {
    const highlighted = isHighlighted(ann.id);
    return (
      <li key={ann.id} className="mb-1">
        <div
          onClick={() => onSelect(ann.id)}
          className={`w-full rounded-lg px-3 py-2 text-left transition ${
            highlighted
              ? "bg-blue-100 ring-1 ring-blue-300"
              : "bg-white hover:bg-slate-100"
          }`}
        >
          <div className="flex items-start gap-2">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border-2 border-red-500 text-xs font-bold text-red-600">
              {ann.number}
            </span>
            <div className="min-w-0 flex-1 space-y-1">
              {valueRow(ann)}
              {labelOptions.length > 0 && (
                <select
                  value={ann.labelId ?? ""}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) =>
                    onUpdate(ann.id, {
                      labelId: e.target.value || undefined,
                      label: labelText(e.target.value) ?? "",
                    })
                  }
                  className="w-full rounded border border-transparent bg-transparent px-1 py-0.5 text-xs text-slate-500 hover:border-slate-200 focus:border-blue-400 focus:bg-white focus:outline-none"
                >
                  <option value="">→ attach to label</option>
                  {labelOptions.map((o) => (
                    <option key={o.id} value={o.id}>
                      → {o.text}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(ann.id);
          }}
          className="mt-0.5 w-full text-center text-[10px] text-red-500 hover:underline"
        >
          Remove
        </button>
      </li>
    );
  };

  return (
    <aside className="flex w-72 shrink-0 flex-col border-l border-slate-200 bg-slate-50">
      <div className="border-b border-slate-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-900">Annotations</h2>
        <p className="text-xs text-slate-500">
          {labels.length} label{labels.length === 1 ? "" : "s"}
          {unmappedValues.length > 0 &&
            ` · ${unmappedValues.length} unmapped value${
              unmappedValues.length === 1 ? "" : "s"
            }`}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {labels.length === 0 && unmappedValues.length === 0 ? (
          <p className="px-2 py-6 text-center text-sm text-slate-500">
            Add a label, then add its value from the drawing.
          </p>
        ) : (
          <>
            {labels.length > 0 && (
              <>
                <p className="px-2 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wide text-indigo-400">
                  Labels
                </p>
                <ul>{labels.map(renderLabel)}</ul>
              </>
            )}
            {unmappedValues.length > 0 && (
              <>
                <p className="px-2 pb-1 pt-3 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                  Unmapped values
                </p>
                <ul>{unmappedValues.map(renderUnmapped)}</ul>
              </>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
