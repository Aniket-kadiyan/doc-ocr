"use client";

import { useEffect, useRef, useState } from "react";
import { classifyDimension } from "@/lib/dimensionClassifier";
import { SPECIAL_SYMBOLS, deriveRange } from "@/lib/valueFields";
import { TOOL_OPTIONS, type Annotation } from "@/types/annotation";

interface ValueEditorProps {
  annotation: Annotation;
  onUpdate: (id: string, patch: Partial<Annotation>) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

/** Edit one ballooned value and its optional checksheet metadata. */
export function ValueEditor({
  annotation,
  onUpdate,
  onDelete,
  onClose,
}: ValueEditorProps) {
  const [value, setValue] = useState(annotation.value);
  const [range, setRange] = useState(annotation.range ?? "");
  const [label, setLabel] = useState(annotation.label ?? "");
  const [method, setMethod] = useState(annotation.method ?? "");
  const [tool, setTool] = useState(annotation.tool ?? "");
  const valueInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setValue(annotation.value);
    setRange(annotation.range ?? deriveRange(annotation.value));
    setLabel(annotation.label ?? "");
    setMethod(annotation.method ?? "");
    setTool(annotation.tool ?? "");
  }, [annotation.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const updateValue = (nextValue: string) => {
    setValue(nextValue);
    setRange(deriveRange(nextValue));
  };

  const insertSymbol = (symbol: string) => {
    const input = valueInputRef.current;
    if (!input) {
      updateValue(value + symbol);
      return;
    }
    const start = input.selectionStart ?? value.length;
    const end = input.selectionEnd ?? value.length;
    updateValue(value.slice(0, start) + symbol + value.slice(end));
    requestAnimationFrame(() => {
      input.focus();
      const position = start + symbol.length;
      input.setSelectionRange(position, position);
    });
  };

  const handleSave = () => {
    const cleanValue = value.trim();
    onUpdate(annotation.id, {
      value: cleanValue,
      type: classifyDimension(cleanValue),
      range: range.trim() || undefined,
      label: label.trim(),
      method: method.trim() || undefined,
      tool: tool || undefined,
    });
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-xl border border-slate-200 bg-white shadow-2xl"
        role="dialog"
        aria-labelledby="value-editor-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="border-b border-slate-100 px-5 py-4">
          <h2
            id="value-editor-title"
            className="text-lg font-semibold text-slate-900"
          >
            Edit Balloon {annotation.number}
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Label, method, and tool are optional metadata for this value.
          </p>
        </div>

        <div className="space-y-4 px-5 py-4">
          <div>
            <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
              Value
            </label>
            <input
              ref={valueInputRef}
              type="text"
              value={value}
              onChange={(event) => updateValue(event.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
            />
            <div className="mt-2 flex flex-wrap gap-1">
              {SPECIAL_SYMBOLS.map((symbol) => (
                <button
                  key={symbol.ch}
                  type="button"
                  title={`Insert ${symbol.title} (${symbol.ch})`}
                  onClick={() => insertSymbol(symbol.ch)}
                  className="flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 font-mono text-sm text-slate-700 hover:border-blue-400 hover:bg-blue-50"
                >
                  {symbol.ch}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
              Tolerance
            </label>
            <input
              type="text"
              value={range}
              onChange={(event) => setRange(event.target.value)}
              placeholder="e.g. ±0.10 or +0.10, -0.20"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
              Label (optional)
            </label>
            <input
              type="text"
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              placeholder="e.g. Shaft diameter"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                Method (optional)
              </label>
              <input
                type="text"
                value={method}
                onChange={(event) => setMethod(event.target.value)}
                placeholder="e.g. Measure"
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                Tool (optional)
              </label>
              <select
                value={tool}
                onChange={(event) => setTool(event.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              >
                <option value="">— Select tool —</option>
                {TOOL_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              onDelete(annotation.id);
              onClose();
            }}
            className="text-xs font-medium text-red-500 hover:underline"
          >
            Remove value and balloon
          </button>
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-100 px-5 py-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={!value.trim()}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
