"use client";

import { useEffect, useRef, useState } from "react";
import { classifyDimension } from "@/lib/dimensionClassifier";
import { SPECIAL_SYMBOLS, deriveRange } from "@/lib/valueFields";
import { TOOL_OPTIONS, type Annotation } from "@/types/annotation";

interface LabelEditorProps {
  label: Annotation;
  /** The value bound one-to-one to this label, if one has been added. */
  value: Annotation | undefined;
  onUpdate: (id: string, patch: Partial<Annotation>) => void;
  onDeleteValue: (id: string) => void;
  /** Close the editor and start drawing a value box for this label. */
  onAddValue: (labelId: string) => void;
  onClose: () => void;
}

/**
 * Click-to-edit modal for a label and all the values associated with it
 * (value, tolerance/range, method, tool). Saving writes back to the
 * annotations, which the project JSON serializes on the next save.
 */
export function LabelEditor({
  label,
  value,
  onUpdate,
  onDeleteValue,
  onAddValue,
  onClose,
}: LabelEditorProps) {
  const [labelText, setLabelText] = useState(label.value);
  const [val, setVal] = useState(value?.value ?? "");
  const [range, setRange] = useState(value?.range ?? "");
  const [method, setMethod] = useState(value?.method ?? "");
  const [tool, setTool] = useState(value?.tool ?? "");
  const valueInputRef = useRef<HTMLInputElement>(null);
  const rangeEditedRef = useRef(false);

  // Re-seed the form whenever a different label is opened.
  useEffect(() => {
    setLabelText(label.value);
    setVal(value?.value ?? "");
    setRange(value?.range ?? "");
    setMethod(value?.method ?? "");
    setTool(value?.tool ?? "");
    rangeEditedRef.current = false;
  }, [label.id, value?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keep range synced to the value's ± tolerance until the user hand-edits it.
  useEffect(() => {
    if (!rangeEditedRef.current) setRange(deriveRange(val));
  }, [val]);

  // Insert a symbol at the caret (or replace the selection) and keep focus.
  const insertSymbol = (sym: string) => {
    const input = valueInputRef.current;
    if (!input) {
      setVal((v) => v + sym);
      return;
    }
    const start = input.selectionStart ?? val.length;
    const end = input.selectionEnd ?? val.length;
    setVal(val.slice(0, start) + sym + val.slice(end));
    requestAnimationFrame(() => {
      input.focus();
      const pos = start + sym.length;
      input.setSelectionRange(pos, pos);
    });
  };

  const handleSave = () => {
    onUpdate(label.id, { value: labelText.trim() });
    if (value) {
      onUpdate(value.id, {
        value: val.trim(),
        type: classifyDimension(val),
        range: range.trim() || undefined,
        method: method.trim() || undefined,
        tool: tool || undefined,
        label: labelText.trim(),
      });
    }
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
        aria-labelledby="label-editor-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-slate-100 px-5 py-4">
          <h2
            id="label-editor-title"
            className="text-lg font-semibold text-slate-900"
          >
            Edit label
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Edit the label and every value associated with it.
          </p>
        </div>

        <div className="space-y-4 px-5 py-4">
          <div>
            <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
              Label text
            </label>
            <input
              type="text"
              value={labelText}
              onChange={(e) => setLabelText(e.target.value)}
              placeholder="e.g. SECTION A-A"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
            />
          </div>

          {value ? (
            <>
              <div>
                <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                  Value
                </label>
                <input
                  ref={valueInputRef}
                  type="text"
                  value={val}
                  onChange={(e) => setVal(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                />
                <div className="mt-2 flex flex-wrap gap-1">
                  {SPECIAL_SYMBOLS.map((s) => (
                    <button
                      key={s.ch}
                      type="button"
                      title={`Insert ${s.title} (${s.ch})`}
                      onClick={() => insertSymbol(s.ch)}
                      className="flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 font-mono text-sm text-slate-700 hover:border-blue-400 hover:bg-blue-50"
                    >
                      {s.ch}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                  Range (± tolerance)
                </label>
                <input
                  type="text"
                  value={range}
                  onChange={(e) => {
                    rangeEditedRef.current = true;
                    setRange(e.target.value);
                  }}
                  placeholder="auto-filled from ± — e.g. +0.50, -0.50"
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                    Method
                  </label>
                  <input
                    type="text"
                    value={method}
                    onChange={(e) => setMethod(e.target.value)}
                    placeholder="e.g. Visual / Functional"
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                    Tool
                  </label>
                  <select
                    value={tool}
                    onChange={(e) => setTool(e.target.value)}
                    className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                  >
                    <option value="">— Select tool —</option>
                    {TOOL_OPTIONS.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                type="button"
                onClick={() => {
                  onDeleteValue(value.id);
                  onClose();
                }}
                className="text-xs font-medium text-red-500 hover:underline"
              >
                Remove value
              </button>
            </>
          ) : (
            <div className="rounded-lg border border-dashed border-slate-200 px-3 py-3 text-center">
              <p className="text-xs text-slate-400">
                No value yet for this label.
              </p>
              <button
                type="button"
                onClick={() => {
                  onUpdate(label.id, { value: labelText.trim() });
                  onClose();
                  onAddValue(label.id);
                }}
                className="mt-2 rounded border border-blue-200 bg-blue-50 px-2 py-1 text-[11px] font-medium text-blue-700 hover:bg-blue-100"
              >
                + Add value
              </button>
            </div>
          )}
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
            disabled={!labelText.trim()}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
