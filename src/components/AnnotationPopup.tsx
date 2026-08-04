"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { useAnnotationStore } from "@/store/annotationStore";
import { classifyDimension } from "@/lib/dimensionClassifier";
import { TOOL_OPTIONS, type Annotation } from "@/types/annotation";
import { SPECIAL_SYMBOLS, deriveRange } from "@/lib/valueFields";

export function AnnotationPopup() {
  const pending = useAnnotationStore((s) => s.pending);
  const annotations = useAnnotationStore((s) => s.annotations);
  const addAnnotation = useAnnotationStore((s) => s.addAnnotation);
  const getNextNumber = useAnnotationStore((s) => s.getNextNumber);
  const setPending = useAnnotationStore((s) => s.setPending);

  const [labelId, setLabelId] = useState("");
  const [value, setValue] = useState("");
  const [method, setMethod] = useState("");
  const [tool, setTool] = useState("");
  const [range, setRange] = useState("");
  const valueInputRef = useRef<HTMLInputElement>(null);

  // Labels created with the Add Label tool — the dimension maps to one of these.
  const labelOptions = useMemo(
    () =>
      annotations
        .filter((a) => a.kind === "label" && a.value.trim())
        .sort((a, b) => a.number - b.number)
        .map((a) => ({ id: a.id, text: a.value.trim() })),
    [annotations]
  );

  // Insert a symbol at the caret (or replace the selection) and keep focus.
  const insertSymbol = (sym: string) => {
    const input = valueInputRef.current;
    if (!input) {
      setValue((v) => v + sym);
      return;
    }
    const start = input.selectionStart ?? value.length;
    const end = input.selectionEnd ?? value.length;
    const next = value.slice(0, start) + sym + value.slice(end);
    setValue(next);
    requestAnimationFrame(() => {
      input.focus();
      const pos = start + sym.length;
      input.setSelectionRange(pos, pos);
    });
  };

  // Becomes true once the user hand-edits the range, which stops auto-syncing
  // it from the value's ± tolerance.
  const rangeEditedRef = useRef(false);

  useEffect(() => {
    if (!pending) return;
    const text = pending.ocrResult.text;
    setValue(text);
    // When adding a value from a label's row, bind to that label up front and
    // inherit the method/tool/range already set on the label.
    setLabelId(pending.labelId ?? "");
    const parent = pending.labelId
      ? annotations.find((a) => a.id === pending.labelId)
      : undefined;
    setMethod(parent?.method ?? "");
    setTool(parent?.tool ?? "");
    rangeEditedRef.current = false;
    // Pre-fill the range from the label or from any ± tolerance in the value.
    setRange(
      pending.kind === "label" ? "" : parent?.range || deriveRange(text)
    );
  }, [pending, annotations]);

  // Keep the range in sync as the user edits a ± value, until they hand-edit it.
  useEffect(() => {
    if (!rangeEditedRef.current) setRange(deriveRange(value));
  }, [value]);

  if (!pending) return null;

  const isLabel = pending.kind === "label";
  const type = classifyDimension(value);
  const confidence = pending.ocrResult.confidence;
  const needsReview = pending.ocrResult.needsReview ?? false;

  // Text of the mapped label, stored alongside labelId for display/export.
  const mappedLabel = labelOptions.find((o) => o.id === labelId)?.text ?? "";

  const handleSave = () => {
    const kind = pending.kind ?? "dimension";
    const annotation: Annotation = {
      id: uuidv4(),
      number: getNextNumber(kind),
      label: isLabel ? "" : mappedLabel,
      value: value.trim(),
      type,
      confidence,
      bbox: pending.ocrResult.valueBox ?? pending.bbox,
      rotation: pending.ocrResult.rotation,
      page: pending.page,
      createdAt: Date.now(),
      kind,
      needsReview,
      labelSource: isLabel ? pending.labelSource ?? "manual" : undefined,
      // Method / tool / range now apply to labels too — a label carries its own
      // inspection method and tool, which its value inherits.
      method: method.trim() || undefined,
      tool: tool || undefined,
      labelId: isLabel ? undefined : labelId || undefined,
      range: range.trim() || undefined,
    };
    addAnnotation(annotation);
    resetForm();
  };

  const handleCancel = () => {
    setPending(null);
    resetForm();
  };

  function resetForm() {
    setLabelId("");
    setValue("");
    setMethod("");
    setTool("");
    setRange("");
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div
        className="w-full max-w-md rounded-xl border border-slate-200 bg-white shadow-2xl"
        role="dialog"
        aria-labelledby="annotation-dialog-title"
      >
        <div className="border-b border-slate-100 px-5 py-4">
          <h2
            id="annotation-dialog-title"
            className="text-lg font-semibold text-slate-900"
          >
            {isLabel ? "Add Label" : "Extracted Value"}
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            {isLabel
              ? pending.labelSource === "ocr"
                ? "Review the OCR'd label text, edit if needed, then add it."
                : "Type the label text for the box you drew."
              : pending.labelId
                ? value.trim()
                  ? "Review the value, set tool / method / tolerance, then save."
                  : "OCR returned empty — type the value for this label manually."
                : value.trim()
                  ? "Review OCR result, edit if needed, then add a label."
                  : "OCR returned empty — type the dimension value manually."}
          </p>
        </div>

        <div className="space-y-4 px-5 py-4">
          {needsReview && (
            <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-800">
              ⚠ Low-confidence read — please verify the text before saving.
            </div>
          )}
          <div>
            <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
              {isLabel ? "Label text" : "Value"}
            </label>
            <input
              ref={valueInputRef}
              type="text"
              value={value}
              autoFocus={isLabel}
              onChange={(e) => setValue(e.target.value)}
              placeholder={isLabel ? "e.g. SECTION A-A" : ""}
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
            <p className="mt-1 text-[11px] text-slate-400">
              Click a symbol to insert it at the cursor.
            </p>
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
            <p className="mt-1 text-[11px] text-slate-400">
              Upper and lower tolerance, comma-separated.
            </p>
          </div>

          {/* Mapping picker: dimensions only (labels don't map to a label). */}
          {!isLabel && (
            <div>
              <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                Label
              </label>
              {pending.labelId ? (
                <p className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-sm font-medium text-blue-800">
                  {mappedLabel || "—"}
                </p>
              ) : labelOptions.length > 0 ? (
                <select
                  value={labelId}
                  onChange={(e) => setLabelId(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                >
                  <option value="">— No label —</option>
                  {labelOptions.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.text}
                    </option>
                  ))}
                </select>
              ) : (
                <p className="rounded-lg border border-dashed border-slate-200 px-3 py-2 text-xs text-slate-400">
                  No labels yet — add some with the Add Label tool to map them
                  here.
                </p>
              )}
            </div>
          )}

          {/* Method + tool apply to both labels and values. */}
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
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-100 px-5 py-4">
          <button
            type="button"
            onClick={handleCancel}
            className="rounded-lg px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={!value.trim()}
            className={`rounded-lg px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50 ${
              isLabel
                ? "bg-indigo-600 hover:bg-indigo-700"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {isLabel ? "Add Label" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
