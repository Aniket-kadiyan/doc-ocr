"use client";

import { useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { useAnnotationStore } from "@/store/annotationStore";
import { classifyDimension } from "@/lib/dimensionClassifier";
import type { Annotation } from "@/types/annotation";
import { SPECIAL_SYMBOLS, deriveRange } from "@/lib/valueFields";

/** Confirm the OCR read before the value receives its balloon number. Optional
 * inspection metadata is deliberately edited later from the sidebar item. */
export function AnnotationPopup() {
  const pending = useAnnotationStore((state) => state.pending);
  const addAnnotation = useAnnotationStore((state) => state.addAnnotation);
  const getNextNumber = useAnnotationStore((state) => state.getNextNumber);
  const setPending = useAnnotationStore((state) => state.setPending);

  const [value, setValue] = useState("");
  const [range, setRange] = useState("");
  const valueInputRef = useRef<HTMLInputElement>(null);
  const rangeEditedRef = useRef(false);

  useEffect(() => {
    if (!pending) return;
    const text = pending.ocrResult.text;
    setValue(text);
    rangeEditedRef.current = false;
    setRange(deriveRange(text));
  }, [pending]);

  // Keep tolerance synchronized with an embedded ± value until it is edited.
  useEffect(() => {
    if (!rangeEditedRef.current) setRange(deriveRange(value));
  }, [value]);

  if (!pending) return null;

  const confidence = pending.ocrResult.confidence;
  const needsReview = pending.ocrResult.needsReview ?? false;

  const insertSymbol = (symbol: string) => {
    const input = valueInputRef.current;
    if (!input) {
      setValue((current) => current + symbol);
      return;
    }
    const start = input.selectionStart ?? value.length;
    const end = input.selectionEnd ?? value.length;
    const next = value.slice(0, start) + symbol + value.slice(end);
    setValue(next);
    requestAnimationFrame(() => {
      input.focus();
      const position = start + symbol.length;
      input.setSelectionRange(position, position);
    });
  };

  const resetForm = () => {
    setValue("");
    setRange("");
    rangeEditedRef.current = false;
  };

  const handleSave = () => {
    const cleanValue = value.trim();
    const annotation: Annotation = {
      id: uuidv4(),
      number: getNextNumber(),
      label: "",
      value: cleanValue,
      type: classifyDimension(cleanValue),
      confidence,
      bbox: pending.ocrResult.valueBox ?? pending.bbox,
      rotation: pending.ocrResult.rotation,
      page: pending.page,
      createdAt: Date.now(),
      kind: "dimension",
      needsReview,
      range: range.trim() || undefined,
    };
    addAnnotation(annotation);
    resetForm();
  };

  const handleCancel = () => {
    setPending(null);
    resetForm();
  };

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
            Extracted Value
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            {value.trim()
              ? "Review the OCR value and tolerance before creating its balloon."
              : "OCR returned empty — type the value manually."}
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
              Value
            </label>
            <input
              ref={valueInputRef}
              type="text"
              value={value}
              autoFocus
              onChange={(event) => setValue(event.target.value)}
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
              onChange={(event) => {
                rangeEditedRef.current = true;
                setRange(event.target.value);
              }}
              placeholder="e.g. ±0.10 or +0.10, -0.20"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
            />
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
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
}
