"use client";

import {
  DOCUMENT_METADATA_FIELDS,
  type DocumentMetadata,
  type DocumentMetadataField,
} from "@/types/documentMetadata";

interface MetadataPanelProps {
  metadata: DocumentMetadata;
  activeField: DocumentMetadataField | null;
  disabled?: boolean;
  drawingAvailable: boolean;
  onChange: (field: DocumentMetadataField, value: string) => void;
  onSelect: (field: DocumentMetadataField) => void;
  onClear: (field: DocumentMetadataField) => void;
}

export function MetadataPanel({
  metadata,
  activeField,
  disabled = false,
  drawingAvailable,
  onChange,
  onSelect,
  onClear,
}: MetadataPanelProps) {
  return (
    <div className="space-y-4 p-3">
      <p className="text-xs leading-5 text-slate-500">
        Enter each value or select it directly from the drawing. Blank fields
        are allowed.
      </p>

      {DOCUMENT_METADATA_FIELDS.map(({ key, label }) => {
        const isActive = activeField === key;
        return (
          <section
            key={key}
            className={`rounded-lg border bg-white p-3 ${
              isActive ? "border-blue-400 ring-1 ring-blue-200" : "border-slate-200"
            }`}
          >
            <label
              htmlFor={`document-metadata-${key}`}
              className="block text-xs font-semibold text-slate-700"
            >
              {label}
            </label>
            <input
              id={`document-metadata-${key}`}
              type="text"
              value={metadata[key]}
              disabled={disabled}
              onChange={(event) => onChange(key, event.target.value)}
              className="mt-1.5 w-full rounded-md border border-slate-300 bg-white px-2.5 py-2 text-sm text-slate-900 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:cursor-not-allowed disabled:bg-slate-100"
            />
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                disabled={disabled || !drawingAvailable}
                onClick={() => onSelect(key)}
                className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
                  isActive
                    ? "bg-blue-600 text-white hover:bg-blue-700"
                    : "border border-blue-200 text-blue-700 hover:bg-blue-50"
                }`}
              >
                {isActive ? "Cancel selection" : "Select from drawing"}
              </button>
              <button
                type="button"
                disabled={disabled || metadata[key] === ""}
                onClick={() => onClear(key)}
                className="rounded-md border border-slate-200 px-2 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Clear
              </button>
            </div>
          </section>
        );
      })}
    </div>
  );
}
