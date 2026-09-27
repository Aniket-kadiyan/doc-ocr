"use client";

import { useEffect, useState, type FormEvent } from "react";

export interface CreateChecksheetValues {
  name: string;
  readingColumns: string[];
}

interface CreateChecksheetDialogProps {
  open: boolean;
  defaultName: string;
  onClose: () => void;
  onCreate: (values: CreateChecksheetValues) => Promise<void>;
}

export function CreateChecksheetDialog({
  open,
  defaultName,
  onClose,
  onCreate,
}: CreateChecksheetDialogProps) {
  const [name, setName] = useState(defaultName);
  const [columns, setColumns] = useState(["Part 1"]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setName(defaultName);
    setColumns(["Part 1"]);
    setSubmitting(false);
    setError("");
  }, [defaultName, open]);

  if (!open) return null;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const normalizedName = name.trim();
    const normalizedColumns = columns.map((column) => column.trim());
    if (!normalizedName) {
      setError("Enter a checksheet name.");
      return;
    }
    if (normalizedColumns.some((column) => !column)) {
      setError("Every measured-part column needs a name.");
      return;
    }
    const folded = normalizedColumns.map((column) =>
      column.toLocaleLowerCase()
    );
    if (new Set(folded).size !== folded.length) {
      setError("Measured-part column names must be unique.");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await onCreate({ name: normalizedName, readingColumns: normalizedColumns });
      onClose();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Could not create checksheet."
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="create-checksheet-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !submitting) onClose();
      }}
    >
      <form
        onSubmit={(event) => void submit(event)}
        className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2
              id="create-checksheet-title"
              className="text-lg font-semibold text-slate-900"
            >
              Create inspection checksheet
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              The drawing and current balloon specifications will be saved as an
              independent snapshot.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="rounded-lg px-2 py-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 disabled:opacity-40"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <label className="mt-5 block text-sm font-medium text-slate-700">
          Checksheet name
          <input
            autoFocus
            value={name}
            onChange={(event) => setName(event.target.value)}
            maxLength={160}
            disabled={submitting}
            className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
          />
        </label>

        <div className="mt-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-700">
                Measured-part columns
              </p>
              <p className="text-xs text-slate-400">
                These are the editable readings in every inspection run.
              </p>
            </div>
            <button
              type="button"
              onClick={() =>
                setColumns((current) => [
                  ...current,
                  `Part ${current.length + 1}`,
                ])
              }
              disabled={submitting || columns.length >= 20}
              className="rounded-lg border border-blue-200 px-2.5 py-1 text-xs font-medium text-blue-700 hover:bg-blue-50 disabled:opacity-40"
            >
              Add column
            </button>
          </div>
          <div className="mt-3 space-y-2">
            {columns.map((column, index) => (
              <div key={index} className="flex items-center gap-2">
                <span className="w-6 text-right text-xs text-slate-400">
                  {index + 1}
                </span>
                <input
                  value={column}
                  onChange={(event) =>
                    setColumns((current) =>
                      current.map((value, currentIndex) =>
                        currentIndex === index ? event.target.value : value
                      )
                    )
                  }
                  maxLength={120}
                  disabled={submitting}
                  className="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                />
                <button
                  type="button"
                  onClick={() =>
                    setColumns((current) =>
                      current.filter((_, currentIndex) => currentIndex !== index)
                    )
                  }
                  disabled={submitting || columns.length === 1}
                  className="rounded-lg px-2 py-1 text-sm text-slate-400 hover:bg-red-50 hover:text-red-600 disabled:opacity-30"
                  aria-label={`Remove ${column || `column ${index + 1}`}`}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        </div>

        {error && (
          <p className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </p>
        )}

        <div className="mt-6 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {submitting ? "Creating…" : "Create and open"}
          </button>
        </div>
      </form>
    </div>
  );
}
