"use client";

import { useEffect, useState, type FormEvent } from "react";

export interface ExcelExportValues {
  partCount: number;
  fileName: string;
}

interface ExcelExportDialogProps {
  open: boolean;
  defaultFileName: string;
  onClose: () => void;
  onExport: (values: ExcelExportValues) => Promise<void>;
}

const MAX_PART_COUNT = 20;

export function ExcelExportDialog({
  open,
  defaultFileName,
  onClose,
  onExport,
}: ExcelExportDialogProps) {
  const [partCount, setPartCount] = useState("1");
  const [fileName, setFileName] = useState(defaultFileName);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setPartCount("1");
    setFileName(defaultFileName);
    setSubmitting(false);
    setError("");
  }, [defaultFileName, open]);

  if (!open) return null;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const normalizedFileName = fileName.trim();
    const normalizedPartCount = Number(partCount);

    if (!normalizedFileName) {
      setError("Enter a file name.");
      return;
    }
    if (
      !Number.isInteger(normalizedPartCount) ||
      normalizedPartCount < 1 ||
      normalizedPartCount > MAX_PART_COUNT
    ) {
      setError(`Number of parts must be a whole number from 1 to ${MAX_PART_COUNT}.`);
      return;
    }

    setSubmitting(true);
    setError("");
    try {
      await onExport({
        partCount: normalizedPartCount,
        fileName: normalizedFileName,
      });
      onClose();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Could not create the Excel file."
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
      aria-labelledby="excel-export-title"
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
              id="excel-export-title"
              className="text-lg font-semibold text-slate-900"
            >
              Export inspection checksheet
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Creates a formatted Excel workbook with the checksheet and full
              ballooned drawing pages.
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
          Number of parts
          <input
            type="number"
            min={1}
            max={MAX_PART_COUNT}
            step={1}
            value={partCount}
            onChange={(event) => setPartCount(event.target.value)}
            disabled={submitting}
            className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
          />
          <span className="mt-1 block text-xs text-slate-400">
            Creates Part 1 through Part {partCount || "N"} reading columns.
          </span>
        </label>

        <label className="mt-4 block text-sm font-medium text-slate-700">
          File name
          <input
            autoFocus
            value={fileName}
            onChange={(event) => setFileName(event.target.value)}
            maxLength={160}
            disabled={submitting}
            className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
          />
          <span className="mt-1 block text-xs text-slate-400">
            Used inside the workbook and for the downloaded .xlsx file.
          </span>
        </label>

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
            {submitting ? "Generating…" : "Generate Excel"}
          </button>
        </div>
      </form>
    </div>
  );
}
