"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ChecksheetPreviewLoader } from "@/components/checksheets/ChecksheetPreviewLoader";
import { ChecksheetRowCharts } from "@/components/checksheets/ChecksheetRowCharts";
import { ChecksheetTable } from "@/components/checksheets/ChecksheetTable";
import { useChecksheetAutosave } from "@/hooks/useChecksheetAutosave";
import {
  completeChecksheetRun,
  getChecksheetRun,
} from "@/lib/checksheetClient";
import type { ChecksheetRunResponse, ChecksheetRow } from "@/types/checksheet";

interface ChecksheetRunProps {
  checksheetId: string;
  runId: string;
}

export function ChecksheetRun({ checksheetId, runId }: ChecksheetRunProps) {
  const [data, setData] = useState<ChecksheetRunResponse | null>(null);
  const [rows, setRows] = useState<ChecksheetRow[]>([]);
  const [activeAnnotationId, setActiveAnnotationId] = useState<string | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [completing, setCompleting] = useState(false);

  const editable = data?.run.status === "draft" && !data.checksheet.archived;
  const {
    status: autosaveStatus,
    error: autosaveError,
    queue: queueAutosave,
    flush: flushAutosave,
  } = useChecksheetAutosave({
    checksheetId,
    runId,
    enabled: Boolean(editable),
  });

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError("");
    void getChecksheetRun(checksheetId, runId)
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setRows(result.rows);
        setActiveAnnotationId((current) =>
          result.rows.some((row) => row.annotation_id === current)
            ? current
            : result.rows[0]?.annotation_id ?? null
        );
      })
      .catch((reason) => {
        if (cancelled) return;
        setLoadError(
          reason instanceof Error ? reason.message : "Could not load checksheet."
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [checksheetId, runId]);

  useEffect(() => {
    if (autosaveStatus !== "unsaved" && autosaveStatus !== "saving") return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [autosaveStatus]);

  const activeIndex = useMemo(
    () => rows.findIndex((row) => row.annotation_id === activeAnnotationId),
    [activeAnnotationId, rows]
  );
  const activeRow = activeIndex >= 0 ? rows[activeIndex] : null;
  const previewRow =
    data?.rows.find((row) => row.annotation_id === activeAnnotationId) ??
    activeRow;

  const navigate = useCallback(
    (offset: number) => {
      if (rows.length === 0) return;
      const current = activeIndex >= 0 ? activeIndex : 0;
      const next = Math.max(0, Math.min(rows.length - 1, current + offset));
      const nextRow = rows[next];
      setActiveAnnotationId(nextRow.annotation_id);
      window.requestAnimationFrame(() => {
        const input = document.querySelector<HTMLInputElement>(
          `[data-checksheet-row="${CSS.escape(nextRow.annotation_id)}"] input`
        );
        input?.focus();
      });
    },
    [activeIndex, rows]
  );

  const changeReading = useCallback(
    (annotationId: string, columnId: string, value: string) => {
      if (!editable) return;
      setRows((current) =>
        current.map((row) =>
          row.annotation_id === annotationId
            ? {
                ...row,
                readings: { ...row.readings, [columnId]: value },
              }
            : row
        )
      );
      queueAutosave({
        annotation_id: annotationId,
        column_id: columnId,
        value,
      });
    },
    [editable, queueAutosave]
  );

  const complete = async () => {
    if (!data || !editable || completing) return;
    const emptyReadings = rows.reduce(
      (count, row) =>
        count +
        data.columns.filter((column) => !(row.readings[column.id] ?? "").trim())
          .length,
      0
    );
    const prompt = emptyReadings
      ? `Complete this inspection with ${emptyReadings} blank reading${
          emptyReadings === 1 ? "" : "s"
        }? Completed runs cannot be edited.`
      : "Complete this inspection? Completed runs cannot be edited.";
    if (!window.confirm(prompt)) return;
    setCompleting(true);
    setLoadError("");
    try {
      await flushAutosave();
      const completed = await completeChecksheetRun(checksheetId, runId);
      setData(completed);
      setRows(completed.rows);
    } catch (reason) {
      setLoadError(
        reason instanceof Error ? reason.message : "Could not complete inspection."
      );
    } finally {
      setCompleting(false);
    }
  };

  const saveLabel =
    autosaveStatus === "unsaved"
      ? "Unsaved changes"
      : autosaveStatus === "saving"
        ? "Saving…"
        : autosaveStatus === "saved"
          ? "Saved"
          : autosaveStatus === "error"
            ? "Save failed"
            : "";

  if (loading) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-100 text-sm text-slate-500">
        Loading checksheet…
      </main>
    );
  }

  if (loadError && !data) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-slate-100 p-8 text-center">
        <p className="max-w-xl rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {loadError}
        </p>
        <Link href="/checksheets" className="text-sm font-medium text-blue-700">
          Return to Saved Checksheets
        </Link>
      </main>
    );
  }

  if (!data) return null;

  return (
    <main className="flex min-h-screen flex-col bg-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white px-4 py-3 shadow-sm sm:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/checksheets"
              className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              Saved Checksheets
            </Link>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold text-slate-900">
                {data.checksheet.name}
              </h1>
              <p className="truncate text-xs text-slate-500">
                {data.checksheet.drawing_name} · Revision{" "}
                {data.revision.revision_number} · Run {data.run.run_number}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span
              className={`text-xs ${
                autosaveStatus === "error" ? "text-red-600" : "text-slate-500"
              }`}
              title={autosaveError}
            >
              {data.run.status === "completed"
                ? `Completed ${new Date(
                    data.run.completed_at ?? data.run.updated_at
                  ).toLocaleString()}`
                : saveLabel}
            </span>
            {autosaveStatus === "error" && (
              <button
                type="button"
                onClick={() => void flushAutosave().catch(() => undefined)}
                className="rounded-lg border border-red-200 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-50"
              >
                Retry save
              </button>
            )}
            <button
              type="button"
              onClick={() => void complete()}
              disabled={!editable || completing}
              className="rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {data.run.status === "completed"
                ? "Completed"
                : completing
                  ? "Completing…"
                  : "Complete"}
            </button>
          </div>
        </div>
        {data.checksheet.archived && (
          <p className="mt-2 rounded-lg bg-amber-50 px-3 py-1.5 text-xs text-amber-800">
            This checksheet is archived. Restore it from Saved Checksheets to edit
            draft runs.
          </p>
        )}
        {loadError && (
          <p className="mt-2 rounded-lg bg-red-50 px-3 py-1.5 text-xs text-red-700">
            {loadError}
          </p>
        )}
      </header>

      <div className="grid min-h-0 flex-1 gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_minmax(340px,38vw)]">
        <section className="min-w-0">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm text-slate-600">
              {rows.length} inspection {rows.length === 1 ? "row" : "rows"}
              {activeRow && ` · Active balloon ${activeRow.balloon_number}`}
            </p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => navigate(-1)}
                disabled={activeIndex <= 0}
                className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
              >
                Previous row
              </button>
              <button
                type="button"
                onClick={() => navigate(1)}
                disabled={activeIndex < 0 || activeIndex >= rows.length - 1}
                className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
              >
                Next row
              </button>
            </div>
          </div>
          <ChecksheetTable
            columns={data.columns}
            rows={rows}
            activeAnnotationId={activeAnnotationId}
            readOnly={!editable}
            onActivate={setActiveAnnotationId}
            onReadingChange={changeReading}
            onNavigate={navigate}
          />
        </section>

        <aside className="min-w-0 xl:sticky xl:top-24 xl:max-h-[calc(100vh-7rem)] xl:self-start xl:overflow-y-auto xl:pr-1">
          <ChecksheetPreviewLoader
            document={data.document}
            pdfRenderScale={data.revision.pdf_render_scale}
            row={previewRow}
          />
          {activeRow && (
            <ChecksheetRowCharts columns={data.columns} row={activeRow} />
          )}
        </aside>
      </div>
    </main>
  );
}
