"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import {
  duplicateChecksheet,
  getChecksheet,
  listChecksheets,
  permanentlyDeleteChecksheet,
  startChecksheetRun,
  updateChecksheet,
} from "@/lib/checksheetClient";
import type { ChecksheetDetail, ChecksheetSummary } from "@/types/checksheet";

const runPath = (checksheetId: string, runId: string) =>
  `/checksheets/${encodeURIComponent(checksheetId)}/runs/${encodeURIComponent(
    runId
  )}`;

export function ChecksheetManager() {
  const router = useRouter();
  const [items, setItems] = useState<ChecksheetSummary[]>([]);
  const [archived, setArchived] = useState(false);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, ChecksheetDetail>>({});
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setItems(await listChecksheets({ search, archived }));
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Could not load checksheets."
      );
    } finally {
      setLoading(false);
    }
  }, [archived, search]);

  useEffect(() => {
    const timer = setTimeout(() => void refresh(), 250);
    return () => clearTimeout(timer);
  }, [refresh]);

  const runAction = async (id: string, action: () => Promise<void>) => {
    setBusyId(id);
    setError("");
    try {
      await action();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "The action could not be completed."
      );
    } finally {
      setBusyId(null);
    }
  };

  const toggleHistory = async (item: ChecksheetSummary) => {
    if (expandedId === item.id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(item.id);
    if (details[item.id]) return;
    await runAction(item.id, async () => {
      const detail = await getChecksheet(item.id);
      setDetails((current) => ({ ...current, [item.id]: detail }));
    });
  };

  const rename = (item: ChecksheetSummary) => {
    const name = window.prompt("Checksheet name:", item.name)?.trim();
    if (!name || name === item.name) return;
    void runAction(item.id, async () => {
      await updateChecksheet(item.id, { name });
      setDetails((current) => {
        const next = { ...current };
        delete next[item.id];
        return next;
      });
      await refresh();
    });
  };

  const duplicate = (item: ChecksheetSummary) => {
    const name = window.prompt("Name for the duplicate:", `${item.name} Copy`)?.trim();
    if (!name) return;
    void runAction(item.id, async () => {
      const created = await duplicateChecksheet(item.id, name);
      router.push(runPath(created.checksheet.id, created.run.id));
    });
  };

  const setArchiveState = (item: ChecksheetSummary, nextArchived: boolean) => {
    const message = nextArchived
      ? `Archive "${item.name}"? Existing runs and its drawing will be retained.`
      : `Restore "${item.name}"?`;
    if (!window.confirm(message)) return;
    void runAction(item.id, async () => {
      await updateChecksheet(item.id, { archived: nextArchived });
      setExpandedId(null);
      await refresh();
    });
  };

  const permanentlyDelete = (item: ChecksheetSummary) => {
    const confirmation = window.prompt(
      `Permanent deletion cannot be undone. Type the checksheet name to delete it:\n\n${item.name}`,
      ""
    );
    if (confirmation !== item.name) return;
    void runAction(item.id, async () => {
      await permanentlyDeleteChecksheet(item.id);
      setExpandedId(null);
      setDetails((current) => {
        const next = { ...current };
        delete next[item.id];
        return next;
      });
      await refresh();
    });
  };

  const startRun = (item: ChecksheetSummary) => {
    void runAction(item.id, async () => {
      const run = await startChecksheetRun(item.id);
      router.push(runPath(item.id, run.run.id));
    });
  };

  return (
    <main className="min-h-screen bg-slate-100">
      <header className="border-b border-slate-200 bg-white px-4 py-4 shadow-sm sm:px-8">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-slate-900">
              Saved Checksheets
            </h1>
            <p className="mt-1 text-sm text-slate-500">
              Durable definitions, draft inspections, and completed history.
            </p>
          </div>
          <Link
            href="/"
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Drawing workspace
          </Link>
        </div>
      </header>

      <div className="mx-auto max-w-7xl p-4 sm:p-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex rounded-lg border border-slate-200 bg-white p-1">
            <button
              type="button"
              onClick={() => setArchived(false)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                !archived
                  ? "bg-blue-600 text-white"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              Active
            </button>
            <button
              type="button"
              onClick={() => setArchived(true)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                archived
                  ? "bg-slate-700 text-white"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              Archive
            </button>
          </div>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search checksheet or drawing…"
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 sm:w-80"
          />
        </div>

        {error && (
          <p className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </p>
        )}

        {loading ? (
          <p className="mt-8 text-center text-sm text-slate-400">Loading…</p>
        ) : items.length === 0 ? (
          <div className="mt-8 rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-14 text-center">
            <p className="font-medium text-slate-600">
              {archived ? "No archived checksheets" : "No saved checksheets"}
            </p>
            <p className="mt-1 text-sm text-slate-400">
              {archived
                ? "Archived checksheets will appear here."
                : "Create one from Export → Checksheet / Web in the drawing workspace."}
            </p>
          </div>
        ) : (
          <div className="mt-5 space-y-3">
            {items.map((item) => {
              const detail = details[item.id];
              const busy = busyId === item.id;
              return (
                <article
                  key={item.id}
                  className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
                >
                  <div className="flex flex-wrap items-center justify-between gap-4 p-4">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h2 className="truncate font-semibold text-slate-900">
                          {item.name}
                        </h2>
                        <span
                          className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase ${
                            item.draft_count > 0
                              ? "bg-amber-100 text-amber-800"
                              : "bg-emerald-100 text-emerald-800"
                          }`}
                        >
                          {item.draft_count > 0
                            ? `${item.draft_count} draft`
                            : "No drafts"}
                        </span>
                      </div>
                      <p className="mt-1 truncate text-sm text-slate-500">
                        {item.drawing_name}
                      </p>
                      <p className="mt-1 text-xs text-slate-400">
                        {item.row_count} rows · Revision {item.revision_number} ·{" "}
                        {item.completed_count} completed · Modified{" "}
                        {new Date(item.updated_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="flex flex-wrap items-center justify-end gap-2">
                      {(item.latest_draft_run_id || item.latest_run_id) && (
                        <Link
                          href={runPath(
                            item.id,
                            item.latest_draft_run_id ?? item.latest_run_id!
                          )}
                          className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
                        >
                          {item.latest_draft_run_id
                            ? "Open draft"
                            : "Open latest"}
                        </Link>
                      )}
                      {!archived && (
                        <button
                          type="button"
                          onClick={() => startRun(item)}
                          disabled={busy}
                          className="rounded-lg border border-emerald-200 px-3 py-1.5 text-sm font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-40"
                        >
                          New inspection
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void toggleHistory(item)}
                        disabled={busy}
                        className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-40"
                      >
                        {expandedId === item.id ? "Hide history" : "History"}
                      </button>
                      {!archived && (
                        <>
                          <button
                            type="button"
                            onClick={() => rename(item)}
                            disabled={busy}
                            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-40"
                          >
                            Rename
                          </button>
                          <button
                            type="button"
                            onClick={() => duplicate(item)}
                            disabled={busy}
                            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-40"
                          >
                            Duplicate
                          </button>
                          <button
                            type="button"
                            onClick={() => setArchiveState(item, true)}
                            disabled={busy}
                            className="rounded-lg border border-amber-200 px-3 py-1.5 text-sm text-amber-700 hover:bg-amber-50 disabled:opacity-40"
                          >
                            Archive
                          </button>
                        </>
                      )}
                      {archived && (
                        <>
                          <button
                            type="button"
                            onClick={() => setArchiveState(item, false)}
                            disabled={busy}
                            className="rounded-lg border border-blue-200 px-3 py-1.5 text-sm text-blue-700 hover:bg-blue-50 disabled:opacity-40"
                          >
                            Restore
                          </button>
                          <button
                            type="button"
                            onClick={() => permanentlyDelete(item)}
                            disabled={busy}
                            className="rounded-lg border border-red-200 px-3 py-1.5 text-sm text-red-700 hover:bg-red-50 disabled:opacity-40"
                          >
                            Delete permanently
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  {expandedId === item.id && (
                    <div className="border-t border-slate-200 bg-slate-50 p-4">
                      {!detail ? (
                        <p className="text-sm text-slate-400">
                          Loading inspection history…
                        </p>
                      ) : detail.runs.length === 0 ? (
                        <p className="text-sm text-slate-400">No inspections.</p>
                      ) : (
                        <div className="overflow-auto rounded-lg border border-slate-200 bg-white">
                          <table className="w-full min-w-[600px] text-left text-sm">
                            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                              <tr>
                                <th className="px-3 py-2">Run</th>
                                <th className="px-3 py-2">Revision</th>
                                <th className="px-3 py-2">Status</th>
                                <th className="px-3 py-2">Last modified</th>
                                <th className="px-3 py-2" />
                              </tr>
                            </thead>
                            <tbody>
                              {detail.runs.map((run) => (
                                <tr
                                  key={run.id}
                                  className="border-t border-slate-100"
                                >
                                  <td className="px-3 py-2 font-medium text-slate-800">
                                    {run.run_number}
                                  </td>
                                  <td className="px-3 py-2 text-slate-600">
                                    {run.revision_number}
                                  </td>
                                  <td className="px-3 py-2">
                                    <span
                                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                                        run.status === "draft"
                                          ? "bg-amber-100 text-amber-800"
                                          : "bg-emerald-100 text-emerald-800"
                                      }`}
                                    >
                                      {run.status}
                                    </span>
                                  </td>
                                  <td className="px-3 py-2 text-slate-500">
                                    {new Date(run.updated_at).toLocaleString()}
                                  </td>
                                  <td className="px-3 py-2 text-right">
                                    <Link
                                      href={runPath(item.id, run.id)}
                                      className="font-medium text-blue-700 hover:underline"
                                    >
                                      Open
                                    </Link>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
