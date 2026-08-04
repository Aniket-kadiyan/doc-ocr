"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { classifyDimension } from "@/lib/dimensionClassifier";
import { TOOL_OPTIONS, type Annotation } from "@/types/annotation";
import { db, loadAnnotations, saveAnnotations } from "@/lib/db";
import { inspectionSheetCSV } from "@/lib/export";
import { buildInspectionSheet } from "@/lib/project";

const CHANNEL = "doc-ocr-box:checksheet";
const kindOf = (a: Annotation) => a.kind ?? "dimension";

/** Read the project id and extra-column list straight off the URL (no Suspense
 * boundary needed, unlike useSearchParams). */
function readParams(): { projectId: string; cols: string[] } {
  if (typeof window === "undefined") return { projectId: "", cols: [] };
  const p = new URLSearchParams(window.location.search);
  const cols = (p.get("cols") ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return { projectId: p.get("project") ?? "", cols };
}

export default function ChecksheetPage() {
  const [projectId, setProjectId] = useState("");
  const [urlCols, setUrlCols] = useState<string[]>([]);
  const [projectName, setProjectName] = useState("");
  const [annos, setAnnos] = useState<Annotation[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [status, setStatus] = useState<"idle" | "saved" | "error">("idle");
  const channelRef = useRef<BroadcastChannel | null>(null);

  // Load the project's annotations from IndexedDB once the URL is known.
  useEffect(() => {
    const { projectId: pid, cols } = readParams();
    setProjectId(pid);
    setUrlCols(cols);
    if (!pid) {
      setLoaded(true);
      return;
    }
    (async () => {
      try {
        const [project, list] = await Promise.all([
          db.projects.get(pid),
          loadAnnotations(pid),
        ]);
        setProjectName(
          (project?.name ?? "Drawing").replace(/\.[^.]+$/, "")
        );
        setAnnos(list);
      } finally {
        setLoaded(true);
      }
    })();
  }, []);

  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return;
    const channel = new BroadcastChannel(CHANNEL);
    channelRef.current = channel;
    return () => {
      channel.close();
      channelRef.current = null;
    };
  }, []);

  // Extra columns = those passed in the URL, plus any already-saved on the data.
  const columns = useMemo(() => {
    const seen = new Set(urlCols);
    const fromData: string[] = [];
    for (const a of annos) {
      for (const key of Object.keys(a.extras ?? {})) {
        if (!seen.has(key)) {
          seen.add(key);
          fromData.push(key);
        }
      }
    }
    return [...urlCols, ...fromData];
  }, [urlCols, annos]);

  // Dimension rows in balloon order, with the label resolved from its parent.
  const rows = useMemo(() => {
    const parentOf = (a: Annotation) =>
      annos.find((x) => x.id === a.labelId);
    return annos
      .filter((a) => kindOf(a) === "dimension")
      .sort((a, b) => a.number - b.number)
      .map((a) => {
        const parent = parentOf(a);
        return {
          a,
          labelId: parent?.id,
          label: parent?.value ?? a.label ?? "",
          method: a.method || parent?.method || "",
          tool: a.tool || parent?.tool || "",
        };
      });
  }, [annos]);

  // Patch a dimension (and, for the label cell, its parent label) in place.
  const setField = useCallback(
    (
      id: string,
      labelId: string | undefined,
      field: "label" | "value" | "range" | "method" | "tool",
      val: string
    ) => {
      setStatus("idle");
      setAnnos((prev) =>
        prev.map((a) => {
          if (field === "label" && labelId && a.id === labelId) {
            return { ...a, value: val };
          }
          if (a.id !== id) return a;
          if (field === "label") return { ...a, label: val };
          if (field === "value")
            return { ...a, value: val, type: classifyDimension(val) };
          return { ...a, [field]: val };
        })
      );
    },
    []
  );

  const setExtra = useCallback((id: string, col: string, val: string) => {
    setStatus("idle");
    setAnnos((prev) =>
      prev.map((a) =>
        a.id === id ? { ...a, extras: { ...(a.extras ?? {}), [col]: val } } : a
      )
    );
  }, []);

  const handleSave = useCallback(async () => {
    if (!projectId) {
      setStatus("error");
      return;
    }
    try {
      await saveAnnotations(projectId, annos);
      channelRef.current?.postMessage({ projectId, annotations: annos });
      setStatus("saved");
    } catch {
      setStatus("error");
    }
  }, [projectId, annos]);

  const handleExportCsv = useCallback(() => {
    const sheet = buildInspectionSheet(annos, columns);
    const blob = new Blob([inspectionSheetCSV(sheet)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${
      projectName.replace(/\s+/g, "_").toLowerCase() || "drawing"
    }_checksheet.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }, [annos, columns, projectName]);

  const cell =
    "w-full rounded border border-transparent bg-transparent px-2 py-1.5 text-sm text-slate-900 outline-none focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-500/20";

  if (loaded && !projectId) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-50 p-8 text-center text-slate-500">
        No project specified. Open the checksheet from the Export menu in the
        drawing view.
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-slate-100">
      <header className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-6 py-3 shadow-sm">
        <div>
          <h1 className="text-base font-semibold text-slate-900">
            Inspection Checksheet
          </h1>
          <p className="text-xs text-slate-500">
            {projectName || "Drawing"} — {rows.length}{" "}
            {rows.length === 1 ? "value" : "values"}
            {columns.length > 0 && ` · ${columns.length} extra column${
              columns.length === 1 ? "" : "s"
            }`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">
            {status === "saved"
              ? "Saved — synced to the drawing."
              : status === "error"
                ? "Save failed."
                : ""}
          </span>
          <button
            type="button"
            onClick={handleExportCsv}
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Download CSV
          </button>
          <button
            type="button"
            onClick={() => void handleSave()}
            className="rounded-lg bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-indigo-700"
          >
            Save
          </button>
        </div>
      </header>

      <div className="p-6">
        {!loaded ? (
          <p className="text-sm text-slate-400">Loading…</p>
        ) : rows.length === 0 ? (
          <p className="rounded-lg border border-dashed border-slate-300 bg-white px-4 py-12 text-center text-sm text-slate-400">
            This project has no values yet. Add dimensions on the drawing, save,
            then reopen the checksheet.
          </p>
        ) : (
          <div className="overflow-auto rounded-xl border border-slate-200 bg-white shadow-sm">
            <table className="w-full min-w-[720px] border-collapse text-left">
              <thead className="sticky top-0 bg-slate-50">
                <tr className="border-b border-slate-200 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  <th className="w-14 px-3 py-2.5">S.no</th>
                  <th className="px-3 py-2.5">Label</th>
                  <th className="px-3 py-2.5">Value</th>
                  <th className="px-3 py-2.5">Tolerance</th>
                  {columns.map((c) => (
                    <th key={c} className="px-3 py-2.5 text-blue-600">
                      {c}
                    </th>
                  ))}
                  <th className="px-3 py-2.5">Method</th>
                  <th className="px-3 py-2.5">Tool</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr
                    key={r.a.id}
                    className={`border-b border-slate-100 ${
                      i % 2 ? "bg-slate-50/40" : ""
                    } hover:bg-blue-50/40`}
                  >
                    <td className="px-3 py-1 text-sm font-medium text-slate-400">
                      {r.a.number}
                    </td>
                    <td className="px-1 py-0.5">
                      <input
                        value={r.label}
                        onChange={(e) =>
                          setField(r.a.id, r.labelId, "label", e.target.value)
                        }
                        className={cell}
                      />
                    </td>
                    <td className="px-1 py-0.5">
                      <input
                        value={r.a.value}
                        onChange={(e) =>
                          setField(r.a.id, r.labelId, "value", e.target.value)
                        }
                        className={`${cell} font-mono`}
                      />
                    </td>
                    <td className="px-1 py-0.5">
                      <input
                        value={r.a.range ?? ""}
                        onChange={(e) =>
                          setField(r.a.id, r.labelId, "range", e.target.value)
                        }
                        className={`${cell} font-mono`}
                      />
                    </td>
                    {columns.map((c) => (
                      <td key={c} className="px-1 py-0.5">
                        <input
                          value={r.a.extras?.[c] ?? ""}
                          onChange={(e) =>
                            setExtra(r.a.id, c, e.target.value)
                          }
                          className={`${cell} bg-blue-50/30 font-mono`}
                        />
                      </td>
                    ))}
                    <td className="px-1 py-0.5">
                      <input
                        value={r.method}
                        onChange={(e) =>
                          setField(r.a.id, r.labelId, "method", e.target.value)
                        }
                        className={cell}
                      />
                    </td>
                    <td className="px-1 py-0.5">
                      <select
                        value={r.tool}
                        onChange={(e) =>
                          setField(r.a.id, r.labelId, "tool", e.target.value)
                        }
                        className={`${cell} bg-white`}
                      >
                        <option value="">—</option>
                        {TOOL_OPTIONS.map((t) => (
                          <option key={t} value={t}>
                            {t}
                          </option>
                        ))}
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
