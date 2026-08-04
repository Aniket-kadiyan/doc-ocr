"use client";

import { useEffect, useRef, useState } from "react";
import {
  buildInspectionJSON,
  exportInspectionCSV,
  exportInspectionJSON,
  exportXML,
  downloadFile,
} from "@/lib/export";
import { saveChecksheetTemplate } from "@/lib/checksheetClient";
import {
  buildVerificationPayload,
  sendForVerification,
  verificationEndpoint,
} from "@/lib/project";
import { useAnnotationStore } from "@/store/annotationStore";
import { saveAnnotations } from "@/lib/db";

// Keep the original IndexedDB /checksheet Web View available for later use,
// but skip it while the Digital Checksheet integration is active.
const USE_LEGACY_CHECKSHEET_WEBVIEW = false;

export function ExportPanel() {
  const annotations = useAnnotationStore((s) => s.annotations);
  const projectName = useAnnotationStore((s) => s.projectName);
  const projectId = useAnnotationStore((s) => s.projectId);
  const [actionStatus, setActionStatus] = useState<{
    kind: "idle" | "saving" | "sending" | "ok" | "error";
    message: string;
  }>({ kind: "idle", message: "" });
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const base = projectName.replace(/\s+/g, "_").toLowerCase() || "drawing";
  const empty = annotations.length === 0;
  const busy =
    actionStatus.kind === "saving" || actionStatus.kind === "sending";
  const busyLabel = actionStatus.kind === "saving" ? "Saving…" : "Sending…";

  // Close the Export dropdown when clicking elsewhere.
  useEffect(() => {
    if (!menuOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [menuOpen]);

  // Ask for extra inspector-filled columns: first how many, then their names.
  // Returns the list of names ([] for none), or null if the user cancels.
  const askExtraColumns = (minimum = 0): string[] | null => {
    const rawN = window.prompt(
      minimum > 0
        ? "How many measured-part columns should the checksheet contain? (minimum 1)"
        : "How many extra columns to add? (e.g. measured readings per part — 0 for none)",
      String(minimum)
    );
    if (rawN === null) return null;
    const n = Math.min(
      20,
      Math.max(minimum, parseInt(rawN, 10) || minimum)
    );
    if (n === 0) return [];
    const defaults = Array.from({ length: n }, (_, i) => `part${i + 1}`).join(
      ", "
    );
    const rawNames = window.prompt(
      `Enter ${n} column name${n === 1 ? "" : "s"}, comma-separated:`,
      defaults
    );
    if (rawNames === null) return null;
    const names = rawNames.split(",").map((s) => s.trim());
    return Array.from({ length: n }, (_, i) => names[i] || `col${i + 1}`);
  };

  // Open the editable checksheet in a new browser tab. Persist the latest
  // annotations to IndexedDB first so the tab loads current data, then pass the
  // project id and the inspector's extra columns through the URL.
  const handleLegacyWebView = async () => {
    setMenuOpen(false);
    const extraColumns = askExtraColumns();
    if (extraColumns === null) return;
    if (projectId && annotations.length > 0) {
      try {
        await saveAnnotations(projectId, annotations);
      } catch {
        // Fall through — the tab will load whatever is already persisted.
      }
    }
    const params = new URLSearchParams({ project: projectId });
    if (extraColumns.length) params.set("cols", extraColumns.join(","));
    window.open(`/checksheet?${params.toString()}`, "_blank", "noopener");
  };

  // Current POC flow: generate the same values-only JSON used by Export JSON,
  // convert it in Python, and save the resulting template on the backend.
  // index.json registration and Digital Checksheet navigation are intentionally
  // deferred to their own milestones.
  const handleTemplateWebView = async () => {
    setMenuOpen(false);
    const extraColumns = askExtraColumns(1);
    if (extraColumns === null) return;

    setActionStatus({
      kind: "saving",
      message: "Generating Digital Checksheet template…",
    });

    try {
      const result = await saveChecksheetTemplate({
        sourceJson: buildInspectionJSON(annotations, extraColumns),
        sourceFileName: `${base}_inspection.json`,
      });
      setActionStatus({
        kind: "ok",
        message: `${result.replaced ? "Replaced" : "Saved"} ${
          result.file_name
        } (${result.row_count} rows).`,
      });
    } catch (error) {
      setActionStatus({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "Failed to save the checksheet template.",
      });
    }
  };

  const handleWebView = async () => {
    if (USE_LEGACY_CHECKSHEET_WEBVIEW) {
      await handleLegacyWebView();
      return;
    }
    await handleTemplateWebView();
  };

  const handleJSON = () => {
    setMenuOpen(false);
    const extraColumns = askExtraColumns();
    if (extraColumns === null) return;
    downloadFile(
      exportInspectionJSON(annotations, extraColumns),
      `${base}_inspection.json`,
      "application/json"
    );
  };

  const handleInspectionCSV = () => {
    setMenuOpen(false);
    const extraColumns = askExtraColumns();
    if (extraColumns === null) return;
    downloadFile(
      exportInspectionCSV(annotations, extraColumns),
      `${base}_inspection.csv`,
      "text/csv"
    );
  };

  const handleXML = () => {
    setMenuOpen(false);
    downloadFile(
      exportXML(annotations),
      `${base}_annotations.xml`,
      "application/xml"
    );
  };

  const handleVerify = async () => {
    setMenuOpen(false);
    const extraColumns = askExtraColumns();
    if (extraColumns === null) return;
    const url =
      verificationEndpoint() ||
      window.prompt("Verification server URL (POST endpoint):", "") ||
      "";
    if (!url) return;
    setActionStatus({ kind: "sending", message: "Sending…" });
    try {
      const payload = buildVerificationPayload(
        annotations,
        projectName,
        url,
        extraColumns
      );
      await sendForVerification(url, payload);
      setActionStatus({
        kind: "ok",
        message: `Sent ${payload.items.length} values for checking.`,
      });
    } catch (err) {
      setActionStatus({
        kind: "error",
        message:
          err instanceof Error ? err.message : "Failed to reach the server.",
      });
    }
  };

  return (
    <div className="relative" ref={menuRef}>
      <button
        type="button"
        onClick={() => setMenuOpen((o) => !o)}
        disabled={empty || busy}
        title="Export annotations or send them for verification"
        className={`rounded-lg px-3 py-1.5 text-sm font-medium disabled:opacity-40 ${
          menuOpen
            ? "bg-blue-600 text-white"
            : "border border-slate-200 text-slate-700 hover:bg-slate-50"
        }`}
      >
        {busy ? busyLabel : "Export ▾"}
      </button>

      {menuOpen && (
        <div className="absolute left-0 top-full z-20 mt-1 w-56 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-lg">
          <button
            type="button"
            onClick={() => void handleWebView()}
            className="block w-full border-b border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">Web View</span>
            <span className="block text-[11px] text-slate-400">
              Generate and save the Digital Checksheet template
            </span>
          </button>
          <button
            type="button"
            onClick={handleJSON}
            className="block w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">JSON</span>
            <span className="block text-[11px] text-slate-400">
              Values only: S.no, Label, Value, Tolerance, …, Method, Tool
            </span>
          </button>
          <button
            type="button"
            onClick={handleInspectionCSV}
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">CSV</span>
            <span className="block text-[11px] text-slate-400">
              Inspection sheet with your extra columns
            </span>
          </button>
          <button
            type="button"
            onClick={handleXML}
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">XML</span>
            <span className="block text-[11px] text-slate-400">
              Annotations as a .xml file
            </span>
          </button>
          <button
            type="button"
            onClick={() => void handleVerify()}
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-blue-700 hover:bg-blue-50"
          >
            <span className="font-medium">Send for Verification</span>
            <span className="block text-[11px] text-blue-400">
              POST values and labels to a server for checking
            </span>
          </button>
        </div>
      )}

      {actionStatus.kind !== "idle" && !busy && (
        <p
          className={`absolute right-0 top-full mt-1 w-56 rounded-lg border bg-white px-3 py-2 text-xs shadow-lg ${
            actionStatus.kind === "ok"
              ? "border-emerald-200 text-emerald-700"
              : "border-red-200 text-red-600"
          }`}
        >
          {actionStatus.message}
        </p>
      )}
    </div>
  );
}
