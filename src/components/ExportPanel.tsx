"use client";

import { useEffect, useRef, useState } from "react";
import {
  exportInspectionCSV,
  exportInspectionJSON,
  exportXML,
  downloadFile,
  findMalformedToleranceAnnotations,
} from "@/lib/export";
import { createChecksheet } from "@/lib/checksheetClient";
import { buildChecksheetCreationSnapshot } from "@/lib/checksheetSnapshot";
import { getProject } from "@/lib/db";
import {
  buildVerificationPayload,
  sendForVerification,
  verificationEndpoint,
} from "@/lib/project";
import { useAnnotationStore } from "@/store/annotationStore";
import {
  CreateChecksheetDialog,
  type CreateChecksheetValues,
} from "@/components/CreateChecksheetDialog";

interface ExportPanelProps {
  disabled?: boolean;
}

export function ExportPanel({ disabled = false }: ExportPanelProps) {
  const annotations = useAnnotationStore((s) => s.annotations);
  const projectName = useAnnotationStore((s) => s.projectName);
  const projectId = useAnnotationStore((s) => s.projectId);
  const [actionStatus, setActionStatus] = useState<{
    kind: "idle" | "saving" | "sending" | "ok" | "error";
    message: string;
  }>({ kind: "idle", message: "" });
  const [menuOpen, setMenuOpen] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
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

  useEffect(() => {
    if (disabled) setMenuOpen(false);
  }, [disabled]);

  const validateToleranceExpressions = (): boolean => {
    const malformed = findMalformedToleranceAnnotations(annotations);
    if (malformed.length === 0) return true;

    const numbers = malformed.map((annotation) => annotation.number).join(", ");
    setMenuOpen(false);
    setActionStatus({
      kind: "error",
      message:
        `Balloon${malformed.length === 1 ? "" : "s"} ${numbers} ` +
        `contain${malformed.length === 1 ? "s" : ""} a malformed value or ` +
        "tolerance expression. Please correct it before saving or exporting.",
    });
    return false;
  };

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

  const handleWebView = () => {
    if (!validateToleranceExpressions()) return;
    setMenuOpen(false);
    setActionStatus({ kind: "idle", message: "" });
    setCreateDialogOpen(true);
  };

  const handleCreateChecksheet = async ({
    name,
    readingColumns,
  }: CreateChecksheetValues) => {
    const newTab = window.open("about:blank", "_blank");
    if (newTab) {
      newTab.opener = null;
      newTab.document.title = "Creating checksheet…";
      newTab.document.body.textContent = "Creating checksheet…";
    }
    setActionStatus({
      kind: "saving",
      message: "Saving checksheet snapshot…",
    });
    try {
      if (!projectId) {
        throw new Error("The current drawing has no project identity.");
      }
      const project = await getProject(projectId);
      if (!project) {
        throw new Error(
          "The original drawing is unavailable. Reopen it before creating a checksheet."
        );
      }
      const creation = buildChecksheetCreationSnapshot({
        checksheetName: name,
        readingColumns,
        annotations,
        projectId,
        projectName,
        project,
      });
      const result = await createChecksheet(creation.payload, creation.document);
      const runPath = `/checksheets/${encodeURIComponent(
        result.checksheet.id
      )}/runs/${encodeURIComponent(result.run.id)}`;
      const runUrl = new URL(runPath, window.location.origin).toString();
      if (newTab) {
        newTab.location.replace(runUrl);
      } else {
        window.open(runUrl, "_blank", "noopener");
      }
      setActionStatus({
        kind: "ok",
        message: `Created ${result.checksheet.name} with ${result.rows.length} rows.`,
      });
    } catch (error) {
      newTab?.close();
      setActionStatus({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "Failed to create the checksheet.",
      });
      throw error;
    }
  };

  const handleJSON = () => {
    if (!validateToleranceExpressions()) return;
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
    if (!validateToleranceExpressions()) return;
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
    if (!validateToleranceExpressions()) return;
    setMenuOpen(false);
    downloadFile(
      exportXML(annotations),
      `${base}_annotations.xml`,
      "application/xml"
    );
  };

  const handleVerify = async () => {
    if (!validateToleranceExpressions()) return;
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
        disabled={disabled || empty || busy}
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
            onClick={handleWebView}
            className="block w-full border-b border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">Checksheet / Web</span>
            <span className="block text-[11px] text-slate-400">
              Save a durable snapshot and start an inspection
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

      <CreateChecksheetDialog
        open={createDialogOpen}
        defaultName={`${projectName || "Drawing"} Checksheet`}
        onClose={() => setCreateDialogOpen(false)}
        onCreate={handleCreateChecksheet}
      />
    </div>
  );
}
