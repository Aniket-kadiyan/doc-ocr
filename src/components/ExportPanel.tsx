"use client";

import { useEffect, useRef, useState } from "react";
import {
  exportInspectionJSON,
  exportXML,
  downloadBlob,
  downloadFile,
  findMalformedToleranceAnnotations,
} from "@/lib/export";
import {
  downloadInspectionWorkbook,
  type ExcelExportOptions,
} from "@/lib/excelExport";
import { createChecksheet } from "@/lib/checksheetClient";
import { buildChecksheetCreationSnapshot } from "@/lib/checksheetSnapshot";
import { getProject } from "@/lib/db";
import {
  buildVerificationPayload,
  sendForVerification,
  verificationEndpoint,
} from "@/lib/project";
import {
  canvasToPngBlob,
  collectBalloonedPageImages,
  collectBalloonedPages,
  pagesToExport,
  renderBalloonedCanvas,
} from "@/lib/ballooned";
import { buildBalloonedPdf } from "@/lib/balloonedPdf";
import { renderReferencePages } from "@/lib/referencePageExport";
import { useAnnotationStore } from "@/store/annotationStore";
import { useDocumentMetadataStore } from "@/store/documentMetadataStore";
import {
  CreateChecksheetDialog,
  type CreateChecksheetValues,
} from "@/components/CreateChecksheetDialog";
import {
  ExcelExportDialog,
  type ExcelExportValues,
} from "@/components/ExcelExportDialog";
import type { PageReferenceTable } from "@/types/referencePoint";
import type { ScanCandidate } from "@/types/scanCandidate";
import { useBalloonStyle } from "@/components/BalloonStyleProvider";
import { balloonArtworkPngDataUrl } from "@/lib/balloonStyle";
import { describeLabelGaps, findLabelGaps } from "@/lib/labelRules";
import { getTitleKeywords } from "@/lib/titleKeywords";

interface ExportPanelProps {
  disabled?: boolean;
  referenceTables?: readonly PageReferenceTable[];
  scanCandidates?: readonly ScanCandidate[];
}

async function loadCompanyLogo(): Promise<string> {
  const response = await fetch("/senior_flexonics_logo.jpeg");
  if (!response.ok) throw new Error("Could not load the Senior Flexonics logo.");
  const blob = await response.blob();
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read the company logo."));
    reader.readAsDataURL(blob);
  });
}

export function ExportPanel({
  disabled = false,
  referenceTables = [],
  scanCandidates = [],
}: ExportPanelProps) {
  const annotations = useAnnotationStore((s) => s.annotations);
  const projectName = useAnnotationStore((s) => s.projectName);
  const projectId = useAnnotationStore((s) => s.projectId);
  const currentPage = useAnnotationStore((s) => s.currentPage);
  const pageCanvasProvider = useAnnotationStore((s) => s.pageCanvasProvider);
  const metadata = useDocumentMetadataStore((s) => s.metadata);
  const {
    style: balloonStyle,
    artworkImage: balloonArtworkImage,
    selection: balloonStyleSelection,
  } = useBalloonStyle();
  const [actionStatus, setActionStatus] = useState<{
    kind:
      | "idle"
      | "saving"
      | "exporting"
      | "sending"
      | "rendering"
      | "ok"
      | "error";
    message: string;
    /** Malformed balloons the user may accept as is, then `retry` the action. */
    malformed?: { ids: string[]; retry: () => void };
  }>({ kind: "idle", message: "" });
  const [menuOpen, setMenuOpen] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [excelDialogOpen, setExcelDialogOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const base = projectName.replace(/\s+/g, "_").toLowerCase() || "drawing";
  const empty = annotations.length === 0 && referenceTables.length === 0;
  // Labels the existing rules call for that the drawing does not carry. Every
  // export still writes them (blank), so this says so where it matters.
  const labelGapNote = menuOpen
    ? describeLabelGaps(findLabelGaps(annotations, getTitleKeywords()))
    : "";
  const busy =
    actionStatus.kind === "saving" ||
    actionStatus.kind === "exporting" ||
    actionStatus.kind === "sending" ||
    actionStatus.kind === "rendering";
  const busyLabel =
    actionStatus.kind === "saving"
      ? "Saving…"
      : actionStatus.kind === "exporting"
        ? "Generating…"
      : actionStatus.kind === "rendering"
        ? "Rendering…"
        : "Sending…";

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

  const updateAnnotation = useAnnotationStore((s) => s.updateAnnotation);

  const validateToleranceExpressions = (retry: () => void): boolean => {
    const malformed = findMalformedToleranceAnnotations(
      useAnnotationStore.getState().annotations
    );
    if (malformed.length === 0) return true;

    const numbers = malformed.map((annotation) => annotation.number).join(", ");
    setMenuOpen(false);
    setActionStatus({
      kind: "error",
      message:
        `Balloon${malformed.length === 1 ? "" : "s"} ${numbers} ` +
        `contain${malformed.length === 1 ? "s" : ""} a malformed value or ` +
        "tolerance expression. Correct it, or accept the value as is.",
      malformed: { ids: malformed.map((annotation) => annotation.id), retry },
    });
    return false;
  };

  const acceptMalformed = () => {
    const pending = actionStatus.malformed;
    if (!pending) return;
    const byId = new Map(
      useAnnotationStore.getState().annotations.map((a) => [a.id, a])
    );
    for (const id of pending.ids) {
      const annotation = byId.get(id);
      if (annotation) updateAnnotation(id, { acceptedValue: annotation.value });
    }
    setActionStatus({ kind: "idle", message: "" });
    pending.retry();
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
    if (!validateToleranceExpressions(handleWebView)) return;
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
        metadata,
        referenceTables,
        scanCandidates: [...scanCandidates],
        balloonStyle: balloonStyleSelection,
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
    if (!validateToleranceExpressions(handleJSON)) return;
    setMenuOpen(false);
    const extraColumns = askExtraColumns();
    if (extraColumns === null) return;
    downloadFile(
      exportInspectionJSON(
        annotations,
        extraColumns,
        metadata,
        referenceTables
      ),
      `${base}_inspection.json`,
      "application/json"
    );
  };

  const handleExcelDialog = () => {
    if (!validateToleranceExpressions(handleExcelDialog)) return;
    setMenuOpen(false);
    setActionStatus({ kind: "idle", message: "" });
    setExcelDialogOpen(true);
  };

  const handleExcelExport = async ({
    partCount,
    fileName,
  }: ExcelExportValues) => {
    setActionStatus({
      kind: "exporting",
      message: "Rendering ballooned pages and generating Excel…",
    });
    try {
      if (!pageCanvasProvider) {
        throw new Error("Open a drawing first — there is nothing to export.");
      }
      const pages =
        annotations.length > 0
          ? await collectBalloonedPageImages(
              pageCanvasProvider,
              annotations,
              pagesToExport(annotations, currentPage),
              { projectName, metadata },
              balloonArtworkImage
                ? { style: balloonStyle, artworkImage: balloonArtworkImage }
                : undefined
            )
          : [];
      const referencePages = await renderReferencePages({
        renderPage: pageCanvasProvider,
        referenceTables,
      });
      if (pages.length === 0 && referencePages.length === 0) {
        throw new Error("Could not render the drawing pages.");
      }
      const options: ExcelExportOptions = {
        partCount,
        fileName,
        companyLogoDataUrl: await loadCompanyLogo(),
        metadata,
      };
      const downloadedFile = await downloadInspectionWorkbook({
        annotations,
        pages,
        referenceTables,
        referencePages,
        options,
      });
      setActionStatus({
        kind: "ok",
        message: `Downloaded ${downloadedFile}.`,
      });
    } catch (error) {
      setActionStatus({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "Failed to generate the Excel file.",
      });
      throw error;
    }
  };

  const handleBalloonedPdf = async () => {
    setMenuOpen(false);
    if (!pageCanvasProvider) {
      setActionStatus({
        kind: "error",
        message: "Open a drawing first — there's nothing to balloon yet.",
      });
      return;
    }
    const extraColumns = askExtraColumns();
    if (extraColumns === null) return;

    setActionStatus({ kind: "rendering", message: "Rendering pages…" });
    try {
      const exportPages = [
        ...new Set([
          ...pagesToExport(annotations, currentPage),
          ...referenceTables.map((table) => table.page),
        ]),
      ].sort((left, right) => left - right);
      const pages = await collectBalloonedPages(
        pageCanvasProvider,
        annotations,
        exportPages
      );
      if (pages.length === 0) {
        throw new Error("Could not render the drawing pages.");
      }
      const pdf = await buildBalloonedPdf({
        projectName,
        metadata,
        pages,
        annotations,
        extraColumns,
        referenceTables,
        balloonStyle,
        balloonArtworkPng: balloonArtworkImage
          ? balloonArtworkPngDataUrl(balloonStyle, balloonArtworkImage)
          : undefined,
      });
      downloadBlob(pdf, `${base}_ballooned.pdf`);
      setActionStatus({
        kind: "ok",
        message: `Saved ${base}_ballooned.pdf (${pages.length} drawing page${
          pages.length === 1 ? "" : "s"
        } + inspection table).`,
      });
    } catch (error) {
      setActionStatus({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "Failed to build the ballooned drawing.",
      });
    }
  };

  // Flat image of the page on screen with its balloons burned in — for pasting
  // into a report or a chat where an .html attachment won't do.
  const handleBalloonedPng = async () => {
    setMenuOpen(false);
    if (!pageCanvasProvider) {
      setActionStatus({
        kind: "error",
        message: "Open a drawing first — there's nothing to balloon yet.",
      });
      return;
    }
    setActionStatus({ kind: "rendering", message: "Rendering page…" });
    try {
      const source = await pageCanvasProvider(currentPage);
      if (!source) throw new Error("Could not render this page.");
      const canvas = renderBalloonedCanvas(
        source,
        annotations.filter((a) => a.page === currentPage),
        { projectName, metadata },
        balloonArtworkImage
          ? { style: balloonStyle, artworkImage: balloonArtworkImage }
          : undefined
      );
      downloadBlob(
        await canvasToPngBlob(canvas),
        `${base}_ballooned_p${currentPage}.png`
      );
      setActionStatus({
        kind: "ok",
        message: `Saved page ${currentPage} as PNG.`,
      });
    } catch (error) {
      setActionStatus({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "Failed to render the ballooned page.",
      });
    }
  };

  const handleXML = () => {
    if (!validateToleranceExpressions(handleXML)) return;
    setMenuOpen(false);
    // Same prompt as CSV/JSON so all three exports carry identical columns.
    const extraColumns = askExtraColumns();
    if (extraColumns === null) return;
    downloadFile(
      exportXML(annotations, extraColumns, metadata, referenceTables),
      `${base}_inspection.xml`,
      "application/xml"
    );
  };

  const handleVerify = async () => {
    if (!validateToleranceExpressions(handleVerify)) return;
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
        extraColumns,
        metadata,
        referenceTables
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
          {labelGapNote && (
            <p
              role="status"
              className="border-b border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800"
            >
              {labelGapNote}
            </p>
          )}
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
              Document metadata and inspection values
            </span>
          </button>
          <button
            type="button"
            onClick={handleExcelDialog}
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">Excel</span>
            <span className="block text-[11px] text-slate-400">
              Checksheet plus a separate References worksheet and image
            </span>
          </button>
          <button
            type="button"
            onClick={() => void handleBalloonedPdf()}
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">Ballooned Drawing (PDF)</span>
            <span className="block text-[11px] text-slate-400">
              Drawing + balloons + values in one printable file
            </span>
          </button>
          <button
            type="button"
            onClick={() => void handleBalloonedPng()}
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">Ballooned Drawing (PNG)</span>
            <span className="block text-[11px] text-slate-400">
              This page as an image with balloons burned in
            </span>
          </button>
          <button
            type="button"
            onClick={handleXML}
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-blue-50"
          >
            <span className="font-medium">XML</span>
            <span className="block text-[11px] text-slate-400">
              Same table as CSV/JSON, as a .xml file
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
        <div
          className={`absolute right-0 top-full mt-1 w-56 rounded-lg border bg-white px-3 py-2 text-xs shadow-lg ${
            actionStatus.kind === "ok"
              ? "border-emerald-200 text-emerald-700"
              : "border-red-200 text-red-600"
          }`}
        >
          <p>{actionStatus.message}</p>
          {actionStatus.malformed && (
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                onClick={acceptMalformed}
                className="rounded-md bg-emerald-600 px-2 py-1 font-medium text-white hover:bg-emerald-700"
              >
                Accept as is
              </button>
              <button
                type="button"
                onClick={() => setActionStatus({ kind: "idle", message: "" })}
                className="rounded-md border border-gray-300 px-2 py-1 text-gray-600 hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}

      <CreateChecksheetDialog
        open={createDialogOpen}
        defaultName={`${projectName || "Drawing"} Checksheet`}
        onClose={() => setCreateDialogOpen(false)}
        onCreate={handleCreateChecksheet}
      />

      <ExcelExportDialog
        open={excelDialogOpen}
        defaultFileName={projectName || "Drawing"}
        onClose={() => setExcelDialogOpen(false)}
        onExport={handleExcelExport}
      />
    </div>
  );
}

