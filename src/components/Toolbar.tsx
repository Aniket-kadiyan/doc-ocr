"use client";

import {
  readOcrDebugDumpToggles,
  setOcrDebugDumpEnabled,
} from "@/lib/ocrDebugDump";
import { AUTO_SEGMENT_ENABLED, DEBUG_DUMP_ENABLED } from "@/lib/featureFlags";
import { useEffect, useRef, useState } from "react";
import type { LabelInputMode } from "@/types/annotation";
import { ExportPanel } from "@/components/ExportPanel";

interface ToolbarProps {
  isSegmenting: boolean;
  isLabeling: boolean;
  isProcessing: boolean;
  currentPage: number;
  totalPages: number;
  scale: number;
  onToggleSegment: () => void;
  onStartLabel: (mode: LabelInputMode) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onPrevPage: () => void;
  onNextPage: () => void;
  onUpload: () => void;
  onSaveProject: () => void;
  onLoadProject: () => void;
  onRemoveDrawing: () => void;
  canSaveProject: boolean;
}

export function Toolbar({
  isSegmenting,
  isLabeling,
  isProcessing,
  currentPage,
  totalPages,
  scale,
  onToggleSegment,
  onStartLabel,
  onZoomIn,
  onZoomOut,
  onPrevPage,
  onNextPage,
  onUpload,
  onSaveProject,
  onLoadProject,
  onRemoveDrawing,
  canSaveProject,
}: ToolbarProps) {
  const [debugDump, setDebugDump] = useState(false);
  const [labelMenuOpen, setLabelMenuOpen] = useState(false);
  const labelMenuRef = useRef<HTMLDivElement>(null);

  // Close the Add Label dropdown when clicking elsewhere.
  useEffect(() => {
    if (!labelMenuOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (!labelMenuRef.current?.contains(e.target as Node)) {
        setLabelMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [labelMenuOpen]);

  const chooseLabelMode = (mode: LabelInputMode) => {
    setLabelMenuOpen(false);
    onStartLabel(mode);
  };

  useEffect(() => {
    const sync = () => {
      const t = readOcrDebugDumpToggles();
      setDebugDump(t.enabled);
    };
    sync();
    window.addEventListener("ocr-debug-dump-changed", sync);
    return () => window.removeEventListener("ocr-debug-dump-changed", sync);
  }, []);

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-white px-4 py-3">
      <button
        type="button"
        onClick={onUpload}
        className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
      >
        Open Drawing
      </button>

      <button
        type="button"
        onClick={onLoadProject}
        title="Open a saved .docbox.json project (drawing + annotations)"
        className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
      >
        Load Project
      </button>

      <button
        type="button"
        onClick={onSaveProject}
        disabled={!canSaveProject}
        title="Save the drawing and its annotations as one .docbox.json file"
        className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
      >
        Save Project
      </button>

      <button
        type="button"
        onClick={onRemoveDrawing}
        disabled={!canSaveProject}
        title="Close the current drawing and clear its annotations"
        className="rounded-lg border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-50 disabled:opacity-40"
      >
        Remove Drawing
      </button>

      <ExportPanel />

      <div className="mx-1 h-6 w-px bg-slate-200" />

      {AUTO_SEGMENT_ENABLED && (
        <button
          type="button"
          onClick={onToggleSegment}
          disabled={isProcessing}
          title="Draw one box around a cluster of values; each is detected, OCR'd and ballooned separately"
          className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
            isSegmenting
              ? "bg-emerald-600 text-white"
              : "border border-slate-200 text-slate-700 hover:bg-slate-50"
          } disabled:opacity-50`}
        >
          {isSegmenting ? "Segmenting…" : "Auto-Segment"}
        </button>
      )}

      <div className="relative" ref={labelMenuRef}>
        <button
          type="button"
          onClick={() => setLabelMenuOpen((o) => !o)}
          disabled={isProcessing}
          title="Add a label — type it manually or OCR it from a box you draw"
          className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
            isLabeling || labelMenuOpen
              ? "bg-indigo-600 text-white"
              : "border border-slate-200 text-slate-700 hover:bg-slate-50"
          } disabled:opacity-50`}
        >
          {isLabeling ? "Labeling…" : "Add Label ▾"}
        </button>

        {labelMenuOpen && (
          <div className="absolute left-0 top-full z-20 mt-1 w-44 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-lg">
            <button
              type="button"
              onClick={() => chooseLabelMode("manual")}
              className="block w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-indigo-50"
            >
              <span className="font-medium">Manually</span>
              <span className="block text-[11px] text-slate-400">
                Draw a box, then type the label
              </span>
            </button>
            <button
              type="button"
              onClick={() => chooseLabelMode("ocr")}
              className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-indigo-50"
            >
              <span className="font-medium">OCR</span>
              <span className="block text-[11px] text-slate-400">
                Draw a box; OCR reads the label
              </span>
            </button>
          </div>
        )}
      </div>

      {isProcessing && (
        <span className="text-sm text-blue-600 animate-pulse">
          Running OCR…
        </span>
      )}

      {DEBUG_DUMP_ENABLED && (
        <button
          type="button"
          onClick={() => setOcrDebugDumpEnabled(!debugDump)}
          title="Save each box OCR pipeline to backend/debug_output/"
          className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
            debugDump
              ? "bg-violet-600 text-white"
              : "border border-slate-200 text-slate-700 hover:bg-slate-50"
          }`}
        >
          {debugDump ? "Debug dump ON" : "Debug dump OFF"}
        </button>
      )}

      <div className="mx-2 h-6 w-px bg-slate-200" />

      <button
        type="button"
        onClick={onZoomOut}
        className="rounded-lg border border-slate-200 px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
      >
        −
      </button>
      <span className="min-w-[4rem] text-center text-sm text-slate-600">
        {Math.round(scale * 100)}%
      </span>
      <button
        type="button"
        onClick={onZoomIn}
        className="rounded-lg border border-slate-200 px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
      >
        +
      </button>

      {totalPages > 1 && (
        <>
          <div className="mx-2 h-6 w-px bg-slate-200" />
          <button
            type="button"
            onClick={onPrevPage}
            disabled={currentPage <= 1}
            className="rounded-lg border border-slate-200 px-2 py-1.5 text-sm disabled:opacity-40"
          >
            ‹
          </button>
          <span className="text-sm text-slate-600">
            Page {currentPage} / {totalPages}
          </span>
          <button
            type="button"
            onClick={onNextPage}
            disabled={currentPage >= totalPages}
            className="rounded-lg border border-slate-200 px-2 py-1.5 text-sm disabled:opacity-40"
          >
            ›
          </button>
        </>
      )}
    </div>
  );
}
