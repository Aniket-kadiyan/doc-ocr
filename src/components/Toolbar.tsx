"use client";

import {
  readOcrDebugDumpToggles,
  setOcrDebugDumpEnabled,
} from "@/lib/ocrDebugDump";
import { AUTO_SEGMENT_ENABLED, DEBUG_DUMP_ENABLED } from "@/lib/featureFlags";
import { useEffect, useState } from "react";
import { ExportPanel } from "@/components/ExportPanel";

interface ToolbarProps {
  isSegmenting: boolean;
  isDrawingValue: boolean;
  isProcessing: boolean;
  currentPage: number;
  totalPages: number;
  scale: number;
  onToggleSegment: () => void;
  onToggleDrawValue: () => void;
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
  isDrawingValue,
  isProcessing,
  currentPage,
  totalPages,
  scale,
  onToggleSegment,
  onToggleDrawValue,
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

      <button
        type="button"
        onClick={onToggleDrawValue}
        disabled={isProcessing}
        title="Draw a box around one value; OCR reads it and creates a numbered balloon"
        className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
          isDrawingValue
            ? "bg-blue-600 text-white"
            : "border border-slate-200 text-slate-700 hover:bg-slate-50"
        } disabled:opacity-50`}
      >
        {isDrawingValue ? "Drawing Value…" : "Draw Value"}
      </button>

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
