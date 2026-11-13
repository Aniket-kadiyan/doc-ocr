"use client";

import { useEffect, useState } from "react";
import {
  isScanProgressVisible,
  SCAN_PROGRESS_VISIBILITY_EVENT,
  setScanProgressVisible,
} from "@/lib/progressVisibility";
import {
  formatScanDuration,
  heartbeatLabel,
  scanLivenessLabel,
  scanWorkCounter,
} from "@/lib/scanProgress";
import type {
  ScanCompletionSummary,
  ScanProgress,
} from "@/types/scanJob";

interface ScanProgressBannerProps {
  progress: ScanProgress | null;
  summary: ScanCompletionSummary | null;
  onStop: () => void;
}

function summaryItems(
  summary: ScanCompletionSummary
): Array<[string, number]> {
  return [
    ["Detected", summary.detected],
    ["Recognized", summary.recognized],
    ["Eligible", summary.eligible],
    ["Other detections", summary.excluded],
    ["Review required", summary.reviewRequired],
    ["Unread", summary.unread],
    ["Balloons added", summary.added],
    ["Existing skipped", summary.skippedExisting],
    ["Duplicates merged", summary.skippedDuplicates],
  ];
}

export function ScanProgressBanner({
  progress,
  summary,
  onStop,
}: ScanProgressBannerProps) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const handleVisibility = (event: Event) => {
      setVisible((event as CustomEvent<boolean>).detail);
    };

    setVisible(true);
    setScanProgressVisible(true);

    window.addEventListener(
      SCAN_PROGRESS_VISIBILITY_EVENT,
      handleVisibility
    );

    return () => {
      window.removeEventListener(
        SCAN_PROGRESS_VISIBILITY_EVENT,
        handleVisibility
      );
    };
  }, []);

  useEffect(() => {
    setVisible(isScanProgressVisible());
  }, [progress?.jobId]);

  if (!visible || (!progress && !summary)) return null;

  if (!progress && summary) {
    return (
      <aside
        className="qtracx-scan-progress-popup border border-emerald-200"
        aria-live="polite"
        aria-label="Last auto-balloon scan summary"
      >
        <header className="flex items-start justify-between gap-3 border-b border-slate-200 px-4 py-3">
          <div>
            <p className="text-sm font-semibold text-slate-900">
              Last scan summary
            </p>

            <p className="mt-0.5 text-xs text-slate-500">
              {summary.scopeKind === "page"
                ? "Whole-page Auto-Balloon scan"
                : "Selected-section Auto-Balloon scan"}
            </p>
          </div>

          <button
            type="button"
            onClick={() => setScanProgressVisible(false)}
            title="Hide summary"
            aria-label="Hide last scan summary"
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-lg text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            ×
          </button>
        </header>

        <div className="grid grid-cols-2 gap-px bg-slate-200 sm:grid-cols-3">
          {summaryItems(summary).map(([label, value]) => (
            <div key={label} className="bg-white px-4 py-3">
              <p className="text-lg font-semibold tabular-nums text-slate-900">
                {value}
              </p>

              <p className="mt-0.5 text-[11px] leading-4 text-slate-500">
                {label}
              </p>
            </div>
          ))}
        </div>

        {summary.reviewRequired > 0 && (
          <div className="border-t border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800">
            {summary.reviewRequired} detection
            {summary.reviewRequired === 1 ? "" : "s"} still require review in
            the right inspector.
          </div>
        )}
      </aside>
    );
  }

  if (!progress) return null;

  const percent = Math.max(0, Math.min(100, progress.percent));
  const counter = scanWorkCounter(progress);
  const isStopping = progress.status === "cancelling";
  const isQueued = progress.status === "queued";
  const isActive =
    progress.status === "queued" ||
    progress.status === "running" ||
    isStopping;

  const needsAttention =
    progress.liveness === "long_running" ||
    progress.liveness === "slow_progress" ||
    progress.liveness === "possibly_stalled" ||
    isStopping;

  const stageTitle: Record<string, string> = {
    queued: "Queued",
    waiting: "Waiting for OCR",
    layout: "Analysing page layout",
    preparing: "Preparing scan",
    proposing: "Finding candidate regions",
    detecting: "Detecting objects",
    refining: "Refining coverage gaps",
    grouping: "Grouping objects",
    recognizing: "Recognizing values",
    recovering: "Recovering uncertain values",
    context: "Reading filter context",
    filtering: "Classifying candidates",
    rereading: "Reading final values",
    splitting: "Separating stacked values",
    slanted: "Reading slanted callouts",
    notes: "Reading drawing notes",
    gdt: "Reading GD&T frames",
    deduplicating: "Removing duplicate values",
    finalizing: "Finalizing balloons",
    cancelling: "Stopping scan",
    cancelled: "Scan stopped",
    complete: "Scan complete",
    failed: "Scan failed",
  };

  const progressTone = needsAttention
    ? "bg-amber-500"
    : "bg-blue-600";

  const cardTone = needsAttention
    ? "border-amber-200"
    : "border-blue-200";

  return (
    <aside
      className={`qtracx-scan-progress-popup border ${cardTone}`}
      aria-live="polite"
      aria-label="Auto-balloon scan progress"
    >
      <header className="flex items-start justify-between gap-3 border-b border-slate-200 px-4 py-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-900">
            {stageTitle[progress.stage] ?? progress.stage}
          </p>

          <p className="mt-0.5 truncate text-xs text-slate-500">
            {progress.message}
          </p>
        </div>

        <button
          type="button"
          onClick={() => setScanProgressVisible(false)}
          title="Hide progress"
          aria-label="Hide scan progress"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-lg text-slate-400 hover:bg-slate-100 hover:text-slate-700"
        >
          ×
        </button>
      </header>

      <div className="space-y-3 px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs font-medium text-slate-600">
            Overall progress
          </span>

          <span className="text-sm font-semibold tabular-nums text-slate-900">
            {percent}%
          </span>
        </div>

        <div className="h-2 overflow-hidden rounded-full bg-slate-100">
          <div
            className={`h-full rounded-full transition-[width] duration-300 ${progressTone}`}
            style={{ width: `${percent}%` }}
          />
        </div>

        {counter && (
          <p className="text-xs tabular-nums text-slate-600">
            {counter}
          </p>
        )}

        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
          <div>
            <dt className="text-slate-400">Elapsed</dt>
            <dd className="mt-0.5 font-medium tabular-nums text-slate-700">
              {formatScanDuration(progress.elapsedSeconds)}
            </dd>
          </div>

          <div>
            <dt className="text-slate-400">Current step</dt>
            <dd className="mt-0.5 font-medium tabular-nums text-slate-700">
              {formatScanDuration(progress.stepElapsedSeconds)}
            </dd>
          </div>

          {progress.estimatedRemainingSeconds !== null && (
            <div>
              <dt className="text-slate-400">Stage ETA</dt>
              <dd className="mt-0.5 font-medium tabular-nums text-slate-700">
                {formatScanDuration(
                  progress.estimatedRemainingSeconds
                )}
              </dd>
            </div>
          )}

          {progress.candidateCount > 0 && (
            <div>
              <dt className="text-slate-400">Candidates</dt>
              <dd className="mt-0.5 font-medium tabular-nums text-slate-700">
                {progress.candidateCount}
              </dd>
            </div>
          )}

          {isQueued && progress.queuePosition !== null && (
            <div>
              <dt className="text-slate-400">Queue position</dt>
              <dd className="mt-0.5 font-medium tabular-nums text-slate-700">
                {progress.queuePosition}/{progress.queueDepth}
              </dd>
            </div>
          )}

          {isQueued &&
            progress.estimatedQueueWaitSeconds !== null && (
              <div>
                <dt className="text-slate-400">
                  Estimated queue wait
                </dt>
                <dd className="mt-0.5 font-medium tabular-nums text-slate-700">
                  {formatScanDuration(
                    progress.estimatedQueueWaitSeconds
                  )}
                </dd>
              </div>
            )}
        </dl>

        <div className="flex items-center justify-between gap-3 border-t border-slate-100 pt-3">
          <span className="inline-flex min-w-0 items-center gap-2 text-xs font-medium text-slate-600">
            <span
              className={`h-2 w-2 shrink-0 rounded-full ${progressTone} ${isActive ? "animate-pulse" : ""
                }`}
            />

            <span className="truncate">
              {scanLivenessLabel[progress.liveness]}

              {(progress.status === "running" || isStopping) && (
                <>
                  {" "}
                  · {heartbeatLabel(progress.heartbeatAgeSeconds)}
                </>
              )}
            </span>
          </span>

          {isActive && (
            <button
              type="button"
              onClick={onStop}
              disabled={!progress.jobId || isStopping}
              className="shrink-0 rounded-lg border border-red-200 px-3 py-1.5 text-xs font-semibold text-red-600 hover:bg-red-50 disabled:cursor-wait disabled:opacity-50"
            >
              {isStopping ? "Stopping…" : "Stop"}
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}