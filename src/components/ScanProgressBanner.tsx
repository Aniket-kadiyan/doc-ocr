import type { ScanProgress } from "@/types/scanJob";
import {
  formatScanDuration,
  heartbeatLabel,
  scanLivenessLabel,
  scanWorkCounter,
} from "@/lib/scanProgress";

interface ScanProgressBannerProps {
  progress: ScanProgress;
  onStop: () => void;
}

export function ScanProgressBanner({
  progress,
  onStop,
}: ScanProgressBannerProps) {
  const percent = Math.max(0, Math.min(100, progress.percent));
  const counter = scanWorkCounter(progress);
  const isStopping = progress.status === "cancelling";
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
    finalizing: "Finalizing balloons",
    cancelling: "Stopping scan",
    cancelled: "Scan stopped",
    complete: "Scan complete",
    failed: "Scan failed",
  };
  const tone = needsAttention
    ? "border-amber-200 bg-amber-50 text-amber-950"
    : "border-blue-200 bg-blue-50 text-blue-950";
  const progressTone = needsAttention ? "bg-amber-500" : "bg-blue-600";

  return (
    <div
      className={`border-b px-4 py-2 ${tone}`}
      aria-live="polite"
      aria-label="Auto-balloon scan progress"
    >
      <div className="mx-auto max-w-5xl text-xs">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="font-semibold">
            {stageTitle[progress.stage] ?? progress.stage}
          </span>
          {counter && (
            <span className="min-w-0 flex-1 truncate tabular-nums">
              {counter}
            </span>
          )}
          {progress.candidateCount > 0 && (
            <span className="tabular-nums">
              Candidates {progress.candidateCount}
            </span>
          )}
          <span className="tabular-nums">
            Elapsed {formatScanDuration(progress.elapsedSeconds)}
          </span>
          {progress.estimatedRemainingSeconds !== null && (
            <span className="tabular-nums">
              Stage ETA {formatScanDuration(progress.estimatedRemainingSeconds)}
            </span>
          )}
          {progress.status === "running" &&
            progress.total > progress.completed &&
            progress.estimatedRemainingSeconds === null && (
              <span>Calculating estimate</span>
            )}
          <span className="w-10 text-right font-medium tabular-nums">
            {percent}%
          </span>
          {isActive && (
            <button
              type="button"
              onClick={onStop}
              disabled={!progress.jobId || isStopping}
              className="rounded border border-current px-2 py-0.5 font-semibold disabled:cursor-wait disabled:opacity-50"
            >
              {isStopping ? "Stopping…" : "Stop"}
            </button>
          )}
        </div>

        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
          <span className="min-w-0 flex-1 truncate">{progress.message}</span>
          <span className="tabular-nums">
            Current step {formatScanDuration(progress.stepElapsedSeconds)}
          </span>
          <span className="inline-flex items-center gap-1.5 font-medium">
            <span
              className={`h-2 w-2 rounded-full ${progressTone} ${
                isActive ? "animate-pulse" : ""
              }`}
            />
            {scanLivenessLabel[progress.liveness]}
            {(progress.status === "running" || isStopping) && (
              <> · {heartbeatLabel(progress.heartbeatAgeSeconds)}</>
            )}
          </span>
        </div>
      </div>

      <div className="mx-auto mt-1.5 h-1.5 max-w-5xl overflow-hidden rounded-full bg-black/10">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${progressTone}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
