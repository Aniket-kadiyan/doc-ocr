import type { ScanProgress } from "@/types/scanJob";

interface ScanProgressBannerProps {
  progress: ScanProgress;
}

export function ScanProgressBanner({ progress }: ScanProgressBannerProps) {
  const percent = Math.max(0, Math.min(100, progress.percent));
  const counter =
    progress.total > 0
      ? `${progress.completed} / ${progress.total}`
      : null;

  return (
    <div className="border-b border-blue-200 bg-blue-50 px-4 py-2 text-blue-900">
      <div className="mx-auto flex max-w-5xl items-center gap-3 text-xs">
        <span className="min-w-0 flex-1 truncate font-medium">
          {progress.message}
        </span>
        {counter && <span className="tabular-nums">{counter}</span>}
        <span className="w-10 text-right tabular-nums">{percent}%</span>
      </div>
      <div className="mx-auto mt-1 h-1.5 max-w-5xl overflow-hidden rounded-full bg-blue-100">
        <div
          className="h-full rounded-full bg-blue-600 transition-[width] duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
