"use client";

import dynamic from "next/dynamic";
import { BalloonStyleProvider } from "@/components/BalloonStyleProvider";
import type { ChecksheetRunResponse, ChecksheetRow } from "@/types/checksheet";

const ChecksheetDrawingPreview = dynamic(
  () =>
    import("@/components/checksheets/ChecksheetDrawingPreview").then(
      (module) => ({ default: module.ChecksheetDrawingPreview })
    ),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-72 items-center justify-center text-sm text-slate-400">
        Loading drawing preview…
      </div>
    ),
  }
);

interface ChecksheetPreviewLoaderProps {
  document: ChecksheetRunResponse["document"];
  pdfRenderScale: number;
  row: ChecksheetRow | null;
  balloonStyle: ChecksheetRunResponse["revision"]["balloon_style"];
}

export function ChecksheetPreviewLoader({
  balloonStyle,
  ...props
}: ChecksheetPreviewLoaderProps) {
  return (
    <BalloonStyleProvider selection={balloonStyle}>
      <ChecksheetDrawingPreview {...props} />
    </BalloonStyleProvider>
  );
}
