"use client";

import dynamic from "next/dynamic";

const DrawingViewer = dynamic(
  () =>
    import("@/components/DrawingViewer").then((m) => ({
      default: m.DrawingViewer,
    })),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-screen items-center justify-center text-slate-600">
        Loading drawing workspace…
      </div>
    ),
  }
);

export function DrawingViewerLoader() {
  return <DrawingViewer />;
}
