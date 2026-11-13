"use client";

import { useEffect, useState } from "react";
import { BalloonStylePicker } from "@/components/BalloonStylePicker";
import { DEBUG_DUMP_ENABLED } from "@/lib/featureFlags";
import {
  readOcrDebugDumpToggles,
  setOcrDebugDumpEnabled,
} from "@/lib/ocrDebugDump";
import {
  isGdtScanEnabled,
  isSlantedScanEnabled,
  SCAN_SETTINGS_EVENT,
  setGdtScanEnabled,
  setSlantedScanEnabled,
} from "@/lib/scanSettings";

interface WorkspaceSettingsPanelProps {
  open: boolean;
  disabled?: boolean;
  findingGdtSymbols: boolean;
  onFindGdtSymbols: () => void;
  onClose: () => void;
}

export function WorkspaceSettingsPanel({
  open,
  disabled = false,
  findingGdtSymbols,
  onFindGdtSymbols,
  onClose,
}: WorkspaceSettingsPanelProps) {
  const [readGdt, setReadGdt] = useState(true);
  const [readSlanted, setReadSlanted] = useState(true);
  const [debugDump, setDebugDump] = useState(false);

  useEffect(() => {
    const syncScanSettings = () => {
      setReadGdt(isGdtScanEnabled());
      setReadSlanted(isSlantedScanEnabled());
    };

    const syncDebugSetting = () => {
      setDebugDump(readOcrDebugDumpToggles().enabled);
    };

    syncScanSettings();
    syncDebugSetting();

    window.addEventListener(
      SCAN_SETTINGS_EVENT,
      syncScanSettings
    );
    window.addEventListener(
      "ocr-debug-dump-changed",
      syncDebugSetting
    );

    return () => {
      window.removeEventListener(
        SCAN_SETTINGS_EVENT,
        syncScanSettings
      );
      window.removeEventListener(
        "ocr-debug-dump-changed",
        syncDebugSetting
      );
    };
  }, []);

  useEffect(() => {
    if (!open) return;

    setReadGdt(isGdtScanEnabled());
    setReadSlanted(isSlantedScanEnabled());
    setDebugDump(readOcrDebugDumpToggles().enabled);
  }, [open]);

  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [open, onClose]);

  if (!open) return null;

  const handleReset = () => {
    setGdtScanEnabled(true);
    setSlantedScanEnabled(true);
    setReadGdt(true);
    setReadSlanted(true);
  };

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center p-4">
      <button
        type="button"
        aria-label="Close settings"
        className="absolute inset-0 bg-slate-950/40 backdrop-blur-[1px]"
        onClick={onClose}
      />

      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="workspace-settings-title"
        className="relative flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl"
      >
        <header className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
          <div>
            <h2
              id="workspace-settings-title"
              className="text-lg font-semibold text-slate-900"
            >
              Drawing workspace settings
            </h2>

            <p className="mt-0.5 text-sm text-slate-500">
              Configure scan behaviour, balloon appearance and diagnostics.
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-xl text-slate-500 hover:bg-slate-50 hover:text-slate-800"
          >
            ×
          </button>
        </header>

        <div className="overflow-y-auto px-6 py-5">
          <div className="grid gap-5 md:grid-cols-2">
            <section className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
              <h3 className="text-sm font-semibold text-slate-900">
                Detection settings
              </h3>

              <p className="mt-1 text-xs leading-relaxed text-slate-500">
                These options apply to Auto-Balloon scans.
              </p>

              <label className="mt-4 flex cursor-pointer items-start gap-3">
                <input
                  type="checkbox"
                  checked={readGdt}
                  onChange={(event) => {
                    const checked = event.target.checked;
                    setReadGdt(checked);
                    setGdtScanEnabled(checked);
                  }}
                  className="mt-0.5 h-4 w-4 accent-blue-600"
                />

                <span>
                  <span className="block text-sm font-medium text-slate-800">
                    Read GD&amp;T frames
                  </span>

                  <span className="mt-0.5 block text-xs leading-relaxed text-slate-500">
                    Include feature control frames during Auto-Balloon scans.
                  </span>
                </span>
              </label>

              <label className="mt-4 flex cursor-pointer items-start gap-3">
                <input
                  type="checkbox"
                  checked={readSlanted}
                  onChange={(event) => {
                    const checked = event.target.checked;
                    setReadSlanted(checked);
                    setSlantedScanEnabled(checked);
                  }}
                  className="mt-0.5 h-4 w-4 accent-blue-600"
                />

                <span>
                  <span className="block text-sm font-medium text-slate-800">
                    Read slanted callouts
                  </span>

                  <span className="mt-0.5 block text-xs leading-relaxed text-slate-500">
                    Analyse rotated neighbourhoods for dimensions drawn along
                    angled leaders.
                  </span>
                </span>
              </label>
            </section>

            <section className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
              <h3 className="text-sm font-semibold text-slate-900">
                Balloon appearance
              </h3>

              <p className="mt-1 text-xs leading-relaxed text-slate-500">
                Choose the SVG balloon style and its display size.
              </p>

              <div className="mt-4">
                <BalloonStylePicker disabled={disabled} />
              </div>
            </section>

            <section className="rounded-xl border border-slate-200 bg-slate-50/60 p-4 md:col-span-2">
              <h3 className="text-sm font-semibold text-slate-900">
                Diagnostics
              </h3>

              <p className="mt-1 text-xs leading-relaxed text-slate-500">
                Optional diagnostic actions for inspecting the current page.
              </p>

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => {
                    onFindGdtSymbols();
                    onClose();
                  }}
                  disabled={disabled || findingGdtSymbols}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                >
                  {findingGdtSymbols
                    ? "Reading GD&T symbols…"
                    : "Analyse page GD&T symbols"}
                </button>

                {DEBUG_DUMP_ENABLED && (
                  <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2">
                    <input
                      type="checkbox"
                      checked={debugDump}
                      onChange={(event) => {
                        setOcrDebugDumpEnabled(event.target.checked);
                        setDebugDump(event.target.checked);
                      }}
                      className="h-4 w-4 accent-violet-600"
                    />

                    <span className="text-sm font-medium text-slate-700">
                      Save OCR debug output
                    </span>
                  </label>
                )}
              </div>
            </section>
          </div>
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 bg-slate-50 px-6 py-4">
          <button
            type="button"
            onClick={handleReset}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            Reset detection settings
          </button>

          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            Close
          </button>
        </footer>
      </section>
    </div>
  );
}