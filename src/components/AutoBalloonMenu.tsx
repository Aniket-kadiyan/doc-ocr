"use client";

import { useEffect, useRef, useState } from "react";
import {
  SCAN_SETTINGS_EVENT,
  isGdtScanEnabled,
  setGdtScanEnabled,
} from "@/lib/scanSettings";

interface AutoBalloonMenuProps {
  disabled: boolean;
  selectingSection: boolean;
  running: boolean;
  onSelectSection: () => void;
  onScanWholePage: () => void;
}

export function AutoBalloonMenu({
  disabled,
  selectingSection,
  running,
  onSelectSection,
  onScanWholePage,
}: AutoBalloonMenuProps) {
  const [open, setOpen] = useState(false);
  const [readGdt, setReadGdt] = useState(true);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sync = () => setReadGdt(isGdtScanEnabled());
    sync();
    window.addEventListener(SCAN_SETTINGS_EVENT, sync);
    return () => window.removeEventListener(SCAN_SETTINGS_EVENT, sync);
  }, []);

  useEffect(() => {
    if (!open) return;
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    return () => document.removeEventListener("mousedown", closeOnOutsideClick);
  }, [open]);

  const choose = (action: () => void) => {
    setOpen(false);
    action();
  };

  return (
    <div ref={menuRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        disabled={disabled || running}
        title="Automatically detect and balloon values in a selected section or the whole page"
        className={`rounded-lg px-3 py-1.5 text-sm font-medium disabled:opacity-50 ${
          selectingSection
            ? "bg-emerald-600 text-white"
            : "border border-slate-200 text-slate-700 hover:bg-slate-50"
        }`}
      >
        {running
          ? "Auto Ballooning…"
          : selectingSection
            ? "Selecting Sections…"
            : "Auto Balloon ▾"}
      </button>

      {open && !running && (
        <div className="absolute left-0 top-full z-30 mt-1 w-56 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-lg">
          <button
            type="button"
            onClick={() => choose(onSelectSection)}
            className="block w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-emerald-50"
          >
            <span className="font-medium">Select Sections</span>
            <span className="block text-[11px] text-slate-400">
              Draw one or more areas, then scan them in order
            </span>
          </button>
          <button
            type="button"
            onClick={() => choose(onScanWholePage)}
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-emerald-50"
          >
            <span className="font-medium">Whole Page</span>
            <span className="block text-[11px] text-slate-400">
              Scan the page, and read its X/Y/Z point table
            </span>
          </button>
          <label className="flex cursor-pointer items-start gap-2 border-t border-slate-100 bg-slate-50 px-3 py-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={readGdt}
              onChange={(event) => setGdtScanEnabled(event.target.checked)}
              className="mt-0.5 h-4 w-4 accent-emerald-600"
            />
            <span>
              <span className="font-medium">Read GD&amp;T frames</span>
              <span className="block text-[11px] text-slate-400">
                Turn off for a faster scan (3–20 s less a page). Frames can
                still be read later with GD&amp;T Symbols.
              </span>
            </span>
          </label>
        </div>
      )}
    </div>
  );
}

