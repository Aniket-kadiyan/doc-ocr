"use client";

import { useEffect, useRef, useState } from "react";

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
  const menuRef = useRef<HTMLDivElement>(null);

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
              Scan the complete current drawing page
            </span>
          </button>
        </div>
      )}
    </div>
  );
}
