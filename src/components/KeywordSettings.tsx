"use client";

import { useEffect, useRef, useState } from "react";
import {
  DEFAULT_KEYWORDS,
  getTitleKeywords,
  parseKeywords,
  setTitleKeywords,
} from "@/lib/titleKeywords";
import {
  SCAN_SETTINGS_EVENT,
  isGdtScanEnabled,
  isSlantedScanEnabled,
  setGdtScanEnabled,
  setSlantedScanEnabled,
} from "@/lib/scanSettings";

/**
 * "Keywords to look for": the title-block labels to pull off each drawing.
 *
 * The list is read during Auto-Segment (sent to the backend, which finds each
 * label in the bottom-right table) and again at export time, where every
 * configured keyword gets a row whether or not it was found.
 */
export function KeywordSettings() {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [saved, setSaved] = useState(false);
  const [readGdt, setReadGdt] = useState(true);
  const [readSlanted, setReadSlanted] = useState(true);
  const panelRef = useRef<HTMLDivElement>(null);

  // Load the stored list when the panel opens (localStorage is client-only, so
  // this cannot run during render).
  useEffect(() => {
    const sync = () => {
      setReadGdt(isGdtScanEnabled());
      setReadSlanted(isSlantedScanEnabled());
    };
    sync();
    window.addEventListener(SCAN_SETTINGS_EVENT, sync);
    return () => window.removeEventListener(SCAN_SETTINGS_EVENT, sync);
  }, []);

  useEffect(() => {
    if (open) {
      setDraft(getTitleKeywords().join(", "));
      setSaved(false);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (!panelRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  const parsed = parseKeywords(draft);

  const handleSave = () => {
    setTitleKeywords(parsed);
    setSaved(true);
  };

  return (
    <div className="relative" ref={panelRef}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title="Choose which title-block fields to pull off the drawing"
        className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
          open
            ? "bg-slate-700 text-white"
            : "border border-slate-200 text-slate-700 hover:bg-slate-50"
        }`}
      >
        Settings
      </button>

      {open && (
        <div className="absolute left-0 top-full z-20 mt-1 w-80 rounded-lg border border-slate-200 bg-white p-3 shadow-lg">
          <label
            htmlFor="title-keywords"
            className="block text-sm font-medium text-slate-700"
          >
            Keywords to look for
          </label>
          <p className="mt-1 text-[11px] leading-snug text-slate-500">
            Title-block labels to read off the drawing, comma separated. Each
            one gets its own row in every export.
          </p>

          <textarea
            id="title-keywords"
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              setSaved(false);
            }}
            rows={3}
            placeholder={DEFAULT_KEYWORDS.join(", ")}
            className="mt-2 w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-900 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
          />

          <div className="mt-2 flex flex-wrap gap-1">
            {parsed.length === 0 ? (
              <span className="text-[11px] text-slate-400">
                No keywords — the title-block section is left out.
              </span>
            ) : (
              parsed.map((k) => (
                <span
                  key={k}
                  className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600"
                >
                  {k}
                </span>
              ))
            )}
          </div>

          <label className="mt-3 flex cursor-pointer items-start gap-2 border-t border-slate-100 pt-3 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={readGdt}
              onChange={(event) => setGdtScanEnabled(event.target.checked)}
              className="mt-0.5 h-4 w-4 accent-emerald-600"
            />
            <span>
              <span className="font-medium">Read GD&amp;T frames</span>
              <span className="block text-[11px] leading-snug text-slate-400">
                Include feature control frames during Auto Balloon scans.
                Turn this off for drawings that do not contain GD&amp;T.
              </span>
            </span>
          </label>

          <label className="mt-3 flex cursor-pointer items-start gap-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={readSlanted}
              onChange={(event) =>
                setSlantedScanEnabled(event.target.checked)
              }
              className="mt-0.5 h-4 w-4 accent-emerald-600"
            />
            <span>
              <span className="font-medium">Read slanted callouts</span>
              <span className="block text-[11px] leading-snug text-slate-400">
                Analyse rotated neighbourhoods for dimensions drawn along
                angled leaders. Turn this off for orthogonal-only drawings.
              </span>
            </span>
          </label>

          <div className="mt-3 flex items-center gap-2">
            <button
              type="button"
              onClick={handleSave}
              className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
            >
              Save
            </button>
            <button
              type="button"
              onClick={() => {
                setDraft(DEFAULT_KEYWORDS.join(", "));
                setSaved(false);
              }}
              className="rounded border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
            >
              Reset
            </button>
            {saved && (
              <span className="text-[11px] text-emerald-600">
                Saved — re-run Auto-Segment to pick up the change.
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
