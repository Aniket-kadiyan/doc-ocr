"use client";

import { useBalloonStyle } from "@/components/BalloonStyleProvider";
import { BALLOON_SIZES, type BalloonSize } from "@/types/balloonStyle";

export function BalloonStylePicker({ disabled = false }: { disabled?: boolean }) {
  const { manifest, selection, setSelection, ready } = useBalloonStyle();
  const definitions = manifest?.styles ?? [];

  return (
    <div className="flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2 py-1">
      <span className="text-xs font-medium text-slate-500">Balloon</span>
      <select
        aria-label="Balloon style"
        title="Balloon shape"
        value={selection.styleId}
        disabled={disabled || !ready || definitions.length === 0}
        onChange={(event) => {
          const definition = definitions.find((item) => item.id === event.target.value);
          if (!definition) return;
          setSelection({
            styleId: definition.id,
            styleVersion: definition.version,
            size: selection.size,
          });
        }}
        className="rounded border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-700 disabled:opacity-50"
      >
        {definitions.map((definition) => (
          <option key={`${definition.id}:${definition.version}`} value={definition.id}>
            {definition.name}
          </option>
        ))}
      </select>
      <select
        aria-label="Balloon size"
        title="Balloon size"
        value={selection.size}
        disabled={disabled || !ready}
        onChange={(event) =>
          setSelection({ ...selection, size: event.target.value as BalloonSize })
        }
        className="rounded border border-slate-200 bg-white px-2 py-1 text-xs font-medium capitalize text-slate-700 disabled:opacity-50"
      >
        {BALLOON_SIZES.map((size) => (
          <option key={size} value={size}>
            {size[0].toUpperCase() + size.slice(1)}
          </option>
        ))}
      </select>
    </div>
  );
}
