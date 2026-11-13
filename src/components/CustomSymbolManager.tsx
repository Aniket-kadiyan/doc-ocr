"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import {
  createCustomSymbol,
  deleteCustomSymbol,
  listCustomSymbols,
  updateCustomSymbol,
} from "@/lib/customSymbolClient";
import type {
  CustomSymbol,
  CustomSymbolLibrary,
  CustomSymbolUpdate,
} from "@/types/customSymbol";

const inputClass =
  "w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-900 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20";

interface SymbolFields {
  name: string;
  label: string;
  category: string;
  threshold: number;
  matchRotations: boolean;
  anySize: boolean;
}

const fieldsOf = (symbol: CustomSymbol): SymbolFields => ({
  name: symbol.name,
  label: symbol.label,
  category: symbol.category,
  threshold: symbol.threshold,
  matchRotations: symbol.match_rotations,
  anySize: symbol.any_size,
});

const errorText = (reason: unknown, fallback: string) =>
  reason instanceof Error ? reason.message : fallback;

/** Name, label, category, strictness and search options for one symbol. */
function SymbolFieldsEditor({
  fields,
  onChange,
  library,
}: {
  fields: SymbolFields;
  onChange: (next: SymbolFields) => void;
  library: CustomSymbolLibrary;
}) {
  const set = <K extends keyof SymbolFields>(key: K, value: SymbolFields[K]) =>
    onChange({ ...fields, [key]: value });
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="block text-sm text-slate-700">
        <span className="font-medium">Name</span>
        <input
          value={fields.name}
          onChange={(event) => set("name", event.target.value)}
          placeholder="e.g. Critical feature mark"
          maxLength={80}
          className={`mt-1 ${inputClass}`}
        />
      </label>
      <label className="block text-sm text-slate-700">
        <span className="font-medium">
          Balloon label <span className="font-normal text-slate-400">(optional)</span>
        </span>
        <input
          value={fields.label}
          onChange={(event) => set("label", event.target.value)}
          placeholder="Leave empty to use the name"
          maxLength={120}
          className={`mt-1 ${inputClass}`}
        />
      </label>
      <label className="block text-sm text-slate-700">
        <span className="font-medium">Category</span>
        <select
          value={fields.category}
          onChange={(event) => set("category", event.target.value)}
          className={`mt-1 ${inputClass}`}
        >
          {library.categories.map((category) => (
            <option key={category} value={category}>
              {category}
            </option>
          ))}
        </select>
      </label>
      <label className="block text-sm text-slate-700">
        <span className="font-medium">
          Match strictness{" "}
          <span className="font-mono text-slate-500">
            {fields.threshold.toFixed(2)}
          </span>
        </span>
        <input
          type="range"
          min={library.defaults.min_threshold}
          max={library.defaults.max_threshold}
          step={0.01}
          value={fields.threshold}
          onChange={(event) => set("threshold", Number(event.target.value))}
          className="mt-2 w-full accent-blue-600"
        />
        <span className="block text-[11px] leading-snug text-slate-400">
          Higher finds fewer, surer matches. Lower it only if real symbols are
          missed; raise it if wrong shapes are ballooned.
        </span>
      </label>
      <label className="flex cursor-pointer items-start gap-2 text-sm text-slate-700">
        <input
          type="checkbox"
          checked={fields.matchRotations}
          onChange={(event) => set("matchRotations", event.target.checked)}
          className="mt-0.5 h-4 w-4 accent-emerald-600"
        />
        <span>
          <span className="font-medium">Also match rotated</span>
          <span className="block text-[11px] leading-snug text-slate-400">
            Look for the symbol turned 90°, 180° and 270° too. Leave off for
            symbols whose direction matters.
          </span>
        </span>
      </label>
      <label className="flex cursor-pointer items-start gap-2 text-sm text-slate-700">
        <input
          type="checkbox"
          checked={fields.anySize}
          onChange={(event) => set("anySize", event.target.checked)}
          className="mt-0.5 h-4 w-4 accent-emerald-600"
        />
        <span>
          <span className="font-medium">Any size</span>
          <span className="block text-[11px] leading-snug text-slate-400">
            For a clean picture not cropped from a drawing. Slower, and more
            likely to balloon look-alike shapes.
          </span>
        </span>
      </label>
    </div>
  );
}

function AddSymbolForm({
  library,
  onAdded,
}: {
  library: CustomSymbolLibrary;
  onAdded: () => Promise<void>;
}) {
  const blank = useCallback(
    (): SymbolFields => ({
      name: "",
      label: "",
      category: library.defaults.category,
      threshold: library.defaults.threshold,
      matchRotations: false,
      anySize: false,
    }),
    [library.defaults]
  );
  const [fields, setFields] = useState<SymbolFields>(blank);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [inputKey, setInputKey] = useState(0);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const chooseFile = (chosen: File | null) => {
    setFile(chosen);
    setError("");
    if (chosen && !fields.name.trim()) {
      setFields((current) => ({
        ...current,
        name: chosen.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " "),
      }));
    }
  };

  const submit = async () => {
    if (!file || !fields.name.trim()) {
      setError("Choose an image and give the symbol a name.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await createCustomSymbol({
        file,
        name: fields.name.trim(),
        label: fields.label.trim(),
        category: fields.category,
        threshold: fields.threshold,
        matchRotations: fields.matchRotations,
        anySize: fields.anySize,
      });
      setFields(blank());
      setFile(null);
      setInputKey((key) => key + 1);
      await onAdded();
    } catch (reason) {
      setError(errorText(reason, "The symbol could not be saved."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
      <h2 className="text-base font-semibold text-slate-900">Add a symbol</h2>
      <p className="mt-1 text-sm text-slate-500">
        Upload a tight crop of the symbol taken from a drawing at normal zoom.
        It is matched from about 0.4× to 2.5× the size of that crop.
      </p>

      <div className="mt-4 flex flex-col gap-4 sm:flex-row">
        <label className="flex h-36 w-full shrink-0 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 text-center text-sm text-slate-500 hover:border-blue-400 sm:w-44">
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={preview}
              alt="Symbol preview"
              className="max-h-28 max-w-[10rem] object-contain"
            />
          ) : (
            <span className="px-3">Choose PNG / JPG image</span>
          )}
          <input
            key={inputKey}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
            onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
            className="sr-only"
          />
        </label>
        <div className="flex-1">
          <SymbolFieldsEditor
            fields={fields}
            onChange={setFields}
            library={library}
          />
        </div>
      </div>

      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      <div className="mt-4 flex justify-end">
        <button
          type="button"
          onClick={() => void submit()}
          disabled={busy}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {busy ? "Saving…" : "Add symbol"}
        </button>
      </div>
    </section>
  );
}

function SymbolRow({
  symbol,
  library,
  onSaved,
  onDeleted,
}: {
  symbol: CustomSymbol;
  library: CustomSymbolLibrary;
  onSaved: (symbol: CustomSymbol) => void;
  onDeleted: (id: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [fields, setFields] = useState<SymbolFields>(() => fieldsOf(symbol));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const save = async (changes: CustomSymbolUpdate) => {
    setBusy(true);
    setError("");
    try {
      onSaved(await updateCustomSymbol(symbol.id, changes));
      return true;
    } catch (reason) {
      setError(errorText(reason, "The change could not be saved."));
      return false;
    } finally {
      setBusy(false);
    }
  };

  const saveEdits = async () => {
    const ok = await save({
      name: fields.name.trim(),
      label: fields.label.trim(),
      category: fields.category,
      threshold: fields.threshold,
      match_rotations: fields.matchRotations,
      any_size: fields.anySize,
    });
    if (ok) setEditing(false);
  };

  const remove = async () => {
    if (
      !window.confirm(
        `Delete "${symbol.name}"? Scans will stop finding it. Balloons already placed stay.`
      )
    )
      return;
    setBusy(true);
    setError("");
    try {
      await deleteCustomSymbol(symbol.id);
      onDeleted(symbol.id);
    } catch (reason) {
      setError(errorText(reason, "The symbol could not be deleted."));
      setBusy(false);
    }
  };

  const options = [
    symbol.match_rotations && "Rotations",
    symbol.any_size && "Any size",
  ].filter(Boolean);

  return (
    <li
      className={`rounded-xl border bg-white p-4 shadow-sm ${
        symbol.enabled ? "border-slate-200" : "border-slate-200 opacity-60"
      }`}
    >
      <div className="flex flex-wrap items-start gap-4">
        <div className="flex h-20 w-20 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-white p-2">
          {symbol.image_data_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={symbol.image_data_url}
              alt={symbol.name}
              className="max-h-full max-w-full object-contain"
            />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold text-slate-900">{symbol.name}</h3>
            <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600">
              {symbol.category}
            </span>
            {!symbol.enabled && (
              <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[11px] text-amber-700">
                Off
              </span>
            )}
          </div>
          <dl className="mt-1 grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs text-slate-500">
            <dt>Balloon label</dt>
            <dd className="text-slate-700">
              {symbol.label || <span className="text-slate-400">{symbol.name} (name)</span>}
            </dd>
            <dt>Strictness</dt>
            <dd className="font-mono text-slate-700">{symbol.threshold.toFixed(2)}</dd>
            <dt>Search</dt>
            <dd className="text-slate-700">
              {options.length ? options.join(", ") : "Upright, near crop size"}
            </dd>
            <dt>Added</dt>
            <dd className="text-slate-700">
              {new Date(symbol.created_at).toLocaleString()}
            </dd>
          </dl>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex cursor-pointer items-center gap-1.5 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={symbol.enabled}
              disabled={busy}
              onChange={(event) => void save({ enabled: event.target.checked })}
              className="h-4 w-4 accent-emerald-600"
            />
            Detect
          </label>
          <button
            type="button"
            onClick={() => {
              setFields(fieldsOf(symbol));
              setEditing((value) => !value);
            }}
            disabled={busy}
            className="rounded border border-slate-200 px-2.5 py-1 text-sm text-slate-700 hover:bg-slate-50"
          >
            {editing ? "Cancel" : "Edit"}
          </button>
          <button
            type="button"
            onClick={() => void remove()}
            disabled={busy}
            className="rounded border border-red-200 px-2.5 py-1 text-sm text-red-600 hover:bg-red-50"
          >
            Delete
          </button>
        </div>
      </div>
      {editing && (
        <div className="mt-4 border-t border-slate-100 pt-4">
          <SymbolFieldsEditor fields={fields} onChange={setFields} library={library} />
          <div className="mt-3 flex justify-end">
            <button
              type="button"
              onClick={() => void saveEdits()}
              disabled={busy || !fields.name.trim()}
              className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              Save
            </button>
          </div>
        </div>
      )}
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </li>
  );
}

/**
 * The shared custom symbol library: add, review, tune and delete the symbols
 * every Auto Balloon scan looks for.
 */
export function CustomSymbolManager() {
  const [library, setLibrary] = useState<CustomSymbolLibrary | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setError("");
    try {
      setLibrary(await listCustomSymbols());
    } catch (reason) {
      setError(errorText(reason, "Could not load the symbol library."));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const replace = (symbol: CustomSymbol) =>
    setLibrary((current) =>
      current && {
        ...current,
        symbols: current.symbols.map((s) => (s.id === symbol.id ? symbol : s)),
      }
    );
  const drop = (id: string) =>
    setLibrary((current) =>
      current && { ...current, symbols: current.symbols.filter((s) => s.id !== id) }
    );

  const enabledCount = library?.symbols.filter((s) => s.enabled).length ?? 0;

  return (
    <main className="min-h-screen bg-slate-100">
      <header className="border-b border-slate-200 bg-white px-4 py-4 shadow-sm sm:px-8">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-slate-900">Custom Symbols</h1>
            <p className="mt-1 text-sm text-slate-500">
              Symbols every Auto Balloon scan looks for. Each match becomes a
              balloon with the symbol&apos;s label. Shared by all users and
              kept until deleted.
            </p>
          </div>
          <Link
            href="/"
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Drawing workspace
          </Link>
        </div>
      </header>

      <div className="mx-auto max-w-5xl space-y-6 p-4 sm:p-8">
        {error && (
          <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </p>
        )}

        {!library ? (
          !error && <p className="text-center text-sm text-slate-400">Loading…</p>
        ) : (
          <>
            <AddSymbolForm library={library} onAdded={refresh} />

            <section>
              <div className="flex items-baseline justify-between">
                <h2 className="text-base font-semibold text-slate-900">
                  Library
                </h2>
                <span className="text-sm text-slate-500">
                  {library.symbols.length} symbol
                  {library.symbols.length === 1 ? "" : "s"}, {enabledCount} detected
                </span>
              </div>
              {library.symbols.length === 0 ? (
                <p className="mt-3 rounded-xl border border-dashed border-slate-300 bg-white px-4 py-8 text-center text-sm text-slate-400">
                  No custom symbols yet.
                </p>
              ) : (
                <ul className="mt-3 space-y-3">
                  {library.symbols.map((symbol) => (
                    <SymbolRow
                      key={`${symbol.id}:${symbol.updated_at}`}
                      symbol={symbol}
                      library={library}
                      onSaved={replace}
                      onDeleted={drop}
                    />
                  ))}
                </ul>
              )}
              <p className="mt-3 text-xs text-slate-400">
                Changes apply to the next scan. Turn detection off for all
                symbols at once under Settings → Detect custom symbols.
              </p>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
