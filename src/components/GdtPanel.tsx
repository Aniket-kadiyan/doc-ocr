"use client";

import { useMemo, useState } from "react";
import {
  categoryLabel,
  describeFrame,
  detectionKey,
  formatConfidence,
  groupByCategory,
  reviewReason,
  statusColor,
  toleranceUnitLabel,
  STATUS_LABEL,
} from "@/lib/gdtSymbols";
import { submitGdtCorrection } from "@/lib/gdtClient";
import {
  GDT_CONFIRMED,
  type GdtCorrectionAction,
  type GdtDetection,
  type GdtFrame,
  type GdtPageResult,
  type GdtRegistry,
} from "@/types/gdt";

interface GdtPanelProps {
  result: GdtPageResult;
  registry: GdtRegistry | null;
  selectedKey: string | null;
  onSelect: (key: string | null) => void;
  onClear: () => void;
  disabled?: boolean;
}

/**
 * The reviewer's view of what the GD&T pass found.
 *
 * Built around the premise that a detection is a *claim awaiting
 * confirmation*, not a result. So the panel leads with what needs attention,
 * shows the engineering meaning rather than only the name, and gives the
 * operator somewhere to disagree — the corrections it records are the only
 * real-drawing ground truth this system will ever accumulate.
 *
 * Nothing here writes into the annotation store. A GD&T reading is not an
 * inspection value: it carries no balloon number and no measured result, and
 * turning one into a value automatically would put a tolerance onto a
 * checksheet that no one measured.
 */
export function GdtPanel({
  result,
  registry,
  selectedKey,
  onSelect,
  onClear,
  disabled = false,
}: GdtPanelProps) {
  const [showAll, setShowAll] = useState(false);

  const keyed = useMemo(
    () =>
      result.symbols.map((detection, index) => ({
        key: detectionKey(detection, index),
        detection,
      })),
    [result.symbols]
  );

  const attention = keyed.filter(
    ({ detection }) =>
      detection.status !== GDT_CONFIRMED || detection.warnings.length > 0
  );
  const visible = showAll ? keyed : attention;
  const selected = keyed.find((entry) => entry.key === selectedKey) ?? null;
  const frameById = new Map(result.frames.map((frame) => [frame.id, frame]));

  return (
    <section className="border-t border-slate-200">
      <div className="flex items-start justify-between gap-2 px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-slate-900">
            GD&amp;T symbols
          </h2>
          <p className="text-xs text-slate-500">
            {result.counts.frames} frame
            {result.counts.frames === 1 ? "" : "s"} ·{" "}
            {result.counts.confirmed} recognised
            {result.counts.ambiguous > 0 && (
              <>
                {" · "}
                <span className="text-amber-700">
                  {result.counts.ambiguous} need review
                </span>
              </>
            )}
          </p>
        </div>
        <button
          type="button"
          onClick={onClear}
          title="Remove the GD&T overlay from the drawing"
          className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-200 hover:text-slate-800"
        >
          Clear
        </button>
      </div>

      {result.errors.length > 0 && (
        <p className="mx-4 mb-2 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
          {result.errors.length} region
          {result.errors.length === 1 ? "" : "s"} could not be processed; the
          rest of the page was read.
        </p>
      )}

      {result.standards_findings.length > 0 && (
        <ul className="mx-4 mb-2 space-y-1">
          {result.standards_findings.map((finding, index) => (
            <li
              key={`finding-${index}`}
              className="rounded bg-slate-100 px-2 py-1 text-[11px] text-slate-700"
            >
              {finding.detail}
            </li>
          ))}
        </ul>
      )}

      {/* Feature control frames, read as structure rather than as a string. */}
      {result.frames.length > 0 && (
        <div className="px-4 pb-2">
          <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            Feature control frames
          </h3>
          <ul className="space-y-1">
            {result.frames.map((frame) => (
              <FrameRow key={frame.id} frame={frame} />
            ))}
          </ul>
        </div>
      )}

      <div className="px-4 pb-2">
        <div className="mb-1 flex items-center justify-between">
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            {showAll ? "All detections" : "Needs review"}
          </h3>
          <button
            type="button"
            onClick={() => setShowAll((value) => !value)}
            className="rounded px-1.5 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-200 hover:text-slate-800"
          >
            {showAll ? `Show ${attention.length} to review` : `Show all ${keyed.length}`}
          </button>
        </div>

        {visible.length === 0 ? (
          <p className="text-xs text-slate-500">
            Nothing needs review on this page.
          </p>
        ) : (
          <ul className="space-y-0.5">
            {groupByCategory(visible.map((v) => v.detection)).map((group) => (
              <li key={group.category}>
                <p className="pt-1 text-[10px] uppercase tracking-wide text-slate-400">
                  {categoryLabel(group.category)}
                </p>
                <ul>
                  {group.items.map((detection) => {
                    const entry = visible.find(
                      (candidate) => candidate.detection === detection
                    );
                    if (!entry) return null;
                    const isSelected = entry.key === selectedKey;
                    return (
                      <li key={entry.key}>
                        <button
                          type="button"
                          onClick={() =>
                            onSelect(isSelected ? null : entry.key)
                          }
                          className={`flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-xs hover:bg-slate-100 ${
                            isSelected ? "bg-slate-100" : ""
                          }`}
                        >
                          <span
                            aria-hidden
                            className="h-2 w-2 shrink-0 rounded-full"
                            style={{
                              backgroundColor: statusColor(detection.status),
                            }}
                          />
                          <span className="min-w-0 flex-1 truncate text-slate-800">
                            {detection.normalized ? `${detection.normalized} ` : ""}
                            {detection.name}
                          </span>
                          <span className="shrink-0 tabular-nums text-slate-500">
                            {formatConfidence(detection.confidence)}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </li>
            ))}
          </ul>
        )}
      </div>

      {selected && (
        <DetectionDetail
          detection={selected.detection}
          frame={
            selected.detection.frame_id
              ? frameById.get(selected.detection.frame_id) ?? null
              : null
          }
          registry={registry}
          documentHash={result.document_hash ?? null}
          disabled={disabled}
        />
      )}
    </section>
  );
}

function FrameRow({ frame }: { frame: GdtFrame }) {
  return (
    <li className="rounded bg-slate-50 px-2 py-1 text-xs">
      <p className="font-medium text-slate-800">{describeFrame(frame)}</p>
      <p className="text-[10px] text-slate-500">
        {frame.tolerance.state === "present"
          ? `Tolerance ${toleranceUnitLabel(frame)}`
          : `Tolerance ${frame.tolerance.state}`}
        {frame.composite_id
          ? ` · composite segment ${frame.segment_index + 1}`
          : ""}
      </p>
      {frame.warnings.length > 0 && (
        <p className="mt-0.5 text-[10px] text-amber-700">{frame.warnings[0]}</p>
      )}
    </li>
  );
}

/**
 * Everything known about one detection, plus a way to disagree with it.
 *
 * Shows the engineering meaning, not just the name: the point of recognising
 * a ⌖ is that the reader learns it controls where a feature sits relative to
 * a datum reference frame, which is what tells them whether the reading is
 * plausible for the feature it is pointing at.
 */
function DetectionDetail({
  detection,
  frame,
  registry,
  documentHash,
  disabled,
}: {
  detection: GdtDetection;
  frame: GdtFrame | null;
  registry: GdtRegistry | null;
  documentHash: string | null;
  disabled: boolean;
}) {
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const entry = registry?.symbols.find((s) => s.id === detection.symbol_id);
  const reason = reviewReason(detection);
  const box = detection.bounding_box;

  async function record(
    action: GdtCorrectionAction,
    correctedSymbolId: string | null
  ) {
    setSaving(true);
    setError(null);
    const response = await submitGdtCorrection({
      action,
      detected_symbol_id: detection.symbol_id,
      corrected_symbol_id: correctedSymbolId,
      page: detection.page,
      bounding_box: box,
      detected_confidence: detection.confidence,
      detected_status: detection.status,
      document_hash: documentHash,
    });
    setSaving(false);
    if (response.ok) {
      setSaved(
        action === "confirm"
          ? "Confirmed"
          : action === "reject"
            ? "Marked as not a symbol"
            : "Correction saved"
      );
    } else {
      setError(response.error ?? "Could not save the correction");
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white px-4 py-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">
          {detection.normalized ? `${detection.normalized} ` : ""}
          {detection.name}
        </h3>
        <span
          className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium text-white"
          style={{ backgroundColor: statusColor(detection.status) }}
        >
          {STATUS_LABEL[detection.status]}
        </span>
      </div>

      {(entry?.description || detection.description) && (
        <p className="mt-1 text-xs text-slate-600">
          {entry?.description ?? detection.description}
        </p>
      )}

      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
        <dt className="text-slate-500">Confidence</dt>
        <dd className="tabular-nums text-slate-800">
          {formatConfidence(detection.confidence)} · {detection.method}
        </dd>

        <dt className="text-slate-500">Page</dt>
        <dd className="text-slate-800">
          {detection.page} at {Math.round(box.x)}, {Math.round(box.y)}
        </dd>

        {detection.associated_text && (
          <>
            <dt className="text-slate-500">Reads</dt>
            <dd className="text-slate-800">{detection.associated_text}</dd>
          </>
        )}

        {detection.value && (
          <>
            <dt className="text-slate-500">Value</dt>
            <dd className="text-slate-800">
              {String(detection.value.value)}
              {detection.value.unit === "drawing_unit"
                ? " (drawing units)"
                : detection.value.unit
                  ? ` ${detection.value.unit}`
                  : ""}
            </dd>
          </>
        )}

        {frame && (
          <>
            <dt className="text-slate-500">In frame</dt>
            <dd className="text-slate-800">{describeFrame(frame)}</dd>
          </>
        )}

        {entry?.standards?.standards?.length ? (
          <>
            <dt className="text-slate-500">Standard</dt>
            <dd className="text-slate-800">
              {entry.standards.standards.join(", ")}
            </dd>
          </>
        ) : null}

        {detection.raw_ocr && (
          <>
            <dt className="text-slate-500">OCR read</dt>
            <dd className="font-mono text-slate-700">{detection.raw_ocr}</dd>
          </>
        )}
      </dl>

      {reason && (
        <p className="mt-2 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-900">
          {reason}
        </p>
      )}

      {detection.warnings.slice(1).map((warning, index) => (
        <p
          key={`warning-${index}`}
          className="mt-1 rounded bg-slate-50 px-2 py-1 text-[11px] text-slate-600"
        >
          {warning}
        </p>
      ))}

      {/* Correcting an ambiguous reading is one click, because the
          recognizer has already named the alternatives it could not
          separate. That is the whole value of reporting them. */}
      {detection.alternatives.length > 0 && (
        <div className="mt-2">
          <p className="text-[10px] uppercase tracking-wide text-slate-400">
            It is actually
          </p>
          <div className="mt-1 flex flex-wrap gap-1">
            {detection.alternatives.map((alternative) => (
              <button
                key={alternative.symbol_id}
                type="button"
                disabled={disabled || saving}
                onClick={() => record("reclassify", alternative.symbol_id)}
                className="rounded border border-slate-300 px-1.5 py-0.5 text-[11px] text-slate-700 hover:bg-slate-100 disabled:opacity-50"
              >
                {alternative.name}
                <span className="ml-1 tabular-nums text-slate-400">
                  {formatConfidence(alternative.confidence)}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mt-2 flex flex-wrap gap-1">
        <button
          type="button"
          disabled={disabled || saving}
          onClick={() => record("confirm", detection.symbol_id)}
          className="rounded bg-emerald-600 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
        >
          Correct
        </button>
        <button
          type="button"
          disabled={disabled || saving}
          onClick={() => record("reject", null)}
          className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:bg-slate-100 disabled:opacity-50"
        >
          Not a symbol
        </button>
      </div>

      {saved && <p className="mt-1 text-[11px] text-emerald-700">{saved}</p>}
      {error && <p className="mt-1 text-[11px] text-red-700">{error}</p>}
    </div>
  );
}
