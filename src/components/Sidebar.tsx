"use client";

import { useState } from "react";
import type { Annotation } from "@/types/annotation";
import { BalloonThumbnail } from "@/components/BalloonThumbnail";

interface SidebarProps {
  annotations: Annotation[];
  reviewCandidates: ReviewSidebarItem[];
  disabled?: boolean;
  selectedId: string | null;
  selectedReviewCandidateId: string | null;
  onSelect: (id: string) => void;
  onSelectReview: (candidateId: string) => void;
  onEdit: (id: string) => void;
  onDelete: (id: string) => void;
  onMove: (id: string, targetNumber: number) => void;
  onToggleVisibility: (id: string) => void;
}

export interface ReviewSidebarItem {
  candidateId: string;
  text: string;
  page: number;
  reviewReason?: string;
  reviewOrder: number;
}

/** One sidebar record per ballooned value. Label, method, and tool are optional
 * metadata on that same record rather than separate annotations. */
export function Sidebar({
  annotations,
  reviewCandidates,
  disabled = false,
  selectedId,
  selectedReviewCandidateId,
  onSelect,
  onSelectReview,
  onEdit,
  onDelete,
  onMove,
  onToggleVisibility,
}: SidebarProps) {
  const [movingId, setMovingId] = useState<string | null>(null);
  const [moveTarget, setMoveTarget] = useState("");
  const values = annotations
    .filter((annotation) => annotation.kind !== "label")
    .sort((left, right) => left.number - right.number);
  const reviews = [...reviewCandidates].sort(
    (left, right) => left.reviewOrder - right.reviewOrder
  );

  return (
    <aside className="flex w-72 shrink-0 flex-col border-l border-slate-200 bg-slate-50">
      <div className="border-b border-slate-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-900">Values</h2>
        <p className="text-xs text-slate-500">
          {values.length} ballooned value{values.length === 1 ? "" : "s"}
          {reviews.length > 0 && (
            <>
              {" · "}
              {reviews.length} review{reviews.length === 1 ? "" : "s"}
            </>
          )}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {values.length === 0 && reviews.length === 0 ? (
          <p className="px-2 py-6 text-center text-sm text-slate-500">
            Choose Draw Value, then draw a box around a value on the drawing.
          </p>
        ) : (
          <>
            {values.length > 0 && (
              <section>
                <h3 className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                  Balloons
                </h3>
                <ul>
                  {values.map((annotation) => {
                    const selected = selectedId === annotation.id;
                    const moving = movingId === annotation.id;
                    const targetNumber = Number(moveTarget);
                    const targetIsValid =
                      moving &&
                      Number.isInteger(targetNumber) &&
                      targetNumber >= 1 &&
                      targetNumber <= values.length;
                    const moveChangesNumber =
                      targetIsValid && targetNumber !== annotation.number;
                    const details = [
                      annotation.type,
                      annotation.range && `Tolerance ${annotation.range}`,
                      annotation.method,
                      annotation.tool,
                    ]
                      .filter(Boolean)
                      .join(" · ");

                    return (
                      <li key={annotation.id} className="mb-2">
                        <div
                          className={`flex items-stretch overflow-hidden rounded-lg transition ${
                            selected
                              ? "bg-blue-100 ring-1 ring-blue-300"
                              : "bg-white"
                          }`}
                        >
                          <button
                            type="button"
                            disabled={disabled}
                            onClick={() => {
                              onSelect(annotation.id);
                              onEdit(annotation.id);
                            }}
                            title="Edit this value and its optional inspection metadata"
                            className="min-w-0 flex-1 px-3 py-2 text-left hover:bg-slate-100/70 disabled:cursor-not-allowed disabled:opacity-60"
                          >
                            <span className="flex items-start gap-2">
                              <BalloonThumbnail
                                number={annotation.number}
                                selected={selected}
                              />
                              <span className="min-w-0 flex-1">
                                <span className="flex items-center gap-1">
                                  {annotation.needsReview && (
                                    <span
                                      title="Low-confidence read — verify"
                                      className="inline-block h-2 w-2 shrink-0 rounded-full bg-amber-500"
                                    />
                                  )}
                                  <span className="truncate font-mono text-xs text-slate-800">
                                    {annotation.value || "Unread — review"}
                                  </span>
                                  {annotation.hidden && (
                                    <span className="rounded bg-slate-200 px-1 text-[9px] font-medium uppercase text-slate-500">
                                      Hidden
                                    </span>
                                  )}
                                </span>
                                {annotation.label && (
                                  <span className="mt-1 block truncate text-xs font-medium text-slate-600">
                                    {annotation.label}
                                  </span>
                                )}
                                {details && (
                                  <span className="mt-1 block text-[10px] text-slate-400">
                                    {details}
                                  </span>
                                )}
                              </span>
                            </span>
                          </button>
                          <button
                            type="button"
                            disabled={disabled}
                            onClick={() => onToggleVisibility(annotation.id)}
                            title={
                              annotation.hidden
                                ? `Show balloon ${annotation.number}`
                                : `Hide balloon ${annotation.number}`
                            }
                            className="border-l border-slate-100 px-2 text-[10px] font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {annotation.hidden ? "Show" : "Hide"}
                          </button>
                        </div>
                        <div className="mt-0.5 flex items-center justify-center gap-2 text-[10px]">
                          <button
                            type="button"
                            disabled={disabled}
                            onClick={() => {
                              setMovingId(annotation.id);
                              setMoveTarget(String(annotation.number));
                            }}
                            className="text-blue-600 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            Move
                          </button>
                          <span className="text-slate-300">·</span>
                          <button
                            type="button"
                            disabled={disabled}
                            onClick={() => onDelete(annotation.id)}
                            className="text-red-500 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            Remove
                          </button>
                        </div>

                        {moving && (
                          <form
                            onSubmit={(event) => {
                              event.preventDefault();
                              if (!moveChangesNumber) return;
                              onMove(annotation.id, targetNumber);
                              setMovingId(null);
                              setMoveTarget("");
                            }}
                            className="mt-2 rounded-lg border border-blue-200 bg-blue-50 p-2"
                          >
                            <label
                              htmlFor={`move-balloon-${annotation.id}`}
                              className="block text-[11px] font-medium text-slate-700"
                            >
                              Move balloon {annotation.number} to number
                            </label>
                            <input
                              id={`move-balloon-${annotation.id}`}
                              type="number"
                              min={1}
                              max={values.length}
                              step={1}
                              value={moveTarget}
                              onChange={(event) =>
                                setMoveTarget(event.target.value)
                              }
                              onKeyDown={(event) => {
                                if (event.key === "Escape") {
                                  setMovingId(null);
                                  setMoveTarget("");
                                }
                              }}
                              autoFocus
                              className="mt-1 w-full rounded border border-slate-300 bg-white px-2 py-1 text-sm text-slate-900 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                            />
                            <p className="mt-1 text-[10px] text-slate-500">
                              {!targetIsValid
                                ? `Enter a whole number from 1 to ${values.length}.`
                                : targetNumber === annotation.number
                                  ? `Balloon ${annotation.number} is already at this number.`
                                  : targetNumber < annotation.number
                                    ? `Balloons ${targetNumber}–${annotation.number - 1} become ${targetNumber + 1}–${annotation.number}.`
                                    : `Balloons ${annotation.number + 1}–${targetNumber} become ${annotation.number}–${targetNumber - 1}.`}
                            </p>
                            <div className="mt-2 flex justify-end gap-2">
                              <button
                                type="button"
                                onClick={() => {
                                  setMovingId(null);
                                  setMoveTarget("");
                                }}
                                className="rounded px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-200"
                              >
                                Cancel
                              </button>
                              <button
                                type="submit"
                                disabled={!moveChangesNumber}
                                className="rounded bg-blue-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
                              >
                                Apply
                              </button>
                            </div>
                          </form>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </section>
            )}

            {reviews.length > 0 && (
              <section
                className={
                  values.length > 0
                    ? "mt-4 border-t border-slate-200 pt-3"
                    : ""
                }
              >
                <h3 className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-wide text-amber-700">
                  Review required · {reviews.length}
                </h3>
                <ul>
                  {reviews.map((candidate) => {
                    const selected =
                      selectedReviewCandidateId === candidate.candidateId;
                    return (
                      <li key={candidate.candidateId} className="mb-2">
                        <button
                          type="button"
                          disabled={disabled}
                          onClick={() => onSelectReview(candidate.candidateId)}
                          title={
                            candidate.reviewReason ||
                            "Review this detected value"
                          }
                          className={`w-full rounded-lg border border-dashed px-3 py-2 text-left transition disabled:cursor-not-allowed disabled:opacity-60 ${
                            selected
                              ? "border-blue-400 bg-blue-50 ring-1 ring-blue-300"
                              : "border-slate-300 bg-white hover:bg-amber-50"
                          }`}
                        >
                          <span className="flex items-start gap-2">
                            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-slate-400 text-xs font-bold text-slate-600">
                              ?
                            </span>
                            <span className="min-w-0 flex-1">
                              <span className="block truncate font-mono text-xs text-slate-800">
                                {candidate.text.trim() || "Unread"}
                              </span>
                              <span className="mt-1 block text-[10px] text-slate-400">
                                Page {candidate.page}
                                {candidate.reviewReason
                                  ? ` · ${candidate.reviewReason}`
                                  : ""}
                              </span>
                            </span>
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </section>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
