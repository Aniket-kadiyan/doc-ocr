"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { saveChecksheetReadings } from "@/lib/checksheetClient";
import type { ReadingPatch } from "@/types/checksheet";

export type AutosaveStatus = "idle" | "unsaved" | "saving" | "saved" | "error";

interface UseChecksheetAutosaveArgs {
  checksheetId: string;
  runId: string;
  enabled: boolean;
  delayMs?: number;
}

const patchKey = (patch: ReadingPatch) =>
  `${patch.annotation_id}\u0000${patch.column_id}`;

export function useChecksheetAutosave({
  checksheetId,
  runId,
  enabled,
  delayMs = 650,
}: UseChecksheetAutosaveArgs) {
  const [status, setStatus] = useState<AutosaveStatus>("idle");
  const [error, setError] = useState("");
  const pendingRef = useRef(new Map<string, ReadingPatch>());
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightRef = useRef<Promise<void> | null>(null);
  const flushRef = useRef<() => Promise<void>>(async () => undefined);

  const flush = useCallback(async (): Promise<void> => {
    if (!enabled) return;
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (inFlightRef.current) {
      await inFlightRef.current;
      if (pendingRef.current.size > 0) return flushRef.current();
      return;
    }
    const batch = [...pendingRef.current.values()];
    if (batch.length === 0) return;
    pendingRef.current.clear();
    setStatus("saving");
    setError("");
    const request = (async () => {
      try {
        await saveChecksheetReadings(checksheetId, runId, batch);
        setStatus("saved");
      } catch (reason) {
        for (const patch of batch) {
          const key = patchKey(patch);
          if (!pendingRef.current.has(key)) pendingRef.current.set(key, patch);
        }
        setStatus("error");
        setError(
          reason instanceof Error ? reason.message : "Autosave failed."
        );
        throw reason;
      } finally {
        inFlightRef.current = null;
      }
    })();
    inFlightRef.current = request;
    await request;
    if (pendingRef.current.size > 0) {
      timerRef.current = setTimeout(
        () => void flushRef.current().catch(() => undefined),
        delayMs
      );
    }
  }, [checksheetId, delayMs, enabled, runId]);

  useEffect(() => {
    flushRef.current = flush;
  }, [flush]);

  useEffect(() => {
    pendingRef.current.clear();
    setStatus("idle");
    setError("");
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [checksheetId, runId]);

  const queue = useCallback(
    (patch: ReadingPatch) => {
      if (!enabled) return;
      pendingRef.current.set(patchKey(patch), patch);
      setStatus("unsaved");
      setError("");
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(
        () => void flushRef.current().catch(() => undefined),
        delayMs
      );
    },
    [delayMs, enabled]
  );

  return { status, error, queue, flush };
}
