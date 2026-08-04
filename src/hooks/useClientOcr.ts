"use client";

import { useEffect, useState } from "react";
import {
  getActiveOcrEngine,
  getOcrApiStatus,
  isOcrReady,
  preloadOcr,
} from "@/lib/clientOcr";

export function useClientOcr() {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [engineLabel, setEngineLabel] = useState("Connecting to PaddleOCR…");

  useEffect(() => {
    let cancelled = false;
    preloadOcr()
      .then(() => {
        if (cancelled) return;
        const engine = getActiveOcrEngine();
        setReady(isOcrReady());
        if (engine === "paddleocr") {
          setEngineLabel("PaddleOCR + symbol vision (local API)");
          setError(null);
        } else {
          setEngineLabel("PaddleOCR API offline");
          setError(
            "Start OCR server: uvicorn main:app --reload --port 8000"
          );
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "OCR init failed");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { ready, error, engineLabel, api: getOcrApiStatus() };
}
