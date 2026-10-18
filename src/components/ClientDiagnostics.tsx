"use client";

import { useEffect } from "react";
import { reportClientError } from "@/lib/clientDiagnostics";
import { getOcrApiUrl } from "@/lib/paddleOcrClient";

/** Captures browser-level failures without rendering or changing the UI. */
export function ClientDiagnostics() {
  useEffect(() => {
    const backendUrl = getOcrApiUrl();
    const onError = (event: ErrorEvent) => {
      void reportClientError(backendUrl, event.error, {
        kind: "runtime_error",
        operation: "window.error",
      });
    };
    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      void reportClientError(backendUrl, event.reason, {
        kind: "unhandled_rejection",
        operation: "window.unhandledrejection",
      });
    };

    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onUnhandledRejection);
    return () => {
      window.removeEventListener("error", onError);
      window.removeEventListener("unhandledrejection", onUnhandledRejection);
    };
  }, []);

  return null;
}
