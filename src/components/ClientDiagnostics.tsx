"use client";

import { useEffect } from "react";
import { reportClientError } from "@/lib/clientDiagnostics";
import { getOcrApiUrl } from "@/lib/paddleOcrClient";

export default function ClientDiagnostics() {
  useEffect(() => {
    const reportRuntimeError = (event: ErrorEvent) => {
      void reportClientError(getOcrApiUrl(), event.error, {
        kind: "runtime_error", operation: "window.error",
      });
    };
    const reportUnhandledRejection = (event: PromiseRejectionEvent) => {
      void reportClientError(getOcrApiUrl(), event.reason, {
        kind: "unhandled_rejection", operation: "window.unhandled-rejection",
      });
    };
    window.addEventListener("error", reportRuntimeError);
    window.addEventListener("unhandledrejection", reportUnhandledRejection);
    return () => {
      window.removeEventListener("error", reportRuntimeError);
      window.removeEventListener("unhandledrejection", reportUnhandledRejection);
    };
  }, []);
  return null;
}
