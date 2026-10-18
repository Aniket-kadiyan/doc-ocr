import {
  ApiRequestError,
  createRequestId,
  normalizeRequestId,
  requestIdFromError,
} from "@/lib/requestDiagnostics";

export type ClientDiagnosticKind =
  | "runtime_error"
  | "unhandled_rejection"
  | "api_error";

export interface ClientDiagnosticContext {
  kind: ClientDiagnosticKind;
  operation: string;
}

const WINDOW_MS = 60_000;
const MAX_EVENTS_PER_WINDOW = 10;
let windowStartedAt = 0;
let sentInWindow = 0;

const safeToken = (value: string, fallback: string) => {
  const token = value
    .trim()
    .replace(/[^A-Za-z0-9._:-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80);
  return token || fallback;
};

function errorType(error: unknown): string {
  if (error instanceof Error) return safeToken(error.name, "Error");
  if (error === null) return "NullRejection";
  return safeToken(typeof error, "UnknownError");
}

function allowDelivery(now: number): boolean {
  if (now - windowStartedAt >= WINDOW_MS) {
    windowStartedAt = now;
    sentInWindow = 0;
  }
  if (sentInWindow >= MAX_EVENTS_PER_WINDOW) return false;
  sentInWindow += 1;
  return true;
}

export function resetClientDiagnosticRateLimitForTests(): void {
  windowStartedAt = 0;
  sentInWindow = 0;
}

/**
 * Report an error using allow-listed metadata only. Error messages, stacks,
 * drawing values, filenames, routes, and browser URLs are never transmitted.
 */
export async function reportClientError(
  backendUrl: string,
  error: unknown,
  context: ClientDiagnosticContext
): Promise<string> {
  const relatedRequestId = normalizeRequestId(requestIdFromError(error));
  const referenceId = relatedRequestId || createRequestId();
  if (!allowDelivery(Date.now())) return referenceId;

  const payload = {
    kind:
      error instanceof ApiRequestError ? "api_error" : context.kind,
    operation: safeToken(context.operation, "frontend.unknown"),
    error_type: errorType(error),
    ...(error instanceof ApiRequestError && error.status
      ? { http_status: error.status }
      : {}),
  };

  try {
    await fetch(`${backendUrl.replace(/\/+$/, "")}/diagnostics/client-events`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Request-ID": referenceId,
      },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  } catch {
    // Diagnostics must never cause another application failure.
  }
  return referenceId;
}
