const REQUEST_ID_HEADER = "X-Request-ID";
const SAFE_REQUEST_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const responseRequestIds = new WeakMap<object, string>();

export interface DiagnosticFetchOptions {
  operation: string;
  networkErrorMessage?: string;
}

export class ApiRequestError extends Error {
  readonly requestId: string;
  readonly status?: number;
  readonly operation: string;

  constructor(
    message: string,
    options: { requestId: string; operation: string; status?: number }
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.requestId = options.requestId;
    this.status = options.status;
    this.operation = options.operation;
  }
}

export function createRequestId(): string {
  const randomUuid = globalThis.crypto?.randomUUID;
  if (typeof randomUuid === "function") return randomUuid.call(globalThis.crypto);
  return `qtx-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 14)}`;
}

export function normalizeRequestId(value: string | null | undefined): string {
  const candidate = value?.trim() ?? "";
  return SAFE_REQUEST_ID.test(candidate) ? candidate : "";
}

export function appendRequestReference(
  message: string,
  requestId: string
): string {
  const cleanMessage = message.trim() || "The request failed.";
  return requestId
    ? `${cleanMessage} Diagnostic reference: ${requestId}`
    : cleanMessage;
}

export function responseRequestId(response: Response): string {
  const exposed = normalizeRequestId(response.headers?.get?.(REQUEST_ID_HEADER));
  return exposed || responseRequestIds.get(response) || "";
}

export function requestIdFromError(error: unknown): string {
  return error instanceof ApiRequestError ? error.requestId : "";
}

export function apiRequestError(
  response: Response,
  message: string,
  operation: string
): ApiRequestError {
  const requestId = responseRequestId(response);
  return new ApiRequestError(appendRequestReference(message, requestId), {
    requestId,
    operation,
    status: response.status,
  });
}

/** Add a safe correlation ID without changing request bodies or content types. */
export async function fetchWithRequestId(
  input: RequestInfo | URL,
  init: RequestInit = {},
  options: DiagnosticFetchOptions
): Promise<Response> {
  const headers = new Headers(
    typeof Request !== "undefined" && input instanceof Request
      ? input.headers
      : undefined
  );
  new Headers(init.headers).forEach((value, key) => headers.set(key, value));

  const supplied = normalizeRequestId(headers.get(REQUEST_ID_HEADER));
  const requestId = supplied || createRequestId();
  headers.set(REQUEST_ID_HEADER, requestId);

  try {
    const response = await fetch(input, { ...init, headers });
    responseRequestIds.set(response, responseRequestId(response) || requestId);
    return response;
  } catch {
    const message =
      options.networkErrorMessage ?? "Could not contact the QTracX backend.";
    throw new ApiRequestError(appendRequestReference(message, requestId), {
      requestId,
      operation: options.operation,
    });
  }
}
