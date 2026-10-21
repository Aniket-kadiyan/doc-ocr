import { getOcrApiUrl } from "@/lib/paddleOcrClient";
import { fetchWithRequestId } from "@/lib/requestDiagnostics";

export interface AuthenticatedUser {
  id: string;
  username: string;
  display_name: string;
  is_active: boolean;
  roles: string[];
  created_at: string;
  updated_at: string;
}

interface LoginResponse {
  user: AuthenticatedUser;
  expires_at: string;
}

export class AuthenticationApiError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "AuthenticationApiError";
    this.status = status;
  }
}

async function responseDetail(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return typeof payload.detail === "string" && payload.detail.trim()
      ? payload.detail
      : fallback;
  } catch {
    return fallback;
  }
}

export async function readAuthenticatedUser(): Promise<AuthenticatedUser | null> {
  const response = await fetchWithRequestId(
    `${getOcrApiUrl()}/auth/me`,
    { method: "GET", cache: "no-store" },
    { operation: "auth.me", notifyAuthenticationRequired: false }
  );
  if (response.status === 401) return null;
  if (!response.ok) {
    throw new AuthenticationApiError(
      await responseDetail(response, "Authentication service unavailable"),
      response.status
    );
  }
  return (await response.json()) as AuthenticatedUser;
}

export async function loginUser(
  username: string,
  password: string
): Promise<AuthenticatedUser> {
  const response = await fetchWithRequestId(
    `${getOcrApiUrl()}/auth/login`,
    {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    },
    { operation: "auth.login", notifyAuthenticationRequired: false }
  );
  if (!response.ok) {
    throw new AuthenticationApiError(
      await responseDetail(response, "Could not sign in"),
      response.status
    );
  }
  return ((await response.json()) as LoginResponse).user;
}

export async function logoutUser(): Promise<void> {
  const response = await fetchWithRequestId(
    `${getOcrApiUrl()}/auth/logout`,
    { method: "POST", cache: "no-store" },
    { operation: "auth.logout", notifyAuthenticationRequired: false }
  );
  if (response.status === 401) return;
  if (!response.ok) {
    throw new AuthenticationApiError(
      await responseDetail(response, "Could not sign out"),
      response.status
    );
  }
}
