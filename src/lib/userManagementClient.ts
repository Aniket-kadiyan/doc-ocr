import { AuthenticationApiError, AuthenticatedUser } from "@/lib/authClient";
import { getOcrApiUrl } from "@/lib/paddleOcrClient";
import { fetchWithRequestId } from "@/lib/requestDiagnostics";

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

async function requireSuccess(
  response: Response,
  fallback: string
): Promise<Response> {
  if (!response.ok) {
    throw new AuthenticationApiError(
      await responseDetail(response, fallback),
      response.status
    );
  }
  return response;
}

export async function listManagedUsers(): Promise<AuthenticatedUser[]> {
  const response = await fetchWithRequestId(
    `${getOcrApiUrl()}/auth/admin/users`,
    { method: "GET", cache: "no-store" },
    { operation: "auth.admin.users.list" }
  );
  await requireSuccess(response, "Could not load QTracX users");
  return (await response.json()) as AuthenticatedUser[];
}

export async function updateManagedUserProfile(
  userId: string,
  profile: { username: string; display_name: string }
): Promise<AuthenticatedUser> {
  const response = await fetchWithRequestId(
    `${getOcrApiUrl()}/auth/admin/users/${encodeURIComponent(userId)}`,
    {
      method: "PATCH",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    },
    { operation: "auth.admin.users.update-profile" }
  );
  await requireSuccess(response, "Could not update the user profile");
  return (await response.json()) as AuthenticatedUser;
}

export async function updateManagedUserPassword(
  userId: string,
  password: string
): Promise<AuthenticatedUser> {
  const response = await fetchWithRequestId(
    `${getOcrApiUrl()}/auth/admin/users/${encodeURIComponent(userId)}/password`,
    {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    },
    { operation: "auth.admin.users.update-password" }
  );
  await requireSuccess(response, "Could not update the user password");
  return (await response.json()) as AuthenticatedUser;
}

export async function updateManagedUserAccess(
  userId: string,
  access: { is_active: boolean; administrator: boolean }
): Promise<AuthenticatedUser> {
  const response = await fetchWithRequestId(
    `${getOcrApiUrl()}/auth/admin/users/${encodeURIComponent(userId)}/access`,
    {
      method: "PATCH",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(access),
    },
    { operation: "auth.admin.users.update-access" }
  );
  await requireSuccess(response, "Could not update user access");
  return (await response.json()) as AuthenticatedUser;
}
