"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { AuthenticationApiError, AuthenticatedUser } from "@/lib/authClient";
import {
  listManagedUsers,
  updateManagedUserAccess,
  updateManagedUserPassword,
  updateManagedUserProfile,
} from "@/lib/userManagementClient";
import { useAuthentication } from "@/components/AuthProvider";

function initials(user: AuthenticatedUser): string {
  const parts = user.display_name.trim().split(/\s+/).filter(Boolean);
  return (parts[0]?.[0] ?? user.username[0] ?? "U")
    .concat(parts[1]?.[0] ?? "")
    .toUpperCase();
}

function friendlyDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "Unknown"
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof AuthenticationApiError ? error.message : fallback;
}

function AccessDenied() {
  return (
    <main className="min-h-screen bg-slate-100 px-6 py-16 text-slate-900">
      <section className="mx-auto max-w-xl rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-blue-700">
          QTraX administration
        </p>
        <h1 className="mt-3 text-3xl font-semibold">Administrator access required</h1>
        <p className="mt-4 leading-7 text-slate-600">
          Your account can use QTraX but cannot manage user credentials. Sign in
          with an administrator account or contact the system administrator.
        </p>
        <Link
          href="/"
          className="mt-7 inline-flex rounded-lg bg-blue-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-800"
        >
          Return to QTraX
        </Link>
      </section>
    </main>
  );
}

export function UserManagementConsole() {
  const { user: administrator } = useAuthentication();
  const [users, setUsers] = useState<AuthenticatedUser[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [query, setQuery] = useState("");

  const isAdministrator = administrator.roles.includes("admin");

  const load = useCallback(async () => {
    if (!isAdministrator) return;
    setLoading(true);
    setLoadError("");
    try {
      const loaded = await listManagedUsers();
      setUsers(loaded);
      setSelectedId((current) =>
        loaded.some((candidate) => candidate.id === current)
          ? current
          : loaded[0]?.id ?? ""
      );
    } catch (error) {
      setLoadError(errorMessage(error, "Could not load QTraX users."));
    } finally {
      setLoading(false);
    }
  }, [isAdministrator]);

  useEffect(() => {
    void load();
  }, [load]);

  const selected = users.find((candidate) => candidate.id === selectedId) ?? null;
  const visibleUsers = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return users;
    return users.filter(
      (candidate) =>
        candidate.username.toLocaleLowerCase().includes(needle) ||
        candidate.display_name.toLocaleLowerCase().includes(needle)
    );
  }, [query, users]);

  const replaceUser = (updated: AuthenticatedUser) => {
    setUsers((current) =>
      current
        .map((candidate) => (candidate.id === updated.id ? updated : candidate))
        .sort((left, right) => left.username.localeCompare(right.username))
    );
  };

  if (!isAdministrator) return <AccessDenied />;

  return (
    <main className="min-h-screen bg-[#f4f7fb] text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-6 px-6 py-5">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-blue-700">
              QTraX administration
            </p>
            <h1 className="mt-1 text-2xl font-semibold">User management</h1>
          </div>
          <Link
            href="/"
            className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          >
            Open QTraX
          </Link>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-6 px-6 py-7 lg:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-200 p-4">
            <label htmlFor="user-search" className="sr-only">
              Search users
            </label>
            <input
              id="user-search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search username or display name"
              className="h-11 w-full rounded-lg border border-slate-300 px-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100"
            />
            <p className="mt-3 text-xs text-slate-500">
              {users.length} existing account{users.length === 1 ? "" : "s"}. New
              accounts are created only by the system administrator.
            </p>
          </div>

          <div className="max-h-[calc(100vh-250px)] min-h-72 overflow-y-auto p-2">
            {loading && <p className="p-4 text-sm text-slate-500">Loading users…</p>}
            {loadError && (
              <div className="m-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                <p>{loadError}</p>
                <button
                  type="button"
                  onClick={() => void load()}
                  className="mt-2 font-semibold underline underline-offset-2"
                >
                  Retry
                </button>
              </div>
            )}
            {!loading && !loadError && visibleUsers.length === 0 && (
              <p className="p-4 text-sm text-slate-500">No matching users.</p>
            )}
            {visibleUsers.map((candidate) => (
              <button
                type="button"
                key={candidate.id}
                onClick={() => setSelectedId(candidate.id)}
                className={`flex w-full items-center gap-3 rounded-xl p-3 text-left transition ${
                  selectedId === candidate.id
                    ? "bg-blue-50 ring-1 ring-blue-200"
                    : "hover:bg-slate-50"
                }`}
              >
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-700 text-sm font-bold text-white">
                  {initials(candidate)}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-semibold text-slate-900">
                    {candidate.display_name}
                  </span>
                  <span className="block truncate text-xs text-slate-500">
                    {candidate.username}
                  </span>
                </span>
                {candidate.roles.includes("admin") && (
                  <span className="rounded-full bg-amber-100 px-2 py-1 text-[10px] font-bold uppercase tracking-wide text-amber-800">
                    Admin
                  </span>
                )}
              </button>
            ))}
          </div>
        </aside>

        <section>
          {selected ? (
            <UserEditor
              key={selected.id}
              user={selected}
              signedInUserId={administrator.id}
              onAccessUpdated={replaceUser}
              onProfileUpdated={replaceUser}
              onPasswordUpdated={replaceUser}
            />
          ) : (
            <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center text-slate-500 shadow-sm">
              Select an existing user to manage their account.
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function UserEditor({
  user,
  signedInUserId,
  onAccessUpdated,
  onProfileUpdated,
  onPasswordUpdated,
}: {
  user: AuthenticatedUser;
  signedInUserId: string;
  onAccessUpdated: (user: AuthenticatedUser) => void;
  onProfileUpdated: (user: AuthenticatedUser) => void;
  onPasswordUpdated: (user: AuthenticatedUser) => void;
}) {
  const [username, setUsername] = useState(user.username);
  const [displayName, setDisplayName] = useState(user.display_name);
  const [active, setActive] = useState(user.is_active);
  const [administrator, setAdministrator] = useState(user.roles.includes("admin"));
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [profileBusy, setProfileBusy] = useState(false);
  const [passwordBusy, setPasswordBusy] = useState(false);
  const [accessBusy, setAccessBusy] = useState(false);
  const [profileMessage, setProfileMessage] = useState("");
  const [profileError, setProfileError] = useState("");
  const [passwordMessage, setPasswordMessage] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [accessMessage, setAccessMessage] = useState("");
  const [accessError, setAccessError] = useState("");

  const saveProfile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setProfileMessage("");
    setProfileError("");
    if (!username.trim() || !displayName.trim()) {
      setProfileError("Username and display name are required.");
      return;
    }
    setProfileBusy(true);
    try {
      const updated = await updateManagedUserProfile(user.id, {
        username: username.trim(),
        display_name: displayName.trim(),
      });
      onProfileUpdated(updated);
      setUsername(updated.username);
      setDisplayName(updated.display_name);
      setProfileMessage("User details updated.");
    } catch (error) {
      setProfileError(errorMessage(error, "Could not update user details."));
    } finally {
      setProfileBusy(false);
    }
  };

  const savePassword = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setPasswordMessage("");
    setPasswordError("");
    if (password.length < 4) {
      setPasswordError("Password must contain at least 4 characters.");
      return;
    }
    if (password !== confirmation) {
      setPasswordError("Password confirmation does not match.");
      return;
    }
    setPasswordBusy(true);
    try {
      const updated = await updateManagedUserPassword(user.id, password);
      onPasswordUpdated(updated);
      setPassword("");
      setConfirmation("");
      setPasswordMessage(
        user.id === signedInUserId
          ? "Password updated. Your current session has been revoked; sign in again to continue."
          : "Password updated. The user's active sessions have been revoked."
      );
    } catch (error) {
      setPasswordError(errorMessage(error, "Could not update the password."));
    } finally {
      setPasswordBusy(false);
    }
  };

  const saveAccess = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAccessMessage("");
    setAccessError("");
    setAccessBusy(true);
    try {
      const updated = await updateManagedUserAccess(user.id, {
        is_active: active,
        administrator,
      });
      onAccessUpdated(updated);
      setActive(updated.is_active);
      setAdministrator(updated.roles.includes("admin"));
      setAccessMessage(
        user.id === signedInUserId &&
          (user.is_active !== updated.is_active ||
            user.roles.includes("admin") !== updated.roles.includes("admin"))
          ? "Access updated. Your current session has been revoked; sign in again to continue."
          : "Roles and permissions updated. The user's active sessions have been revoked."
      );
    } catch (error) {
      setAccessError(errorMessage(error, "Could not update user access."));
    } finally {
      setAccessBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-200 pb-5">
          <div>
            <h2 className="text-xl font-semibold">{user.display_name}</h2>
            <p className="mt-1 text-sm text-slate-500">Account ID: {user.id}</p>
          </div>
          <div className="flex gap-2">
            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold ${
                user.is_active
                  ? "bg-emerald-100 text-emerald-800"
                  : "bg-slate-200 text-slate-700"
              }`}
            >
              {user.is_active ? "Active" : "Disabled"}
            </span>
            {user.roles.includes("admin") && (
              <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-800">
                Administrator
              </span>
            )}
          </div>
        </div>

        <form onSubmit={saveProfile} className="mt-6">
          <h3 className="font-semibold">Account details</h3>
          <div className="mt-4 grid gap-5 md:grid-cols-2">
            <label className="text-sm font-medium text-slate-700">
              Username
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                maxLength={64}
                autoComplete="off"
                className="mt-2 h-11 w-full rounded-lg border border-slate-300 px-3 font-normal text-slate-900 outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100"
              />
              <span className="mt-1.5 block text-xs font-normal text-slate-500">
                3–64 letters, numbers, periods, underscores, hyphens or @.
              </span>
            </label>
            <label className="text-sm font-medium text-slate-700">
              Display name
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                maxLength={120}
                autoComplete="off"
                className="mt-2 h-11 w-full rounded-lg border border-slate-300 px-3 font-normal text-slate-900 outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100"
              />
            </label>
          </div>
          <div className="mt-5 flex flex-wrap items-center gap-4">
            <button
              type="submit"
              disabled={profileBusy}
              className="rounded-lg bg-blue-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-800 disabled:opacity-60"
            >
              {profileBusy ? "Saving…" : "Save user details"}
            </button>
            {profileMessage && <p className="text-sm text-emerald-700">{profileMessage}</p>}
            {profileError && <p role="alert" className="text-sm text-red-700">{profileError}</p>}
          </div>
        </form>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <form onSubmit={saveAccess}>
          <h3 className="font-semibold">Roles and permissions</h3>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Permissions are assigned through the roles currently supported by
            QTraX. Changing access or administrator status signs the affected
            account out of all current sessions.
          </p>

          <div className="mt-5 space-y-3">
            <label className="flex cursor-pointer items-start justify-between gap-5 rounded-xl border border-slate-200 p-4 hover:bg-slate-50">
              <span>
                <span className="block text-sm font-semibold text-slate-900">
                  Application access
                </span>
                <span className="mt-1 block text-sm leading-5 text-slate-600">
                  Allows this account to sign in and use the QTraX application.
                </span>
              </span>
              <input
                type="checkbox"
                checked={active}
                onChange={(event) => setActive(event.target.checked)}
                className="mt-1 h-5 w-5 shrink-0 accent-blue-700"
              />
            </label>

            <div className="rounded-xl border border-slate-200 p-4">
              <div className="flex items-start justify-between gap-5">
                <span>
                  <span className="block text-sm font-semibold text-slate-900">
                    Standard user
                  </span>
                  <span className="mt-1 block text-sm leading-5 text-slate-600">
                    Drawing, review, checksheet and export workflows.
                  </span>
                </span>
                <input
                  type="checkbox"
                  checked
                  disabled
                  aria-label="Standard user role is required"
                  className="mt-1 h-5 w-5 shrink-0 accent-blue-700"
                />
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Required for every QTraX account and cannot be removed here.
              </p>
            </div>

            <label className="flex cursor-pointer items-start justify-between gap-5 rounded-xl border border-slate-200 p-4 hover:bg-slate-50">
              <span>
                <span className="block text-sm font-semibold text-slate-900">
                  Administrator
                </span>
                <span className="mt-1 block text-sm leading-5 text-slate-600">
                  Can open this management console and change existing users,
                  credentials, roles and application access.
                </span>
              </span>
              <input
                type="checkbox"
                checked={administrator}
                onChange={(event) => setAdministrator(event.target.checked)}
                className="mt-1 h-5 w-5 shrink-0 accent-blue-700"
              />
            </label>
          </div>

          <div className="mt-5 rounded-lg bg-blue-50 px-4 py-3 text-sm text-blue-900">
            Effective access: {active ? "Application enabled" : "Sign-in disabled"}
            {administrator ? " · User administration enabled" : " · Standard workflows only"}
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-4">
            <button
              type="submit"
              disabled={accessBusy}
              className="rounded-lg bg-blue-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-800 disabled:opacity-60"
            >
              {accessBusy ? "Saving…" : "Save roles and permissions"}
            </button>
            {accessMessage && <p className="text-sm text-emerald-700">{accessMessage}</p>}
            {accessError && <p role="alert" className="text-sm text-red-700">{accessError}</p>}
          </div>
        </form>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <form onSubmit={savePassword}>
          <h3 className="font-semibold">Set a new password</h3>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Existing passwords cannot be viewed. Setting a new password signs this
            account out of all current sessions.
          </p>
          <div className="mt-4 grid gap-5 md:grid-cols-2">
            <label className="text-sm font-medium text-slate-700">
              New password
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={4}
                maxLength={128}
                autoComplete="new-password"
                className="mt-2 h-11 w-full rounded-lg border border-slate-300 px-3 font-normal text-slate-900 outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100"
              />
            </label>
            <label className="text-sm font-medium text-slate-700">
              Confirm new password
              <input
                type="password"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                minLength={4}
                maxLength={128}
                autoComplete="new-password"
                className="mt-2 h-11 w-full rounded-lg border border-slate-300 px-3 font-normal text-slate-900 outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100"
              />
            </label>
          </div>
          <div className="mt-5 flex flex-wrap items-center gap-4">
            <button
              type="submit"
              disabled={passwordBusy || !password || !confirmation}
              className="rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {passwordBusy ? "Updating…" : "Update password"}
            </button>
            {passwordMessage && <p className="text-sm text-emerald-700">{passwordMessage}</p>}
            {passwordError && <p role="alert" className="text-sm text-red-700">{passwordError}</p>}
          </div>
        </form>
        <div className="mt-6 border-t border-slate-200 pt-4 text-xs leading-5 text-slate-500">
          Created {friendlyDate(user.created_at)} · Last account update {friendlyDate(user.updated_at)}
        </div>
      </div>
    </div>
  );
}
