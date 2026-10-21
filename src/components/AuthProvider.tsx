"use client";

import {
  createContext,
  FormEvent,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  AuthenticationApiError,
  AuthenticatedUser,
  loginUser,
  logoutUser,
  readAuthenticatedUser,
} from "@/lib/authClient";
import { AUTHENTICATION_REQUIRED_EVENT } from "@/lib/requestDiagnostics";

type AuthenticationState =
  | "loading"
  | "authenticated"
  | "unauthenticated"
  | "unavailable";

interface AuthenticationContextValue {
  user: AuthenticatedUser;
  logout: () => Promise<void>;
}

const AuthenticationContext = createContext<AuthenticationContextValue | null>(null);

export function useAuthentication(): AuthenticationContextValue {
  const value = useContext(AuthenticationContext);
  if (!value) throw new Error("Authentication context is unavailable");
  return value;
}

function LoadingScreen() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-100 p-6">
      <div className="rounded-xl border border-slate-200 bg-white px-8 py-7 text-center shadow-sm">
        <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-slate-200 border-t-blue-600" />
        <p className="text-sm font-medium text-slate-700">Checking your QTracX session…</p>
      </div>
    </main>
  );
}

interface LoginScreenProps {
  serviceUnavailable: boolean;
  onLogin: (username: string, password: string) => Promise<void>;
  onRetry: () => Promise<void>;
}

function LoginScreen({ serviceUnavailable, onLogin, onRetry }: LoginScreenProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      await onLogin(username, password);
    } catch (caught) {
      if (caught instanceof AuthenticationApiError && caught.status === 401) {
        setError("Invalid username or password.");
      } else {
        setError("QTracX could not contact the authentication service.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-100 p-6">
      <section className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-8 shadow-xl shadow-slate-200/60">
        <div className="mb-8">
          <p className="text-sm font-semibold uppercase tracking-[0.22em] text-blue-600">
            QTracX
          </p>
          <h1 className="mt-2 text-2xl font-semibold text-slate-900">Sign in</h1>
          <p className="mt-2 text-sm text-slate-500">
            Use your QTracX account to access drawings and checksheets.
          </p>
        </div>

        {serviceUnavailable && (
          <div className="mb-5 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
            The authentication service is currently unavailable.
            <button
              type="button"
              onClick={() => void onRetry()}
              className="ml-2 font-semibold underline underline-offset-2"
            >
              Retry
            </button>
          </div>
        )}

        <form onSubmit={submit} className="space-y-5">
          <label className="block text-sm font-medium text-slate-700">
            Username
            <input
              autoFocus
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
              maxLength={64}
              className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5 text-slate-900 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </label>
          <label className="block text-sm font-medium text-slate-700">
            Password
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              maxLength={128}
              className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5 text-slate-900 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </label>

          {error && (
            <p role="alert" className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={submitting || !username.trim() || !password}
            className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}

function SessionControl() {
  const { user, logout } = useAuthentication();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(false);

  const signOut = async () => {
    setBusy(true);
    setError(false);
    try {
      await logout();
    } catch {
      setError(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-[100] flex items-center gap-3 rounded-xl border border-slate-200 bg-white/95 px-3 py-2 shadow-lg backdrop-blur">
      <div className="min-w-0 text-right">
        <p className="max-w-48 truncate text-xs font-semibold text-slate-800">
          {user.display_name}
        </p>
        <p className={`text-[11px] ${error ? "text-red-600" : "text-slate-500"}`}>
          {error ? "Sign out failed" : user.username}
        </p>
      </div>
      <button
        type="button"
        onClick={() => void signOut()}
        disabled={busy}
        className="rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
      >
        {busy ? "Signing out…" : "Sign out"}
      </button>
    </div>
  );
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthenticationState>("loading");
  const [user, setUser] = useState<AuthenticatedUser | null>(null);

  const restore = useCallback(async () => {
    setState("loading");
    try {
      const restored = await readAuthenticatedUser();
      setUser(restored);
      setState(restored ? "authenticated" : "unauthenticated");
    } catch {
      setUser(null);
      setState("unavailable");
    }
  }, []);

  useEffect(() => {
    void restore();
    const expire = () => {
      setUser(null);
      setState("unauthenticated");
    };
    window.addEventListener(AUTHENTICATION_REQUIRED_EVENT, expire);
    return () => window.removeEventListener(AUTHENTICATION_REQUIRED_EVENT, expire);
  }, [restore]);

  const login = useCallback(async (username: string, password: string) => {
    const authenticated = await loginUser(username, password);
    setUser(authenticated);
    setState("authenticated");
  }, []);

  const logout = useCallback(async () => {
    await logoutUser();
    setUser(null);
    setState("unauthenticated");
  }, []);

  const context = useMemo(
    () => (user ? { user, logout } : null),
    [logout, user]
  );

  if (state === "loading") return <LoadingScreen />;
  if (state !== "authenticated" || !context) {
    return (
      <LoginScreen
        serviceUnavailable={state === "unavailable"}
        onLogin={login}
        onRetry={restore}
      />
    );
  }

  return (
    <AuthenticationContext.Provider value={context}>
      {children}
      <SessionControl />
    </AuthenticationContext.Provider>
  );
}
