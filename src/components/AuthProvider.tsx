"use client";

import Image from "next/image";
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

function UserIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" className="h-5 w-5">
      <circle cx="12" cy="8" r="3.25" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M5.75 19c.43-3.17 2.63-5 6.25-5s5.82 1.83 6.25 5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function LockIcon({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" className={className}>
      <rect
        x="5.5"
        y="10"
        width="13"
        height="10"
        rx="2"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path
        d="M8.5 10V7.5a3.5 3.5 0 0 1 7 0V10"
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  );
}

function EyeIcon({ hidden }: { hidden: boolean }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" className="h-5 w-5">
      <path
        d="M3.5 12s3.1-5 8.5-5 8.5 5 8.5 5-3.1 5-8.5 5-8.5-5-8.5-5Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="2.25" stroke="currentColor" strokeWidth="1.5" />
      {hidden && (
        <path
          d="M5 4.5 19 19.5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      )}
    </svg>
  );
}

function ErrorIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 20 20" fill="none" className="h-5 w-5 shrink-0">
      <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M10 5.8v5.1M10 14.1v.1"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

function BrandPanel() {
  return (
    <aside className="qtracx-auth-grid relative hidden min-h-screen overflow-hidden bg-[#f1f7fd] lg:block">
      <div className="absolute left-1/2 top-[4.8%] w-[60.2%] -translate-x-1/2">
        <Image
          src="/auth/qtracx-logo.png"
          alt="QTraX — Drawing to Quality Intelligence"
          width={419}
          height={209}
          priority
          className="h-auto w-full"
        />
      </div>

      <div className="absolute left-1/2 top-[31.8%] w-full -translate-x-1/2">
        <Image
          src="/auth/engineering-login-illustration.svg"
          alt="Annotated engineering drawing"
          width={697}
          height={440}
          priority
          className="h-auto w-full"
        />
      </div>

      <div className="absolute inset-x-8 top-[80.9%] text-center text-[#263342]">
        <p className="text-[14px] font-semibold tracking-[0.09em]">
          DRAWING TO HIGH QUALITY INTELLIGENCE
        </p>
        <p className="mt-2 text-[10px] font-medium tracking-[0.18em] text-[#647387]">
          SCHEMATIC • ILLUSTRATIVE ONLY
        </p>
      </div>

      <div className="absolute bottom-6 left-0 right-0 flex items-center justify-center gap-1.5 text-[12px] text-[#68778a]">
        <span aria-hidden="true">©</span>
        <span>2026 QTraX</span>
      </div>
    </aside>
  );
}

function MobileBrand() {
  return (
    <div className="mx-auto mb-8 w-[250px] lg:hidden">
      <Image
        src="/auth/qtracx-logo.png"
        alt="QTraX — Drawing to Quality Intelligence"
        width={419}
        height={209}
        priority
        className="h-auto w-full"
      />
    </div>
  );
}

function LoadingScreen() {
  return (
    <main className="qtracx-auth flex min-h-screen items-center justify-center bg-white p-6">
      <div className="text-center">
        <Image
          src="/auth/qtracx-logo.png"
          alt="QTraX"
          width={210}
          height={105}
          priority
          className="mx-auto h-auto w-[210px]"
        />
        <div className="mx-auto mt-5 h-7 w-7 animate-spin rounded-full border-[3px] border-[#dce5f2] border-t-[#3573c9]" />
        <p className="mt-4 text-sm font-medium text-[#5d6672]">
          Checking your QTraX session…
        </p>
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
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [usernameError, setUsernameError] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [recoveryHelp, setRecoveryHelp] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedUsername = username.trim();
    const nextUsernameError = normalizedUsername ? "" : "Enter your User ID.";
    const nextPasswordError = password ? "" : "Enter your password.";

    setUsernameError(nextUsernameError);
    setPasswordError(nextPasswordError);
    setError("");
    if (nextUsernameError || nextPasswordError) return;

    setSubmitting(true);
    try {
      await onLogin(normalizedUsername, password);
    } catch (caught) {
      if (caught instanceof AuthenticationApiError && caught.status === 401) {
        setError("Unable to sign in. Check your credentials and try again.");
      } else {
        setError("QTraX could not contact the authentication service.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="qtracx-auth min-h-screen bg-white text-black lg:grid lg:grid-cols-[48.4%_51.6%]">
      <BrandPanel />

      <section className="relative flex min-h-screen flex-col px-6 pb-10 pt-10 sm:px-10 lg:px-0 lg:pb-8 lg:pt-[21.5vh]">
        <MobileBrand />

        <div className="mx-auto w-full max-w-[501px]">
          <header>
            <h1 className="text-[40px] font-medium leading-[1.12] tracking-[-0.025em] text-black sm:text-[48px]">
              Welcome back
            </h1>
            <p className="mt-3 text-[18px] font-normal leading-6 text-[#727272] sm:text-[20px]">
              Sign in to continue to QTraX
            </p>
          </header>

          {serviceUnavailable && (
            <div
              role="alert"
              className="mt-7 flex items-start gap-2.5 rounded-lg border border-[#e9b7ae] bg-[#fff5f3] px-4 py-3 text-[14px] text-[#9d3024]"
            >
              <ErrorIcon />
              <p className="leading-5">
                The authentication service is currently unavailable.{" "}
                <button
                  type="button"
                  onClick={() => void onRetry()}
                  className="font-semibold underline underline-offset-2 hover:text-[#772018]"
                >
                  Retry
                </button>
              </p>
            </div>
          )}

          <form onSubmit={submit} noValidate className="mt-8">
            <div>
              <label
                htmlFor="qtracx-user-id"
                className="block text-[18px] font-medium leading-6 text-black sm:text-[20px]"
              >
                User ID
              </label>
              <div className="relative mt-4">
                <span className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2 text-[#575d65]">
                  <UserIcon />
                </span>
                <input
                  id="qtracx-user-id"
                  autoFocus
                  autoComplete="username"
                  value={username}
                  onChange={(event) => {
                    setUsername(event.target.value);
                    if (usernameError) setUsernameError("");
                  }}
                  aria-invalid={Boolean(usernameError)}
                  aria-describedby={usernameError ? "qtracx-user-id-error" : undefined}
                  maxLength={64}
                  placeholder="Enter your User ID"
                  className={`h-[50px] w-full rounded-lg border bg-white pl-[54px] pr-4 text-[16px] text-black outline-none transition placeholder:text-[#8b8b8b] focus:ring-[3px] ${
                    usernameError
                      ? "border-[#c64a3d] focus:border-[#c64a3d] focus:ring-[#f6d8d4]"
                      : "border-[#b6b6b6] focus:border-[#3573c9] focus:ring-[#d9e8fb]"
                  }`}
                />
              </div>
              {usernameError && (
                <p
                  id="qtracx-user-id-error"
                  role="alert"
                  className="mt-2 flex items-center gap-2 text-[14px] text-[#b53b30]"
                >
                  <ErrorIcon />
                  {usernameError}
                </p>
              )}
            </div>

            <div className="mt-8">
              <label
                htmlFor="qtracx-password"
                className="block text-[18px] font-medium leading-6 text-black sm:text-[20px]"
              >
                Password
              </label>
              <div className="relative mt-4">
                <span className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2 text-[#575d65]">
                  <LockIcon />
                </span>
                <input
                  id="qtracx-password"
                  type={passwordVisible ? "text" : "password"}
                  autoComplete="current-password"
                  value={password}
                  onChange={(event) => {
                    setPassword(event.target.value);
                    if (passwordError) setPasswordError("");
                  }}
                  aria-invalid={Boolean(passwordError)}
                  aria-describedby={passwordError ? "qtracx-password-error" : undefined}
                  maxLength={128}
                  placeholder="Enter your password"
                  className={`h-[50px] w-full rounded-lg border bg-white pl-[54px] pr-[54px] text-[16px] text-black outline-none transition placeholder:text-[#8b8b8b] focus:ring-[3px] ${
                    passwordError
                      ? "border-[#c64a3d] focus:border-[#c64a3d] focus:ring-[#f6d8d4]"
                      : "border-[#b6b6b6] focus:border-[#3573c9] focus:ring-[#d9e8fb]"
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setPasswordVisible((visible) => !visible)}
                  aria-label={passwordVisible ? "Hide password" : "Show password"}
                  className="absolute right-5 top-1/2 -translate-y-1/2 rounded text-[#575d65] transition hover:text-black focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3573c9]"
                >
                  <EyeIcon hidden={passwordVisible} />
                </button>
              </div>
              {passwordError && (
                <p
                  id="qtracx-password-error"
                  role="alert"
                  className="mt-2 flex items-center gap-2 text-[14px] text-[#b53b30]"
                >
                  <ErrorIcon />
                  {passwordError}
                </p>
              )}
            </div>

            {error && (
              <div
                role="alert"
                className="mt-6 flex items-start gap-2.5 rounded-lg border border-[#e9b7ae] bg-[#fff5f3] px-4 py-3 text-[14px] text-[#9d3024]"
              >
                <ErrorIcon />
                <p className="leading-5">{error}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={submitting || serviceUnavailable}
              className="mt-8 flex h-[50px] w-full items-center justify-center gap-2 rounded-lg bg-[#8ba9df] px-4 text-[16px] font-medium text-white transition hover:bg-[#7899d5] focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[#d9e8fb] disabled:cursor-not-allowed disabled:opacity-70"
            >
              {submitting && (
                <span
                  aria-hidden="true"
                  className="h-4 w-4 animate-spin rounded-full border-2 border-white/45 border-t-white"
                />
              )}
              {submitting ? "Signing in" : "Sign in"}
            </button>

            <div className="mt-8 text-center">
              <button
                type="button"
                onClick={() => setRecoveryHelp(true)}
                className="text-[16px] font-medium text-[#3978d3] underline-offset-4 hover:underline focus-visible:rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3978d3]"
              >
                Forgot password?
              </button>
            </div>
          </form>

          <div
            className="mt-8 border-t border-[#d6d6d6] pt-7 text-center text-[14px] leading-5 text-[#666d76]"
            aria-live="polite"
          >
            {recoveryHelp
              ? "To reset your password, contact your system administrator."
              : "Having trouble signing in? Contact your system administrator."}
          </div>
        </div>

        <div className="mx-auto mt-auto flex w-full max-w-[501px] items-center justify-center gap-2 pt-10 text-[13px] text-[#6d7680]">
          <LockIcon className="h-4 w-4" />
          <span>Secure access to your workspace</span>
        </div>
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

  const context = useMemo(() => (user ? { user, logout } : null), [logout, user]);

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
