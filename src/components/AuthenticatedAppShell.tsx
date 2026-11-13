"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
    createContext,
    type ReactNode,
    useContext,
    useEffect,
    useMemo,
    useState,
} from "react";
import type { AuthenticatedUser } from "@/lib/authClient";
import { useNotifications } from "@/components/NotificationProvider";

interface AuthenticatedAppShellProps {
    user: AuthenticatedUser;
    logout: () => Promise<void>;
    children: ReactNode;
}

interface AppShellContextValue {
    documentName: string | null;
    setDocumentName: (name: string | null) => void;
}

const AppShellContext =
    createContext<AppShellContextValue | null>(null);

export function useAppShell(): AppShellContextValue {
    const value = useContext(AppShellContext);

    if (!value) {
        throw new Error("Application shell context is unavailable");
    }

    return value;
}

function routeDetails(pathname: string): {
    title: string;
    subtitle: string;
} {
    if (pathname.startsWith("/drawing")) {
        return {
            title: "Drawing Workspace",
            subtitle: "Engineering drawing intelligence",
        };
    }

    if (pathname.startsWith("/checksheets")) {
        return {
            title: "Digital Checksheets",
            subtitle: "Inspection definitions and history",
        };
    }

    if (pathname.startsWith("/user-management")) {
        return {
            title: "User Management",
            subtitle: "Administration",
        };
    }

    return {
        title: "Modules",
        subtitle: "Choose a QTraX workspace",
    };
}

function HomeIcon() {
    return (
        <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            fill="none"
            className="h-4 w-4"
        >
            <path
                d="m4 11 8-7 8 7v9H4v-9Z"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinejoin="round"
            />

            <path
                d="M9 20v-6h6v6"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinejoin="round"
            />
        </svg>
    );
}

function ChevronIcon() {
    return (
        <svg
            aria-hidden="true"
            viewBox="0 0 20 20"
            fill="none"
            className="h-4 w-4"
        >
            <path
                d="m6 8 4 4 4-4"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

function SignOutIcon() {
    return (
        <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            fill="none"
            className="h-4 w-4"
        >
            <path
                d="M14 8V5H5v14h9v-3M11 12h9M17 9l3 3-3 3"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

function initials(user: AuthenticatedUser): string {
    const names = user.display_name
        .trim()
        .split(/\s+/)
        .filter(Boolean);

    return `${names[0]?.[0] ?? user.username[0] ?? "U"}${names[1]?.[0] ?? ""
        }`.toUpperCase();
}

export function AuthenticatedAppShell({
    user,
    logout,
    children,
}: AuthenticatedAppShellProps) {
    const pathname = usePathname();
    const { notifyError } = useNotifications();

    const [documentName, setDocumentName] =
        useState<string | null>(null);
    const [signingOut, setSigningOut] = useState(false);

    const details = routeDetails(pathname);
    const isModuleNavigation = pathname === "/";
    const isDrawingWorkspace = pathname.startsWith("/drawing");
    const isAdministrator = user.roles.includes("admin");

    useEffect(() => {
        if (!isDrawingWorkspace) {
            setDocumentName(null);
        }
    }, [isDrawingWorkspace]);

    const context = useMemo(
        () => ({
            documentName,
            setDocumentName,
        }),
        [documentName]
    );

    const signOut = async () => {
        setSigningOut(true);

        try {
            await logout();
        } catch {
            notifyError("Sign out failed. Please try again.", {
                id: "sign-out-error",
                title: "Could not sign out",
            });

            setSigningOut(false);
        }
    };

    return (
        <AppShellContext.Provider value={context}>
            <div className="qtracx-app-shell flex min-h-screen flex-col bg-[#f4f8fc] text-[#182231]">
                <header className="relative z-50 flex h-[72px] shrink-0 items-center border-b border-[#dce4ee] bg-white px-4 shadow-[0_1px_8px_rgba(34,55,80,0.05)] sm:px-6 lg:px-8">
                    <div className="flex min-w-0 flex-1 items-center">
                        <Link
                            href="/"
                            aria-label="Open QTraX modules"
                            className="shrink-0 rounded-lg focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[#d7e7fa]"
                        >
                            <Image
                                src="/auth/qtracx-logo.png"
                                alt="QTraX"
                                width={419}
                                height={209}
                                priority
                                className="h-auto w-[116px]"
                            />
                        </Link>

                        <div className="mx-4 hidden h-8 w-px bg-[#dce4ee] sm:block" />

                        {!isModuleNavigation && (
                            <Link
                                href="/"
                                className="mr-4 hidden items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-[#5d6b7d] transition hover:bg-[#f1f5fa] hover:text-[#233247] focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[#d7e7fa] md:flex"
                            >
                                <HomeIcon />
                                <span>Modules</span>
                            </Link>
                        )}

                        <div className="min-w-0">
                            <p className="truncate text-[15px] font-semibold text-[#1c2838]">
                                {details.title}
                            </p>

                            <p className="truncate text-xs text-[#788597]">
                                {isDrawingWorkspace && documentName
                                    ? documentName
                                    : details.subtitle}
                            </p>
                        </div>
                    </div>

                    <details className="qtracx-user-menu relative shrink-0">
                        <summary className="flex cursor-pointer list-none items-center gap-3 rounded-xl px-2 py-1.5 transition hover:bg-[#f2f6fb] focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[#d7e7fa] sm:px-3">
                            <span className="flex h-9 w-9 items-center justify-center rounded-full bg-[#3978c9] text-xs font-bold text-white">
                                {initials(user)}
                            </span>

                            <span className="hidden min-w-0 text-left sm:block">
                                <span className="block max-w-44 truncate text-sm font-semibold text-[#253143]">
                                    {user.display_name}
                                </span>

                                <span className="block max-w-44 truncate text-xs text-[#788597]">
                                    {isAdministrator
                                        ? "Administrator"
                                        : user.username}
                                </span>
                            </span>

                            <span className="text-[#6f7d8e]">
                                <ChevronIcon />
                            </span>
                        </summary>

                        <div className="absolute right-0 top-[calc(100%+10px)] w-64 overflow-hidden rounded-2xl border border-[#dce4ee] bg-white shadow-[0_18px_50px_rgba(31,48,70,0.16)]">
                            <div className="border-b border-[#e5ebf2] px-4 py-4">
                                <p className="truncate text-sm font-semibold text-[#1d2939]">
                                    {user.display_name}
                                </p>

                                <p className="mt-1 truncate text-xs text-[#748195]">
                                    {user.username}
                                </p>

                                <div className="mt-3 flex flex-wrap gap-1.5">
                                    {user.roles.map((role) => (
                                        <span
                                            key={role}
                                            className="rounded-full bg-[#edf4fd] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.08em] text-[#326cad]"
                                        >
                                            {role}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            <div className="p-2">
                                <Link
                                    href="/"
                                    className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-[#4e5c6e] hover:bg-[#f2f6fb] hover:text-[#1d2939]"
                                >
                                    <HomeIcon />
                                    Module navigation
                                </Link>

                                {isAdministrator && (
                                    <Link
                                        href="/user-management"
                                        className="flex w-full items-center rounded-lg px-3 py-2 text-sm font-medium text-[#4e5c6e] hover:bg-[#f2f6fb] hover:text-[#1d2939]"
                                    >
                                        User management
                                    </Link>
                                )}

                                <button
                                    type="button"
                                    onClick={() => void signOut()}
                                    disabled={signingOut}
                                    className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium text-[#a53b32] hover:bg-[#fff2f0] disabled:cursor-not-allowed disabled:opacity-60"
                                >
                                    <SignOutIcon />
                                    {signingOut ? "Signing out…" : "Sign out"}
                                </button>
                            </div>
                        </div>
                    </details>
                </header>

                <div className="min-h-0 flex-1">
                    {children}
                </div>
            </div>
        </AppShellContext.Provider>
    );
}