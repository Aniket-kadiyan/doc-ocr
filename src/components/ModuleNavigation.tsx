"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { useAuthentication } from "@/components/AuthProvider";

interface ModuleCardProps {
    href: string;
    eyebrow: string;
    title: string;
    description: string;
    action: string;
    icon: ReactNode;
    iconClassName: string;
}

function ArrowIcon() {
    return (
        <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            fill="none"
            className="h-5 w-5"
        >
            <path
                d="M5 12h14M14 7l5 5-5 5"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

function DrawingIcon() {
    return (
        <svg
            aria-hidden="true"
            viewBox="0 0 32 32"
            fill="none"
            className="h-8 w-8"
        >
            <path
                d="M6 4.5h14l6 6V27.5H6V4.5Z"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinejoin="round"
            />
            <path
                d="M20 4.5v6h6M10 16h12M10 20h8M10 12h5"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />
            <circle cx="23.5" cy="23" r="3.5" fill="currentColor" />
            <path
                d="M23.5 21.2v3.6M21.7 23h3.6"
                stroke="white"
                strokeWidth="1.3"
                strokeLinecap="round"
            />
        </svg>
    );
}

function ChecksheetIcon() {
    return (
        <svg
            aria-hidden="true"
            viewBox="0 0 32 32"
            fill="none"
            className="h-8 w-8"
        >
            <rect
                x="6"
                y="5"
                width="20"
                height="23"
                rx="2"
                stroke="currentColor"
                strokeWidth="1.7"
            />
            <path
                d="M11 11h10M11 16h10M11 21h6"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />
            <path
                d="m19.5 22.5 2 2 4-4"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

function AdministrationIcon() {
    return (
        <svg
            aria-hidden="true"
            viewBox="0 0 32 32"
            fill="none"
            className="h-8 w-8"
        >
            <circle
                cx="12"
                cy="11"
                r="4"
                stroke="currentColor"
                strokeWidth="1.7"
            />
            <path
                d="M5.5 24c.6-4.2 2.8-6.5 6.5-6.5 2.2 0 3.9.8 5 2.3"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />
            <path
                d="M23 17.5 24.1 19l1.9.1.6 1.8-1.4 1.2.4 1.9-1.7.9-1.5-1.1-1.7.8-1.4-1.2.6-1.8-1.1-1.5 1.1-1.5 1.9.2L23 17.5Z"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinejoin="round"
            />
            <circle
                cx="22.7"
                cy="21.4"
                r="1.7"
                fill="currentColor"
            />
        </svg>
    );
}

function ModuleCard({
    href,
    eyebrow,
    title,
    description,
    action,
    icon,
    iconClassName,
}: ModuleCardProps) {
    return (
        <Link
            href={href}
            className="group flex min-h-[290px] flex-col rounded-[22px] border border-[#dce4ee] bg-white p-7 shadow-[0_12px_40px_rgba(32,55,85,0.07)] transition duration-200 hover:-translate-y-1 hover:border-[#b8cbe3] hover:shadow-[0_18px_50px_rgba(32,55,85,0.13)] focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[#cfe1f7]"
        >
            <div
                className={`flex h-14 w-14 items-center justify-center rounded-2xl ${iconClassName}`}
            >
                {icon}
            </div>

            <p className="mt-7 text-xs font-bold uppercase tracking-[0.18em] text-[#6f7d8e]">
                {eyebrow}
            </p>

            <h2 className="mt-2 text-[25px] font-semibold tracking-[-0.02em] text-[#182231]">
                {title}
            </h2>

            <p className="mt-3 max-w-sm text-[15px] leading-6 text-[#647183]">
                {description}
            </p>

            <div className="mt-auto flex items-center justify-between pt-8 text-[15px] font-semibold text-[#2f6fc4]">
                <span>{action}</span>

                <span className="transition-transform duration-200 group-hover:translate-x-1">
                    <ArrowIcon />
                </span>
            </div>
        </Link>
    );
}

export function ModuleNavigation() {
    const { user } = useAuthentication();
    const isAdministrator = user.roles.includes("admin");

    return (
        <main className="qtracx-module-grid flex min-h-full flex-col bg-[#f4f8fc] text-[#182231]">
            <section className="mx-auto w-full max-w-[1320px] flex-1 px-6 pb-16 pt-14 sm:px-10 lg:px-14 lg:pt-20">
                <div className="max-w-3xl">
                    <p className="text-xs font-bold uppercase tracking-[0.22em] text-[#3978c9]">
                        QTraX workspace
                    </p>

                    <h1 className="mt-4 text-[38px] font-semibold leading-tight tracking-[-0.035em] text-[#172131] sm:text-[48px]">
                        Choose a module
                    </h1>

                    <p className="mt-4 max-w-2xl text-[17px] leading-7 text-[#657286]">
                        Welcome, {user.display_name}. Select the workspace you
                        want to open.
                    </p>
                </div>

                <div
                    className={`mt-12 grid gap-6 ${isAdministrator
                            ? "lg:grid-cols-3"
                            : "mx-auto max-w-[900px] md:grid-cols-2"
                        }`}
                >
                    <ModuleCard
                        href="/drawing"
                        eyebrow="Drawing intelligence"
                        title="Drawing Workspace"
                        description="Open engineering drawings, detect inspection characteristics, review values and create ballooned outputs."
                        action="Open workspace"
                        icon={<DrawingIcon />}
                        iconClassName="bg-[#eaf3ff] text-[#3978c9]"
                    />

                    <ModuleCard
                        href="/checksheets"
                        eyebrow="Quality inspection"
                        title="Digital Checksheets"
                        description="Open saved checksheets, continue draft inspections and review completed inspection history."
                        action="Open checksheets"
                        icon={<ChecksheetIcon />}
                        iconClassName="bg-[#eaf8f1] text-[#27845d]"
                    />

                    {isAdministrator && (
                        <ModuleCard
                            href="/user-management"
                            eyebrow="Administration"
                            title="User Management"
                            description="Manage existing user profiles, passwords, roles and application permissions."
                            action="Manage users"
                            icon={<AdministrationIcon />}
                            iconClassName="bg-[#f3edff] text-[#7752b7]"
                        />
                    )}
                </div>
            </section>

            <footer className="border-t border-[#dce4ee] bg-white/75">
                <div className="mx-auto flex max-w-[1320px] flex-wrap items-center justify-between gap-3 px-6 py-5 text-xs text-[#7a8798] sm:px-10 lg:px-14">
                    <span>QTraX — Drawing to Quality Intelligence</span>
                    <span>Secure access to your workspace</span>
                </div>
            </footer>
        </main>
    );
}