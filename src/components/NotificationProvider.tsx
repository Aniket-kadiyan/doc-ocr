"use client";

import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useState,
    type ReactNode,
} from "react";

type NotificationTone = "error" | "warning" | "success" | "info";

interface NotificationAction {
    label: string;
    onClick: () => void;
}

interface NotificationOptions {
    id?: string;
    title?: string;
    durationMs?: number | null;
    action?: NotificationAction;
}

interface NotificationRecord {
    id: string;
    tone: NotificationTone;
    title: string;
    message: string;
    durationMs: number | null;
    action?: NotificationAction;
    createdAt: number;
}

interface NotificationContextValue {
    notifyError: (
        message: string,
        options?: NotificationOptions
    ) => string;
    notifyWarning: (
        message: string,
        options?: NotificationOptions
    ) => string;
    notifySuccess: (
        message: string,
        options?: NotificationOptions
    ) => string;
    notifyInfo: (
        message: string,
        options?: NotificationOptions
    ) => string;
    dismissNotification: (id: string) => void;
}

const NotificationContext =
    createContext<NotificationContextValue | null>(null);

let notificationSequence = 0;

const defaultTitle: Record<NotificationTone, string> = {
    error: "Error",
    warning: "Attention required",
    success: "Completed",
    info: "Information",
};

const defaultDuration: Record<
    NotificationTone,
    number | null
> = {
    error: null,
    warning: 10_000,
    success: 5_000,
    info: 6_000,
};

const toneClasses: Record<NotificationTone, string> = {
    error: "border-red-200 bg-white text-red-700",
    warning: "border-amber-200 bg-white text-amber-800",
    success: "border-emerald-200 bg-white text-emerald-700",
    info: "border-blue-200 bg-white text-blue-700",
};

const indicatorClasses: Record<NotificationTone, string> = {
    error: "bg-red-500",
    warning: "bg-amber-500",
    success: "bg-emerald-500",
    info: "bg-blue-500",
};

export function useNotifications(): NotificationContextValue {
    const value = useContext(NotificationContext);

    if (!value) {
        throw new Error("Notification context is unavailable");
    }

    return value;
}

function NotificationItem({
    notification,
    onDismiss,
}: {
    notification: NotificationRecord;
    onDismiss: (id: string) => void;
}) {
    useEffect(() => {
        if (
            notification.durationMs === null ||
            notification.durationMs <= 0
        ) {
            return;
        }

        const timeout = window.setTimeout(() => {
            onDismiss(notification.id);
        }, notification.durationMs);

        return () => window.clearTimeout(timeout);
    }, [
        notification.createdAt,
        notification.durationMs,
        notification.id,
        onDismiss,
    ]);

    return (
        <article
            role={notification.tone === "error" ? "alert" : "status"}
            className={`pointer-events-auto w-full overflow-hidden rounded-xl border shadow-[0_14px_38px_rgba(15,23,42,0.18)] ${toneClasses[notification.tone]}`}
        >
            <div className="flex items-start gap-3 p-4">
                <span
                    className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${indicatorClasses[notification.tone]}`}
                />

                <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-slate-900">
                        {notification.title}
                    </p>

                    <p className="mt-1 whitespace-pre-line text-sm leading-5 text-slate-600">
                        {notification.message}
                    </p>

                    {notification.action && (
                        <button
                            type="button"
                            onClick={() => {
                                notification.action?.onClick();
                                onDismiss(notification.id);
                            }}
                            className="mt-3 rounded-lg border border-current px-3 py-1.5 text-xs font-semibold hover:bg-slate-50"
                        >
                            {notification.action.label}
                        </button>
                    )}
                </div>

                <button
                    type="button"
                    onClick={() => onDismiss(notification.id)}
                    aria-label="Dismiss notification"
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-lg leading-none text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                >
                    ×
                </button>
            </div>
        </article>
    );
}

export function NotificationProvider({
    children,
}: {
    children: ReactNode;
}) {
    const [notifications, setNotifications] = useState<
        NotificationRecord[]
    >([]);

    const dismissNotification = useCallback((id: string) => {
        setNotifications((current) =>
            current.filter((notification) => notification.id !== id)
        );
    }, []);

    const notify = useCallback(
        (
            tone: NotificationTone,
            message: string,
            options: NotificationOptions = {}
        ) => {
            const id =
                options.id ??
                `notification-${Date.now()}-${++notificationSequence}`;

            const next: NotificationRecord = {
                id,
                tone,
                title: options.title ?? defaultTitle[tone],
                message,
                durationMs:
                    options.durationMs === undefined
                        ? defaultDuration[tone]
                        : options.durationMs,
                action: options.action,
                createdAt: Date.now(),
            };

            setNotifications((current) => {
                const withoutDuplicate = current.filter(
                    (notification) => notification.id !== id
                );

                return [...withoutDuplicate, next].slice(-5);
            });

            return id;
        },
        []
    );

    const value = useMemo<NotificationContextValue>(
        () => ({
            notifyError: (message, options) =>
                notify("error", message, options),
            notifyWarning: (message, options) =>
                notify("warning", message, options),
            notifySuccess: (message, options) =>
                notify("success", message, options),
            notifyInfo: (message, options) =>
                notify("info", message, options),
            dismissNotification,
        }),
        [dismissNotification, notify]
    );

    return (
        <NotificationContext.Provider value={value}>
            {children}

            <div
                aria-live="polite"
                aria-relevant="additions text"
                className="pointer-events-none fixed bottom-5 left-1/2 z-[120] flex w-[min(440px,calc(100vw-32px))] -translate-x-1/2 flex-col-reverse gap-3"
            >
                {notifications.map((notification) => (
                    <NotificationItem
                        key={notification.id}
                        notification={notification}
                        onDismiss={dismissNotification}
                    />
                ))}
            </div>
        </NotificationContext.Provider>
    );
}