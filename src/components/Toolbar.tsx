"use client";

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { ExportPanel } from "@/components/ExportPanel";
import { WorkspaceSettingsPanel } from "@/components/WorkspaceSettingsPanel";
import {
  isScanProgressVisible,
  SCAN_PROGRESS_VISIBILITY_EVENT,
  setScanProgressVisible,
} from "@/lib/progressVisibility";
import type { PageReferenceTable } from "@/types/referencePoint";
import type { ScanCandidate } from "@/types/scanCandidate";

interface ToolbarProps {
  isSelectingScanArea: boolean;
  isScanRunning: boolean;
  hasProgressDetails: boolean;
  isDrawingValue: boolean;
  isProcessing: boolean;
  currentPage: number;
  totalPages: number;
  scale: number;
  balloonsVisible: boolean;
  okBalloonsVisible: boolean;
  reviewBalloonsVisible: boolean;
  hasOkBalloons: boolean;
  hasReviewBalloons: boolean;
  hasBalloons: boolean;
  onSelectScanSection: () => void;
  onScanWholePage: () => void;
  findingReferencePoints: boolean;
  onFindGdtSymbols: () => void;
  findingGdtSymbols: boolean;
  onToggleDrawValue: () => void;
  onToggleBalloons: () => void;
  onSetOkBalloonsVisible: (visible: boolean) => void;
  onSetReviewBalloonsVisible: (visible: boolean) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onPrevPage: () => void;
  onNextPage: () => void;
  onUpload: () => void;
  onSaveProject: () => void;
  onLoadProject: () => void;
  onRemoveDrawing: () => void;
  canSaveProject: boolean;
  referenceTables: readonly PageReferenceTable[];
  scanCandidates: readonly ScanCandidate[];
  fileName: string | null;
}

type IconName =
  | "menu"
  | "new"
  | "load"
  | "save"
  | "clear"
  | "detect"
  | "manual"
  | "export"
  | "settings"
  | "balloon"
  | "progress"
  | "previous"
  | "next";

function ActionIcon({ name }: { name: IconName }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="h-5 w-5 shrink-0"
    >
      {name === "menu" && (
        <>
          <path d="M4 6h16" />
          <path d="M4 12h16" />
          <path d="M4 18h16" />
        </>
      )}

      {name === "new" && (
        <>
          <path d="M6 3h8l4 4v14H6z" />
          <path d="M14 3v5h5" />
          <path d="M12 12v6" />
          <path d="M9 15h6" />
        </>
      )}

      {name === "load" && (
        <>
          <path d="M3 7h7l2 2h9v10H3z" />
          <path d="M12 11v6" />
          <path d="m9 14 3 3 3-3" />
        </>
      )}

      {name === "save" && (
        <>
          <path d="M5 3h12l2 2v16H5z" />
          <path d="M8 3v6h8V3" />
          <path d="M8 14h8v7H8z" />
        </>
      )}

      {name === "clear" && (
        <>
          <path d="M4 7h16" />
          <path d="M9 7V4h6v3" />
          <path d="m7 7 1 14h8l1-14" />
          <path d="M10 11v6" />
          <path d="M14 11v6" />
        </>
      )}

      {name === "detect" && (
        <>
          <path d="M4 9V4h5" />
          <path d="M15 4h5v5" />
          <path d="M20 15v5h-5" />
          <path d="M9 20H4v-5" />
          <circle cx="12" cy="12" r="3" />
        </>
      )}

      {name === "manual" && (
        <>
          <path d="m4 20 4.5-1 10-10-3.5-3.5-10 10z" />
          <path d="m13.5 6.5 3.5 3.5" />
          <path d="M4 20h5" />
        </>
      )}

      {name === "export" && (
        <>
          <path d="M12 3v12" />
          <path d="m8 11 4 4 4-4" />
          <path d="M5 19h14" />
        </>
      )}

      {name === "settings" && (
        <>
          <circle cx="12" cy="12" r="3" />
          <path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.5-2.4 1a7 7 0 0 0-1.7-1L14.5 3h-5L9 6a7 7 0 0 0-1.7 1L5 6 3 9.5 5.1 11a7 7 0 0 0 0 2L3 14.5 5 18l2.3-1a7 7 0 0 0 1.7 1l.5 3h5l.5-3a7 7 0 0 0 1.7-1l2.3 1 2-3.5-2.1-1.5a7 7 0 0 0 .1-1z" />
        </>
      )}

      {name === "balloon" && (
        <>
          <circle cx="10" cy="10" r="6" />
          <path d="m14.5 14.5 5 5" />
          <path d="M7.5 10h5" />
        </>
      )}

      {name === "progress" && (
        <>
          <path d="M4 19V5" />
          <path d="M4 19h16" />
          <path d="m7 15 4-4 3 2 5-6" />
        </>
      )}

      {name === "previous" && <path d="m15 18-6-6 6-6" />}
      {name === "next" && <path d="m9 18 6-6-6-6" />}
    </svg>
  );
}

interface DrawerActionProps {
  label: string;
  icon: IconName;
  expanded: boolean;
  disabled?: boolean;
  active?: boolean;
  danger?: boolean;
  onClick: () => void;
}

function DrawerAction({
  label,
  icon,
  expanded,
  disabled = false,
  active = false,
  danger = false,
  onClick,
}: DrawerActionProps) {
  const tone = active
    ? "bg-blue-600 text-white shadow-sm"
    : danger
      ? "text-red-600 hover:bg-red-50"
      : "text-slate-700 hover:bg-slate-100";

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={expanded ? undefined : label}
      className={`flex h-11 w-full items-center rounded-lg px-3 text-left text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${tone} ${expanded ? "gap-3" : "justify-center"
        }`}
    >
      <ActionIcon name={icon} />
      {expanded && <span className="truncate">{label}</span>}
    </button>
  );
}

function ToolbarDivider() {
  return <span className="mx-1 h-6 w-px bg-slate-200" />;
}

const DRAWER_STORAGE_KEY = "qtracx.workspace-drawer-expanded";

export function Toolbar({
  isSelectingScanArea,
  isScanRunning,
  hasProgressDetails,
  isDrawingValue,
  isProcessing,
  currentPage,
  totalPages,
  scale,
  balloonsVisible,
  okBalloonsVisible,
  reviewBalloonsVisible,
  hasOkBalloons,
  hasReviewBalloons,
  hasBalloons,
  onSelectScanSection,
  onScanWholePage,
  findingReferencePoints,
  onFindGdtSymbols,
  findingGdtSymbols,
  onToggleDrawValue,
  onToggleBalloons,
  onSetOkBalloonsVisible,
  onSetReviewBalloonsVisible,
  onZoomIn,
  onZoomOut,
  onPrevPage,
  onNextPage,
  onSaveProject,
  onLoadProject,
  onRemoveDrawing,
  canSaveProject,
  referenceTables,
  scanCandidates,
}: ToolbarProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const visibilityMenuRef = useRef<HTMLDivElement>(null);
  const [drawerExpanded, setDrawerExpanded] = useState(true);
  const [drawerTop, setDrawerTop] = useState(0);
  const [detectionOpen, setDetectionOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [progressVisible, setProgressVisibleState] = useState(true);
  const [visibilityOpen, setVisibilityOpen] = useState(false);

  useEffect(() => {
    setDrawerExpanded(
      window.localStorage.getItem(DRAWER_STORAGE_KEY) !== "false"
    );

    setProgressVisibleState(isScanProgressVisible());

    const handleProgressVisibility = (event: Event) => {
      setProgressVisibleState(
        (event as CustomEvent<boolean>).detail
      );
    };

    window.addEventListener(
      SCAN_PROGRESS_VISIBILITY_EVENT,
      handleProgressVisibility
    );

    return () => {
      window.removeEventListener(
        SCAN_PROGRESS_VISIBILITY_EVENT,
        handleProgressVisibility
      );
    };
  }, []);

  useEffect(() => {
    if (!visibilityOpen) return;

    const closeOutside = (event: MouseEvent) => {
      if (
        event.target instanceof Node &&
        !visibilityMenuRef.current?.contains(event.target)
      ) {
        setVisibilityOpen(false);
      }
    };

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setVisibilityOpen(false);
    };

    document.addEventListener("mousedown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);

    return () => {
      document.removeEventListener("mousedown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [visibilityOpen]);

  useLayoutEffect(() => {
    const measure = () => {
      const bottom =
        rootRef.current?.getBoundingClientRect().bottom ?? 0;

      const roundedBottom = Math.round(bottom);

      setDrawerTop(roundedBottom);

      document.documentElement.style.setProperty(
        "--qtracx-workspace-toolbar-bottom",
        `${roundedBottom}px`
      );
    };

    measure();

    const animationFrame = window.requestAnimationFrame(measure);
    const observer = new ResizeObserver(measure);

    if (rootRef.current) observer.observe(rootRef.current);

    window.addEventListener("resize", measure);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  });

  const toggleDrawer = () => {
    setDrawerExpanded((current) => {
      const next = !current;
      window.localStorage.setItem(
        DRAWER_STORAGE_KEY,
        next ? "true" : "false"
      );
      return next;
    });
  };

  const runWholePageScan = () => {
    setDetectionOpen(false);
    onScanWholePage();
  };

  const selectScanSections = () => {
    setDetectionOpen(false);
    onSelectScanSection();
  };

  const toggleProgress = () => {
    if (!hasProgressDetails) return;

    setScanProgressVisible(!progressVisible);
  };

  const toggleAllBalloons = () => {
    setVisibilityOpen(false);
    onToggleBalloons();
  };

  return (
    <>
      <div
        ref={rootRef}
        data-drawer-expanded={drawerExpanded ? "true" : "false"}
        className="qtracx-workspace-toolbar"
      >
        {/* <div className="qtracx-workspace-topbar">
          <button
            type="button"
            onClick={toggleDrawer}
            title={
              drawerExpanded
                ? "Collapse action drawer"
                : "Expand action drawer"
            }
            aria-label={
              drawerExpanded
                ? "Collapse action drawer"
                : "Expand action drawer"
            }
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
          >
            <ActionIcon name="menu" />
          </button>

          <div className="flex min-w-0 flex-1 items-center gap-2">
            {isDrawingValue && (
              <button
                type="button"
                onClick={onToggleDrawValue}
                className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-700"
              >
                Cancel manual balloon
              </button>
            )}

            {isSelectingScanArea && (
              <span className="truncate rounded-lg bg-emerald-100 px-3 py-1.5 text-xs font-semibold text-emerald-800">
                Selecting scan sections
              </span>
            )}

            {findingReferencePoints && (
              <span className="truncate text-xs font-medium text-teal-700">
                Reading reference points…
              </span>
            )}

            {findingGdtSymbols && (
              <span className="truncate text-xs font-medium text-indigo-700">
                Reading GD&amp;T symbols…
              </span>
            )}

            {isProcessing &&
              !isScanRunning &&
              !findingReferencePoints &&
              !findingGdtSymbols && (
                <span className="truncate text-xs font-medium text-blue-700">
                  Running OCR…
                </span>
              )}
          </div>

          <div className="flex shrink-0 items-center gap-1.5">
            <button
              type="button"
              onClick={onZoomOut}
              title="Zoom out"
              className="qtracx-topbar-square"
            >
              −
            </button>

            <span className="min-w-[3.5rem] text-center text-xs font-semibold tabular-nums text-slate-600">
              {Math.round(scale * 100)}%
            </span>

            <button
              type="button"
              onClick={onZoomIn}
              title="Zoom in"
              className="qtracx-topbar-square"
            >
              +
            </button>

            <ToolbarDivider />

            <button
              type="button"
              onClick={onPrevPage}
              disabled={
                currentPage <= 1 ||
                isProcessing ||
                isSelectingScanArea
              }
              title="Previous page"
              className="qtracx-topbar-square"
            >
              <ActionIcon name="previous" />
            </button>

            <span className="min-w-[5.5rem] text-center text-xs font-medium tabular-nums text-slate-600">
              {currentPage} / {Math.max(totalPages, 1)}
            </span>

            <button
              type="button"
              onClick={onNextPage}
              disabled={
                currentPage >= totalPages ||
                isProcessing ||
                isSelectingScanArea
              }
              title="Next page"
              className="qtracx-topbar-square"
            >
              <ActionIcon name="next" />
            </button>

            <ToolbarDivider />

            <button
              type="button"
              onClick={onToggleBalloons}
              disabled={!canSaveProject || !hasBalloons}
              title={
                balloonsVisible
                  ? "Hide balloons and white out their values"
                  : "Show balloons"
              }
              className={`qtracx-topbar-action ${balloonsVisible
                  ? ""
                  : "border-blue-200 bg-blue-50 text-blue-700"
                }`}
            >
              <ActionIcon name="balloon" />
              <span className="hidden xl:inline">
                {balloonsVisible ? "Hide balloons" : "Show balloons"}
              </span>
            </button>

            <button
              type="button"
              onClick={toggleProgress}
              disabled={!isScanRunning}
              title={
                progressVisible
                  ? "Hide scan progress"
                  : "Show scan progress"
              }
              className={`qtracx-topbar-action ${isScanRunning && !progressVisible
                  ? "border-blue-200 bg-blue-50 text-blue-700"
                  : ""
                }`}
            >
              <ActionIcon name="progress" />
              <span className="hidden xl:inline">
                {progressVisible ? "Hide progress" : "Show progress"}
              </span>
            </button>
          </div>
        </div> */}

        <div className="qtracx-workspace-topbar">
          <button
            type="button"
            onClick={toggleDrawer}
            title={
              drawerExpanded
                ? "Collapse action drawer"
                : "Expand action drawer"
            }
            aria-label={
              drawerExpanded
                ? "Collapse action drawer"
                : "Expand action drawer"
            }
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
          >
            <ActionIcon name="menu" />
          </button>

          <ToolbarDivider />

          <button
            type="button"
            onClick={onZoomOut}
            title="Zoom out"
            className="qtracx-topbar-square"
          >
            −
          </button>

          <span className="min-w-[3.5rem] text-center text-xs font-semibold tabular-nums text-slate-600">
            {Math.round(scale * 100)}%
          </span>

          <button
            type="button"
            onClick={onZoomIn}
            title="Zoom in"
            className="qtracx-topbar-square"
          >
            +
          </button>

          <ToolbarDivider />

          <button
            type="button"
            onClick={onPrevPage}
            disabled={
              currentPage <= 1 ||
              isProcessing ||
              isSelectingScanArea
            }
            title="Previous page"
            className="qtracx-topbar-square"
          >
            <ActionIcon name="previous" />
          </button>

          <span className="min-w-[5.5rem] text-center text-xs font-medium tabular-nums text-slate-600">
            {currentPage} / {Math.max(totalPages, 1)}
          </span>

          <button
            type="button"
            onClick={onNextPage}
            disabled={
              currentPage >= totalPages ||
              isProcessing ||
              isSelectingScanArea
            }
            title="Next page"
            className="qtracx-topbar-square"
          >
            <ActionIcon name="next" />
          </button>

          <ToolbarDivider />

          <div ref={visibilityMenuRef} className="relative flex">
            <button
              type="button"
              onClick={toggleAllBalloons}
              disabled={!canSaveProject || !hasBalloons}
              title={
                balloonsVisible
                  ? "Hide all balloons and white out their values"
                  : "Show all balloons"
              }
              className={`qtracx-topbar-action rounded-r-none border-r-0 ${balloonsVisible
                  ? ""
                  : "border-blue-200 bg-blue-50 text-blue-700"
                }`}
            >
              <ActionIcon name="balloon" />
              <span>
                {balloonsVisible ? "Hide balloons" : "Show balloons"}
              </span>
            </button>

            <button
              type="button"
              onClick={() =>
                setVisibilityOpen((current) => !current)
              }
              disabled={!canSaveProject || !hasBalloons}
              aria-haspopup="menu"
              aria-expanded={visibilityOpen}
              title="More balloon visibility options"
              className={`qtracx-topbar-action rounded-l-none px-2 ${visibilityOpen
                  ? "border-blue-200 bg-blue-50 text-blue-700"
                  : ""
                }`}
            >
              <svg
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
                className="h-4 w-4"
              >
                <path d="m5 7.5 5 5 5-5" />
              </svg>
              <span className="sr-only">
                More balloon visibility options
              </span>
            </button>

            {visibilityOpen && (
              <div
                role="menu"
                className="absolute left-0 top-[calc(100%+0.5rem)] z-[80] w-56 rounded-xl border border-slate-200 bg-white p-2 shadow-xl"
              >
                <button
                  type="button"
                  role="menuitemcheckbox"
                  aria-checked={okBalloonsVisible}
                  disabled={!hasOkBalloons}
                  onClick={() =>
                    onSetOkBalloonsVisible(!okBalloonsVisible)
                  }
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <span
                    aria-hidden="true"
                    className={`flex h-4 w-4 items-center justify-center rounded border text-[10px] ${okBalloonsVisible
                        ? "border-emerald-600 bg-emerald-600 text-white"
                        : "border-slate-300 bg-white"
                      }`}
                  >
                    {okBalloonsVisible ? "✓" : ""}
                  </span>
                  <span>Show OK values</span>
                </button>

                <button
                  type="button"
                  role="menuitemcheckbox"
                  aria-checked={reviewBalloonsVisible}
                  disabled={!hasReviewBalloons}
                  onClick={() =>
                    onSetReviewBalloonsVisible(
                      !reviewBalloonsVisible
                    )
                  }
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <span
                    aria-hidden="true"
                    className={`flex h-4 w-4 items-center justify-center rounded border text-[10px] ${reviewBalloonsVisible
                        ? "border-red-600 bg-red-600 text-white"
                        : "border-slate-300 bg-white"
                      }`}
                  >
                    {reviewBalloonsVisible ? "✓" : ""}
                  </span>
                  <span>Show review values</span>
                </button>

                <p className="mt-2 border-t border-slate-100 px-3 pt-2 text-xs text-slate-500">
                  The main button controls both categories.
                </p>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={toggleProgress}
            disabled={!hasProgressDetails}
            title={
              progressVisible
                ? "Hide scan progress and summary"
                : isScanRunning
                  ? "Show scan progress"
                  : "Show last scan summary"
            }
            className={`qtracx-topbar-action ${hasProgressDetails && !progressVisible
                ? "border-blue-200 bg-blue-50 text-blue-700"
                : ""
              }`}
          >
            <ActionIcon name="progress" />
            <span>
              {progressVisible
                ? "Hide progress"
                : isScanRunning
                  ? "Show progress"
                  : "Show summary"}
            </span>
          </button>

          {(isDrawingValue ||
            isSelectingScanArea ||
            findingReferencePoints ||
            findingGdtSymbols ||
            (isProcessing && !isScanRunning)) && (
              <ToolbarDivider />
            )}

          <div className="flex min-w-0 items-center gap-2">
            {isDrawingValue && (
              <button
                type="button"
                onClick={onToggleDrawValue}
                className="whitespace-nowrap rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-700"
              >
                Cancel manual balloon
              </button>
            )}

            {isSelectingScanArea && (
              <span className="whitespace-nowrap rounded-lg bg-emerald-100 px-3 py-1.5 text-xs font-semibold text-emerald-800">
                Selecting scan sections
              </span>
            )}

            {findingReferencePoints && (
              <span className="whitespace-nowrap text-xs font-medium text-teal-700">
                Reading reference points…
              </span>
            )}

            {findingGdtSymbols && (
              <span className="whitespace-nowrap text-xs font-medium text-indigo-700">
                Reading GD&amp;T symbols…
              </span>
            )}

            {isProcessing &&
              !isScanRunning &&
              !findingReferencePoints &&
              !findingGdtSymbols && (
                <span className="whitespace-nowrap text-xs font-medium text-blue-700">
                  Running OCR…
                </span>
              )}
          </div>
        </div>

        <aside
          className="qtracx-action-drawer"
          style={{
            top: drawerTop,
            visibility: drawerTop > 0 ? "visible" : "hidden",
          }}
        >
          <div className="qtracx-drawer-content">
            {drawerExpanded && (
              <p className="qtracx-drawer-group-title">Document</p>
            )}

            <DrawerAction
              label="New Project"
              icon="new"
              expanded={drawerExpanded}
              disabled={isProcessing}
              onClick={onRemoveDrawing}
            />

            <DrawerAction
              label="Load Project"
              icon="load"
              expanded={drawerExpanded}
              disabled={isProcessing}
              onClick={onLoadProject}
            />

            <DrawerAction
              label="Save Project"
              icon="save"
              expanded={drawerExpanded}
              disabled={!canSaveProject || isProcessing}
              onClick={onSaveProject}
            />

            <DrawerAction
              label="Clear Workspace"
              icon="clear"
              expanded={drawerExpanded}
              disabled={!canSaveProject || isProcessing}
              danger
              onClick={onRemoveDrawing}
            />

            <div className="my-3 border-t border-slate-200" />

            {drawerExpanded && (
              <p className="qtracx-drawer-group-title">Detection</p>
            )}

            <div className="relative">
              <DrawerAction
                label="Auto-Balloon"
                icon="detect"
                expanded={drawerExpanded}
                active={detectionOpen || isSelectingScanArea}
                disabled={
                  isProcessing ||
                  isScanRunning ||
                  !canSaveProject
                }
                onClick={() =>
                  setDetectionOpen((current) => !current)
                }
              />

              {detectionOpen && (
                <div
                  className={
                    drawerExpanded
                      ? "mt-1 space-y-1 rounded-lg bg-slate-100 p-2"
                      : "absolute left-[calc(100%+0.5rem)] top-0 z-50 w-52 rounded-xl border border-slate-200 bg-white p-2 shadow-xl"
                  }
                >
                  <button
                    type="button"
                    onClick={runWholePageScan}
                    className="block w-full rounded-lg px-3 py-2 text-left text-sm font-medium text-slate-700 hover:bg-white"
                  >
                    Whole Page
                  </button>

                  <button
                    type="button"
                    onClick={selectScanSections}
                    className="block w-full rounded-lg px-3 py-2 text-left text-sm font-medium text-slate-700 hover:bg-white"
                  >
                    Select Sections
                  </button>
                </div>
              )}
            </div>

            <DrawerAction
              label={
                isDrawingValue
                  ? "Cancel Manual Balloon"
                  : "Manual Balloon"
              }
              icon="manual"
              expanded={drawerExpanded}
              active={isDrawingValue}
              disabled={isProcessing || !canSaveProject}
              onClick={onToggleDrawValue}
            />

            <div className="my-3 border-t border-slate-200" />

            <div className="qtracx-drawer-export">
              <span className="qtracx-drawer-export-icon">
                <ActionIcon name="export" />
              </span>

              <ExportPanel
                disabled={isProcessing}
                referenceTables={referenceTables}
                scanCandidates={scanCandidates}
              />
            </div>

            <DrawerAction
              label="Settings"
              icon="settings"
              expanded={drawerExpanded}
              active={settingsOpen}
              onClick={() => setSettingsOpen(true)}
            />
          </div>
        </aside>
      </div>

      <WorkspaceSettingsPanel
        open={settingsOpen}
        disabled={isProcessing}
        findingGdtSymbols={findingGdtSymbols}
        onFindGdtSymbols={onFindGdtSymbols}
        onClose={() => setSettingsOpen(false)}
      />
    </>
  );
}
