"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  Stage,
  Layer,
  Group,
  Image as KonvaImage,
  Rect,
  Text,
} from "react-konva";
import type Konva from "konva";
import { v4 as uuidv4 } from "uuid";
import { useAnnotationStore } from "@/store/annotationStore";
import { normalizeBBox } from "@/lib/canvasUtils";
import {
  isScanJobCancelledError,
  preloadOcr,
  runAutoBalloonScan,
  runOCR,
  stopAutoBalloonScan,
} from "@/lib/clientOcr";
import { classifyDimension } from "@/lib/dimensionClassifier";
import { filterNewScanRegions } from "@/lib/scanCandidates";
import { mergePageReviewCandidates } from "@/lib/scanReviewCandidates";
import {
  runSectionScanQueue,
  type QueuedScanSection,
  type ScanRunOutcome,
  type SectionQueuePosition,
} from "@/lib/sectionScanQueue";
import { deriveRange } from "@/lib/valueFields";
import { useClientOcr } from "@/hooks/useClientOcr";
import {
  loadPdfDocument,
  renderPdfPage,
  loadImageFile,
} from "@/lib/pdfLoader";
import {
  saveAnnotations,
  loadAnnotations,
  saveProject,
  getMostRecentProject,
  deleteProject,
} from "@/lib/db";
import {
  buildProjectBundle,
  normalizeLegacyAnnotations,
  parseProjectBundle,
  fileToDataUrl,
  dataUrlToFile,
  type ProjectSource,
} from "@/lib/project";
import { downloadFile } from "@/lib/export";
import { Balloon } from "@/components/Balloon";
import { Toolbar } from "@/components/Toolbar";
import { Sidebar } from "@/components/Sidebar";
import { AnnotationPopup } from "@/components/AnnotationPopup";
import { ValueEditor } from "@/components/ValueEditor";
import { ScanProgressBanner } from "@/components/ScanProgressBanner";
import { ScanDebugOverlay } from "@/components/ScanDebugOverlay";
import { ScanReviewOverlay } from "@/components/ScanReviewOverlay";
import { SCAN_DEBUG_OVERLAY_ENABLED } from "@/lib/featureFlags";
import type { Annotation, BBox } from "@/types/annotation";
import type { SegmentRegion } from "@/lib/paddleOcrClient";
import type {
  ScanCompletionSummary,
  ScanDebugOverlay as ScanDebugOverlayModel,
  ScanProgress,
  ScanScopeKind,
} from "@/types/scanJob";
import type { PDFDocumentProxy } from "pdfjs-dist";

const MIN_BOX = 8;
const DRAWING_BACKGROUND_NAME = "drawing-background";

type PageScanReviewCandidate = Omit<SegmentRegion, "candidateId"> & {
  candidateId: string;
  page: number;
  reviewOrder: number;
};

interface AutoBalloonRunOptions {
  manageProcessing?: boolean;
  sectionPosition?: SectionQueuePosition;
}

const waitForBrowserPaint = () =>
  new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));

const legacyMigrationWarning = (orphanLabelCount: number) =>
  orphanLabelCount > 0
    ? `${orphanLabelCount} legacy label${
        orphanLabelCount === 1 ? "" : "s"
      } had no associated value and ${
        orphanLabelCount === 1 ? "was" : "were"
      } not imported.`
    : null;

// PDFs are rasterized once at this fixed resolution; this canvas is the base
// coordinate system every bbox is stored in. Zoom is applied as a Konva Stage
// transform on top — never by re-rendering — so boxes stay aligned at any zoom.
const PDF_RENDER_SCALE = 1.5;

export function DrawingViewer() {
  const { ready: ocrReady, error: ocrError, engineLabel } = useClientOcr();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadProjectInputRef = useRef<HTMLInputElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  /** Original drawing bytes, kept so a saved project can embed them. */
  const sourceFileRef = useRef<ProjectSource | null>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const projectIdRef = useRef("");

  const [konvaImage, setKonvaImage] = useState<HTMLImageElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [projectId, setProjectId] = useState(() => uuidv4());
  const restoredRef = useRef(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedReviewCandidateId, setSelectedReviewCandidateId] = useState<
    string | null
  >(null);
  const [currentBox, setCurrentBox] = useState<BBox | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [lastDebugDump, setLastDebugDump] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null);
  const [scanOverlay, setScanOverlay] =
    useState<ScanDebugOverlayModel | null>(null);
  const [scanSummary, setScanSummary] =
    useState<ScanCompletionSummary | null>(null);
  const [scanReviewCandidates, setScanReviewCandidates] = useState<
    PageScanReviewCandidate[]
  >([]);
  const scanReviewCandidatesRef = useRef<PageScanReviewCandidate[]>([]);
  const reviewOrderRef = useRef(0);
  const [selectedScanSections, setSelectedScanSections] = useState<
    QueuedScanSection[]
  >([]);
  const drawStartRef = useRef<{ x: number; y: number } | null>(null);
  const currentBoxRef = useRef<BBox | null>(null);

  const annotations = useAnnotationStore((s) => s.annotations);
  const setAnnotations = useAnnotationStore((s) => s.setAnnotations);
  const addAnnotations = useAnnotationStore((s) => s.addAnnotations);
  const updateAnnotation = useAnnotationStore((s) => s.updateAnnotation);
  const setAllBalloonVisibility = useAnnotationStore(
    (s) => s.setAllBalloonVisibility
  );
  const removeAnnotation = useAnnotationStore((s) => s.removeAnnotation);
  const setPending = useAnnotationStore((s) => s.setPending);
  const currentPage = useAnnotationStore((s) => s.currentPage);
  const setCurrentPage = useAnnotationStore((s) => s.setCurrentPage);
  const totalPages = useAnnotationStore((s) => s.totalPages);
  const setTotalPages = useAnnotationStore((s) => s.setTotalPages);
  const scale = useAnnotationStore((s) => s.scale);
  const setScale = useAnnotationStore((s) => s.setScale);
  const isDrawingValue = useAnnotationStore((s) => s.isDrawingValue);
  const setIsDrawingValue = useAnnotationStore((s) => s.setIsDrawingValue);
  const editingValueId = useAnnotationStore((s) => s.editingValueId);
  const setEditingValueId = useAnnotationStore((s) => s.setEditingValueId);
  const isSegmenting = useAnnotationStore((s) => s.isSegmenting);
  const setIsSegmenting = useAnnotationStore((s) => s.setIsSegmenting);
  const isProcessing = useAnnotationStore((s) => s.isProcessing);
  const setIsProcessing = useAnnotationStore((s) => s.setIsProcessing);
  const setProjectName = useAnnotationStore((s) => s.setProjectName);
  const projectName = useAnnotationStore((s) => s.projectName);
  const setStoreProjectId = useAnnotationStore((s) => s.setProjectId);

  const pageAnnotations = annotations.filter(
    (annotation) =>
      annotation.page === currentPage && annotation.kind !== "label"
  );
  const visiblePageAnnotations = pageAnnotations.filter(
    (annotation) => !annotation.hidden
  );
  const valueAnnotations = annotations.filter(
    (annotation) => annotation.kind !== "label"
  );
  const balloonsVisible =
    valueAnnotations.length > 0 &&
    valueAnnotations.every((annotation) => !annotation.hidden);
  const pageScanReviewCandidates = scanReviewCandidates.filter(
    (candidate) => candidate.page === currentPage
  );
  const pageSelectedScanSections = selectedScanSections.filter(
    (section) => section.page === currentPage
  );

  const replaceScanReviewCandidates = useCallback(
    (next: PageScanReviewCandidate[]) => {
      scanReviewCandidatesRef.current = next;
      setScanReviewCandidates(next);
    },
    []
  );

  const canvasToKonvaImage = useCallback((canvas: HTMLCanvasElement) => {
    const img = new window.Image();
    img.src = canvas.toDataURL("image/png");
    img.onload = () => {
      setKonvaImage(img);
      setStageSize({ width: canvas.width, height: canvas.height });
    };
  }, []);

  const renderCurrentPage = useCallback(
    async (doc: PDFDocumentProxy, page: number) => {
      const { canvas, width, height } = await renderPdfPage(
        doc,
        page,
        PDF_RENDER_SCALE
      );
      sourceCanvasRef.current = canvas;
      canvasToKonvaImage(canvas);
      setStageSize({ width, height });
    },
    [canvasToKonvaImage]
  );

  // Re-render only when the document or page changes — never on zoom. Zoom is a
  // pure display transform applied to the Stage below.
  useEffect(() => {
    if (!pdfDoc) return;
    void renderCurrentPage(pdfDoc, currentPage);
  }, [pdfDoc, currentPage, renderCurrentPage]);

  useEffect(() => {
    // Overlay geometry belongs to one exact page/project and is never persisted.
    setScanOverlay(null);
  }, [currentPage, projectId]);

  useEffect(() => {
    // Review boxes are temporary scan output, never project data.
    replaceScanReviewCandidates([]);
    reviewOrderRef.current = 0;
    setSelectedReviewCandidateId(null);
    setSelectedScanSections([]);
  }, [projectId, replaceScanReviewCandidates]);

  useEffect(() => {
    void preloadOcr();
  }, []);

  useEffect(() => {
    if (annotations.length === 0) return;
    const t = setTimeout(() => {
      void saveAnnotations(projectId, annotations);
    }, 500);
    return () => clearTimeout(t);
  }, [annotations, projectId]);

  // Expose the current project id to the store so the Export menu can hand it to
  // the checksheet web view (which opens in a separate browser tab).
  useEffect(() => {
    setStoreProjectId(projectId);
    projectIdRef.current = projectId;
  }, [projectId, setStoreProjectId]);

  // Pull edits made in the checksheet tab back into the live drawing. The tab
  // saves to IndexedDB and broadcasts the updated annotations on this channel.
  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return;
    const channel = new BroadcastChannel("doc-ocr-box:checksheet");
    channel.onmessage = (e) => {
      const msg = e.data as { projectId?: string; annotations?: Annotation[] };
      if (msg?.projectId === projectId && Array.isArray(msg.annotations)) {
        setAnnotations(normalizeLegacyAnnotations(msg.annotations).annotations);
      }
    };
    return () => channel.close();
  }, [projectId, setAnnotations]);

  // Render a drawing onto the stage. Shared by fresh uploads and project loads
  // so both paths render at the same scale and produce matching bbox coords.
  const loadSource = useCallback(
    async (file: File) => {
      if (file.type === "application/pdf") {
        const doc = await loadPdfDocument(file);
        setPdfDoc(doc);
        setTotalPages(doc.numPages);
        setCurrentPage(1);
        setScale(1);
        await renderCurrentPage(doc, 1);
      } else {
        const { canvas, width, height } = await loadImageFile(file);
        sourceCanvasRef.current = canvas;
        setPdfDoc(null);
        setTotalPages(1);
        setCurrentPage(1);
        canvasToKonvaImage(canvas);
        setStageSize({ width, height });
      }
    },
    [
      canvasToKonvaImage,
      renderCurrentPage,
      setCurrentPage,
      setScale,
      setTotalPages,
    ]
  );

  // On load, reopen the most recent project so its drawing and annotations come
  // back together (persisted in IndexedDB across reloads).
  const restoreLastProject = useCallback(async () => {
    const recent = await getMostRecentProject();
    if (!recent?.fileBlob) return;
    const file = new File([recent.fileBlob], recent.fileName, {
      type: recent.mimeType ?? recent.fileBlob.type,
    });
    setProjectId(recent.id);
    setProjectName(recent.name.replace(/\.[^.]+$/, ""));
    sourceFileRef.current = {
      fileName: recent.fileName,
      mimeType: file.type,
      fileType: recent.fileType,
      dataUrl: await fileToDataUrl(file),
    };
    await loadSource(file);
    const migration = normalizeLegacyAnnotations(
      await loadAnnotations(recent.id)
    );
    setAnnotations(migration.annotations);
    await saveAnnotations(recent.id, migration.annotations);
    const warning = legacyMigrationWarning(migration.orphanLabelCount);
    if (warning) setSelectionError(warning);
  }, [loadSource, setAnnotations, setProjectName]);

  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    void restoreLastProject();
  }, [restoreLastProject]);

  const handleFile = async (file: File) => {
    // A new drawing is a new project — fresh id, no stale annotations.
    const id = uuidv4();
    setProjectId(id);
    setAnnotations([]);
    setSelectedId(null);
    setSelectedReviewCandidateId(null);
    setSelectedScanSections([]);
    setEditingValueId(null);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setScanProgress(null);
    setScanOverlay(null);
    setScanSummary(null);
    setProjectName(file.name.replace(/\.[^.]+$/, ""));
    const fileType = file.type === "application/pdf" ? "pdf" : "image";
    sourceFileRef.current = {
      fileName: file.name,
      mimeType: file.type,
      fileType,
      dataUrl: await fileToDataUrl(file),
    };
    await saveProject({
      id,
      name: file.name,
      fileName: file.name,
      fileType,
      mimeType: file.type,
      fileBlob: file,
      updatedAt: Date.now(),
    });
    await loadSource(file);
  };

  // Save the whole project — drawing + annotations + verification block — as one
  // portable `.docbox.json` the app can reload later.
  const handleSaveProject = () => {
    const source = sourceFileRef.current;
    if (!source || !source.dataUrl) {
      setSelectionError(
        "Can't save yet — reopen the drawing, then Save Project."
      );
      return;
    }
    const base = projectName.replace(/\s+/g, "_").toLowerCase() || "drawing";
    const json = buildProjectBundle({
      projectName,
      source,
      annotations,
      savedAt: Date.now(),
    });
    downloadFile(json, `${base}.docbox.json`, "application/json");
  };

  // Close the current drawing: clear the canvas + annotations and forget the
  // project so it doesn't auto-restore on the next reload.
  const handleRemoveDrawing = async () => {
    const id = projectId;
    setKonvaImage(null);
    setPdfDoc(null);
    sourceCanvasRef.current = null;
    sourceFileRef.current = null;
    setAnnotations([]);
    setSelectedId(null);
    setSelectedReviewCandidateId(null);
    setSelectedScanSections([]);
    setEditingValueId(null);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setPending(null);
    setSelectionError(null);
    setScanProgress(null);
    setScanSummary(null);
    setCurrentPage(1);
    setTotalPages(1);
    setProjectName("Untitled Drawing");
    setProjectId(uuidv4());
    await deleteProject(id);
  };

  const handleLoadProject = async (file: File) => {
    setSelectionError(null);
    setScanProgress(null);
    setScanSummary(null);
    setSelectedReviewCandidateId(null);
    setSelectedScanSections([]);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    try {
      const bundle = parseProjectBundle(await file.text());
      const id = uuidv4();
      setProjectId(id);
      sourceFileRef.current = bundle.source;
      setProjectName(bundle.projectName);
      const srcFile = dataUrlToFile(
        bundle.source.dataUrl,
        bundle.source.fileName,
        bundle.source.mimeType
      );
      await loadSource(srcFile);
      setSelectedId(null);
      setEditingValueId(null);
      const migration = normalizeLegacyAnnotations(bundle.annotations);
      setAnnotations(migration.annotations);
      const warning = legacyMigrationWarning(migration.orphanLabelCount);
      if (warning) setSelectionError(warning);
      // Persist so the loaded project reopens on the next visit too.
      await saveProject({
        id,
        name: bundle.projectName,
        fileName: bundle.source.fileName,
        fileType: bundle.source.fileType,
        mimeType: bundle.source.mimeType,
        fileBlob: srcFile,
        updatedAt: Date.now(),
      });
      await saveAnnotations(id, migration.annotations);
    } catch (err) {
      setSelectionError(
        err instanceof Error ? err.message : "Could not open project file."
      );
    }
  };

  // Select an annotation and follow it to its page so it's actually on screen.
  const handleSelect = useCallback(
    (id: string | null) => {
      setSelectedReviewCandidateId(null);
      if (!id) {
        setSelectedId(null);
        return;
      }
      const ann = annotations.find((a) => a.id === id);
      setSelectedId(ann?.hidden ? null : id);
      if (ann && ann.page !== currentPage) setCurrentPage(ann.page);
    },
    [annotations, currentPage, setCurrentPage]
  );

  // Clear the local selection with the deleted value so the remaining balloons
  // do not stay dimmed against an id that no longer exists.
  const handleDelete = useCallback(
    (id: string) => {
      removeAnnotation(id);
      void saveAnnotations(
        projectId,
        useAnnotationStore.getState().annotations
      );
      setSelectedId((current) => (current === id ? null : current));
      if (editingValueId === id) setEditingValueId(null);
    },
    [editingValueId, projectId, removeAnnotation, setEditingValueId]
  );

  const toggleAnnotationVisibility = useCallback(
    (id: string) => {
      const annotation = useAnnotationStore
        .getState()
        .annotations.find((candidate) => candidate.id === id);
      if (!annotation) return;

      const hidden = !annotation.hidden;
      updateAnnotation(id, { hidden });
      if (hidden) {
        setSelectedId((current) => (current === id ? null : current));
      }
      void saveAnnotations(
        projectId,
        useAnnotationStore.getState().annotations
      );
    },
    [projectId, updateAnnotation]
  );

  // Read one drawn value and open the value/tolerance confirmation popup.
  const finishBox = useCallback(
    async (bbox: BBox) => {
      if (bbox.width < MIN_BOX || bbox.height < MIN_BOX) return;
      const source = sourceCanvasRef.current;
      if (!source) {
        setIsDrawingValue(false);
        return;
      }

      setIsProcessing(true);
      setSelectionError(null);
      setScanSummary(null);
      try {
        const ocrResult = await runOCR(source, bbox, 1);
        if (ocrResult.debugDumpDir) {
          setLastDebugDump(ocrResult.debugDumpDir);
        } else if (ocrResult.debugDumpSkipped) {
          setLastDebugDump(`skipped: ${ocrResult.debugDumpSkipped}`);
        }
        if (!ocrResult.text.trim()) {
          setSelectionError(
            "No text detected — type the value manually or draw a tighter box."
          );
        }
        setPending({
          bbox,
          page: currentPage,
          ocrResult,
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "OCR failed unexpectedly";
        setSelectionError(message);
        setPending({
          bbox,
          page: currentPage,
          ocrResult: {
            text: "",
            confidence: 0,
            rotation: 0,
            orientation: "horizontal",
            words: [],
            engine: "paddleocr",
          },
        });
      } finally {
        setIsProcessing(false);
        setIsDrawingValue(false);
      }
    },
    [currentPage, setPending, setIsDrawingValue, setIsProcessing]
  );

  const runAutoBalloon = useCallback(
    async (
      bbox: BBox,
      scopeKind: ScanScopeKind,
      scanPage: number,
      options: AutoBalloonRunOptions = {}
    ): Promise<ScanRunOutcome> => {
      if (bbox.width < MIN_BOX || bbox.height < MIN_BOX) {
        setSelectionError("The selected scan section is too small.");
        return { status: "failed" };
      }
      const source = sourceCanvasRef.current;
      if (!source) {
        setSelectionError("No drawing is available to scan.");
        return { status: "failed" };
      }

      const projectAtStart = projectIdRef.current;
      const scanRunId = uuidv4();
      const manageProcessing = options.manageProcessing ?? true;
      const sectionLabel = options.sectionPosition
        ? "Section " +
          options.sectionPosition.current +
          " of " +
          options.sectionPosition.total
        : "";
      if (manageProcessing) setIsProcessing(true);
      setSelectionError(null);
      setScanSummary(null);
      setScanOverlay(null);
      setScanProgress({
        jobId: "",
        status: "queued",
        stage: "preparing",
        message: sectionLabel
          ? sectionLabel + " · Preparing selected area"
          : "Preparing selected area",
        percent: 0,
        completed: 0,
        total: 0,
        passCurrent: 0,
        passTotal: 0,
        tileCurrent: 0,
        tileTotal: 0,
        objectCurrent: 0,
        objectTotal: 0,
        batchCurrent: 0,
        batchTotal: 0,
        candidateCount: 0,
        operationLabel: sectionLabel,
        elapsedSeconds: 0,
        stepElapsedSeconds: 0,
        heartbeatAgeSeconds: 0,
        progressAgeSeconds: 0,
        estimatedRemainingSeconds: null,
        liveness: "queued",
        overlay: null,
      });
      try {
        const scanResult = await runAutoBalloonScan({
          sourceCanvas: source,
          bbox,
          page: scanPage,
          scopeKind,
          displayScale: 1,
          existingValueBoxes: useAnnotationStore
            .getState()
            .annotations.filter(
              (annotation) =>
                annotation.kind !== "label" && annotation.page === scanPage
            )
            .map((annotation) => annotation.bbox),
          onProgress: (progress) => {
            const displayedProgress = sectionLabel
              ? {
                  ...progress,
                  message: sectionLabel + " · " + progress.message,
                  operationLabel: progress.operationLabel
                    ? sectionLabel + " · " + progress.operationLabel
                    : sectionLabel,
                }
              : progress;
            setScanProgress(displayedProgress);
            if (SCAN_DEBUG_OVERLAY_ENABLED && progress.overlay) {
              setScanOverlay(progress.overlay);
            }
          },
        });
        if (projectIdRef.current !== projectAtStart) {
          throw new Error(
            "The drawing changed while the scan was running. No balloons were added."
          );
        }

        // Recheck against the live store immediately before the single batch
        // insertion so existing manual or edited balloons always win.
        const liveAnnotations = useAnnotationStore.getState().annotations;
        const filtered = filterNewScanRegions(
          scanResult.regions,
          liveAnnotations,
          scanPage,
          undefined,
          scopeKind
        );
        const now = Date.now();
        const newAnnotations: Annotation[] = filtered.accepted
          .map((r) => {
            const cleanValue = r.text.trim();
            const type = classifyDimension(cleanValue);
            return {
              id: uuidv4(),
              // number is assigned sequentially by addAnnotations
              number: 0,
              label: "",
              value: cleanValue,
              type,
              confidence: r.confidence,
              bbox: r.valueBox,
              rotation: r.rotation,
              page: scanPage,
              createdAt: now,
              kind: "dimension",
              needsReview: r.needsReview || !r.recognized,
              range: deriveRange(cleanValue) || undefined,
            };
          });

        // Review candidates stay outside the annotation store. Filter them
        // against both existing balloons and this scan's atomic insert so a
        // resolved object is never shown again as a grey review box.
        const reviewExisting = [...liveAnnotations, ...newAnnotations];
        const filteredReview = filterNewScanRegions(
          scanResult.reviewCandidates,
          reviewExisting,
          scanPage,
          undefined,
          scopeKind
        );
        const reviewCandidates: PageScanReviewCandidate[] =
          filteredReview.accepted.map((candidate) => ({
            ...candidate,
            candidateId: [
              projectAtStart,
              scanPage,
              scanRunId,
              candidate.candidateId ?? uuidv4(),
            ].join(":"),
            page: scanPage,
            reviewOrder: reviewOrderRef.current++,
          }));

        const mergedReviews = mergePageReviewCandidates({
          existing: scanReviewCandidatesRef.current,
          incoming: reviewCandidates,
          newAnnotations,
          page: scanPage,
          scopeKind,
        });
        const incomingReviewIds = new Set(
          reviewCandidates.map((candidate) => candidate.candidateId)
        );
        const addedReviewCount = mergedReviews.candidates.filter((candidate) =>
          incomingReviewIds.has(candidate.candidateId)
        ).length;
        const pageReviewCount = mergedReviews.candidates.filter(
          (candidate) => candidate.page === scanPage
        ).length;

        if (newAnnotations.length > 0) {
          // Atomic per-section commit: save these balloons before the queue is
          // allowed to start its next section.
          addAnnotations(newAnnotations);
          await saveAnnotations(
            projectAtStart,
            useAnnotationStore.getState().annotations
          );
        }
        replaceScanReviewCandidates(mergedReviews.candidates);
        setScanOverlay(null);
        const summary: ScanCompletionSummary = {
          scopeKind,
          added: newAnnotations.length,
          detected: scanResult.detected,
          recognized: scanResult.recognized,
          eligible: scanResult.eligible,
          excluded: scanResult.excluded,
          reviewRequired:
            scopeKind === "section" ? pageReviewCount : addedReviewCount,
          unread: scanResult.unread,
          skippedExisting:
            scanResult.skippedExisting +
            filtered.skippedExisting +
            filteredReview.skippedExisting,
          skippedDuplicates:
            filtered.skippedDuplicates +
            filteredReview.skippedDuplicates +
            mergedReviews.skippedDuplicates,
        };
        setScanSummary(summary);
        if (newAnnotations.length === 0 && addedReviewCount === 0) {
          setSelectionError(
            scopeKind === "page"
              ? "No eligible numeric values remained after whole-page filtering."
              : "No values were detected in the selected section."
          );
        }
        return { status: "succeeded", summary };
      } catch (err) {
        if (isScanJobCancelledError(err)) {
          setScanOverlay(null);
          setSelectionError(
            options.sectionPosition
              ? sectionLabel +
                  " stopped. Earlier completed sections remain saved; no results from this section were added."
              : "Scan stopped. No balloons or review candidates were added."
          );
          return { status: "cancelled" };
        } else {
          const message =
            err instanceof Error
              ? err.message
              : "Auto-balloon scan failed unexpectedly";
          setSelectionError(message);
          return { status: "failed" };
        }
      } finally {
        setScanProgress(null);
        if (manageProcessing) setIsProcessing(false);
        setIsSegmenting(false);
      }
    },
    [
      addAnnotations,
      replaceScanReviewCandidates,
      setIsProcessing,
      setIsSegmenting,
    ]
  );

  const stopActiveScan = useCallback(async () => {
    const active = scanProgress;
    if (
      !active?.jobId ||
      !["queued", "running"].includes(active.status)
    ) {
      return;
    }

    setScanProgress((current) =>
      current?.jobId === active.jobId
        ? {
            ...current,
            status: "cancelling",
            stage: "cancelling",
            message: "Stopping after the current OCR operation",
            operationLabel: "Cancellation requested",
            liveness: "cancelling",
          }
        : current
    );
    try {
      const updated = await stopAutoBalloonScan(active.jobId);
      setScanProgress((current) =>
        current?.jobId === active.jobId ? updated : current
      );
    } catch (err) {
      setSelectionError(
        err instanceof Error ? err.message : "Could not stop the active scan"
      );
    }
  }, [scanProgress]);

  const toggleBalloons = useCallback(() => {
    setSelectedId(null);
    setAllBalloonVisibility(!balloonsVisible);
    void saveAnnotations(
      projectId,
      useAnnotationStore.getState().annotations
    );
  }, [balloonsVisible, projectId, setAllBalloonVisibility]);

  const finishSegmentBox = useCallback(
    (bbox: BBox) => {
      if (bbox.width < MIN_BOX || bbox.height < MIN_BOX) return;
      setSelectedScanSections((current) => [
        ...current,
        { id: uuidv4(), bbox, page: currentPage },
      ]);
    },
    [currentPage]
  );

  const cancelSectionSelection = useCallback(() => {
    drawStartRef.current = null;
    currentBoxRef.current = null;
    setCurrentBox(null);
    setSelectedScanSections([]);
    setIsSegmenting(false);
  }, [setIsSegmenting]);

  const startSelectedSectionScan = useCallback(async () => {
    const sections = [...selectedScanSections];
    if (sections.length === 0) return;

    setSelectedScanSections([]);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setIsProcessing(true);
    try {
      const result = await runSectionScanQueue({
        sections,
        runSection: (section, position) =>
          runAutoBalloon(section.bbox, "section", section.page, {
            manageProcessing: false,
            sectionPosition: position,
          }),
        afterSection: async (_section, _position, aggregate) => {
          setScanSummary(aggregate);
          // Yield after the atomic insert/save so this section's balloons are
          // visibly painted before the next OCR job starts.
          await waitForBrowserPaint();
        },
      });
      if (result.completedSections > 0) {
        setScanSummary(result.summary);
      }
    } finally {
      setIsProcessing(false);
    }
  }, [
    runAutoBalloon,
    selectedScanSections,
    setIsDrawingValue,
    setIsProcessing,
    setIsSegmenting,
  ]);

  const scanWholePage = useCallback(async () => {
    const source = sourceCanvasRef.current;
    if (!source) return;
    setSelectedScanSections([]);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    await runAutoBalloon(
      { x: 0, y: 0, width: source.width, height: source.height },
      "page",
      currentPage
    );
  }, [currentPage, runAutoBalloon, setIsDrawingValue, setIsSegmenting]
  );

  const openScanReviewCandidate = useCallback(
    (candidate: PageScanReviewCandidate) => {
      setSelectionError(null);
      setSelectedId(null);
      setSelectedReviewCandidateId(candidate.candidateId);
      if (candidate.page !== currentPage) setCurrentPage(candidate.page);
      setPending({
        bbox: candidate.valueBox,
        page: candidate.page,
        source: "scan_review",
        reviewCandidateId: candidate.candidateId,
        reviewReason:
          candidate.reviewReason ||
          "Recognition remained uncertain after bounded recovery.",
        ocrResult: {
          text: candidate.text,
          confidence: candidate.confidence,
          rotation: candidate.rotation,
          orientation: candidate.orientation,
          words: [],
          engine: "paddleocr",
          agreement: undefined,
          needsReview: true,
          valueBox: candidate.valueBox,
        },
      });
    },
    [currentPage, setCurrentPage, setPending]
  );

  const selectScanReviewCandidate = useCallback(
    (candidateId: string) => {
      const candidate = scanReviewCandidatesRef.current.find(
        (item) => item.candidateId === candidateId
      );
      if (candidate) openScanReviewCandidate(candidate);
    },
    [openScanReviewCandidate]
  );

  const resolveScanReviewCandidate = useCallback(
    (candidateId: string, action: "accepted" | "ignored") => {
      const resolved = scanReviewCandidatesRef.current.find(
        (candidate) => candidate.candidateId === candidateId
      );
      replaceScanReviewCandidates(
        scanReviewCandidatesRef.current.filter(
          (candidate) => candidate.candidateId !== candidateId
        )
      );
      setSelectedReviewCandidateId((current) =>
        current === candidateId ? null : current
      );
      if (resolved?.page === currentPage) {
        setScanSummary((summary) =>
          summary
            ? {
                ...summary,
                reviewRequired: Math.max(0, summary.reviewRequired - 1),
                added: summary.added + (action === "accepted" ? 1 : 0),
              }
            : null
        );
      }
    },
    [currentPage, replaceScanReviewCandidates]
  );

  const isCompletingDraw = useRef(false);

  const drawingActive = isDrawingValue || isSegmenting;

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!drawingActive) {
      if (
        e.target === stage ||
        e.target.name() === DRAWING_BACKGROUND_NAME
      ) {
        handleSelect(null);
      }
      return;
    }
    // Relative pointer position undoes the Stage zoom transform, so the box is
    // captured in base (unzoomed) canvas coords — the system bboxes are stored in.
    const pos = stage?.getRelativePointerPosition();
    if (!pos) return;
    drawStartRef.current = pos;
    const box = { x: pos.x, y: pos.y, width: 0, height: 0 };
    currentBoxRef.current = box;
    setCurrentBox(box);
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!drawStartRef.current || !drawingActive) return;
    const stage = e.target.getStage();
    const pos = stage?.getRelativePointerPosition();
    if (!pos) return;
    const box = normalizeBBox(drawStartRef.current, pos);
    currentBoxRef.current = box;
    setCurrentBox(box);
  };

  const completeDraw = useCallback(async () => {
    if (isCompletingDraw.current || !drawStartRef.current || !currentBoxRef.current) {
      return;
    }
    isCompletingDraw.current = true;
    const box = { ...currentBoxRef.current };
    const segmenting = isSegmenting;
    const drawingValue = isDrawingValue;
    drawStartRef.current = null;
    currentBoxRef.current = null;
    setCurrentBox(null);
    try {
      if (segmenting) {
        await finishSegmentBox(box);
      } else if (drawingValue) {
        await finishBox(box);
      }
    } finally {
      isCompletingDraw.current = false;
    }
  }, [
    finishBox,
    finishSegmentBox,
    isSegmenting,
    isDrawingValue,
  ]);

  useEffect(() => {
    if (!drawingActive) return;
    const onWindowMouseUp = () => {
      if (drawStartRef.current) void completeDraw();
    };
    window.addEventListener("mouseup", onWindowMouseUp);
    return () => window.removeEventListener("mouseup", onWindowMouseUp);
  }, [drawingActive, completeDraw]);

  return (
    <div className="flex h-screen flex-col bg-slate-100">
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf,image/png,image/jpeg,image/webp,image/tiff"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void handleFile(f);
        }}
      />

      <input
        ref={loadProjectInputRef}
        type="file"
        accept="application/json,.json,.docbox.json"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void handleLoadProject(f);
          e.target.value = "";
        }}
      />

      {!ocrReady && !ocrError && (
        <div className="bg-amber-50 px-4 py-1.5 text-center text-xs text-amber-800">
          {engineLabel}
        </div>
      )}
      {ocrReady && !ocrError && (
        <div className="bg-emerald-50 px-4 py-1.5 text-center text-xs text-emerald-800">
          Active OCR: {engineLabel}
        </div>
      )}
      {ocrError && (
        <div className="bg-red-50 px-4 py-1.5 text-center text-xs text-red-700">
          OCR failed to load: {ocrError}
        </div>
      )}
      {selectionError && (
        <div className="flex items-center justify-center gap-3 bg-amber-50 px-4 py-1.5 text-center text-xs text-amber-900">
          <span>{selectionError}</span>
          {scanOverlay && (
            <button
              type="button"
              onClick={() => setScanOverlay(null)}
              className="rounded border border-amber-400 px-2 py-0.5 font-medium hover:bg-amber-100"
            >
              Dismiss scan overlay
            </button>
          )}
        </div>
      )}
      {scanSummary && (
        <div
          className={`px-4 py-1.5 text-center text-xs ${
            scanSummary.reviewRequired > 0
              ? "bg-slate-100 text-slate-900"
              : "bg-emerald-50 text-emerald-900"
          }`}
        >
          {scanSummary.detected} detected
          {" · "}
          {scanSummary.recognized} recognized
          {" · "}
          {scanSummary.eligible} eligible
          {" · "}
          {scanSummary.excluded} excluded
          {" · "}
          {scanSummary.reviewRequired} review required
          {" · "}
          {scanSummary.unread} unread
          {" · "}
          {scanSummary.added} balloon{scanSummary.added === 1 ? "" : "s"} added
          {" · "}
          {scanSummary.skippedExisting} existing object
          {scanSummary.skippedExisting === 1 ? "" : "s"} skipped
          {scanSummary.skippedDuplicates > 0 && (
            <>
              {" · "}
              {scanSummary.skippedDuplicates} duplicate candidate
              {scanSummary.skippedDuplicates === 1 ? "" : "s"} skipped
            </>
          )}
        </div>
      )}
      {pageScanReviewCandidates.length > 0 && scanProgress === null && (
        <div className="bg-slate-100 px-4 py-1.5 text-center text-xs text-slate-700">
          Click a grey dashed box to correct and accept it, ignore it, or cancel
          and leave it for later.
        </div>
      )}
      {scanProgress && (
        <ScanProgressBanner
          progress={scanProgress}
          onStop={() => void stopActiveScan()}
        />
      )}
      {lastDebugDump && (
        <div className="bg-violet-50 px-4 py-1.5 text-center text-xs text-violet-900">
          OCR debug: {lastDebugDump}
        </div>
      )}
      {isDrawingValue && (
        <div className="flex items-center justify-center gap-3 bg-blue-600 px-4 py-2.5 text-center text-sm font-medium text-white">
          <span>
            ✏️ Drag a box around one value. OCR will read it and open the
            value/tolerance confirmation.
          </span>
          <button
            type="button"
            onClick={() => setIsDrawingValue(false)}
            className="rounded border border-white/60 px-2 py-0.5 font-medium text-white hover:bg-white/20"
          >
            Cancel
          </button>
        </div>
      )}
      {isSegmenting && !isProcessing && (
        <div className="flex flex-wrap items-center justify-center gap-2 bg-emerald-600 px-4 py-2.5 text-center text-sm font-medium text-white">
          <span>
            Draw sections in scan order. {selectedScanSections.length} selected.
          </span>
          <button
            type="button"
            onClick={() => void startSelectedSectionScan()}
            disabled={selectedScanSections.length === 0}
            className="rounded bg-white px-2 py-0.5 font-semibold text-emerald-700 hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Start Scan
          </button>
          <button
            type="button"
            onClick={() =>
              setSelectedScanSections((current) => current.slice(0, -1))
            }
            disabled={selectedScanSections.length === 0}
            className="rounded border border-white/60 px-2 py-0.5 font-medium text-white hover:bg-white/20 disabled:opacity-50"
          >
            Undo Last
          </button>
          <button
            type="button"
            onClick={() => setSelectedScanSections([])}
            disabled={selectedScanSections.length === 0}
            className="rounded border border-white/60 px-2 py-0.5 font-medium text-white hover:bg-white/20 disabled:opacity-50"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={cancelSectionSelection}
            className="rounded border border-white/60 px-2 py-0.5 font-medium text-white hover:bg-white/20"
          >
            Cancel
          </button>
        </div>
      )}

      <Toolbar
        isSelectingScanArea={isSegmenting}
        isScanRunning={scanProgress !== null}
        isDrawingValue={isDrawingValue}
        isProcessing={isProcessing}
        currentPage={currentPage}
        totalPages={totalPages}
        scale={scale}
        balloonsVisible={balloonsVisible}
        hasBalloons={valueAnnotations.length > 0}
        onSelectScanSection={() => {
          setSelectionError(null);
          setScanSummary(null);
          setSelectedScanSections([]);
          setIsDrawingValue(false);
          setIsSegmenting(true);
        }}
        onScanWholePage={() => void scanWholePage()}
        onToggleDrawValue={() => {
          setSelectionError(null);
          setScanSummary(null);
          setSelectedScanSections([]);
          setIsSegmenting(false);
          setIsDrawingValue(!isDrawingValue);
        }}
        onToggleBalloons={toggleBalloons}
        onZoomIn={() => setScale(Math.min(scale + 0.25, 4))}
        onZoomOut={() => setScale(Math.max(scale - 0.25, 0.5))}
        onPrevPage={() => setCurrentPage(Math.max(1, currentPage - 1))}
        onNextPage={() =>
          setCurrentPage(Math.min(totalPages, currentPage + 1))
        }
        onUpload={() => fileInputRef.current?.click()}
        onSaveProject={handleSaveProject}
        onLoadProject={() => loadProjectInputRef.current?.click()}
        onRemoveDrawing={() => void handleRemoveDrawing()}
        canSaveProject={konvaImage !== null}
      />

      <div className="flex min-h-0 flex-1">
        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <div
            className="flex-1 overflow-auto p-4"
            onMouseDown={(event) => {
              if (event.target === event.currentTarget && !drawingActive) {
                handleSelect(null);
              }
            }}
          >
            {!konvaImage ? (
              <div className="flex h-full min-h-[400px] flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-white text-slate-500">
                <p className="text-lg font-medium">No drawing loaded</p>
                <p className="mt-2 text-sm">
                  Open a PDF or image to start annotating dimensions.
                </p>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-4 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
                >
                  Open Drawing
                </button>
              </div>
            ) : (
              <div className="inline-block rounded-lg bg-white shadow-lg">
                <Stage
                  ref={stageRef}
                  width={stageSize.width * scale}
                  height={stageSize.height * scale}
                  scaleX={scale}
                  scaleY={scale}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  style={{
                    cursor: drawingActive ? "crosshair" : "default",
                  }}
                >
                  <Layer>
                    <KonvaImage
                      name={DRAWING_BACKGROUND_NAME}
                      image={konvaImage}
                      width={stageSize.width}
                      height={stageSize.height}
                      listening={!drawingActive}
                    />

                    {SCAN_DEBUG_OVERLAY_ENABLED && scanOverlay && (
                      <ScanDebugOverlay overlay={scanOverlay} scale={scale} />
                    )}

                    {pageScanReviewCandidates.length > 0 &&
                      scanProgress === null && (
                      <ScanReviewOverlay
                        candidates={pageScanReviewCandidates}
                        scale={scale}
                        disabled={drawingActive || isProcessing}
                        selectedCandidateId={selectedReviewCandidateId}
                        onSelect={openScanReviewCandidate}
                      />
                    )}

                    {visiblePageAnnotations.map((ann) => {
                      const highlighted = selectedId === ann.id;
                      const stroke = highlighted ? "#2563eb" : "#dc2626";
                      return (
                        <Rect
                          key={ann.id}
                          x={ann.bbox.x}
                          y={ann.bbox.y}
                          width={ann.bbox.width}
                          height={ann.bbox.height}
                          stroke={stroke}
                          // Divide by zoom so stroke + dash keep a constant
                          // on-screen size while the Stage scales the geometry.
                          strokeWidth={(highlighted ? 3 : 2) / scale}
                          dash={[6 / scale, 4 / scale]}
                          // When something is selected, fade the other values.
                          opacity={selectedId && !highlighted ? 0.15 : 1}
                          listening={!drawingActive}
                          onClick={() => handleSelect(ann.id)}
                        />
                      );
                    })}

                    {pageSelectedScanSections.map((section) => {
                      const order =
                        selectedScanSections.findIndex(
                          (candidate) => candidate.id === section.id
                        ) + 1;
                      return (
                        <Group key={section.id} listening={false}>
                          <Rect
                            {...section.bbox}
                            fill="#10b981"
                            opacity={0.08}
                            stroke="#059669"
                            strokeWidth={2.5 / scale}
                            dash={[8 / scale, 4 / scale]}
                          />
                          <Text
                            x={section.bbox.x + 4 / scale}
                            y={section.bbox.y + 3 / scale}
                            text={String(order)}
                            fill="#047857"
                            fontSize={14 / scale}
                            fontStyle="bold"
                          />
                        </Group>
                      );
                    })}

                    {currentBox && (
                      <Rect
                        x={currentBox.x}
                        y={currentBox.y}
                        width={currentBox.width}
                        height={currentBox.height}
                        stroke="#2563eb"
                        strokeWidth={2 / scale}
                        dash={[4 / scale, 4 / scale]}
                      />
                    )}

                    {visiblePageAnnotations.map((ann) => (
                      <Balloon
                        key={`balloon-${ann.id}`}
                        annotation={ann}
                        scale={scale}
                        selected={selectedId === ann.id}
                        dimmed={!!selectedId && selectedId !== ann.id}
                        listening={!drawingActive}
                        onSelect={handleSelect}
                      />
                    ))}
                  </Layer>
                </Stage>
              </div>
            )}
          </div>
        </div>

        <Sidebar
          annotations={annotations}
          reviewCandidates={scanReviewCandidates}
          disabled={isSegmenting || isProcessing}
          selectedId={selectedId}
          selectedReviewCandidateId={selectedReviewCandidateId}
          onSelect={(id) => handleSelect(id)}
          onSelectReview={selectScanReviewCandidate}
          onDelete={handleDelete}
          onToggleVisibility={toggleAnnotationVisibility}
          onEdit={(id) => {
            handleSelect(id);
            setEditingValueId(id);
          }}
        />
      </div>

      <AnnotationPopup
        onReviewResolved={resolveScanReviewCandidate}
        onReviewCancelled={(candidateId) =>
          setSelectedReviewCandidateId((current) =>
            current === candidateId ? null : current
          )
        }
      />

      {(() => {
        const editingValue = annotations.find(
          (annotation) =>
            annotation.id === editingValueId && annotation.kind !== "label"
        );
        if (!editingValue) return null;
        return (
          <ValueEditor
            annotation={editingValue}
            onUpdate={updateAnnotation}
            onDelete={handleDelete}
            onClose={() => setEditingValueId(null)}
          />
        );
      })()}
    </div>
  );
}
