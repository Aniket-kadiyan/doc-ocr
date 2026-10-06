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
import { useDocumentMetadataStore } from "@/store/documentMetadataStore";
import { normalizeBBox } from "@/lib/canvasUtils";
import {
  isScanJobCancelledError,
  preloadOcr,
  runAutoBalloonScan,
  runPageTitleFields,
  runOCR,
  stopAutoBalloonScan,
} from "@/lib/clientOcr";
import { classifyDimension } from "@/lib/dimensionClassifier";
import { filterNewScanRegions } from "@/lib/scanCandidates";
import {
  ignoreScanCandidate,
  mergeScanCandidates,
  restoreScanCandidate,
} from "@/lib/scanCandidateLifecycle";
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
  PDF_RENDER_SCALE,
} from "@/lib/pdfLoader";
import { detectSourceFileKind, isPdfKind } from "@/lib/sourceFile";
import {
  saveAnnotations,
  loadAnnotations,
  saveScanCandidates,
  loadScanCandidates,
  saveProject,
  saveProjectMetadata,
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
import {
  mergeTitleAnnotationsIntoMetadata,
  metadataFieldForTitleLabel,
} from "@/lib/titleMetadata";
import type { Annotation, BBox, DimensionType } from "@/types/annotation";
import type { ScanCandidate } from "@/types/scanCandidate";
import {
  DOCUMENT_METADATA_FIELDS,
  normalizeDocumentMetadata,
  type DocumentMetadataField,
} from "@/types/documentMetadata";
import type { SegmentCandidateOutcome } from "@/lib/paddleOcrClient";
import type {
  ScanCompletionSummary,
  ScanDebugOverlay as ScanDebugOverlayModel,
  ScanProgress,
  ScanScopeKind,
} from "@/types/scanJob";
import type { PDFDocumentProxy } from "pdfjs-dist";

/**
 * Scale the page is re-rendered at for scanning, independent of the viewer's
 * display scale. The display scale is chosen for a screen; OCR wants pixels.
 * Measured on the benchmark drawings, reading the displayed canvas rather than
 * a 250dpi render costs a match or two per sheet (47630 16/19 -> 15/19,
 * BS1801006.020 11/12 -> 10/12) and loses slanted callouts outright: "0.5x45°"
 * came back as "6". Coordinates map back through displayScale, which exists
 * for exactly this, so annotations still land in viewer space.
 *
 * 250dpi specifically: it is what the accuracy fixtures render at and it
 * measured best of the scales tried. 216dpi was uneven — better on two
 * drawings, worse on 56103-0182B (14/24 -> 12/24).
 */
const OCR_RENDER_SCALE = 250 / 72;

const MIN_BOX = 8;
const DRAWING_BACKGROUND_NAME = "drawing-background";

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

export function DrawingViewer() {
  const { ready: ocrReady, error: ocrError, engineLabel } = useClientOcr();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadProjectInputRef = useRef<HTMLInputElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  /** Original drawing bytes, kept so a saved project can embed them. */
  const sourceFileRef = useRef<ProjectSource | null>(null);
  /** Original PDF file supplied to native-text evidence during page scans. */
  const sourceDocumentRef = useRef<File | null>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const projectIdRef = useRef("");

  const [konvaImage, setKonvaImage] = useState<HTMLImageElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [rasterFile, setRasterFile] = useState<File | null>(null);
  const [projectId, setProjectId] = useState(() => uuidv4());
  const restoredRef = useRef(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedScanCandidateId, setSelectedScanCandidateId] = useState<
    string | null
  >(null);
  const [currentBox, setCurrentBox] = useState<BBox | null>(null);
  const [activeMetadataField, setActiveMetadataField] =
    useState<DocumentMetadataField | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [lastDebugDump, setLastDebugDump] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null);
  const [scanOverlay, setScanOverlay] =
    useState<ScanDebugOverlayModel | null>(null);
  const [scanSummary, setScanSummary] =
    useState<ScanCompletionSummary | null>(null);
  const [scanCandidates, setScanCandidates] = useState<ScanCandidate[]>([]);
  const scanCandidatesRef = useRef<ScanCandidate[]>([]);
  const candidateOrderRef = useRef(0);
  const [selectedScanSections, setSelectedScanSections] = useState<
    QueuedScanSection[]
  >([]);
  const drawStartRef = useRef<{ x: number; y: number } | null>(null);
  const currentBoxRef = useRef<BBox | null>(null);

  const annotations = useAnnotationStore((s) => s.annotations);
  const setAnnotations = useAnnotationStore((s) => s.setAnnotations);
  const addAnnotations = useAnnotationStore((s) => s.addAnnotations);
  const updateAnnotation = useAnnotationStore((s) => s.updateAnnotation);
  const moveAnnotationToNumber = useAnnotationStore(
    (s) => s.moveAnnotationToNumber
  );
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
  const metadata = useDocumentMetadataStore((s) => s.metadata);
  const setDocumentMetadata = useDocumentMetadataStore((s) => s.setMetadata);
  const updateDocumentMetadataField = useDocumentMetadataStore(
    (s) => s.updateField
  );
  const resetDocumentMetadata = useDocumentMetadataStore(
    (s) => s.resetMetadata
  );
  const setPageCanvasProvider = useAnnotationStore(
    (s) => s.setPageCanvasProvider
  );

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
  const pageScanCandidates = scanCandidates.filter(
    (candidate) =>
      candidate.page === currentPage && candidate.state !== "ignored"
  );
  const pageSelectedScanSections = selectedScanSections.filter(
    (section) => section.page === currentPage
  );
  const activeMetadataLabel = DOCUMENT_METADATA_FIELDS.find(
    ({ key }) => key === activeMetadataField
  )?.label;

  const replaceScanCandidates = useCallback(
    (next: ScanCandidate[]) => {
      scanCandidatesRef.current = next;
      setScanCandidates(next);
    },
    []
  );

  const restoreScanCandidates = useCallback(
    (next: ScanCandidate[]) => {
      replaceScanCandidates(next);
      candidateOrderRef.current =
        next.reduce(
          (largest, candidate) => Math.max(largest, candidate.order + 1),
          0
        );
    },
    [replaceScanCandidates]
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

  const renderCurrentImagePage = useCallback(
    async (file: File, page: number) => {
      const { canvas, width, height } = await loadImageFile(file, page);
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
    if (!rasterFile) return;
    void renderCurrentImagePage(rasterFile, currentPage).catch((reason) => {
      setSelectionError(
        reason instanceof Error
          ? reason.message
          : "Could not render the selected image page."
      );
    });
  }, [currentPage, rasterFile, renderCurrentImagePage]);

  useEffect(() => {
    // Overlay geometry belongs to one exact page/project and is never persisted.
    setScanOverlay(null);
  }, [currentPage, projectId]);

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

  // Hand the Export menu a way to re-render any page. The ballooned PDF/PNG
  // exports draw balloons over the drawing, so they need the page bitmap
  // without owning the PDF document themselves.
  const renderPageCanvas = useCallback(
    async (page: number): Promise<HTMLCanvasElement | null> => {
      if (pdfDoc) {
        const { canvas } = await renderPdfPage(pdfDoc, page, PDF_RENDER_SCALE);
        return canvas;
      }
      if (rasterFile) {
        const { canvas } = await loadImageFile(rasterFile, page);
        return canvas;
      }
      return null;
    },
    [pdfDoc, rasterFile]
  );

  useEffect(() => {
    setPageCanvasProvider(konvaImage ? renderPageCanvas : null);
    return () => setPageCanvasProvider(null);
  }, [konvaImage, renderPageCanvas, setPageCanvasProvider]);

  // Expose the current project id so Export can retrieve the original drawing
  // blob and create a durable backend checksheet snapshot.
  useEffect(() => {
    setStoreProjectId(projectId);
    projectIdRef.current = projectId;
  }, [projectId, setStoreProjectId]);

  // Render a drawing onto the stage. Shared by fresh uploads and project loads
  // so both paths render at the same scale and produce matching bbox coords.
  const loadSource = useCallback(
    async (file: File) => {
      const sourceKind = await detectSourceFileKind(file);
      const isPdf = isPdfKind(sourceKind);
      sourceDocumentRef.current = isPdf ? file : null;
      if (isPdf) {
        const doc = await loadPdfDocument(file);
        setRasterFile(null);
        setPdfDoc(doc);
        setTotalPages(doc.numPages);
        setCurrentPage(1);
        setScale(1);
        await renderCurrentPage(doc, 1);
      } else {
        const { canvas, width, height, totalPages } = await loadImageFile(file, 1);
        sourceCanvasRef.current = canvas;
        setPdfDoc(null);
        setRasterFile(file);
        setTotalPages(totalPages);
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
    const [storedAnnotations, storedCandidates] = await Promise.all([
      loadAnnotations(recent.id),
      loadScanCandidates(recent.id),
    ]);
    const legacyMigration = normalizeLegacyAnnotations(storedAnnotations);
    const migration = mergeTitleAnnotationsIntoMetadata(
      legacyMigration.annotations,
      recent.metadata
    );
    setDocumentMetadata(migration.metadata);
    setAnnotations(migration.annotations);
    restoreScanCandidates(storedCandidates);
    await Promise.all([
      saveAnnotations(recent.id, migration.annotations),
      saveProjectMetadata(recent.id, migration.metadata),
    ]);
    const warning = legacyMigrationWarning(legacyMigration.orphanLabelCount);
    if (warning) setSelectionError(warning);
  }, [
    loadSource,
    restoreScanCandidates,
    setAnnotations,
    setDocumentMetadata,
    setProjectName,
  ]);

  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    void restoreLastProject();
  }, [restoreLastProject]);

  const handleFile = async (file: File) => {
    const sourceKind = await detectSourceFileKind(file);
    // A new drawing is a new project — fresh id, no stale annotations.
    const id = uuidv4();
    setProjectId(id);
    setAnnotations([]);
    setSelectedId(null);
    replaceScanCandidates([]);
    candidateOrderRef.current = 0;
    setSelectedScanCandidateId(null);
    setSelectedScanSections([]);
    setEditingValueId(null);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setActiveMetadataField(null);
    resetDocumentMetadata();
    setScanProgress(null);
    setScanOverlay(null);
    setScanSummary(null);
    setProjectName(file.name.replace(/\.[^.]+$/, ""));
    const fileType = isPdfKind(sourceKind) ? "pdf" : "image";
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
      metadata: normalizeDocumentMetadata(),
      updatedAt: Date.now(),
    });
    await loadSource(file);
  };

  // Save the drawing, accepted balloons, and unresolved candidate lifecycle as
  // one portable `.docbox.json` the app can reload later.
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
      scanCandidates,
      metadata,
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
    setRasterFile(null);
    sourceCanvasRef.current = null;
    sourceFileRef.current = null;
    sourceDocumentRef.current = null;
    setAnnotations([]);
    setSelectedId(null);
    replaceScanCandidates([]);
    candidateOrderRef.current = 0;
    setSelectedScanCandidateId(null);
    setSelectedScanSections([]);
    setEditingValueId(null);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setActiveMetadataField(null);
    resetDocumentMetadata();
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
    setSelectedScanCandidateId(null);
    replaceScanCandidates([]);
    candidateOrderRef.current = 0;
    setSelectedScanSections([]);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setActiveMetadataField(null);
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
      const legacyMigration = normalizeLegacyAnnotations(bundle.annotations);
      const migration = mergeTitleAnnotationsIntoMetadata(
        legacyMigration.annotations,
        bundle.metadata
      );
      // Importing the same portable project twice creates two local projects.
      // Give candidate rows fresh local keys so Dexie's global primary key
      // cannot move the first project's audit records into the second one.
      const importedCandidates = bundle.scanCandidates.map((candidate) => ({
        ...candidate,
        id: `${id}:${uuidv4()}`,
      }));
      setDocumentMetadata(migration.metadata);
      setAnnotations(migration.annotations);
      restoreScanCandidates(importedCandidates);
      const warning = legacyMigrationWarning(legacyMigration.orphanLabelCount);
      if (warning) setSelectionError(warning);
      // Persist so the loaded project reopens on the next visit too.
      await saveProject({
        id,
        name: bundle.projectName,
        fileName: bundle.source.fileName,
        fileType: bundle.source.fileType,
        mimeType: bundle.source.mimeType,
        fileBlob: srcFile,
        metadata: migration.metadata,
        updatedAt: Date.now(),
      });
      await Promise.all([
        saveAnnotations(id, migration.annotations),
        saveScanCandidates(id, importedCandidates),
      ]);
    } catch (err) {
      setSelectionError(
        err instanceof Error ? err.message : "Could not open project file."
      );
    }
  };

  // Select an annotation and follow it to its page so it's actually on screen.
  const handleSelect = useCallback(
    (id: string | null) => {
      setSelectedScanCandidateId(null);
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

  const handleMoveAnnotation = useCallback(
    (id: string, targetNumber: number) => {
      moveAnnotationToNumber(id, targetNumber);
      void saveAnnotations(
        projectId,
        useAnnotationStore.getState().annotations
      );
    },
    [moveAnnotationToNumber, projectId]
  );

  const handleMetadataChange = useCallback(
    (field: DocumentMetadataField, value: string) => {
      const next = {
        ...useDocumentMetadataStore.getState().metadata,
        [field]: value,
      };
      updateDocumentMetadataField(field, value);
      void saveProjectMetadata(projectId, next).catch(() => {
        setSelectionError("Could not save the document metadata locally.");
      });
    },
    [projectId, updateDocumentMetadataField]
  );

  const handleMetadataSelect = useCallback(
    (field: DocumentMetadataField) => {
      if (activeMetadataField === field) {
        setActiveMetadataField(null);
        return;
      }
      setSelectionError(null);
      setScanSummary(null);
      setSelectedScanSections([]);
      setSelectedId(null);
      setSelectedScanCandidateId(null);
      setPending(null);
      setIsDrawingValue(false);
      setIsSegmenting(false);
      setActiveMetadataField(field);
    }, [activeMetadataField, setIsDrawingValue, setIsSegmenting, setPending]);

  const handleMetadataClear = useCallback(
    (field: DocumentMetadataField) => {
      if (activeMetadataField === field) setActiveMetadataField(null);
      handleMetadataChange(field, "");
    },
    [activeMetadataField, handleMetadataChange]
  );

  const finishMetadataBox = useCallback(
    async (bbox: BBox, field: DocumentMetadataField) => {
      if (bbox.width < MIN_BOX || bbox.height < MIN_BOX) return;
      const source = sourceCanvasRef.current;
      if (!source) {
        setActiveMetadataField(null);
        return;
      }

      setActiveMetadataField(null);
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
        const detectedValue = ocrResult.text.trim().replace(/\s+/g, " ");
        if (!detectedValue) {
          setSelectionError(
            "No metadata text detected — select the field and draw a tighter box, or enter the value manually."
          );
          return;
        }
        handleMetadataChange(field, detectedValue);
      } catch (err) {
        setSelectionError(
          err instanceof Error ? err.message : "Metadata OCR failed unexpectedly"
        );
      } finally {
        setIsProcessing(false);
        setActiveMetadataField(null);
      }
    },
    [handleMetadataChange, setIsProcessing]
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
          suggestedType: ocrResult.category as DimensionType | undefined,
          suggestedSubtype: ocrResult.subtype,
          suggestedLabel: ocrResult.label,
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

      // Scan a higher-resolution render than the one on screen. Falls back to
      // the displayed canvas for an image file, or if the re-render fails.
      let scanCanvas = source;
      let scanScale = 1;
      if (pdfDoc) {
        try {
          const { canvas: hi } = await renderPdfPage(
            pdfDoc,
            scanPage,
            OCR_RENDER_SCALE
          );
          scanCanvas = hi;
          scanScale = OCR_RENDER_SCALE / PDF_RENDER_SCALE;
        } catch {
          scanCanvas = source;
          scanScale = 1;
        }
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
          sourceCanvas: scanCanvas,
          sourceDocument: sourceDocumentRef.current ?? undefined,
          bbox,
          page: scanPage,
          scopeKind,
          displayScale: scanScale,
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
            // Prefer the backend rule engine, which also sees the detected
            // symbols and geometry; fall back to the local text-only rules.
            const type = ((r.category as DimensionType) ||
              classifyDimension(cleanValue)) as DimensionType;
            return {
              id: uuidv4(),
              // number is assigned sequentially by addAnnotations
              number: 0,
              label: r.label?.trim() || "",
              value: cleanValue,
              type,
              subtype: r.subtype,
              confidence: r.confidence,
              bbox: r.valueBox,
              rotation: r.rotation,
              // Slanted callouts draw along their leader line; the loose
              // axis-aligned valueBox stays the anchor for overlap tests.
              orientedBox: r.orientedBox,
              page: scanPage,
              createdAt: now,
              kind: "dimension",
              needsReview: r.needsReview || !r.recognized,
              recognitionEvidence: r.recognitionEvidence,
              range: deriveRange(cleanValue) || undefined,
            };
          });

        // Keep every unaccepted detector outcome outside the annotation store.
        // Review and filtered "Other" objects share one durable lifecycle;
        // geometry-equivalent rescans merge into an audit trail rather than
        // disappearing silently.
        const toScanCandidate = (
          candidate: SegmentCandidateOutcome,
          state: "review" | "other"
        ): ScanCandidate => ({
          id: [
            projectAtStart,
            scanPage,
            scanRunId,
            candidate.candidateId ?? uuidv4(),
          ].join(":"),
          sourceCandidateId: candidate.candidateId,
          page: scanPage,
          order: candidateOrderRef.current++,
          state,
          text: candidate.text,
          rawText: candidate.rawText,
          preliminaryText: candidate.preliminaryText,
          confidence: candidate.confidence,
          recognized: candidate.recognized,
          reason:
            candidate.outcomeReason ||
            candidate.reviewReason ||
            candidate.pageFilterReason ||
            "Automatic scan did not accept this object.",
          rule: candidate.outcomeRule || candidate.pageFilterRule,
          type: candidate.type,
          category: candidate.category,
          subtype: candidate.subtype,
          label: candidate.label,
          orientation: candidate.orientation,
          rotation: candidate.rotation,
          recoveryAttempted: candidate.recoveryAttempted,
          authoritativeReread: candidate.authoritativeReread,
          recognitionEvidence: candidate.recognitionEvidence,
          valueBox: candidate.valueBox,
          orientedBox: candidate.orientedBox,
          createdAt: now,
          updatedAt: now,
        });
        const incomingCandidates = [
          ...scanResult.reviewCandidates.map((candidate) =>
            toScanCandidate(candidate, "review")
          ),
          ...scanResult.otherCandidates.map((candidate) =>
            toScanCandidate(candidate, "other")
          ),
        ].sort(
          (left, right) =>
            left.valueBox.y - right.valueBox.y ||
            left.valueBox.x - right.valueBox.x
        );
        const acceptedAnnotations = [...liveAnnotations, ...newAnnotations];
        const mergedCandidates = mergeScanCandidates({
          existing: scanCandidatesRef.current,
          incoming: incomingCandidates,
          acceptedAnnotations,
        });
        const incomingCandidateIds = new Set(
          incomingCandidates.map((candidate) => candidate.id)
        );
        const addedCandidateCount = mergedCandidates.candidates.filter(
          (candidate) => incomingCandidateIds.has(candidate.id)
        ).length;
        const addedReviewCount = mergedCandidates.candidates.filter(
          (candidate) =>
            incomingCandidateIds.has(candidate.id) &&
            candidate.state === "review"
        ).length;

        // Title-block fields ("DWG NO.", "REV") are read from the WHOLE page,
        // not this scan's scope: the title block sits at a fixed spot on the
        // sheet, so which fields turn up must not depend on the section being
        // scanned. Fields already present are skipped, so running more than one
        // section — or re-scanning — never duplicates them.
        if (scopeKind === "page") {
          const existingFields = new Set(
            useAnnotationStore
              .getState()
              .annotations.filter((a) => a.type === "Title Block")
              .map((a) => (a.label ?? "").trim().toUpperCase())
          );
          try {
            const fields = await runPageTitleFields(source);
            const nextMetadata = {
              ...useDocumentMetadataStore.getState().metadata,
            };
            let metadataChanged = false;
            for (const f of fields) {
              if (!f.value.trim()) continue;
              const metadataField = metadataFieldForTitleLabel(f.label);
              if (metadataField) {
                if (!nextMetadata[metadataField].trim()) {
                  nextMetadata[metadataField] = f.value.trim();
                  metadataChanged = true;
                }
                continue;
              }
              if (!f.bbox) continue;
              if (existingFields.has(f.label.trim().toUpperCase())) continue;
              newAnnotations.push({
                id: uuidv4(),
                number: 0,
                label: f.label,
                value: f.value.trim(),
                type: "Title Block" as DimensionType,
                confidence: f.confidence,
                bbox: f.bbox,
                rotation: 0,
                page: scanPage,
                createdAt: now,
                kind: "dimension",
              });
            }
            if (metadataChanged) {
              setDocumentMetadata(nextMetadata);
              await saveProjectMetadata(projectAtStart, nextMetadata);
            }
          } catch {
            // The balloons are the point of the scan; a failed title-block read
            // must not throw them away. The export still lists every configured
            // keyword, with an empty value to fill in by hand.
          }
        }

        if (newAnnotations.length > 0) {
          // Atomic per-section commit: save these balloons before the queue is
          // allowed to start its next section.
          addAnnotations(newAnnotations);
          await saveAnnotations(
            projectAtStart,
            useAnnotationStore.getState().annotations
          );
        }
        replaceScanCandidates(mergedCandidates.candidates);
        await saveScanCandidates(projectAtStart, mergedCandidates.candidates);
        setScanOverlay(null);
        const summary: ScanCompletionSummary = {
          scopeKind,
          added: newAnnotations.length,
          detected: scanResult.detected,
          recognized: scanResult.recognized,
          eligible: scanResult.eligible,
          excluded: scanResult.excluded,
          reviewRequired: addedReviewCount,
          unread: scanResult.unread,
          skippedExisting:
            scanResult.skippedExisting +
            filtered.skippedExisting +
            mergedCandidates.skippedAccepted,
          skippedDuplicates:
            filtered.skippedDuplicates +
            mergedCandidates.mergedDuplicates,
        };
        setScanSummary(summary);
        if (newAnnotations.length === 0 && addedCandidateCount === 0) {
          setSelectionError(
            scopeKind === "page"
              ? "No new accepted or unresolved detections were found."
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
      replaceScanCandidates,
      setIsProcessing,
      setIsSegmenting,
      pdfDoc,
      setDocumentMetadata,
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
    setActiveMetadataField(null);
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
    setActiveMetadataField(null);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    await runAutoBalloon(
      { x: 0, y: 0, width: source.width, height: source.height },
      "page",
      currentPage
    );
  }, [currentPage, runAutoBalloon, setIsDrawingValue, setIsSegmenting]
  );

  const openScanCandidate = useCallback(
    (candidate: ScanCandidate) => {
      setSelectionError(null);
      setSelectedId(null);
      setSelectedScanCandidateId(candidate.id);
      if (candidate.page !== currentPage) setCurrentPage(candidate.page);
      setPending({
        bbox: candidate.valueBox,
        page: candidate.page,
        source: "scan_candidate",
        reviewCandidateId: candidate.id,
        reviewReason: candidate.reason || "Review this detected object.",
        scanCandidateState:
          candidate.state === "review" ? "review" : "other",
        suggestedType: candidate.category as DimensionType | undefined,
        suggestedSubtype: candidate.subtype,
        suggestedLabel: candidate.label,
        orientedBox: candidate.orientedBox,
        ocrResult: {
          text: candidate.text,
          confidence: candidate.confidence,
          rotation: candidate.rotation,
          orientation: candidate.orientation,
          words: [],
          engine: "paddleocr",
          agreement: undefined,
          needsReview: candidate.state === "review",
          category: candidate.category,
          subtype: candidate.subtype,
          label: candidate.label,
          valueBox: candidate.valueBox,
          recognitionEvidence: candidate.recognitionEvidence,
        },
      });
    },
    [currentPage, setCurrentPage, setPending]
  );

  const selectScanCandidate = useCallback(
    (candidateId: string) => {
      const candidate = scanCandidatesRef.current.find(
        (item) => item.id === candidateId
      );
      if (candidate && candidate.state !== "ignored") {
        openScanCandidate(candidate);
      }
    },
    [openScanCandidate]
  );

  const resolveScanCandidate = useCallback(
    (candidateId: string, action: "accepted" | "ignored") => {
      const resolved = scanCandidatesRef.current.find(
        (candidate) => candidate.id === candidateId
      );
      const next =
        action === "accepted"
          ? scanCandidatesRef.current.filter(
              (candidate) => candidate.id !== candidateId
            )
          : ignoreScanCandidate(scanCandidatesRef.current, candidateId);
      replaceScanCandidates(next);
      void saveScanCandidates(projectIdRef.current, next).catch(() => {
        setSelectionError("Could not save the candidate decision locally.");
      });
      setSelectedScanCandidateId((current) =>
        current === candidateId ? null : current
      );
      if (resolved?.page === currentPage) {
        setScanSummary((summary) =>
          summary
            ? {
                ...summary,
                reviewRequired:
                  resolved.state === "review"
                    ? Math.max(0, summary.reviewRequired - 1)
                    : summary.reviewRequired,
                added: summary.added + (action === "accepted" ? 1 : 0),
              }
            : null
        );
      }
    },
    [currentPage, replaceScanCandidates]
  );

  const handleRestoreScanCandidate = useCallback(
    (candidateId: string) => {
      const restored = scanCandidatesRef.current.find(
        (candidate) => candidate.id === candidateId
      );
      const next = restoreScanCandidate(
        scanCandidatesRef.current,
        candidateId
      );
      replaceScanCandidates(next);
      void saveScanCandidates(projectIdRef.current, next).catch(() => {
        setSelectionError("Could not restore the candidate locally.");
      });
      if (
        restored?.page === currentPage &&
        restored.restoreState === "review"
      ) {
        setScanSummary((summary) =>
          summary
            ? { ...summary, reviewRequired: summary.reviewRequired + 1 }
            : null
        );
      }
    },
    [currentPage, replaceScanCandidates]
  );

  const isCompletingDraw = useRef(false);

  const drawingActive =
    isDrawingValue || isSegmenting || activeMetadataField !== null;

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
    const metadataField = activeMetadataField;
    drawStartRef.current = null;
    currentBoxRef.current = null;
    setCurrentBox(null);
    try {
      if (segmenting) {
        await finishSegmentBox(box);
      } else if (metadataField) {
        await finishMetadataBox(box, metadataField);
      } else if (drawingValue) {
        await finishBox(box);
      }
    } finally {
      isCompletingDraw.current = false;
    }
  }, [
    activeMetadataField,
    finishBox,
    finishMetadataBox,
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
        accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp,.bmp,application/pdf,image/png,image/jpeg,image/tiff,image/webp,image/bmp"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) {
            void handleFile(f).catch((reason) => {
              setSelectionError(
                reason instanceof Error
                  ? reason.message
                  : "Could not open the selected drawing."
              );
            });
          }
          e.target.value = "";
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
          {scanSummary.excluded} other detection
          {scanSummary.excluded === 1 ? "" : "s"}
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
              {scanSummary.skippedDuplicates === 1 ? "" : "s"} merged
            </>
          )}
        </div>
      )}
      {pageScanCandidates.length > 0 && scanProgress === null && (
        <div className="bg-slate-100 px-4 py-1.5 text-center text-xs text-slate-700">
          Amber boxes need review. Muted boxes are filtered detections retained
          under Other. Click either to accept, edit, ignore, or leave it for later.
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
      {activeMetadataField && (
        <div className="flex items-center justify-center gap-3 bg-violet-600 px-4 py-2.5 text-center text-sm font-medium text-white">
          <span>
            Drag a box around the {activeMetadataLabel}. OCR will fill the
            metadata field.
          </span>
          <button
            type="button"
            onClick={() => setActiveMetadataField(null)}
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
          setActiveMetadataField(null);
          setIsDrawingValue(false);
          setIsSegmenting(true);
        }}
        onScanWholePage={() => void scanWholePage()}
        onToggleDrawValue={() => {
          setSelectionError(null);
          setScanSummary(null);
          setSelectedScanSections([]);
          setActiveMetadataField(null);
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
        scanCandidates={scanCandidates}
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

                    {visiblePageAnnotations.map((ann) => {
                      const highlighted = selectedId === ann.id;
                      const stroke = highlighted ? "#2563eb" : "#dc2626";
                      // Slanted callouts carry a tight rotated rectangle; draw
                      // that (Konva rotates about x,y clockwise) instead of the
                      // loose axis-aligned bbox so the box hugs the diagonal text.
                      const box = ann.orientedBox ?? ann.bbox;
                      const boxRotation = ann.orientedBox?.rotation ?? 0;
                      return (
                        <Rect
                          key={ann.id}
                          x={box.x}
                          y={box.y}
                          width={box.width}
                          height={box.height}
                          rotation={boxRotation}
                          stroke={stroke}
                          // Divide by zoom so stroke + dash keep a constant
                          // on-screen size while the Stage scales the geometry.
                          strokeWidth={(highlighted ? 3 : 2) / scale}
                          dash={[6 / scale, 4 / scale]}
                          // When something is selected, fade the other values.
                          opacity={
                            (selectedId && !highlighted) ||
                            selectedScanCandidateId !== null
                              ? 0.15
                              : 1
                          }
                          listening={!drawingActive}
                          onClick={() => handleSelect(ann.id)}
                        />
                      );
                    })}

                    {pageScanCandidates.length > 0 &&
                      scanProgress === null && (
                      <ScanReviewOverlay
                        candidates={pageScanCandidates}
                        scale={scale}
                        disabled={drawingActive || isProcessing}
                        selectedCandidateId={selectedScanCandidateId}
                        onSelect={openScanCandidate}
                      />
                    )}

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
                        dimmed={
                          (!!selectedId && selectedId !== ann.id) ||
                          selectedScanCandidateId !== null
                        }
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
          scanCandidates={scanCandidates}
          metadata={metadata}
          activeMetadataField={activeMetadataField}
          drawingAvailable={konvaImage !== null}
          disabled={isSegmenting || isProcessing || konvaImage === null}
          selectedId={selectedId}
          selectedScanCandidateId={selectedScanCandidateId}
          onSelect={(id) => handleSelect(id)}
          onSelectScanCandidate={selectScanCandidate}
          onRestoreScanCandidate={handleRestoreScanCandidate}
          onDelete={handleDelete}
          onMove={handleMoveAnnotation}
          onToggleVisibility={toggleAnnotationVisibility}
          onMetadataChange={handleMetadataChange}
          onMetadataSelect={handleMetadataSelect}
          onMetadataClear={handleMetadataClear}
          onEdit={(id) => {
            handleSelect(id);
            setEditingValueId(id);
          }}
        />
      </div>

      <AnnotationPopup
        onReviewResolved={resolveScanCandidate}
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
