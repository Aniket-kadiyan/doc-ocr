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
  abandonAutoBalloonScan,
  isScanJobCancelledError,
  preloadOcr,
  runAutoBalloonScan,
  runPageReferencePoints,
  runPageTitleFields,
  runOCR,
  stopAutoBalloonScan,
} from "@/lib/clientOcr";
import { filterNewScanRegions } from "@/lib/scanCandidates";
import {
  ignoreScanCandidate,
  mergeScanCandidates,
  restoreScanCandidate,
} from "@/lib/scanCandidateLifecycle";
import {
  areAllValueBalloonsHidden,
  areAllValueBalloonsVisible,
} from "@/lib/balloonVisibility";
import {
  runSectionScanQueue,
  type QueuedScanSection,
  type ScanRunOutcome,
  type SectionQueuePosition,
} from "@/lib/sectionScanQueue";
import {
  completedReferencePages,
  referenceScanNotices,
  referenceTablesAfterPageResult,
  runReferenceScanQueue,
} from "@/lib/referenceScanQueue";
import { normalizePageReferenceTables } from "@/lib/referenceValues";
import { annotationsFromRegions } from "@/lib/scanAnnotations";
import { readPdfTextLayer, regionsWithin } from "@/lib/pdfTextLayer";
import { titleFieldAnnotations } from "@/lib/labelRules";
import { useClientOcr } from "@/hooks/useClientOcr";
import {
  loadPdfDocument,
  renderPdfPage,
  loadImageFile,
  openableDrawing,
  PDF_RENDER_SCALE,
  displayRenderScale,
} from "@/lib/pdfLoader";
import { detectSourceFileKind, isPdfKind } from "@/lib/sourceFile";
import {
  saveAnnotations,
  loadAnnotations,
  saveScanCandidates,
  loadScanCandidates,
  saveProject,
  saveProjectMetadata,
  saveProjectReferenceTables,
  saveProjectBalloonStyle,
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
import {
  ReferenceMarker,
  referenceTagSize,
} from "@/components/ReferenceMarker";
import { layoutReferenceTags } from "@/lib/referenceTagLayout";
import { ReferencePanel } from "@/components/ReferencePanel";
import { referenceColor } from "@/lib/referenceColors";
import { Toolbar } from "@/components/Toolbar";
import { Sidebar } from "@/components/Sidebar";
import { AnnotationPopup } from "@/components/AnnotationPopup";
import { ValueEditor } from "@/components/ValueEditor";
import { ScanProgressBanner } from "@/components/ScanProgressBanner";
import { GdtOverlay } from "@/components/GdtOverlay";
import { GdtPanel } from "@/components/GdtPanel";
import {
  fetchGdtRegistry,
  recognizeGdtSymbols,
  scaleResult,
} from "@/lib/gdtClient";
import { framesForPage, overlayItems } from "@/lib/gdtSymbols";
import type { GdtPageResult, GdtRegistry } from "@/types/gdt";
import { ScanDebugOverlay } from "@/components/ScanDebugOverlay";
import { ScanReviewOverlay } from "@/components/ScanReviewOverlay";
import { SCAN_DEBUG_OVERLAY_ENABLED } from "@/lib/featureFlags";
import type { Annotation, BBox } from "@/types/annotation";
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
import type { PageReferenceTable } from "@/types/referencePoint";
import type { ScanCandidate } from "@/types/scanCandidate";
import type { PDFDocumentProxy } from "pdfjs-dist";
import { useBalloonStyle } from "@/components/BalloonStyleProvider";
import { DEFAULT_BALLOON_STYLE_SELECTION } from "@/types/balloonStyle";

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

/**
 * Resolution the page is re-rendered at to look for reference points, and the
 * pixel budget that caps it.
 *
 * Far above {@link OCR_RENDER_SCALE} on purpose. A point name is printed at
 * the sheet's note size — a tenth the height of a dimension callout — and
 * recall on the benchmark drawing went from 10 of 17 markers at 576dpi to 16
 * of 17 at 720dpi, with no false positives at either. The budget is what
 * keeps a large sheet from asking the browser for a canvas it will refuse;
 * an E-size sheet comes back at a lower resolution rather than not at all.
 */
const REFERENCE_RENDER_SCALE = 10;
// Finer than the displayed canvas, coarser than the reference-point pass: a
// GD&T glyph is larger than a point name, and the whole page is read rather
// than matched against a handful of templates.
const GDT_RENDER_SCALE = 4;
const REFERENCE_MAX_PIXELS = 48_000_000;

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

/**
 * Title-block fields for the configured keywords, as annotations.
 *
 * Read from the WHOLE page, not the scan's scope: the title block sits at a
 * fixed spot on the sheet, so which fields turn up must not depend on the
 * section scanned. Shared by the scan and the vector text-layer routes.
 */
async function readTitleFieldAnnotations(
  pageCanvas: HTMLCanvasElement,
  page: number,
  createdAt: number
): Promise<Annotation[]> {
  try {
    const fields = await runPageTitleFields(pageCanvas);
    return titleFieldAnnotations(
      fields,
      useAnnotationStore.getState().annotations,
      page,
      createdAt,
      uuidv4
    );
  } catch {
    // The balloons are the point of the scan; a failed title-block read must
    // not throw them away. The export still lists every configured keyword,
    // with an empty value to fill in by hand, and the export warns about it.
    return [];
  }
}

export function DrawingViewer() {
  const { ready: ocrReady, error: ocrError, engineLabel } = useClientOcr();
  const {
    selection: balloonStyleSelection,
    setSelection: setBalloonStyleSelection,
  } = useBalloonStyle();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadProjectInputRef = useRef<HTMLInputElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  /** Original drawing bytes, kept so a saved project can embed them. */
  const sourceFileRef = useRef<ProjectSource | null>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const projectIdRef = useRef("");

  const [konvaImage, setKonvaImage] = useState<
    HTMLImageElement | HTMLCanvasElement | null
  >(null);
  /** Guards the async display render against a page flip landing after it. */
  const renderSeqRef = useRef(0);
  const renderedRasterPageRef = useRef<{
    file: File;
    page: number;
  } | null>(null);
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
  // The area the overlay is allowed to paint in: the selection for a section
  // scan, the whole drawing for a page scan. Kept beside the overlay so the
  // two are always cleared together.
  const [scanScope, setScanScope] = useState<BBox | null>(null);
  const [scanSummary, setScanSummary] =
    useState<ScanCompletionSummary | null>(null);
  const [scanCandidates, setScanCandidates] = useState<ScanCandidate[]>([]);
  const scanCandidatesRef = useRef<ScanCandidate[]>([]);
  const candidateOrderRef = useRef(0);
  const [selectedScanSections, setSelectedScanSections] = useState<
    QueuedScanSection[]
  >([]);
  const [referenceTables, setReferenceTables] = useState<
    PageReferenceTable[]
  >([]);
  const referenceTablesRef = useRef<PageReferenceTable[]>([]);
  const [selectedReferenceSymbol, setSelectedReferenceSymbol] = useState<
    string | null
  >(null);
  const [findingReferencePoints, setFindingReferencePoints] = useState(false);
  // GD&T recognition is presentation-only: it never writes into the
  // annotation store, so it lives in component state beside the reference
  // table rather than in the project.
  const [gdtResult, setGdtResult] = useState<GdtPageResult | null>(null);
  const [gdtRegistry, setGdtRegistry] = useState<GdtRegistry | null>(null);
  const [gdtSelectedKey, setGdtSelectedKey] = useState<string | null>(null);
  const [findingGdtSymbols, setFindingGdtSymbols] = useState(false);
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
  const setPageCanvasProvider = useAnnotationStore(
    (s) => s.setPageCanvasProvider
  );
  const metadata = useDocumentMetadataStore((s) => s.metadata);
  const setDocumentMetadata = useDocumentMetadataStore((s) => s.setMetadata);
  const updateDocumentMetadataField = useDocumentMetadataStore(
    (s) => s.updateField
  );
  const resetDocumentMetadata = useDocumentMetadataStore(
    (s) => s.resetMetadata
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
  const balloonsVisible = areAllValueBalloonsVisible(annotations);
  const balloonsGloballyHidden = areAllValueBalloonsHidden(annotations);
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
  // Carried with their table position, because that is what fixes a point's
  // colour: the row keeps its hue whether or not its name was found.
  const referenceTable =
    referenceTables.find((table) => table.page === currentPage) ?? null;
  const pageReferencePoints = referenceTable
    ? referenceTable.points.map((point, index) => ({ point, index }))
    : [];
  const selectedReferenceRow = pageReferencePoints
    .filter(({ point }) => point.symbol === selectedReferenceSymbol)
    .map(({ point, index }) => ({
      box: point.rowBox,
      color: referenceColor(index),
    }))[0];
  // Tags hold their screen size, so where they fit depends on the zoom: the
  // placement is recomputed with it rather than baked in once.
  const referenceTagPlacements = (() => {
    const anchors = pageReferencePoints.flatMap(({ point }) =>
      point.markers.map((marker, place) => ({
        id: `${point.symbol}-${place}`,
        glyph: marker.bbox,
      }))
    );
    if (anchors.length === 0) return new Map<string, { x: number; y: number }>();
    const axes = referenceTable?.points[0]?.coordinates ?? [];
    const size = referenceTagSize(axes);
    return layoutReferenceTags(anchors, {
      width: size.width / scale,
      height: size.height / scale,
      gap: 7 / scale,
      bounds: stageSize,
    });
  })();

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
      candidateOrderRef.current = next.reduce(
        (largest, candidate) => Math.max(largest, candidate.order + 1),
        0
      );
    },
    [replaceScanCandidates]
  );

  const replaceReferenceTables = useCallback(
    (next: readonly PageReferenceTable[]) => {
      const normalized = normalizePageReferenceTables(next);
      referenceTablesRef.current = normalized;
      setReferenceTables(normalized);
      return normalized;
    },
    []
  );

  const persistReferenceTables = useCallback(
    async (next: readonly PageReferenceTable[], targetProjectId: string) => {
      const normalized = replaceReferenceTables(next);
      await saveProjectReferenceTables(targetProjectId, normalized);
    },
    [replaceReferenceTables]
  );

  /**
   * Put a page bitmap on the stage.
   *
   * The canvas goes to Konva as it is. Encoding it to a PNG data URL first
   * cost a multi-megabyte base64 round-trip on every page change, and at
   * display resolution it would be several times worse; Konva draws a canvas
   * as readily as an image. It also made the hand-off asynchronous, so the
   * bitmap and the stage size it belongs to were set a frame apart.
   */
  const renderCurrentPage = useCallback(
    async (doc: PDFDocumentProxy, page: number) => {
      const seq = ++renderSeqRef.current;

      // Two renders of the same page, for two different jobs. The first is the
      // annotation coordinate space — every stored bbox is measured in it, and
      // it is what OCR and the scanners crop from — so its scale is fixed and
      // it goes up first, to get the sheet on screen. The second is only ever
      // looked at, so it is rendered as fine as this sheet's size allows and
      // swapped in behind it.
      const { canvas, width, height } = await renderPdfPage(
        doc,
        page,
        PDF_RENDER_SCALE
      );
      if (renderSeqRef.current !== seq) return;
      sourceCanvasRef.current = canvas;
      setStageSize({ width, height });
      setKonvaImage(canvas);

      const unscaled = (await doc.getPage(page)).getViewport({ scale: 1 });
      const displayScale = displayRenderScale(unscaled.width, unscaled.height);
      if (displayScale <= PDF_RENDER_SCALE) return;
      try {
        const { canvas: display } = await renderPdfPage(
          doc,
          page,
          displayScale
        );
        if (renderSeqRef.current !== seq) return;
        setKonvaImage(display);
      } catch {
        // A canvas the browser refuses costs sharpness, not the drawing: the
        // coordinate-space render is already on screen and stays there.
      }
    },
    []
  );

  const renderCurrentImagePage = useCallback(
    async (file: File, page: number) => {
      const seq = ++renderSeqRef.current;
      const { canvas, width, height, totalPages } = await loadImageFile(
        file,
        page
      );
      if (renderSeqRef.current !== seq) return;
      sourceCanvasRef.current = canvas;
      renderedRasterPageRef.current = { file, page };
      setTotalPages(totalPages);
      setStageSize({ width, height });
      setKonvaImage(canvas);
    },
    [setTotalPages]
  );

  // Re-render only when the document or page changes — never on zoom. Zoom is a
  // pure display transform applied to the Stage below.
  useEffect(() => {
    if (!pdfDoc) return;
    void renderCurrentPage(pdfDoc, currentPage);
  }, [pdfDoc, currentPage, renderCurrentPage]);

  useEffect(() => {
    if (!rasterFile) return;
    const rendered = renderedRasterPageRef.current;
    if (rendered?.file === rasterFile && rendered.page === currentPage) return;
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
    setScanScope(null);
  }, [currentPage, projectId]);

  useEffect(() => {
    // Candidate and reference data are restored/reset by each project load
    // path. Only transient selection state is cleared on an id change.
    setSelectedScanCandidateId(null);
    setSelectedScanSections([]);
    setSelectedReferenceSymbol(null);
  }, [projectId]);

  useEffect(() => {
    void preloadOcr();
  }, []);

  // A closed tab leaves its scan running on the backend, and the backend has
  // one scan worker, so the next scan queues behind a job nobody is watching
  // for as long as a whole page takes. Tell it on the way out.
  useEffect(() => {
    const jobId = scanProgress?.jobId;
    if (!jobId) return;
    const stop = () => abandonAutoBalloonScan(jobId);
    window.addEventListener("pagehide", stop);
    return () => window.removeEventListener("pagehide", stop);
  }, [scanProgress?.jobId]);

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

  useEffect(() => {
    void saveProjectBalloonStyle(projectId, balloonStyleSelection).catch(() => {
      // A fresh in-memory id may not have a project row yet. The subsequent
      // saveProject call writes the same selection.
    });
  }, [balloonStyleSelection, projectId]);

  // Render a drawing onto the stage. Shared by fresh uploads and project loads
  // so both paths render at the same scale and produce matching bbox coords.
  const loadSource = useCallback(
    async (file: File) => {
      const sourceKind = await detectSourceFileKind(file);
      if (isPdfKind(sourceKind)) {
        const doc = await loadPdfDocument(file);
        renderedRasterPageRef.current = null;
        setRasterFile(null);
        setPdfDoc(doc);
        setTotalPages(doc.numPages);
        setCurrentPage(1);
        setScale(1);
        await renderCurrentPage(doc, 1);
      } else {
        setPdfDoc(null);
        setCurrentPage(1);
        setScale(1);
        await renderCurrentImagePage(file, 1);
        setRasterFile(file);
      }
    },
    [
      renderCurrentImagePage,
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
    setBalloonStyleSelection(
      recent.balloonStyle ?? DEFAULT_BALLOON_STYLE_SELECTION
    );
    setProjectId(recent.id);
    setProjectName(recent.name.replace(/\.[^.]+$/, ""));
    setDocumentMetadata(recent.metadata);
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
    const migration = normalizeLegacyAnnotations(storedAnnotations);
    setAnnotations(migration.annotations);
    restoreScanCandidates(storedCandidates);
    replaceReferenceTables(recent.referenceTables ?? []);
    await saveAnnotations(recent.id, migration.annotations);
    const warning = legacyMigrationWarning(migration.orphanLabelCount);
    if (warning) setSelectionError(warning);
  }, [
    loadSource,
    replaceReferenceTables,
    restoreScanCandidates,
    setBalloonStyleSelection,
    setAnnotations,
    setDocumentMetadata,
    setProjectName,
  ]);

  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    void restoreLastProject();
  }, [restoreLastProject]);

  const handleFile = async (selected: File) => {
    const file = await openableDrawing(selected);
    const sourceKind = await detectSourceFileKind(file);
    // A new drawing is a new project — fresh id, no stale annotations.
    const id = uuidv4();
    setProjectId(id);
    setBalloonStyleSelection(DEFAULT_BALLOON_STYLE_SELECTION);
    setAnnotations([]);
    setSelectedId(null);
    replaceScanCandidates([]);
    candidateOrderRef.current = 0;
    setSelectedScanCandidateId(null);
    setSelectedScanSections([]);
    replaceReferenceTables([]);
    setEditingValueId(null);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setActiveMetadataField(null);
    resetDocumentMetadata();
    setScanProgress(null);
    setScanOverlay(null);
    setScanScope(null);
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
      referenceTables: [],
      balloonStyle: DEFAULT_BALLOON_STYLE_SELECTION,
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
      scanCandidates,
      metadata,
      referenceTables,
      balloonStyle: balloonStyleSelection,
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
    renderedRasterPageRef.current = null;
    sourceCanvasRef.current = null;
    sourceFileRef.current = null;
    setAnnotations([]);
    setSelectedId(null);
    replaceScanCandidates([]);
    candidateOrderRef.current = 0;
    setSelectedScanCandidateId(null);
    setSelectedScanSections([]);
    replaceReferenceTables([]);
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
    setBalloonStyleSelection(DEFAULT_BALLOON_STYLE_SELECTION);
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
    replaceReferenceTables([]);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setActiveMetadataField(null);
    try {
      const bundle = parseProjectBundle(await file.text());
      const id = uuidv4();
      setBalloonStyleSelection(bundle.balloonStyle);
      setProjectId(id);
      sourceFileRef.current = bundle.source;
      setProjectName(bundle.projectName);
      setDocumentMetadata(bundle.metadata);
      replaceReferenceTables(bundle.referenceTables);
      const srcFile = dataUrlToFile(
        bundle.source.dataUrl,
        bundle.source.fileName,
        bundle.source.mimeType
      );
      await loadSource(srcFile);
      setSelectedId(null);
      setEditingValueId(null);
      const migration = normalizeLegacyAnnotations(bundle.annotations);
      // Candidate ids are global Dexie primary keys. Fresh ids prevent
      // importing the same portable project twice from moving the first
      // project's candidate rows into the second one.
      const importedCandidates = bundle.scanCandidates.map((candidate) => ({
        ...candidate,
        id: `${id}:${uuidv4()}`,
      }));
      setAnnotations(migration.annotations);
      restoreScanCandidates(importedCandidates);
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
        metadata: bundle.metadata,
        referenceTables: bundle.referenceTables,
        balloonStyle: bundle.balloonStyle,
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
        // Intentionally use the same single-value reader as Draw Value. This
        // fills metadata without creating an annotation or balloon.
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

      // Scan a higher-resolution PDF render when available. Raster drawings
      // use their original pixels; a non-current TIFF page is decoded here.
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
      } else if (rasterFile && scanPage !== currentPage) {
        scanCanvas = (await loadImageFile(rasterFile, scanPage)).canvas;
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
      setScanScope(bbox);
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
        // A vector PDF already carries every callout as text, with exact
        // digits and exact glyph boxes, so there is nothing to recognise:
        // the values come back perfect in milliseconds instead of the tens
        // of seconds a scan takes. Most drawings are scans with no text at
        // all, and those report back unusable so the scan runs as before.
        const sourceFile = sourceFileRef.current;
        if (sourceFile?.fileType === "pdf" && source) {
          const pageBounds = {
            x: 0,
            y: 0,
            width: source.width,
            height: source.height,
          };
          const textLayer = await readPdfTextLayer(
            await (await fetch(sourceFile.dataUrl)).blob(),
            scanPage,
            pageBounds
          );
          if (textLayer.usable) {
            const scoped =
              scopeKind === "page"
                ? textLayer.regions
                : regionsWithin(textLayer.regions, bbox);
            const live = useAnnotationStore.getState().annotations;
            const kept = filterNewScanRegions(
              scoped,
              live,
              scanPage,
              undefined,
              scopeKind
            );
            const fromText = annotationsFromRegions(kept.accepted, scanPage);
            // A vector PDF has a title block too. This branch returned before
            // the title-field step, so on every drawing with a text layer the
            // configured keywords were never read and their labels were
            // missing from the drawing and the exports alike.
            if (scopeKind === "page") {
              fromText.push(
                ...(await readTitleFieldAnnotations(source, scanPage, Date.now()))
              );
            }
            if (fromText.length > 0) addAnnotations(fromText);
            const textSummary: ScanCompletionSummary = {
              scopeKind,
              added: fromText.length,
              detected: scoped.length,
              recognized: scoped.length,
              eligible: scoped.length,
              excluded: 0,
              // A value the text layer read exactly but could not say what
              // it measures — a lone deviation, a number whose symbol the
              // drawing drew rather than wrote — is ballooned and flagged.
              reviewRequired: fromText.filter(
                (annotation) => annotation.needsReview
              ).length,
              unread: 0,
              skippedExisting: kept.skippedExisting,
              skippedDuplicates: kept.skippedDuplicates,
            };
            setScanSummary(textSummary);
            if (fromText.length === 0) {
              setSelectionError(
                scopeKind === "page"
                  ? "The drawing's text carries no values that are not already ballooned."
                  : "No values were found in the selected section."
              );
            }
            return { status: "succeeded", summary: textSummary };
          }
        }
        const scanResult = await runAutoBalloonScan({
          sourceCanvas: scanCanvas,
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
        const newAnnotations: Annotation[] = annotationsFromRegions(
          filtered.accepted,
          scanPage,
          now
        );

        // Keep every unaccepted detector outcome outside the annotation store.
        // Review and filtered "Other" objects share one durable lifecycle;
        // geometry-equivalent rescans merge into an audit trail instead of
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
          newAnnotations.push(
            ...(await readTitleFieldAnnotations(source, scanPage, now))
          );
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
        setScanScope(null);
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
          setScanScope(null);
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
      currentPage,
      pdfDoc,
      rasterFile,
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

  const renderReferencePage = useCallback(
    async (
      page: number
    ): Promise<{ canvas: HTMLCanvasElement; displayScale: number }> => {
      const displayedCanvas =
        page === currentPage ? sourceCanvasRef.current : null;

      if (pdfDoc) {
        const pdfPage = await pdfDoc.getPage(page);
        const unscaled = pdfPage.getViewport({ scale: 1 });
        const pixelBudgetScale = Math.sqrt(
          REFERENCE_MAX_PIXELS / (unscaled.width * unscaled.height)
        );
        const renderScale = Math.min(
          REFERENCE_RENDER_SCALE,
          pixelBudgetScale
        );

        if (renderScale > PDF_RENDER_SCALE) {
          try {
            const rendered = await renderPdfPage(pdfDoc, page, renderScale);
            return {
              canvas: rendered.canvas,
              displayScale: renderScale / PDF_RENDER_SCALE,
            };
          } catch {
            // Fall through to the displayed or standard-resolution page.
          }
        }

        if (displayedCanvas) {
          return { canvas: displayedCanvas, displayScale: 1 };
        }
        const fallbackScale = Math.min(PDF_RENDER_SCALE, pixelBudgetScale);
        const rendered = await renderPdfPage(pdfDoc, page, fallbackScale);
        return {
          canvas: rendered.canvas,
          displayScale: fallbackScale / PDF_RENDER_SCALE,
        };
      }

      if (displayedCanvas) {
        return { canvas: displayedCanvas, displayScale: 1 };
      }
      if (rasterFile) {
        const rendered = await loadImageFile(rasterFile, page);
        return { canvas: rendered.canvas, displayScale: 1 };
      }
      throw new Error("No drawing is available for reference detection.");
    },
    [currentPage, pdfDoc, rasterFile]
  );

  /** Run reference detection after balloon results have already been saved. */
  const runAutomaticReferencePass = useCallback(
    async (pages: readonly number[]) => {
      if (pages.length === 0) return;

      const projectAtStart = projectIdRef.current;
      setSelectedReferenceSymbol(null);
      setFindingReferencePoints(true);
      try {
        const outcomes = await runReferenceScanQueue({
          pages,
          runPage: async (page) => {
            if (projectIdRef.current !== projectAtStart) {
              throw new Error("The drawing changed before reference detection.");
            }
            const { canvas, displayScale } = await renderReferencePage(page);
            const result = await runPageReferencePoints(canvas, displayScale);
            if (projectIdRef.current !== projectAtStart) {
              throw new Error("The drawing changed during reference detection.");
            }
            return result;
          },
          applyResult: async (page, result) => {
            if (projectIdRef.current !== projectAtStart) {
              throw new Error("The drawing changed before references were saved.");
            }
            await persistReferenceTables(
              referenceTablesAfterPageResult(
                referenceTablesRef.current,
                page,
                result
              ),
              projectAtStart
            );
          },
        });

        if (projectIdRef.current !== projectAtStart) return;
        const notices = referenceScanNotices(outcomes);
        if (notices.length > 0) {
          setSelectionError((current) =>
            [...new Set([...(current ? [current] : []), ...notices])].join(" ")
          );
        }
      } finally {
        setFindingReferencePoints(false);
      }
    },
    [persistReferenceTables, renderReferencePage]
  );

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
        await runAutomaticReferencePass(
          completedReferencePages(sections, result.completedSections)
        );
      }
    } finally {
      setIsProcessing(false);
    }
  }, [
    runAutoBalloon,
    runAutomaticReferencePass,
    selectedScanSections,
    setIsDrawingValue,
    setIsProcessing,
    setIsSegmenting,
  ]);

  const scanWholePage = useCallback(async () => {
    const source = sourceCanvasRef.current;
    if (!source) return;
    setSelectedScanSections([]);
    setSelectedReferenceSymbol(null);
    setActiveMetadataField(null);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setIsProcessing(true);
    try {
      const outcome = await runAutoBalloon(
        { x: 0, y: 0, width: source.width, height: source.height },
        "page",
        currentPage,
        { manageProcessing: false }
      );
      if (outcome.status === "succeeded") {
        await runAutomaticReferencePass([currentPage]);
      }
    } finally {
      setIsProcessing(false);
    }
  }, [
    currentPage,
    runAutoBalloon,
    runAutomaticReferencePass,
    setIsDrawingValue,
    setIsProcessing,
    setIsSegmenting,
  ]);

  const clearReferencePoints = useCallback(() => {
    const next = referenceTablesRef.current.filter(
      (table) => table.page !== currentPage
    );
    void persistReferenceTables(next, projectIdRef.current).catch(() => {
      setSelectionError((current) =>
        current ?? "Reference points could not be saved locally."
      );
    });
    setSelectedReferenceSymbol(null);
  }, [currentPage, persistReferenceTables]);

  /**
   * Find feature control frames and GD&T symbols on this page.
   *
   * Like the reference-point pass, and for the same reason, the sheet is
   * re-rendered finer than the displayed canvas before it is read. GD&T
   * modifiers are printed at note size — a circled Ⓜ is around 20px at a
   * typical frame height, with a 10px letter inside it — and below that the
   * recognizer correctly refuses to name them. Resolution is the cheapest
   * accuracy available here.
   *
   * These are not inspection values. They carry no tolerance of their own, no
   * balloon number and no measurement, they are not saved into the project
   * and not exported; they are shown for an operator to read and correct.
   */
  const findGdtSymbols = useCallback(async () => {
    const source = sourceCanvasRef.current;
    if (!source) return;

    setSelectionError(null);
    setScanSummary(null);
    setSelectedScanSections([]);
    setActiveMetadataField(null);
    setIsDrawingValue(false);
    setIsSegmenting(false);
    setGdtSelectedKey(null);
    setFindingGdtSymbols(true);
    setIsProcessing(true);

    const projectAtStart = projectIdRef.current;
    const page = currentPage;
    try {
      let readCanvas = source;
      let readScale = 1;
      if (pdfDoc) {
        const base = await pdfDoc.getPage(page);
        const unscaled = base.getViewport({ scale: 1 });
        const budget = Math.sqrt(
          REFERENCE_MAX_PIXELS / (unscaled.width * unscaled.height)
        );
        const renderScale = Math.min(GDT_RENDER_SCALE, budget);
        if (renderScale > PDF_RENDER_SCALE) {
          try {
            const { canvas } = await renderPdfPage(pdfDoc, page, renderScale);
            readCanvas = canvas;
            readScale = renderScale / PDF_RENDER_SCALE;
          } catch {
            readCanvas = source;
            readScale = 1;
          }
        }
      }

      const raw = await recognizeGdtSymbols(readCanvas, null, { page });
      if (projectIdRef.current !== projectAtStart) return;
      if (!raw) {
        setGdtResult(null);
        setSelectionError(
          "The OCR service could not be reached for GD&T recognition."
        );
        return;
      }

      // Boxes come back in the upscaled render's pixels; the overlay draws in
      // page coordinates.
      const result = scaleResult(raw, readScale);
      setGdtResult(result);
      void fetchGdtRegistry().then(setGdtRegistry);

      if (result.counts.frames === 0 && result.counts.symbols === 0) {
        setSelectionError(
          "No feature control frames or GD&T symbols were found on this page."
        );
      } else if (result.counts.ambiguous > 0) {
        setSelectionError(
          `${result.counts.ambiguous} detection${
            result.counts.ambiguous === 1 ? "" : "s"
          } need review — see the GD&T panel.`
        );
      }
    } catch (error) {
      setGdtResult(null);
      setSelectionError(
        error instanceof Error
          ? error.message
          : "GD&T recognition failed."
      );
    } finally {
      setFindingGdtSymbols(false);
      setIsProcessing(false);
    }
  }, [
    currentPage,
    pdfDoc,
    setIsDrawingValue,
    setIsProcessing,
    setIsSegmenting,
  ]);

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
        ocrResult: {
          text: candidate.text,
          confidence: candidate.confidence,
          rotation: candidate.rotation,
          orientation: candidate.orientation,
          words: [],
          engine: "paddleocr",
          agreement: undefined,
          needsReview: candidate.state === "review",
          valueBox: candidate.valueBox,
          // The scan already ran the labelling rules on this text.
          rule:
            candidate.label || candidate.category
              ? {
                  text: candidate.text,
                  category: candidate.category ?? "",
                  label: candidate.label ?? "",
                }
              : undefined,
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
        accept=".pdf,.dwg,.dxf,.png,.jpg,.jpeg,.tif,.tiff,.webp,application/pdf,image/png,image/jpeg,image/tiff,image/webp"
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
              onClick={() => {
                setScanOverlay(null);
                setScanScope(null);
              }}
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
            Drag a box around the {activeMetadataLabel}. The detected text will
            fill the metadata field without creating a balloon.
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
        findingReferencePoints={findingReferencePoints}
        onFindGdtSymbols={() => void findGdtSymbols()}
        findingGdtSymbols={findingGdtSymbols}
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
        referenceTables={referenceTables}
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

                    {balloonsGloballyHidden &&
                      pageAnnotations.map((ann) => {
                        const box = ann.orientedBox ?? ann.bbox;
                        const rotation = ann.orientedBox?.rotation ?? 0;
                        return (
                          <Rect
                            key={`whiteout-${ann.id}`}
                            x={box.x}
                            y={box.y}
                            width={box.width}
                            height={box.height}
                            rotation={rotation}
                            fill="#ffffff"
                            stroke="#ffffff"
                            strokeWidth={1 / scale}
                            listening={false}
                          />
                        );
                      })}

                    {gdtResult && (
                      <GdtOverlay
                        frames={framesForPage(gdtResult, currentPage)}
                        items={overlayItems(gdtResult, currentPage)}
                        scale={scale}
                        selectedKey={gdtSelectedKey}
                        onSelect={(key) => setGdtSelectedKey(key)}
                      />
                    )}

                    {SCAN_DEBUG_OVERLAY_ENABLED && scanOverlay && (
                      <ScanDebugOverlay
                        overlay={scanOverlay}
                        scale={scale}
                        scope={scanScope ?? { x: 0, y: 0, ...stageSize }}
                        pageSize={stageSize}
                      />
                    )}

                    {visiblePageAnnotations.map((ann) => {
                      const highlighted = selectedId === ann.id;
                      const stroke = highlighted ? "#2563eb" : "#dc2626";
                      const box = ann.orientedBox ?? ann.bbox;
                      const rotation = ann.orientedBox?.rotation ?? 0;
                      return (
                        <Rect
                          key={ann.id}
                          x={box.x}
                          y={box.y}
                          width={box.width}
                          height={box.height}
                          rotation={rotation}
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

                    {selectedReferenceRow && (
                      <Rect
                        x={selectedReferenceRow.box.x}
                        y={selectedReferenceRow.box.y}
                        width={selectedReferenceRow.box.width}
                        height={selectedReferenceRow.box.height}
                        fill={selectedReferenceRow.color.fill}
                        opacity={0.22}
                        stroke={selectedReferenceRow.color.fill}
                        strokeWidth={2 / scale}
                        cornerRadius={2 / scale}
                        listening={false}
                      />
                    )}

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

                    {pageReferencePoints.map(({ point, index }) =>
                      point.markers.map((marker, place) => (
                        <ReferenceMarker
                          key={`ref-${point.symbol}-${index}-${place}`}
                          symbol={point.symbol}
                          coordinates={point.coordinates}
                          marker={marker}
                          placement={
                            referenceTagPlacements.get(
                              `${point.symbol}-${place}`
                            ) ?? { x: marker.bbox.x, y: marker.bbox.y }
                          }
                          color={referenceColor(index)}
                          scale={scale}
                          selected={
                            selectedReferenceSymbol === point.symbol
                          }
                          dimmed={
                            selectedReferenceSymbol !== null &&
                            selectedReferenceSymbol !== point.symbol
                          }
                          onSelect={(symbol) =>
                            setSelectedReferenceSymbol((current) =>
                              current === symbol ? null : symbol
                            )
                          }
                        />
                      ))
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
          referencePanel={
            referenceTable ? (
              <ReferencePanel
                table={referenceTable}
                selectedSymbol={selectedReferenceSymbol}
                disabled={isProcessing}
                onSelect={setSelectedReferenceSymbol}
                onClear={clearReferencePoints}
              />
            ) : null
          }
          gdtPanel={
            gdtResult && gdtResult.page === currentPage ? (
              <GdtPanel
                result={gdtResult}
                registry={gdtRegistry}
                selectedKey={gdtSelectedKey}
                disabled={isProcessing}
                onSelect={setGdtSelectedKey}
                onClear={() => {
                  setGdtResult(null);
                  setGdtSelectedKey(null);
                }}
              />
            ) : null
          }
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

      <AnnotationPopup onReviewResolved={resolveScanCandidate} />

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
