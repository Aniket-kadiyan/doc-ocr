"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Stage, Layer, Image as KonvaImage, Rect } from "react-konva";
import type Konva from "konva";
import { v4 as uuidv4 } from "uuid";
import { useAnnotationStore } from "@/store/annotationStore";
import { normalizeBBox } from "@/lib/canvasUtils";
import { runOCR, runSegment, preloadOcr } from "@/lib/clientOcr";
import { classifyDimension } from "@/lib/dimensionClassifier";
import { suggestLabel } from "@/lib/labelSuggestions";
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
import { LabelEditor } from "@/components/LabelEditor";
import type { Annotation, BBox } from "@/types/annotation";
import type { PDFDocumentProxy } from "pdfjs-dist";

const MIN_BOX = 8;

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

  const [konvaImage, setKonvaImage] = useState<HTMLImageElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [projectId, setProjectId] = useState(() => uuidv4());
  const restoredRef = useRef(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [currentBox, setCurrentBox] = useState<BBox | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [lastDebugDump, setLastDebugDump] = useState<string | null>(null);
  const drawStartRef = useRef<{ x: number; y: number } | null>(null);
  const currentBoxRef = useRef<BBox | null>(null);

  const annotations = useAnnotationStore((s) => s.annotations);
  const setAnnotations = useAnnotationStore((s) => s.setAnnotations);
  const addAnnotations = useAnnotationStore((s) => s.addAnnotations);
  const updateAnnotation = useAnnotationStore((s) => s.updateAnnotation);
  const removeAnnotation = useAnnotationStore((s) => s.removeAnnotation);
  const setPending = useAnnotationStore((s) => s.setPending);
  const currentPage = useAnnotationStore((s) => s.currentPage);
  const setCurrentPage = useAnnotationStore((s) => s.setCurrentPage);
  const totalPages = useAnnotationStore((s) => s.totalPages);
  const setTotalPages = useAnnotationStore((s) => s.setTotalPages);
  const scale = useAnnotationStore((s) => s.scale);
  const setScale = useAnnotationStore((s) => s.setScale);
  const addValueLabelId = useAnnotationStore((s) => s.addValueLabelId);
  const setAddValueLabelId = useAnnotationStore((s) => s.setAddValueLabelId);
  const editingLabelId = useAnnotationStore((s) => s.editingLabelId);
  const setEditingLabelId = useAnnotationStore((s) => s.setEditingLabelId);
  const isSegmenting = useAnnotationStore((s) => s.isSegmenting);
  const setIsSegmenting = useAnnotationStore((s) => s.setIsSegmenting);
  const isLabeling = useAnnotationStore((s) => s.isLabeling);
  const setIsLabeling = useAnnotationStore((s) => s.setIsLabeling);
  const labelInputMode = useAnnotationStore((s) => s.labelInputMode);
  const setLabelInputMode = useAnnotationStore((s) => s.setLabelInputMode);
  const isProcessing = useAnnotationStore((s) => s.isProcessing);
  const setIsProcessing = useAnnotationStore((s) => s.setIsProcessing);
  const setProjectName = useAnnotationStore((s) => s.setProjectName);
  const projectName = useAnnotationStore((s) => s.projectName);
  const setStoreProjectId = useAnnotationStore((s) => s.setProjectId);

  const pageAnnotations = annotations.filter((a) => a.page === currentPage);

  // The selected annotation plus its mapped partner(s): a dimension maps to its
  // label via labelId; a label maps to every dimension that points at it. Both
  // sides stay highlighted (not dimmed) when either is selected.
  const relatedIds = useMemo(() => {
    const set = new Set<string>();
    if (!selectedId) return set;
    set.add(selectedId);
    const sel = annotations.find((a) => a.id === selectedId);
    if (!sel) return set;
    if (sel.kind === "label") {
      annotations.forEach((a) => {
        if (a.labelId === sel.id) set.add(a.id);
      });
    } else if (sel.labelId) {
      set.add(sel.labelId);
    }
    return set;
  }, [selectedId, annotations]);

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
  }, [projectId, setStoreProjectId]);

  // Pull edits made in the checksheet tab back into the live drawing. The tab
  // saves to IndexedDB and broadcasts the updated annotations on this channel.
  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return;
    const channel = new BroadcastChannel("doc-ocr-box:checksheet");
    channel.onmessage = (e) => {
      const msg = e.data as { projectId?: string; annotations?: Annotation[] };
      if (msg?.projectId === projectId && Array.isArray(msg.annotations)) {
        setAnnotations(msg.annotations);
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
    setAnnotations(await loadAnnotations(recent.id));
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
    setPending(null);
    setSelectionError(null);
    setCurrentPage(1);
    setTotalPages(1);
    setProjectName("Untitled Drawing");
    setProjectId(uuidv4());
    await deleteProject(id);
  };

  const handleLoadProject = async (file: File) => {
    setSelectionError(null);
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
      setAnnotations(bundle.annotations);
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
      await saveAnnotations(id, bundle.annotations);
    } catch (err) {
      setSelectionError(
        err instanceof Error ? err.message : "Could not open project file."
      );
    }
  };

  // Select an annotation and follow it to its page so it's actually on screen.
  const handleSelect = useCallback(
    (id: string | null) => {
      setSelectedId(id);
      if (!id) return;
      const ann = annotations.find((a) => a.id === id);
      if (ann && ann.page !== currentPage) setCurrentPage(ann.page);
    },
    [annotations, currentPage, setCurrentPage]
  );

  // Read a value box (drawn via a label's "Add value" action) and open the
  // popup. labelId binds the value one-to-one to the label it was added from.
  const finishBox = useCallback(
    async (bbox: BBox, labelId?: string) => {
      if (bbox.width < MIN_BOX || bbox.height < MIN_BOX) return;
      const source = sourceCanvasRef.current;
      if (!source) return;

      setIsProcessing(true);
      setSelectionError(null);
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
          kind: "dimension",
          labelId,
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "OCR failed unexpectedly";
        setSelectionError(message);
        setPending({
          bbox,
          page: currentPage,
          kind: "dimension",
          labelId,
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
        setAddValueLabelId(null);
      }
    },
    [currentPage, setPending, setIsProcessing, setAddValueLabelId]
  );

  // Finish an Add Label box. "manual" skips OCR and opens an empty label form;
  // "ocr" reads the box and pre-fills the label text. Either way the result is a
  // label-kind annotation (its own color + numbering).
  const finishLabelBox = useCallback(
    async (bbox: BBox, mode: typeof labelInputMode) => {
      if (bbox.width < MIN_BOX || bbox.height < MIN_BOX) {
        setIsLabeling(false);
        return;
      }
      const source = sourceCanvasRef.current;
      if (!source) {
        setIsLabeling(false);
        return;
      }
      setSelectionError(null);

      const emptyOcr = {
        text: "",
        confidence: 0,
        rotation: 0,
        orientation: "horizontal" as const,
        words: [],
        engine: "paddleocr" as const,
      };

      if (mode === "manual") {
        setPending({
          bbox,
          page: currentPage,
          kind: "label",
          labelSource: "manual",
          ocrResult: emptyOcr,
        });
        setIsLabeling(false);
        return;
      }

      setIsProcessing(true);
      try {
        const ocrResult = await runOCR(source, bbox, 1);
        if (!ocrResult.text.trim()) {
          setSelectionError(
            "No label text detected — type it in or draw a tighter box."
          );
        }
        setPending({
          bbox: ocrResult.valueBox ?? bbox,
          page: currentPage,
          kind: "label",
          labelSource: "ocr",
          ocrResult,
        });
      } catch (err) {
        setSelectionError(
          err instanceof Error ? err.message : "Label OCR failed unexpectedly"
        );
        setPending({
          bbox,
          page: currentPage,
          kind: "label",
          labelSource: "ocr",
          ocrResult: emptyOcr,
        });
      } finally {
        setIsProcessing(false);
        setIsLabeling(false);
      }
    },
    [currentPage, setPending, setIsProcessing, setIsLabeling]
  );

  const finishSegmentBox = useCallback(
    async (bbox: BBox) => {
      if (bbox.width < MIN_BOX || bbox.height < MIN_BOX) return;
      const source = sourceCanvasRef.current;
      if (!source) return;

      setIsProcessing(true);
      setSelectionError(null);
      try {
        const regions = await runSegment(source, bbox, 1);
        if (regions.length === 0) {
          setSelectionError(
            "No values detected in that area — draw a tighter box around the cluster."
          );
          return;
        }
        const now = Date.now();
        const newAnnotations: Annotation[] = regions
          .filter((r) => r.text.trim())
          .map((r) => {
            const type = classifyDimension(r.text);
            return {
              id: uuidv4(),
              // number is assigned sequentially by addAnnotations
              number: 0,
              label: suggestLabel(type, r.text),
              value: r.text.trim(),
              type,
              confidence: r.confidence,
              bbox: r.valueBox,
              rotation: r.rotation,
              page: currentPage,
              createdAt: now,
              needsReview: r.needsReview,
            };
          });
        if (newAnnotations.length === 0) {
          setSelectionError("Detected regions but read no text — try a tighter box.");
          return;
        }
        addAnnotations(newAnnotations);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Auto-segment failed unexpectedly";
        setSelectionError(message);
      } finally {
        setIsProcessing(false);
        setIsSegmenting(false);
      }
    },
    [currentPage, addAnnotations, setIsProcessing, setIsSegmenting]
  );

  const isCompletingDraw = useRef(false);

  const drawingActive = addValueLabelId !== null || isSegmenting || isLabeling;

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!drawingActive) return;
    const stage = e.target.getStage();
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
    const labeling = isLabeling;
    const valueLabelId = addValueLabelId;
    const mode = labelInputMode;
    drawStartRef.current = null;
    currentBoxRef.current = null;
    setCurrentBox(null);
    try {
      if (labeling) {
        await finishLabelBox(box, mode);
      } else if (segmenting) {
        await finishSegmentBox(box);
      } else if (valueLabelId) {
        await finishBox(box, valueLabelId);
      }
    } finally {
      isCompletingDraw.current = false;
    }
  }, [
    finishBox,
    finishSegmentBox,
    finishLabelBox,
    isSegmenting,
    isLabeling,
    addValueLabelId,
    labelInputMode,
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
        <div className="bg-amber-50 px-4 py-1.5 text-center text-xs text-amber-900">
          {selectionError}
        </div>
      )}
      {lastDebugDump && (
        <div className="bg-violet-50 px-4 py-1.5 text-center text-xs text-violet-900">
          OCR debug: {lastDebugDump}
        </div>
      )}
      {isLabeling && (
        <div className="flex items-center justify-center gap-3 bg-indigo-50 px-4 py-1.5 text-center text-xs text-indigo-900">
          <span>
            {labelInputMode === "manual"
              ? "Draw a box where the label goes, then type it."
              : "Draw a box around the label text — OCR will read it."}
          </span>
          <button
            type="button"
            onClick={() => setIsLabeling(false)}
            className="rounded border border-indigo-300 px-2 py-0.5 font-medium text-indigo-700 hover:bg-indigo-100"
          >
            Cancel
          </button>
        </div>
      )}
      {addValueLabelId && (
        <div className="flex items-center justify-center gap-3 bg-blue-600 px-4 py-2.5 text-center text-sm font-medium text-white">
          <span>
            ✏️ Now drag a box on the drawing around the value for “
            {annotations.find((a) => a.id === addValueLabelId)?.value ?? "label"}
            ” — OCR will read it.
          </span>
          <button
            type="button"
            onClick={() => setAddValueLabelId(null)}
            className="rounded border border-white/60 px-2 py-0.5 font-medium text-white hover:bg-white/20"
          >
            Cancel
          </button>
        </div>
      )}

      <Toolbar
        isSegmenting={isSegmenting}
        isLabeling={isLabeling}
        isProcessing={isProcessing}
        currentPage={currentPage}
        totalPages={totalPages}
        scale={scale}
        onToggleSegment={() => {
          setAddValueLabelId(null);
          setIsLabeling(false);
          setIsSegmenting(!isSegmenting);
        }}
        onStartLabel={(mode) => {
          setAddValueLabelId(null);
          setIsSegmenting(false);
          setLabelInputMode(mode);
          setIsLabeling(true);
        }}
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
          <div className="flex-1 overflow-auto p-4">
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
                      image={konvaImage}
                      width={stageSize.width}
                      height={stageSize.height}
                      listening={!drawingActive}
                    />

                    {pageAnnotations.map((ann) => {
                      const isLabel = ann.kind === "label";
                      // Highlight the selection AND its mapped partner.
                      const highlighted = relatedIds.has(ann.id);
                      // Labels read indigo, dimensions red; highlight brightens.
                      const stroke = highlighted
                        ? isLabel
                          ? "#4f46e5"
                          : "#2563eb"
                        : isLabel
                          ? "#7c3aed"
                          : "#dc2626";
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
                          dash={(isLabel ? [3, 3] : [6, 4]).map((d) => d / scale)}
                          // When something is selected, fade everything that
                          // isn't the selection or its mapped partner.
                          opacity={selectedId && !highlighted ? 0.15 : 1}
                          listening={!drawingActive}
                          onClick={() => handleSelect(ann.id)}
                        />
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

                    {pageAnnotations.map((ann) => (
                      <Balloon
                        key={`balloon-${ann.id}`}
                        annotation={ann}
                        scale={scale}
                        selected={relatedIds.has(ann.id)}
                        dimmed={!!selectedId && !relatedIds.has(ann.id)}
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
          selectedId={selectedId}
          highlightedIds={relatedIds}
          onSelect={handleSelect}
          onUpdate={updateAnnotation}
          onDelete={removeAnnotation}
          onAddValue={(labelId) => {
            setIsSegmenting(false);
            setIsLabeling(false);
            setAddValueLabelId(labelId);
          }}
          onEditLabel={(labelId) => {
            setSelectedId(labelId);
            setEditingLabelId(labelId);
          }}
        />
      </div>

      <AnnotationPopup />

      {(() => {
        const editingLabel = annotations.find(
          (a) => a.id === editingLabelId && a.kind === "label"
        );
        if (!editingLabel) return null;
        const editingValue = annotations.find(
          (a) =>
            (a.kind ?? "dimension") === "dimension" &&
            a.labelId === editingLabel.id
        );
        return (
          <LabelEditor
            label={editingLabel}
            value={editingValue}
            onUpdate={updateAnnotation}
            onDeleteValue={removeAnnotation}
            onAddValue={(labelId) => {
              setIsSegmenting(false);
              setIsLabeling(false);
              setAddValueLabelId(labelId);
            }}
            onClose={() => setEditingLabelId(null)}
          />
        );
      })()}
    </div>
  );
}
