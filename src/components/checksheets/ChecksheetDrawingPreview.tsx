"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Image as KonvaImage, Layer, Rect, Stage } from "react-konva";
import type Konva from "konva";
import type { PDFDocumentProxy } from "pdfjs-dist";
import { Balloon } from "@/components/Balloon";
import {
  ReferenceMarker,
  referenceTagSize,
} from "@/components/ReferenceMarker";
import { checksheetDocumentUrl } from "@/lib/checksheetClient";
import {
  checksheetFocusBox,
  checksheetItemCaption,
  isChecksheetReferenceRow,
  referenceDisplayCoordinates,
} from "@/lib/checksheetReference";
import {
  loadImageFile,
  loadPdfDocument,
  renderPdfPage,
} from "@/lib/pdfLoader";
import { referenceColor } from "@/lib/referenceColors";
import { layoutReferenceTags } from "@/lib/referenceTagLayout";
import type { Annotation } from "@/types/annotation";
import type { ChecksheetRunResponse, ChecksheetRow } from "@/types/checksheet";

interface ChecksheetDrawingPreviewProps {
  document: ChecksheetRunResponse["document"];
  pdfRenderScale: number;
  row: ChecksheetRow | null;
}

type ViewMode = "focus" | "fit";

interface Transform {
  x: number;
  y: number;
  scale: number;
}

const canvasImage = (canvas: HTMLCanvasElement) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new window.Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not render drawing preview."));
    image.src = canvas.toDataURL("image/png");
  });

function fitTransform(
  viewport: { width: number; height: number },
  page: { width: number; height: number }
): Transform {
  const scale = Math.max(
    0.02,
    Math.min(
      (viewport.width - 24) / page.width,
      (viewport.height - 24) / page.height
    )
  );
  return {
    scale,
    x: (viewport.width - page.width * scale) / 2,
    y: (viewport.height - page.height * scale) / 2,
  };
}

function focusTransform(
  viewport: { width: number; height: number },
  page: { width: number; height: number },
  row: ChecksheetRow
): Transform {
  const focusBox = checksheetFocusBox(row);
  const reference = isChecksheetReferenceRow(row);
  const contextWidth = Math.min(
    page.width,
    Math.max(280, focusBox.width * (reference ? 1.25 : 5.5))
  );
  const contextHeight = Math.min(
    page.height,
    Math.max(190, focusBox.height * (reference ? 1.4 : 7))
  );
  const scale = Math.max(
    0.05,
    Math.min(
      (viewport.width - 28) / contextWidth,
      (viewport.height - 28) / contextHeight,
      4
    )
  );
  const centerX = focusBox.x + focusBox.width / 2;
  const centerY = focusBox.y + focusBox.height / 2;
  return {
    scale,
    x: viewport.width / 2 - centerX * scale,
    y: viewport.height / 2 - centerY * scale,
  };
}

export function ChecksheetDrawingPreview({
  document,
  pdfRenderScale,
  row,
}: ChecksheetDrawingPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [pdfDocument, setPdfDocument] = useState<PDFDocumentProxy | null>(null);
  const [drawingImage, setDrawingImage] = useState<HTMLImageElement | null>(null);
  const [pageSize, setPageSize] = useState({ width: 1, height: 1 });
  const [viewport, setViewport] = useState({ width: 640, height: 288 });
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, scale: 1 });
  const [expanded, setExpanded] = useState(false);
  const [mode, setMode] = useState<ViewMode>("fit");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    setSourceFile(null);
    setPdfDocument(null);
    setDrawingImage(null);
    void (async () => {
      try {
        const response = await fetch(checksheetDocumentUrl(document.id), {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Drawing request failed with HTTP ${response.status}.`);
        }
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        const file = new File([blob], document.file_name, {
          type: document.mime_type || blob.type,
        });
        setSourceFile(file);
        if (document.file_type === "pdf") {
          setPdfDocument(await loadPdfDocument(file));
        }
      } catch (reason) {
        if (controller.signal.aborted) return;
        setError(
          reason instanceof Error ? reason.message : "Could not load drawing."
        );
        setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [document.file_name, document.file_type, document.id, document.mime_type]);

  useEffect(() => {
    if (!sourceFile || !row) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    void (async () => {
      try {
        const rendered =
          document.file_type === "pdf"
            ? pdfDocument
              ? await renderPdfPage(pdfDocument, row.page, pdfRenderScale)
              : null
            : await loadImageFile(sourceFile);
        if (!rendered || cancelled) return;
        const image = await canvasImage(rendered.canvas);
        if (cancelled) return;
        setDrawingImage(image);
        setPageSize({ width: rendered.width, height: rendered.height });
        setLoading(false);
      } catch (reason) {
        if (cancelled) return;
        setError(
          reason instanceof Error
            ? reason.message
            : "Could not render the selected drawing page."
        );
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [document.file_type, pdfDocument, pdfRenderScale, row, sourceFile]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const update = () => {
      const bounds = element.getBoundingClientRect();
      setViewport({
        width: Math.max(1, bounds.width),
        height: Math.max(1, bounds.height),
      });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [expanded]);

  const applyView = useCallback(
    (nextMode: ViewMode) => {
      if (!row) return;
      setMode(nextMode);
      setTransform(
        nextMode === "fit"
          ? fitTransform(viewport, pageSize)
          : focusTransform(viewport, pageSize, row)
      );
    },
    [pageSize, row, viewport]
  );

  useEffect(() => {
    if (!drawingImage || !row) return;
    setTransform(
      mode === "fit"
        ? fitTransform(viewport, pageSize)
        : focusTransform(viewport, pageSize, row)
    );
  }, [drawingImage, mode, pageSize, row, viewport]);

  useEffect(() => {
    setMode("fit");
  }, [row?.annotation_id]);

  useEffect(() => {
    if (!expanded) return;
    const previousOverflow = window.document.body.style.overflow;
    window.document.body.style.overflow = "hidden";
    return () => {
      window.document.body.style.overflow = previousOverflow;
    };
  }, [expanded]);

  const annotation = useMemo<Annotation | null>(() => {
    if (!row || isChecksheetReferenceRow(row)) return null;
    return {
      id: row.annotation_id,
      number: row.balloon_number,
      label: row.label,
      value: row.specification,
      type: row.dimension_type,
      confidence: 1,
      bbox: row.bbox,
      rotation: row.rotation,
      page: row.page,
      createdAt: 0,
      kind: "dimension",
      range: row.tolerance,
      method: row.method,
      tool: row.tool,
    };
  }, [row]);

  const referenceCoordinates = useMemo(
    () => (row ? referenceDisplayCoordinates(row) : []),
    [row]
  );
  const selectedReference =
    row && isChecksheetReferenceRow(row) ? row.reference : null;
  const selectedReferenceColor = selectedReference
    ? referenceColor(selectedReference.point_index)
    : null;
  const referenceTagPlacements = useMemo(() => {
    if (!selectedReference || selectedReference.markers.length === 0) {
      return new Map<string, { x: number; y: number }>();
    }
    const size = referenceTagSize(referenceCoordinates);
    return layoutReferenceTags(
      selectedReference.markers.map((marker, index) => ({
        id: `marker:${index}`,
        glyph: marker.bbox,
      })),
      {
        width: size.width / transform.scale,
        height: size.height / transform.scale,
        gap: 7 / transform.scale,
        bounds: pageSize,
      }
    );
  }, [pageSize, referenceCoordinates, selectedReference, transform.scale]);
  const itemCaption = row ? checksheetItemCaption(row) : "No row selected";

  const zoom = useCallback(
    (factor: number) => {
      const oldScale = transform.scale;
      const nextScale = Math.max(0.05, Math.min(8, oldScale * factor));
      const center = { x: viewport.width / 2, y: viewport.height / 2 };
      const point = {
        x: (center.x - transform.x) / oldScale,
        y: (center.y - transform.y) / oldScale,
      };
      setMode("focus");
      setTransform({
        scale: nextScale,
        x: center.x - point.x * nextScale,
        y: center.y - point.y * nextScale,
      });
    },
    [transform, viewport]
  );

  const content = (
    <div
      ref={containerRef}
      className={`relative overflow-hidden bg-slate-200 ${
        expanded ? "min-h-0 flex-1" : "h-72"
      }`}
    >
      {!row && (
        <div className="flex h-full items-center justify-center px-6 text-center text-sm text-slate-500">
          Select a checksheet row to review its drawing location.
        </div>
      )}
      {row && loading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/75 text-sm text-slate-500">
          Loading page {row.page}…
        </div>
      )}
      {error && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-red-50 p-6 text-center text-sm text-red-700">
          {error}
        </div>
      )}
      {drawingImage && row && (
        <Stage
          ref={stageRef}
          width={viewport.width}
          height={viewport.height}
          x={transform.x}
          y={transform.y}
          scaleX={transform.scale}
          scaleY={transform.scale}
          draggable={expanded}
          onDragEnd={(event) =>
            setTransform((current) => ({
              ...current,
              x: event.target.x(),
              y: event.target.y(),
            }))
          }
          onWheel={(event) => {
            if (!expanded) return;
            event.evt.preventDefault();
            const stage = stageRef.current;
            const pointer = stage?.getPointerPosition();
            if (!pointer) return;
            const oldScale = transform.scale;
            const nextScale = Math.max(
              0.05,
              Math.min(8, oldScale * (event.evt.deltaY > 0 ? 0.9 : 1.1))
            );
            const point = {
              x: (pointer.x - transform.x) / oldScale,
              y: (pointer.y - transform.y) / oldScale,
            };
            setMode("focus");
            setTransform({
              scale: nextScale,
              x: pointer.x - point.x * nextScale,
              y: pointer.y - point.y * nextScale,
            });
          }}
        >
          <Layer>
            <KonvaImage
              image={drawingImage}
              width={pageSize.width}
              height={pageSize.height}
              listening={false}
            />
            {annotation && (
              <>
                <Rect
                  {...annotation.bbox}
                  stroke="#dc2626"
                  strokeWidth={2.5 / transform.scale}
                  dash={[7 / transform.scale, 4 / transform.scale]}
                  listening={false}
                />
                <Balloon
                  annotation={annotation}
                  scale={transform.scale}
                  selected
                  listening={false}
                />
              </>
            )}
            {selectedReference && selectedReferenceColor && (
              <>
                <Rect
                  {...selectedReference.row_box}
                  fill={selectedReferenceColor.wash}
                  stroke={selectedReferenceColor.fill}
                  strokeWidth={2 / transform.scale}
                  listening={false}
                />
                <Rect
                  {...selectedReference.name_box}
                  stroke={selectedReferenceColor.fill}
                  strokeWidth={3 / transform.scale}
                  dash={[6 / transform.scale, 3 / transform.scale]}
                  listening={false}
                />
                {selectedReference.markers.map((marker, index) => {
                  const placement = referenceTagPlacements.get(
                    `marker:${index}`
                  ) ?? {
                    x:
                      marker.bbox.x +
                      marker.bbox.width +
                      7 / transform.scale,
                    y: marker.bbox.y,
                  };
                  return (
                    <ReferenceMarker
                      key={`${row.annotation_id}:marker:${index}`}
                      symbol={selectedReference.symbol}
                      coordinates={referenceCoordinates}
                      marker={marker}
                      placement={placement}
                      color={selectedReferenceColor}
                      scale={transform.scale}
                      selected
                      dimmed={false}
                      onSelect={() => undefined}
                    />
                  );
                })}
              </>
            )}
          </Layer>
        </Stage>
      )}
    </div>
  );

  if (expanded) {
    return createPortal(
      <div className="fixed inset-0 z-[100] flex flex-col bg-slate-950/80 p-3 sm:p-6">
        <div className="mx-auto flex h-full w-full max-w-[1500px] flex-col overflow-hidden rounded-2xl bg-white shadow-2xl">
          <div className="relative z-10 flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-white px-4 py-3">
            <div>
              <p className="text-sm font-semibold text-slate-900">
                {document.file_name}
              </p>
              <p className="text-xs text-slate-500">
                {row ? `${itemCaption} · Page ${row.page}` : itemCaption}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => applyView("fit")}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
              >
                Fit Page
              </button>
              <button
                type="button"
                onClick={() => applyView("focus")}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
              >
                {row && isChecksheetReferenceRow(row)
                  ? "Focus Reference"
                  : "Focus Balloon"}
              </button>
              <button
                type="button"
                onClick={() => zoom(0.8)}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
              >
                −
              </button>
              <button
                type="button"
                onClick={() => zoom(1.25)}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
              >
                +
              </button>
              <button
                type="button"
                onClick={() => {
                  setMode("fit");
                  setExpanded(false);
                }}
                className="rounded-lg bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-700"
              >
                Close
              </button>
            </div>
          </div>
          {content}
        </div>
      </div>,
      window.document.body
    );
  }

  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
        <div>
          <p className="text-sm font-semibold text-slate-800">Drawing preview</p>
          <p className="text-xs text-slate-400">
            {row ? `${itemCaption} · Page ${row.page}` : "Select a row"}
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            setMode("fit");
            setExpanded(true);
          }}
          disabled={!row}
          className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
        >
          Expand
        </button>
      </div>
      {content}
    </section>
  );
}
