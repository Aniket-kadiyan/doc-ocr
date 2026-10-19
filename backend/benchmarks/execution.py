"""Production-faithful execution helpers for the real-drawing benchmark.

The application treats trustworthy PDF text as authoritative, uses OCR only
when the PDF is raster/untrusted, and merges OCR additively for hybrid pages.
Keeping that policy in the benchmark matters: calling ``segment_page`` on a
rendered PDF unconditionally measures a different product path.
"""

from __future__ import annotations

from dataclasses import replace
import time
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from gdt_config import GdtConfig, get_config as get_gdt_config
from gdt_scan_integration import augment_page_scan_result, page_scan_enabled
from native_symbol_enrichment import enrich_native_result_with_geometry
from page_layout import analyze_page_layout
from pdf_evidence import PdfPageEvidence, extract_pdf_page_evidence
from pdf_text_first import (
    merge_native_and_ocr_results,
    native_text_segment_result,
    native_vector_result_is_primary,
)
from pdf_text_layer import text_layer_regions


def render_pages(path: Path, dpi: int) -> Iterator[Image.Image]:
    """Render every page in the same coordinate space used by the scan API."""

    if path.suffix.lower() != ".pdf":
        with Image.open(path) as source:
            yield source.convert("RGB")
        return

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(document)):
            yield document[index].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        document.close()


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars, tuples, and nested scan data to JSON values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _native_pdf_inputs(
    pdf_bytes: bytes,
    *,
    page_number: int,
    target_size: tuple[int, int],
) -> tuple[dict[str, Any] | None, PdfPageEvidence]:
    """Extract the two native-PDF inputs used by the production page route."""

    try:
        text_layer = text_layer_regions(
            pdf_bytes,
            page_number - 1,
            target_size=target_size,
        )
        evidence = extract_pdf_page_evidence(
            pdf_bytes,
            page_number=page_number,
            target_size=target_size,
        )
        if not bool(text_layer.get("usable")):
            evidence = PdfPageEvidence.raster_fallback(
                page_number=page_number,
                target_size=target_size,
                error=str(
                    text_layer.get("fallback_reason")
                    or "PDF text layer is not trustworthy"
                ),
            )
        return text_layer, evidence
    except (ImportError, IndexError, RuntimeError, ValueError, OSError) as error:
        return None, PdfPageEvidence.raster_fallback(
            page_number=page_number,
            target_size=target_size,
            error=str(error)[:240],
        )


def segment_production_page(
    image: Image.Image,
    *,
    page_number: int,
    pipeline: Any,
    source_pdf_bytes: bytes | None = None,
    gdt_config: GdtConfig | None = None,
) -> dict[str, Any]:
    """Run one whole page with the application's complete production policy."""

    native_text_layer: dict[str, Any] | None = None
    native_evidence: PdfPageEvidence | None = None
    if source_pdf_bytes is not None:
        native_text_layer, native_evidence = _native_pdf_inputs(
            source_pdf_bytes,
            page_number=page_number,
            target_size=image.size,
        )

    layout = analyze_page_layout(image)
    native_result = (
        native_text_segment_result(
            native_text_layer,
            page_number=page_number,
            scope_kind="page",
            table_masks=[box.to_dict() for box in layout.table_masks],
            existing_value_boxes=(),
            source_profile=(
                native_evidence.profile_dict()
                if native_evidence is not None
                else None
            ),
        )
        if native_text_layer and native_text_layer.get("usable")
        else None
    )
    if native_result is not None:
        native_result = (
            enrich_native_result_with_geometry(image, native_result)
            or native_result
        )

    if native_vector_result_is_primary(
        native_result,
        evidence_profile=(
            native_evidence.profile if native_evidence is not None else None
        ),
    ):
        result = native_result
    else:
        result = pipeline.segment_page(
            image,
            layout=layout,
            existing_value_boxes=(),
            native_evidence=native_evidence,
        )
        if native_result is not None:
            result = merge_native_and_ocr_results(native_result, result)

    resolved_gdt_config = gdt_config or get_gdt_config()
    if page_scan_enabled(resolved_gdt_config):

        def recognize_gdt_compartment(crop: Image.Image) -> tuple[str, float]:
            recognized = pipeline.recognize(
                crop,
                compute_text_bbox=False,
                max_paddle_predictions=3,
                allow_prefix_ocr=False,
            )
            return (
                str(recognized.get("text") or ""),
                float(recognized.get("confidence") or 0.0),
            )

        result = augment_page_scan_result(
            image,
            result,
            ocr_fn=recognize_gdt_compartment,
            page=page_number,
            existing_value_boxes=(),
            config=resolved_gdt_config,
        )

    result["coordinate_space"] = "page"
    return result


def segment_document(
    path: Path,
    dpi: int,
    *,
    route: str = "page",
    pipeline: Any | None = None,
    gdt_page_scan: bool | None = None,
) -> dict[str, Any]:
    """Run a complete document and retain the full candidate lifecycle."""

    if pipeline is None:
        from ocr_pipeline import get_pipeline

        pipeline = get_pipeline()

    resolved_gdt_config = get_gdt_config()
    if gdt_page_scan is not None:
        resolved_gdt_config = replace(
            resolved_gdt_config,
            page_scan_enabled=bool(gdt_page_scan),
        )
    configured_gdt_enabled = page_scan_enabled(resolved_gdt_config)
    applied_gdt_enabled = route == "page" and configured_gdt_enabled

    source_pdf_bytes = path.read_bytes() if path.suffix.lower() == ".pdf" else None
    pages: list[dict[str, Any]] = []
    started = time.time()
    for page_number, image in enumerate(render_pages(path, dpi), start=1):
        result = (
            segment_production_page(
                image,
                page_number=page_number,
                pipeline=pipeline,
                source_pdf_bytes=source_pdf_bytes,
                gdt_config=resolved_gdt_config,
            )
            if route == "page"
            else pipeline.segment(image)
        )
        pages.append(
            {
                "page": page_number,
                "width": image.width,
                "height": image.height,
                "result": json_safe(result),
            }
        )
    return {
        "schema_version": 2,
        "route": route,
        "execution_policy": (
            "native_pdf_first" if route == "page" else "legacy_segment"
        ),
        "gdt_page_scan": {
            "policy": (
                "configured"
                if gdt_page_scan is None
                else ("forced_on" if gdt_page_scan else "forced_off")
            ),
            "enabled": applied_gdt_enabled,
            "configured_enabled": configured_gdt_enabled,
            "gdt_enabled": bool(resolved_gdt_config.enabled),
            "page_scan_enabled": bool(resolved_gdt_config.page_scan_enabled),
        },
        "source_format": path.suffix.lower().lstrip(".") or "unknown",
        "dpi": dpi,
        "seconds": time.time() - started,
        "pages": pages,
    }
