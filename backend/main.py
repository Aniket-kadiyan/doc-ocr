"""
Local in-house OCR API — PaddleOCR + TrOCR (no cloud APIs).

Run: uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
from enum import Enum
from time import perf_counter
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from checksheet_converter import (
    ChecksheetConfigurationError,
    save_checksheet_template,
)
from checksheet_index import (
    ChecksheetIndexError,
    register_template_if_missing,
)
from checksheet_routes import router as checksheet_router
from checksheet_storage import initialize_checksheet_storage
from app_logging import (
    configure_logging,
    get_logger,
    log_context,
    request_id,
    safe_exception_details,
    shutdown_logging,
)
from client_diagnostics import ClientDiagnosticEvent, record_client_diagnostic
from config_env import load_env_files, load_runtime_config
from debug_dump import dump_status
from gdt_routes import router as gdt_router
from ocr_pipeline import get_pipeline
from page_layout import (
    analyze_page_layout,
    crop_section,
)
from page_layout_cache import PageLayoutCache
from page_value_filters import normalize_page_value_text
from scan_jobs import ProgressReporter, ScanJobManager
from source_images import SourceImageError, render_source_page

# Draw config from the project's root .env.local / .env (same file the frontend
# uses). Done before reading any OCR_* setting below.
load_env_files()
runtime_config = load_runtime_config()
log_path = configure_logging(runtime_config)
logger = get_logger("api")

app = FastAPI(
    title="Doc OCR Box — PaddleOCR + TrOCR",
    description="Self-hosted engineering drawing OCR",
    version="0.2.0",
)
app.include_router(checksheet_router)
app.include_router(gdt_router)


@app.middleware("http")
async def request_diagnostics(request: Request, call_next):
    """Correlate each request without recording its query string or body."""

    correlation_id = request_id(request.headers.get("X-Request-ID"))
    started = perf_counter()
    client_ip = request.client.host if request.client else ""
    fields = {
        "request_id": correlation_id,
        "method": request.method,
        "path": request.url.path,
        "client_ip": client_ip,
    }
    with log_context(**fields):
        quiet_request = request.url.path == "/health" or (
            request.method == "GET"
            and request.url.path.startswith("/ocr/scan-jobs/")
        )
        logger.debug("HTTP request started", extra={"event": "request_started"})
        try:
            response = await call_next(request)
        except Exception as error:
            logger.error(
                "HTTP request failed",
                extra={
                    "event": "request_failed",
                    "status_code": 500,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    **safe_exception_details(error),
                },
            )
            raise
        response.headers["X-Request-ID"] = correlation_id
        completion_level = (
            logging.ERROR if response.status_code >= 500
            else logging.WARNING if response.status_code >= 400
            else logging.DEBUG if quiet_request
            else logging.INFO
        )
        logger.log(
            completion_level,
            "HTTP request completed",
            extra={
                "event": "request_completed",
                "status_code": response.status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return response

# PaddleOCR is expensive and the local service normally serves one operator.
# A single worker prevents concurrent scans from competing for the same model
# while still returning the HTTP request immediately.
scan_job_manager = ScanJobManager(max_workers=1)
page_layout_cache = PageLayoutCache(max_entries=8)
MAX_EXISTING_VALUE_BOXES = 5000

# CORS origins: an explicit OCR_CORS_ORIGINS allow-list (comma-separated) when
# set; otherwise a dev-friendly fallback that accepts any localhost port — so a
# frontend on :3000, :4000, etc. all work without editing this file.
_cors_origins = runtime_config.cors_origins
_raster_response_headers = [
    "X-Document-Page-Count",
    "X-Document-Page-Width",
    "X-Document-Page-Height",
    "X-Document-Format",
    "X-Request-ID",
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=_raster_response_headers,
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=_raster_response_headers,
    )


class DimensionType(str, Enum):
    diameter = "Diameter"
    radius = "Radius"
    angle = "Angle"
    tolerance = "Tolerance"
    linear = "Linear"
    note = "Note"
    unknown = "Unknown"


class ChecksheetTemplateRequest(BaseModel):
    """Inspection export and filename used to generate one template."""

    source_json: dict[str, Any]
    source_file_name: str = Field(min_length=1, max_length=255)


@app.post("/diagnostics/client-events", status_code=204)
def client_diagnostic(event: ClientDiagnosticEvent) -> Response:
    """Persist allow-listed frontend failure metadata for local support."""

    record_client_diagnostic(event, get_logger("frontend"))
    return Response(status_code=204)


@app.post("/documents/raster-page")
async def raster_document_page(
    file: UploadFile = File(...),
    page: int = Form(1),
) -> Response:
    """Return one PNG page from a validated PNG, JPEG, or TIFF drawing.

    PNG and JPEG normally decode in the browser. The same endpoint supports
    them as a validated fallback, while TIFF always uses this route so that
    multi-page drawings work consistently in Chromium-based browsers.
    """

    data = await file.read()
    try:
        rendered = await run_in_threadpool(
            render_source_page,
            data,
            page=page,
            filename=file.filename or "",
            content_type=file.content_type or "",
        )
    except SourceImageError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return Response(
        content=rendered.png,
        media_type="image/png",
        headers={
            "X-Document-Page-Count": str(rendered.page_count),
            "X-Document-Page-Width": str(rendered.width),
            "X-Document-Page-Height": str(rendered.height),
            "X-Document-Format": rendered.format,
        },
    )


def classify_dimension(text: str) -> DimensionType:
    t = text.strip()
    if re.search(r"[Øø]", t):
        return DimensionType.diameter
    if re.search(r"\bR\d", t) or re.match(r"^R[\d.]", t):
        return DimensionType.radius
    if re.search(r"°|['\"]", t) and re.search(r"\d", t):
        return DimensionType.angle
    if "±" in t:
        return DimensionType.tolerance
    if re.match(r"^\d+(\.\d+)?", t):
        return DimensionType.linear
    if t:
        return DimensionType.note
    return DimensionType.unknown


@app.on_event("startup")
def startup() -> None:
    configure_logging(runtime_config)
    logger.info(
        "QTracX backend starting",
        extra={
            "event": "backend_starting",
            "details": {
                **runtime_config.public_summary(),
                "log_file": str(log_path),
            },
        },
    )
    initialize_checksheet_storage()
    get_pipeline()
    st = dump_status()
    if st["enabled"]:
        logger.warning(
            "OCR debug dump is enabled",
            extra={"event": "debug_dump_enabled", "details": {"root": st["root"]}},
        )
    else:
        logger.info("OCR debug dump is disabled", extra={"event": "debug_dump_disabled"})
    logger.info("QTracX backend ready", extra={"event": "backend_ready"})


@app.on_event("shutdown")
def shutdown() -> None:
    logger.info("QTracX backend stopping", extra={"event": "backend_stopping"})
    scan_job_manager.shutdown(wait=False)
    logger.info("QTracX backend stopped", extra={"event": "backend_stopped"})
    shutdown_logging()


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {"status": "ok", **pipeline.status, "debug_dump": dump_status()}


@app.post("/checksheet/templates")
def create_checksheet_template(
    request: ChecksheetTemplateRequest,
) -> dict[str, Any]:
    """Convert an inspection export and create/replace its template JSON.

    The destination is deliberately read only from CHECKSHEET_TEMPLATE_DIR on
    the backend.  A browser request cannot choose an arbitrary filesystem path.
    """
    try:
        saved = save_checksheet_template(
            source_json=request.source_json,
            source_file_name=request.source_file_name,
            overwrite=True,
        )

        # index.json is located beside the generated template files.
        index_path = saved.path.parent / "index.json"

        try:
            # Always call the idempotent registration function. It returns
            # False without rewriting index.json when the entry already exists.
            index_updated = register_template_if_missing(
                index_path=index_path,
                template_id=saved.template["template_id"],
                file_name=saved.path.name,
                template_name=saved.template["template_name"]["en"],
                users=("admin_m",),
            )
        except (ChecksheetIndexError, OSError) as exc:
            # The template has already been saved. Retrying the request is safe:
            # the helper will not create duplicate index entries.
            raise HTTPException(
                status_code=500,
                detail=(
                    "The template file was saved, but index.json could not "
                    f"be updated: {exc}"
                ),
            ) from exc

    except ChecksheetConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        # Preserve the OS message because permission and invalid-path errors are
        # important diagnostics for an on-premises Windows deployment.
        raise HTTPException(
            status_code=500,
            detail=f"Could not save checksheet template: {exc}",
        ) from exc

    return {
            "template_id": saved.template["template_id"],
            "template_name": saved.template["template_name"]["en"],
            "file_name": saved.path.name,
            "row_count": max(len(saved.template["items"]) - 1, 0),
            "replaced": saved.replaced,
            "index_updated": index_updated,
        }


@app.post("/ocr/recognize")
async def recognize_region(
    file: UploadFile = File(...),
    debug_dump: bool = Query(
        False,
        description="Write pipeline steps to backend/debug_output/",
    ),
    debug_dump_force: bool = Query(
        False,
        description="Re-dump even if this crop was saved before (.done)",
    ),
    x_debug_dump: str | None = Header(None, alias="X-Debug-Dump"),
    x_debug_dump_force: str | None = Header(None, alias="X-Debug-Dump-Force"),
    gdt: bool = Query(
        False,
        description=(
            "Also run GD&T symbol recognition over this region and return it "
            "under a 'gdt' key. Off by default so the existing response and "
            "the existing cost are unchanged."
        ),
    ),
) -> dict[str, Any]:
    """PaddleOCR primary; TrOCR when confidence < 95%."""
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    pipeline = get_pipeline()
    truthy = {"1", "true", "yes", "on"}
    req_dump = debug_dump or (x_debug_dump or "").strip().lower() in truthy
    req_force = debug_dump_force or (x_debug_dump_force or "").strip().lower() in truthy
    result = pipeline.recognize(
        image,
        image_bytes=raw,
        debug_dump=req_dump,
        debug_dump_force=req_force,
    )
    dim_type = result.get("type") or classify_dimension(result["text"]).value

    response: dict[str, Any] = {
        **result,
        "type": dim_type if isinstance(dim_type, str) else dim_type.value,
    }

    # Additive: present only when asked for. A failure here degrades this one
    # key to an error record and leaves the OCR result intact, because an
    # operator mid-inspection needs the value they drew a box around even when
    # symbol recognition cannot run.
    if gdt:
        try:
            import gdt_pipeline
            from gdt_routes import _ocr_hook

            page_result = await run_in_threadpool(
                gdt_pipeline.analyze_page,
                image,
                ocr_fn=_ocr_hook(),
                source="ocr/recognize",
            )
            response["gdt"] = page_result.to_dict()
        except Exception as exc:  # noqa: BLE001
            response["gdt"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "frames": [],
                "symbols": [],
            }

    return response


@app.post("/ocr/segment")
async def segment_region(
    file: UploadFile = File(...),
    debug_dump: bool = Query(
        False,
        description="Write pipeline steps to backend/debug_output/",
    ),
    debug_dump_force: bool = Query(
        False,
        description="Re-dump even if a sub-crop was saved before (.done)",
    ),
    x_debug_dump: str | None = Header(None, alias="X-Debug-Dump"),
    x_debug_dump_force: str | None = Header(None, alias="X-Debug-Dump-Force"),
    keywords: str = Form(
        "",
        description=(
            "Comma-separated title-block labels to look for "
            "(the client's 'Keywords to look for' setting), e.g. 'DWG,REV'"
        ),
    ),
) -> dict[str, Any]:
    """Auto-segment a multi-value selection into per-dimension OCR results."""
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    pipeline = get_pipeline()
    truthy = {"1", "true", "yes", "on"}
    req_dump = debug_dump or (x_debug_dump or "").strip().lower() in truthy
    req_force = debug_dump_force or (x_debug_dump_force or "").strip().lower() in truthy

    title_keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    seg = pipeline.segment(
        image,
        debug_dump=req_dump,
        debug_dump_force=req_force,
        title_keywords=title_keywords,
    )

    return _serialize_segment_result(seg)


def _serialize_segment_result(seg: dict[str, Any]) -> dict[str, Any]:
    """Apply the API's dimension classification to one complete scan result."""

    def serialize_regions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for r in items:
            text = str(r.get("text") or "")
            dim_type = r.get("type") or classify_dimension(text).value
            serialized.append(
                {
                    **r,
                    "text": text,
                    "type": (
                        dim_type if isinstance(dim_type, str) else dim_type.value
                    ),
                }
            )
        return serialized

    regions = serialize_regions(list(seg.get("regions", [])))
    review_candidates = serialize_regions(
        list(seg.get("review_candidates", []))
    )

    recognized_count = int(
        seg.get(
            "recognized_count",
            sum(
                1
                for region in regions
                if region.get("recognized", bool(region["text"].strip()))
            ),
        )
    )
    detected_count = int(seg.get("detected_count", len(regions)))
    unread_count = int(
        seg.get("unread_count", max(0, detected_count - recognized_count))
    )
    eligible_count = int(seg.get("eligible_count", len(regions)))
    excluded_count = int(
        seg.get("excluded_count", max(0, recognized_count - eligible_count))
    )
    review_count = int(seg.get("review_count", len(review_candidates)))
    filter_rule_counts = {
        str(name): int(count)
        for name, count in dict(seg.get("filter_rule_counts", {})).items()
    }
    return {
        "count": len(regions),
        "detected_count": detected_count,
        "recognized_count": recognized_count,
        "eligible_count": eligible_count,
        "excluded_count": excluded_count,
        "review_count": review_count,
        "unread_count": unread_count,
        "skipped_existing_count": int(seg.get("skipped_existing_count", 0)),
        "filter_rule_counts": filter_rule_counts,
        "coordinate_space": str(seg.get("coordinate_space", "scope")),
        "regions": regions,
        "review_candidates": review_candidates,
        "candidate_outcomes": list(seg.get("candidate_outcomes", [])),
    }


def _map_section_result_to_page(
    seg: dict[str, Any],
    *,
    origin: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Restore light-filtered section values to page coordinates."""

    origin_x, origin_y = origin
    mapped_regions: list[dict[str, Any]] = []
    overlay_candidates: list[dict[str, object]] = []
    source_regions = list(seg.get("regions", []))

    for region in source_regions:
        local = dict(region.get("bbox", {}))
        page_bbox = {
            "x": round(origin_x + float(local.get("x", 0.0)), 1),
            "y": round(origin_y + float(local.get("y", 0.0)), 1),
            "width": round(float(local.get("width", 0.0)), 1),
            "height": round(float(local.get("height", 0.0)), 1),
        }
        text = normalize_page_value_text(str(region.get("text") or ""))
        if not text:
            continue
        rule = str(region.get("page_filter_rule") or "section_engineering_value")
        reason = str(
            region.get("page_filter_reason")
            or "Matches the light section engineering-value syntax gate"
        )
        overlay_candidates.append(
            {
                "bbox": page_bbox,
                "state": "eligible",
                "text": text,
                "reason": reason,
                "rule": rule,
            }
        )
        mapped_regions.append(
            {
                **region,
                "bbox": page_bbox,
                "text": text,
                "page_filter_rule": rule,
                "page_filter_reason": reason,
            }
        )

    mapped_outcomes: list[dict[str, Any]] = []
    for outcome in list(seg.get("candidate_outcomes", [])):
        local = dict(outcome.get("bbox", {}))
        mapped_outcome = {
            **outcome,
            "bbox": {
                "x": round(origin_x + float(local.get("x", 0.0)), 1),
                "y": round(origin_y + float(local.get("y", 0.0)), 1),
                "width": round(float(local.get("width", 0.0)), 1),
                "height": round(float(local.get("height", 0.0)), 1),
            },
        }
        oriented = outcome.get("oriented_box")
        if isinstance(oriented, dict):
            mapped_outcome["oriented_box"] = {
                **oriented,
                "x": round(origin_x + float(oriented.get("x", 0.0)), 1),
                "y": round(origin_y + float(oriented.get("y", 0.0)), 1),
            }
        mapped_outcomes.append(mapped_outcome)

    # Review candidates carry geometry too. They were always empty while the
    # section scan used the light pipeline, so passing them through untouched
    # was harmless; the page pipeline produces real ones and they would
    # otherwise reach the client still in section coordinates.
    mapped_reviews: list[dict[str, Any]] = []
    for candidate in list(seg.get("review_candidates", [])):
        local = dict(candidate.get("bbox", {}))
        mapped_candidate = {
            **candidate,
            "bbox": {
                "x": round(origin_x + float(local.get("x", 0.0)), 1),
                "y": round(origin_y + float(local.get("y", 0.0)), 1),
                "width": round(float(local.get("width", 0.0)), 1),
                "height": round(float(local.get("height", 0.0)), 1),
            },
        }
        oriented = candidate.get("oriented_box")
        if isinstance(oriented, dict):
            mapped_candidate["oriented_box"] = {
                **oriented,
                "x": round(origin_x + float(oriented.get("x", 0.0)), 1),
                "y": round(origin_y + float(oriented.get("y", 0.0)), 1),
            }
        mapped_reviews.append(mapped_candidate)

    return (
        {
            **seg,
            "count": len(mapped_regions),
            "eligible_count": len(mapped_regions),
            "coordinate_space": "page",
            "regions": mapped_regions,
            "candidate_outcomes": mapped_outcomes,
            "review_candidates": mapped_reviews,
        },
        overlay_candidates,
    )


def _parse_existing_value_boxes(payload: str) -> list[dict[str, float]]:
    """Validate saved balloon geometry supplied for pre-OCR exclusion."""

    try:
        decoded = json.loads(payload or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="existing_value_boxes must be valid JSON",
        ) from exc
    if not isinstance(decoded, list):
        raise HTTPException(
            status_code=422,
            detail="existing_value_boxes must be a JSON array",
        )
    if len(decoded) > MAX_EXISTING_VALUE_BOXES:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_EXISTING_VALUE_BOXES} existing boxes are allowed",
        )

    boxes: list[dict[str, float]] = []
    for item in decoded:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=422,
                detail="Each existing value box must be an object",
            )
        try:
            box = {
                key: float(item[key])
                for key in ("x", "y", "width", "height")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="Each existing value box needs numeric x, y, width, and height",
            ) from exc
        if not all(math.isfinite(value) for value in box.values()):
            raise HTTPException(
                status_code=422,
                detail="Existing value box coordinates must be finite",
            )
        if box["width"] <= 0 or box["height"] <= 0:
            raise HTTPException(
                status_code=422,
                detail="Existing value box dimensions must be positive",
            )
        boxes.append(box)
    return boxes


@app.post("/ocr/scan-jobs", status_code=202)
async def create_scan_job(
    file: UploadFile = File(...),
    scope_kind: str = Form("section"),
    page: int = Form(1),
    scope_x: float = Form(0),
    scope_y: float = Form(0),
    scope_width: float = Form(...),
    scope_height: float = Form(...),
    existing_value_boxes: str = Form("[]"),
    debug_dump: bool = Query(
        False,
        description="Write pipeline steps to backend/debug_output/",
    ),
    debug_dump_force: bool = Query(
        False,
        description="Re-dump even if a sub-crop was saved before (.done)",
    ),
    x_debug_dump: str | None = Header(None, alias="X-Debug-Dump"),
    x_debug_dump_force: str | None = Header(None, alias="X-Debug-Dump-Force"),
) -> dict[str, Any]:
    """Queue an atomic section or whole-page scan and return its job id."""
    if scope_kind not in {"section", "page"}:
        raise HTTPException(status_code=422, detail="Invalid scan scope")
    if page < 1 or scope_width <= 0 or scope_height <= 0:
        raise HTTPException(status_code=422, detail="Invalid scan page or bounds")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="The scan image is empty")
    saved_value_boxes = _parse_existing_value_boxes(existing_value_boxes)

    truthy = {"1", "true", "yes", "on"}
    req_dump = debug_dump or (x_debug_dump or "").strip().lower() in truthy
    req_force = (
        debug_dump_force
        or (x_debug_dump_force or "").strip().lower() in truthy
    )
    fingerprint = hashlib.sha256(raw).hexdigest()
    scope_bbox = {
        "x": scope_x,
        "y": scope_y,
        "width": scope_width,
        "height": scope_height,
    }

    def run_scan(report_progress: ProgressReporter) -> dict[str, Any]:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if (
            scope_x >= image.width
            or scope_y >= image.height
            or scope_x + scope_width <= 0
            or scope_y + scope_height <= 0
        ):
            raise ValueError("The scan scope does not intersect the uploaded page")

        report_progress(
            stage="layout",
            message="Analysing page tables and processing panels",
            percent=2,
            completed=0,
            total=1,
            operation_label="Page layout analysis",
        )
        layout, cache_hit = page_layout_cache.get_or_create(
            fingerprint,
            lambda: analyze_page_layout(image),
        )
        initial_overlay = layout.overlay(scope_kind=scope_kind)
        report_progress(
            stage="layout",
            message=(
                f"{'Reused' if cache_hit else 'Prepared'} page layout: "
                f"{len(layout.table_masks)} table masks and "
                f"{len(layout.panels)} adaptive panels"
            ),
            percent=8,
            completed=1,
            total=1,
            tile_total=len(layout.panels) if scope_kind == "page" else 0,
            operation_label=(
                "Cached page layout" if cache_hit else "Page layout ready"
            ),
            overlay=initial_overlay,
        )

        pipeline = get_pipeline()
        if scope_kind == "page":
            seg = pipeline.segment_page(
                image,
                layout=layout,
                debug_dump=req_dump,
                debug_dump_force=req_force,
                progress_callback=report_progress,
                existing_value_boxes=saved_value_boxes,
            )
            seg["coordinate_space"] = "page"
        else:
            section_image, section_origin, _clipped_scope = crop_section(
                image,
                scope_bbox,
            )
            section_existing_boxes = [
                {
                    "x": box["x"] - section_origin[0],
                    "y": box["y"] - section_origin[1],
                    "width": box["width"],
                    "height": box["height"],
                }
                for box in saved_value_boxes
                if (
                    box["x"] + box["width"] > section_origin[0]
                    and box["y"] + box["height"] > section_origin[1]
                    and box["x"] < section_origin[0] + section_image.width
                    and box["y"] < section_origin[1] + section_image.height
                )
            ]

            def report_section_progress(**event: Any) -> None:
                # Keep the section detector/recognizer unchanged while fitting
                # its existing 1–99% progress into the post-layout range.
                raw_percent = int(event.pop("percent", 0))
                report_progress(
                    **event,
                    percent=max(10, 10 + int(raw_percent * 0.88)),
                )

            # Both scopes run the same pipeline. Measured on the benchmark
            # drawings at fixture resolution, the page route reads far more of
            # the same sheet than the light section route does — 16/19 against
            # 11/19 on 47630, 16/24 against 10/24 on 56103-0182B, 11/12 against
            # 8/12 on BS1801006.020 — and it is the route that now matches or
            # beats the pre-merge pipeline. A section scan is the same problem
            # on a smaller image, so it gets the same treatment.
            seg = pipeline.segment_page(
                section_image,
                debug_dump=req_dump,
                debug_dump_force=req_force,
                progress_callback=report_section_progress,
                existing_value_boxes=section_existing_boxes,
            )
            seg, section_overlay_candidates = _map_section_result_to_page(
                seg,
                origin=section_origin,
            )
            report_progress(
                stage="finalizing",
                message=(
                    f"Prepared {seg['eligible_count']} section values; "
                    f"ignored {seg['excluded_count']} non-value OCR objects"
                ),
                percent=99,
                completed=int(seg["detected_count"]),
                total=int(seg["detected_count"]),
                candidate_count=int(seg["eligible_count"]),
                operation_label="Section light filtering complete",
                overlay=layout.overlay(
                    scope_kind="section",
                    candidates=section_overlay_candidates,
                ),
            )
        return _serialize_segment_result(seg)

    return scan_job_manager.submit(
        run_scan,
        metadata={
            "scope_kind": scope_kind,
            "page": page,
            "scope_bbox": scope_bbox,
            # Liveness thresholds scale with the actual selected drawing area.
            # This is telemetry only; it does not impose a scan timeout.
            "scope_pixel_area": scope_width * scope_height,
            # This is not a resume checkpoint yet. It gives the future
            # stop/resume design a stable identity for the complete page.
            "page_fingerprint": fingerprint,
            "existing_value_box_count": len(saved_value_boxes),
        },
    )


@app.get("/ocr/scan-jobs/{job_id}")
def get_scan_job(job_id: str) -> dict[str, Any]:
    """Poll genuine scan progress; candidates appear only after success."""
    job = scan_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Scan job not found")
    return job


@app.post("/ocr/scan-jobs/{job_id}/cancel")
def cancel_scan_job(job_id: str) -> dict[str, Any]:
    """Stop a scan cooperatively without publishing partial results."""

    job = scan_job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Scan job not found")
    return job


@app.post("/ocr/reference-points")
async def read_reference_points(
    file: UploadFile = File(..., description="The whole rendered sheet"),
    display_scale: float = Form(
        1.0,
        description=(
            "Submitted pixels per annotation pixel, so geometry comes back "
            "in the coordinates the drawing's annotations already use"
        ),
    ),
) -> dict[str, Any]:
    """
    Read the sheet's coordinate reference table and locate its point names.

    Separate from /ocr/scan-jobs because nothing here is a dimension: the
    result is one table of named points plus every place those names are
    printed in the views, and the page scanner masks ruled tables out before
    it starts.

    Submit the sheet at the highest resolution available. The point names are
    printed at note size, and recall on the benchmark drawing went from 10 of
    17 markers at 576dpi to 16 of 17 at 720dpi with no false positives at
    either.
    """

    scale = display_scale if display_scale > 0 else 1.0
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    # Off the event loop. This is tens of seconds of blocking CPU on a sheet
    # rendered for point names, and run inline it stops the server answering
    # anything — including the progress polls of the scan that just finished
    # and asked for this, which then looks like the app has hung.
    result = await run_in_threadpool(get_pipeline().reference_points, image)

    def box(layout: Any) -> dict[str, float]:
        return {
            "x": round(layout.x / scale, 1),
            "y": round(layout.y / scale, 1),
            "width": round(layout.width / scale, 1),
            "height": round(layout.height / scale, 1),
        }

    table = result["table"]
    if table is None:
        return {
            "found": False,
            "tableCount": result["table_count"],
            "table": None,
            "unmatchedSymbols": [],
        }

    markers_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for marker in result["markers"]:
        markers_by_symbol.setdefault(marker.symbol, []).append(
            {
                "bbox": box(marker.bbox),
                "score": marker.score,
                "margin": marker.margin,
            }
        )

    points = [
        {
            "symbol": point.symbol,
            "coordinates": [
                {"axis": axis, "value": value}
                for axis, value in point.coordinates
            ],
            "nameBox": box(point.name_box),
            "rowBox": box(point.row_box),
            "markers": markers_by_symbol.get(point.symbol, []),
        }
        for point in table.points
    ]
    return {
        "found": True,
        "tableCount": result["table_count"],
        "table": {
            "bbox": box(table.bbox),
            "headerBox": box(table.header_box),
            "axes": list(table.axes),
            "nameHeader": table.name_header,
            "points": points,
        },
        # Named in the table but never found in the views. Surfaced rather
        # than hidden: the row is still worth showing, and a blank marker
        # count is the only honest way to say the callout was not located.
        "unmatchedSymbols": [
            point["symbol"] for point in points if not point["markers"]
        ],
    }


@app.post("/ocr/text-layer")
async def read_pdf_text_layer(
    file: UploadFile = File(..., description="The drawing PDF itself"),
    page: int = Form(1, description="1-based page number"),
    render_scale: float = Form(
        1.5,
        description=(
            "The client's PDF raster scale, so boxes come back in the same "
            "pixels its annotations already use"
        ),
    ),
) -> dict[str, Any]:
    """
    Read one page's callouts from the PDF's own text layer.

    A vector drawing already carries every value as text with exact glyph
    boxes, so this returns them with no OCR at all — perfectly accurate and
    about three orders of magnitude faster. ``usable`` is False for a raster
    scan, which is the signal to run the ordinary scan instead.
    """
    from pdf_text_layer import text_layer_regions

    if page < 1:
        raise HTTPException(status_code=400, detail="page must be 1 or greater")
    if not 0.1 <= render_scale <= 20:
        raise HTTPException(status_code=400, detail="render_scale is out of range")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    try:
        result = text_layer_regions(data, page - 1, dpi=72.0 * render_scale)
    except IndexError:
        raise HTTPException(status_code=400, detail="That page is not in the PDF")
    except Exception as error:  # a damaged or non-PDF upload
        raise HTTPException(
            status_code=400, detail=f"Could not read the PDF text layer: {error}"
        )

    width, height = result["page_size"]
    serialized = _serialize_segment_result(
        {
            "regions": result["regions"],
            "review_candidates": [],
            "coordinate_space": "page",
            "detected_count": len(result["regions"]) + len(result["excluded"]),
            "eligible_count": len(result["regions"]),
            "excluded_count": len(result["excluded"]),
            "review_count": 0,
        }
    )
    return {
        **serialized,
        "usable": result["usable"],
        "page_width": width,
        "page_height": height,
        "fonts": result["fonts"],
    }


@app.post("/ocr/title-fields")
async def read_title_fields(
    file: UploadFile = File(...),
    keywords: str = Form(
        "",
        description="Comma-separated title-block labels, e.g. 'DWG,REV'",
    ),
) -> dict[str, Any]:
    """
    Read the title block of a WHOLE sheet for the configured keywords.

    Separate from /ocr/segment because the title block's position does not
    depend on whatever rectangle the user drew for Auto-Segment.
    """
    wanted = [k.strip() for k in keywords.split(",") if k.strip()]
    if not wanted:
        return {"fields": []}

    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    fields = get_pipeline().title_fields(image, wanted)

    return {
        "fields": [
            {
                "keyword": f["keyword"],
                "label": f["label"],
                "value": f["value"],
                "confidence": f["confidence"],
                "bbox": (
                    {
                        "x": round(f["bbox"]["x"], 1),
                        "y": round(f["bbox"]["y"], 1),
                        "width": round(f["bbox"]["w"], 1),
                        "height": round(f["bbox"]["h"], 1),
                    }
                    if f["bbox"]
                    else None
                ),
            }
            for f in fields
        ]
    }


@app.post("/training/export-labels")
async def export_training_labels(
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "count": len(annotations),
        "format": "engineering_dimension_v1",
        "labels": [
            {
                "text": a.get("value", ""),
                "label": a.get("label", ""),
                "type": a.get("type", "Unknown"),
                "bbox": a.get("bbox"),
            }
            for a in annotations
        ],
    }


if __name__ == "__main__":
    # Launch the server using host/port/reload from the config file.
    # Run from this directory so uvicorn's reloader can import "main:app".
    import uvicorn

    uvicorn.run(
        "main:app",
        host=runtime_config.backend_host,
        port=runtime_config.backend_port,
        reload=runtime_config.backend_reload,
    )
