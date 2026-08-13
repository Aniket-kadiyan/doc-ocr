"""
Local in-house OCR API — PaddleOCR + TrOCR (no cloud APIs).

Run: uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from enum import Enum
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
from config_env import get_cors_origins, load_env_files
from debug_dump import dump_status
from ocr_pipeline import get_pipeline
from page_layout import (
    analyze_page_layout,
    crop_section,
)
from page_layout_cache import PageLayoutCache
from page_value_filters import normalize_page_value_text
from scan_jobs import ProgressReporter, ScanJobManager

# Draw config from the project's root .env.local / .env (same file the frontend
# uses). Done before reading any OCR_* setting below.
load_env_files()

app = FastAPI(
    title="Doc OCR Box — PaddleOCR + TrOCR",
    description="Self-hosted engineering drawing OCR",
    version="0.2.0",
)

# PaddleOCR is expensive and the local service normally serves one operator.
# A single worker prevents concurrent scans from competing for the same model
# while still returning the HTTP request immediately.
scan_job_manager = ScanJobManager(max_workers=1)
page_layout_cache = PageLayoutCache(max_entries=8)
MAX_EXISTING_VALUE_BOXES = 5000

# CORS origins: an explicit OCR_CORS_ORIGINS allow-list (comma-separated) when
# set; otherwise a dev-friendly fallback that accepts any localhost port — so a
# frontend on :3000, :4000, etc. all work without editing this file.
_cors_origins = get_cors_origins()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
    get_pipeline()
    st = dump_status()
    if st["enabled"]:
        print(f"[debug_dump] ON → {st['root']}")  # noqa: T201
    else:
        print(  # noqa: T201
            "[debug_dump] OFF — set DEBUG_DUMP=1 on server, or use ?debug_dump=1 per request"
        )


@app.on_event("shutdown")
def shutdown() -> None:
    scan_job_manager.shutdown(wait=False)


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

    return {
        **result,
        "type": dim_type if isinstance(dim_type, str) else dim_type.value,
    }


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
) -> dict[str, Any]:
    """Auto-segment a multi-value selection into per-dimension OCR results."""
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    pipeline = get_pipeline()
    truthy = {"1", "true", "yes", "on"}
    req_dump = debug_dump or (x_debug_dump or "").strip().lower() in truthy
    req_force = debug_dump_force or (x_debug_dump_force or "").strip().lower() in truthy

    seg = pipeline.segment(
        image,
        debug_dump=req_dump,
        debug_dump_force=req_force,
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
        mapped_outcomes.append(
            {
                **outcome,
                "bbox": {
                    "x": round(origin_x + float(local.get("x", 0.0)), 1),
                    "y": round(origin_y + float(local.get("y", 0.0)), 1),
                    "width": round(float(local.get("width", 0.0)), 1),
                    "height": round(float(local.get("height", 0.0)), 1),
                },
            }
        )

    return (
        {
            **seg,
            "count": len(mapped_regions),
            "eligible_count": len(mapped_regions),
            "coordinate_space": "page",
            "regions": mapped_regions,
            "candidate_outcomes": mapped_outcomes,
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

            seg = pipeline.segment(
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

    from config_env import get_bool, get_int, get_str

    uvicorn.run(
        "main:app",
        host=get_str("OCR_HOST", "127.0.0.1"),
        port=get_int("OCR_PORT", 8000),
        reload=get_bool("OCR_RELOAD", True),
    )
