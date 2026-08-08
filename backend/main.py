"""
Local in-house OCR API — PaddleOCR + TrOCR (no cloud APIs).

Run: uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import io
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
    regions: list[dict[str, Any]] = []
    for r in seg.get("regions", []):
        text = str(r.get("text") or "")
        dim_type = r.get("type") or classify_dimension(text).value
        regions.append(
            {
                **r,
                "text": text,
                "type": dim_type if isinstance(dim_type, str) else dim_type.value,
            }
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
    return {
        "count": len(regions),
        "detected_count": detected_count,
        "recognized_count": recognized_count,
        "unread_count": unread_count,
        "regions": regions,
    }


@app.post("/ocr/scan-jobs", status_code=202)
async def create_scan_job(
    file: UploadFile = File(...),
    scope_kind: str = Form("section"),
    page: int = Form(1),
    scope_x: float = Form(0),
    scope_y: float = Form(0),
    scope_width: float = Form(...),
    scope_height: float = Form(...),
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

    truthy = {"1", "true", "yes", "on"}
    req_dump = debug_dump or (x_debug_dump or "").strip().lower() in truthy
    req_force = (
        debug_dump_force
        or (x_debug_dump_force or "").strip().lower() in truthy
    )

    def run_scan(report_progress: ProgressReporter) -> dict[str, Any]:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        pipeline = get_pipeline()
        if scope_kind == "page":
            seg = pipeline.segment_page(
                image,
                debug_dump=req_dump,
                debug_dump_force=req_force,
                progress_callback=report_progress,
            )
        else:
            seg = pipeline.segment(
                image,
                debug_dump=req_dump,
                debug_dump_force=req_force,
                progress_callback=report_progress,
            )
        return _serialize_segment_result(seg)

    return scan_job_manager.submit(
        run_scan,
        metadata={
            "scope_kind": scope_kind,
            "page": page,
            "scope_bbox": {
                "x": scope_x,
                "y": scope_y,
                "width": scope_width,
                "height": scope_height,
            },
            # Liveness thresholds scale with the actual selected drawing area.
            # This is telemetry only; it does not impose a scan timeout.
            "scope_pixel_area": scope_width * scope_height,
            # This is not a resume checkpoint yet.  It gives the future
            # stop/resume design a stable identity for the exact scanned crop.
            "crop_fingerprint": hashlib.sha256(raw).hexdigest(),
        },
    )


@app.get("/ocr/scan-jobs/{job_id}")
def get_scan_job(job_id: str) -> dict[str, Any]:
    """Poll genuine scan progress; candidates appear only after success."""
    job = scan_job_manager.get(job_id)
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
