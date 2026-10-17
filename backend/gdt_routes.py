"""
FastAPI routes for GD&T symbol recognition.

Additive by design
------------------
Every route here is new. No existing response shape changes, so a client that
knows nothing about GD&T keeps working unchanged — which is the backward-
compatibility requirement, met by not touching the contracts rather than by
versioning them.

``/ocr/recognize`` gains one *optional* query flag, ``gdt=1``. Default off, so
the existing behaviour and the existing cost are untouched; when set, the
response carries one extra ``gdt`` key alongside everything it already
returned.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, Field

import gdt_corrections
import gdt_pipeline
import gdt_registry as reg
from gdt_config import get_config

router = APIRouter(tags=["gdt"])


def _ocr_hook():
    """The project's own OCR pipeline, wrapped for ``fcf_parse``.

    A cell that fails to read must not take the frame down with it, so this
    returns empty text rather than raising — the parser then reports the
    compartment as unreadable, which is the honest outcome.
    """
    from ocr_pipeline import get_pipeline

    pipeline = get_pipeline()

    def ocr_fn(image: Image.Image) -> tuple[str, float]:
        try:
            result = pipeline.recognize(image)
            return (result.get("text") or "", float(result.get("confidence") or 0.0))
        except Exception:  # noqa: BLE001
            return ("", 0.0)

    return ocr_fn


# ── The registry ─────────────────────────────────────────────────────────────


@router.get("/gdt/registry")
def get_registry() -> dict[str, Any]:
    """The complete symbol registry.

    Served rather than duplicated in the frontend: the names, meanings,
    standards and confusions a reviewer sees in the UI come from the same
    table the recognizer decides with, so the two cannot drift.
    """
    return reg.to_dict()


@router.get("/gdt/registry/{symbol_id}")
def get_symbol(symbol_id: str) -> dict[str, Any]:
    symbol = reg.find(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"unknown symbol {symbol_id!r}")
    return {
        **symbol.to_dict(),
        "confusable_with_resolved": [
            reg.get(other).to_dict()["name"]
            for other in reg.confusion_set(symbol_id)
        ],
    }


@router.get("/gdt/config")
def get_gdt_config() -> dict[str, Any]:
    """The thresholds in force, so a reviewer can see why something was
    reported ambiguous rather than confirmed."""
    return get_config().to_dict()


# ── Recognition ──────────────────────────────────────────────────────────────


@router.post("/ocr/gdt-symbols")
async def recognize_gdt_symbols(
    file: UploadFile = File(...),
    page: int = Query(1, ge=1, description="Page number, for the result"),
    declared_standard: str = Form(
        "",
        description=(
            "The standard the drawing is released to, e.g. 'ASME Y14.5-2018' "
            "or 'ISO 1101'. Used only to report symbols that do not belong to "
            "it; never to rewrite a reading."
        ),
    ),
    read_text: bool = Query(
        True,
        description="Run OCR on frame compartments (needed for tolerances and datums)",
    ),
    detect_loose_symbols: bool = Query(
        True, description="Also look for symbols outside feature control frames"
    ),
) -> dict[str, Any]:
    """Find feature control frames and GD&T symbols on one rendered page.

    The image is whatever the client renders — a PDF page, a scan, a PNG, or
    a high-resolution crop. Coordinates come back in that image's own pixel
    space, so the caller maps them with the same transform it already uses for
    OCR boxes.
    """
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"could not read the image: {exc}"
        ) from exc

    ocr_fn = _ocr_hook() if read_text else None
    result = await run_in_threadpool(
        gdt_pipeline.analyze_page,
        image,
        page=page,
        ocr_fn=ocr_fn,
        declared_standard=declared_standard.strip() or None,
        source=file.filename,
        detect_loose_symbols=detect_loose_symbols,
    )
    payload = result.to_dict()
    # The hash lets a correction be tied back to this drawing without this
    # service storing the drawing.
    payload["document_hash"] = hashlib.sha256(raw).hexdigest()[:32]
    return payload


# ── Corrections ──────────────────────────────────────────────────────────────


class CorrectionRequest(BaseModel):
    """One operator judgement about one detection."""

    action: str = Field(
        description="reclassify | reject | confirm | add",
    )
    detected_symbol_id: str | None = None
    corrected_symbol_id: str | None = None
    page: int = Field(default=1, ge=1)
    bounding_box: dict[str, float] | None = None
    detected_confidence: float | None = None
    detected_status: str | None = None
    document_hash: str | None = Field(
        default=None,
        max_length=128,
        description="Content hash from the recognition response",
    )
    note: str | None = Field(default=None, max_length=2000)


@router.post("/gdt/corrections", status_code=201)
def post_correction(request: CorrectionRequest) -> dict[str, Any]:
    """Record a correction so it can inform evaluation and future training."""
    correction = gdt_corrections.Correction(
        action=request.action,
        detected_symbol_id=request.detected_symbol_id,
        corrected_symbol_id=request.corrected_symbol_id,
        page=request.page,
        bounding_box=request.bounding_box,
        detected_confidence=request.detected_confidence,
        detected_status=request.detected_status,
        document_hash=request.document_hash,
        note=request.note,
    )
    try:
        saved = gdt_corrections.record(correction)
    except gdt_corrections.CorrectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return saved.to_dict()


@router.get("/gdt/corrections")
def list_corrections(
    document_hash: str | None = Query(None, max_length=128),
) -> dict[str, Any]:
    entries = gdt_corrections.read_all(document_hash=document_hash)
    return {"count": len(entries), "corrections": entries}


@router.get("/gdt/corrections/summary")
def corrections_summary() -> dict[str, Any]:
    """Counts by action, and which symbol pairs are actually confused.

    The confusion counts are the operationally useful number: they say what
    the recognizer gets wrong on real drawings, which no synthetic benchmark
    can tell you.
    """
    return gdt_corrections.summary()
