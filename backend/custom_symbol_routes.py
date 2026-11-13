"""FastAPI routes for the shared library of user-defined symbols."""

from __future__ import annotations

from typing import NoReturn

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from auth_routes import require_protected_request
from auth_service import AuthenticatedSession
from custom_symbols import (
    ALLOWED_CATEGORIES,
    DEFAULT_CATEGORY,
    DEFAULT_THRESHOLD,
    MAX_THRESHOLD,
    MAX_UPLOAD_BYTES,
    MIN_THRESHOLD,
    CustomSymbol,
    CustomSymbolConflictError,
    CustomSymbolError,
    CustomSymbolNotFoundError,
    CustomSymbolValidationError,
    get_custom_symbol_store,
)

router = APIRouter(tags=["custom-symbols"])


def _raise(error: CustomSymbolError) -> NoReturn:
    if isinstance(error, CustomSymbolNotFoundError):
        status_code = 404
    elif isinstance(error, CustomSymbolConflictError):
        status_code = 409
    elif isinstance(error, CustomSymbolValidationError):
        status_code = 422
    else:
        status_code = 500
    raise HTTPException(status_code=status_code, detail=str(error)) from error


def _public(symbol: CustomSymbol) -> dict:
    store = get_custom_symbol_store()
    try:
        image = store.image_png(symbol.id)
    except CustomSymbolError:
        image = None
    return symbol.to_public(image)


class CustomSymbolUpdate(BaseModel):
    name: str | None = Field(None, max_length=200)
    label: str | None = Field(None, max_length=500)
    category: str | None = None
    threshold: float | None = None
    match_rotations: bool | None = None
    any_size: bool | None = None
    enabled: bool | None = None


@router.get("/custom-symbols")
def list_custom_symbols() -> dict:
    symbols = get_custom_symbol_store().list()
    return {
        "symbols": [_public(symbol) for symbol in symbols],
        "categories": sorted(ALLOWED_CATEGORIES),
        "defaults": {
            "category": DEFAULT_CATEGORY,
            "threshold": DEFAULT_THRESHOLD,
            "min_threshold": MIN_THRESHOLD,
            "max_threshold": MAX_THRESHOLD,
        },
    }


@router.post("/custom-symbols", status_code=201)
async def create_custom_symbol(
    file: UploadFile = File(...),
    name: str = Form(...),
    label: str = Form(""),
    category: str = Form(DEFAULT_CATEGORY),
    threshold: float = Form(DEFAULT_THRESHOLD),
    match_rotations: bool = Form(False),
    any_size: bool = Form(False),
    authenticated: AuthenticatedSession = Depends(require_protected_request),
) -> dict:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    store = get_custom_symbol_store()
    try:
        symbol = await run_in_threadpool(
            store.create,
            name=name,
            image_bytes=raw,
            label=label,
            category=category,
            threshold=threshold,
            match_rotations=match_rotations,
            any_size=any_size,
            created_by=str(authenticated.user.id),
        )
    except CustomSymbolError as error:
        _raise(error)
    return _public(symbol)


@router.patch("/custom-symbols/{symbol_id}")
def update_custom_symbol(symbol_id: str, payload: CustomSymbolUpdate) -> dict:
    try:
        symbol = get_custom_symbol_store().update(
            symbol_id, payload.model_dump(exclude_unset=True)
        )
    except CustomSymbolError as error:
        _raise(error)
    return _public(symbol)


@router.delete("/custom-symbols/{symbol_id}", status_code=204)
def delete_custom_symbol(symbol_id: str) -> Response:
    try:
        get_custom_symbol_store().delete(symbol_id)
    except CustomSymbolError as error:
        _raise(error)
    return Response(status_code=204)
