"""FastAPI routes for durable, internal inspection checksheets."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from checksheet_models import (
    ChecksheetSnapshot,
    ChecksheetUpdateRequest,
    DuplicateChecksheetRequest,
    ReadingPatchRequest,
    StartRunRequest,
)
from checksheet_storage import (
    ChecksheetConflictError,
    ChecksheetNotFoundError,
    ChecksheetStorageError,
    ChecksheetValidationError,
    get_checksheet_storage,
)

router = APIRouter(tags=["checksheets"])


def _raise_storage_error(error: ChecksheetStorageError) -> NoReturn:
    if isinstance(error, ChecksheetNotFoundError):
        status_code = 404
    elif isinstance(error, ChecksheetConflictError):
        status_code = 409
    elif isinstance(error, ChecksheetValidationError):
        status_code = 422
    else:
        status_code = 500
    raise HTTPException(status_code=status_code, detail=str(error)) from error


def _parse_snapshot(raw: str) -> ChecksheetSnapshot:
    try:
        return ChecksheetSnapshot.model_validate_json(raw)
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=error.errors(include_context=False),
        ) from error


def _validate_uploaded_document(
    snapshot: ChecksheetSnapshot,
    document: UploadFile,
) -> None:
    file_name = Path(document.filename or "").name
    suffix = Path(file_name).suffix.casefold()
    mime_type = (document.content_type or "").casefold()
    if snapshot.source_file_type == "pdf":
        if mime_type not in {"application/pdf", "application/octet-stream"} and suffix != ".pdf":
            raise HTTPException(
                status_code=422,
                detail="A PDF checksheet snapshot must include the original PDF file.",
            )
        return
    allowed_image_suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
    }
    if not mime_type.startswith("image/") and suffix not in allowed_image_suffixes:
        raise HTTPException(
            status_code=422,
            detail="An image checksheet snapshot must include the original image file.",
        )


@router.post("/checksheets", status_code=201)
def create_checksheet(
    snapshot_json: str = Form(...),
    document: UploadFile = File(...),
) -> dict:
    snapshot = _parse_snapshot(snapshot_json)
    _validate_uploaded_document(snapshot, document)
    try:
        document.file.seek(0)
        return get_checksheet_storage().create_checksheet(
            snapshot,
            source=document.file,
            original_name=document.filename or snapshot.drawing_name,
            mime_type=document.content_type or "application/octet-stream",
        )
    except ChecksheetStorageError as error:
        _raise_storage_error(error)
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not store the checksheet source drawing: {error}",
        ) from error


@router.post("/checksheets/{checksheet_id}/revisions", status_code=201)
def create_checksheet_revision(
    checksheet_id: str,
    snapshot_json: str = Form(...),
    document: UploadFile = File(...),
) -> dict:
    snapshot = _parse_snapshot(snapshot_json)
    _validate_uploaded_document(snapshot, document)
    try:
        document.file.seek(0)
        return get_checksheet_storage().create_revision(
            checksheet_id,
            snapshot,
            source=document.file,
            original_name=document.filename or snapshot.drawing_name,
            mime_type=document.content_type or "application/octet-stream",
        )
    except ChecksheetStorageError as error:
        _raise_storage_error(error)
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not store the revised source drawing: {error}",
        ) from error


@router.get("/checksheets")
def list_checksheets(
    search: str = Query(default="", max_length=200),
    archived: bool = Query(default=False),
) -> dict:
    try:
        return {
            "items": get_checksheet_storage().list_checksheets(
                search=search,
                archived=archived,
            )
        }
    except ChecksheetStorageError as error:
        _raise_storage_error(error)


@router.get("/checksheets/{checksheet_id}")
def get_checksheet(checksheet_id: str) -> dict:
    try:
        return get_checksheet_storage().get_checksheet(checksheet_id)
    except ChecksheetStorageError as error:
        _raise_storage_error(error)


@router.patch("/checksheets/{checksheet_id}")
def update_checksheet(
    checksheet_id: str,
    request: ChecksheetUpdateRequest,
) -> dict:
    try:
        return get_checksheet_storage().update_checksheet(
            checksheet_id,
            name=request.name,
            archived=request.archived,
        )
    except ChecksheetStorageError as error:
        _raise_storage_error(error)


@router.post("/checksheets/{checksheet_id}/duplicate", status_code=201)
def duplicate_checksheet(
    checksheet_id: str,
    request: DuplicateChecksheetRequest,
) -> dict:
    try:
        return get_checksheet_storage().duplicate_checksheet(
            checksheet_id,
            name=request.name,
        )
    except ChecksheetStorageError as error:
        _raise_storage_error(error)


@router.delete("/checksheets/{checksheet_id}", status_code=204)
def permanently_delete_checksheet(checksheet_id: str) -> Response:
    try:
        get_checksheet_storage().delete_checksheet_permanently(checksheet_id)
    except ChecksheetStorageError as error:
        _raise_storage_error(error)
    return Response(status_code=204)


@router.post("/checksheets/{checksheet_id}/runs", status_code=201)
def start_checksheet_run(
    checksheet_id: str,
    request: StartRunRequest,
) -> dict:
    try:
        return get_checksheet_storage().start_run(
            checksheet_id,
            revision_id=request.revision_id,
        )
    except ChecksheetStorageError as error:
        _raise_storage_error(error)


@router.get("/checksheets/{checksheet_id}/runs/{run_id}")
def get_checksheet_run(checksheet_id: str, run_id: str) -> dict:
    try:
        return get_checksheet_storage().get_run(checksheet_id, run_id)
    except ChecksheetStorageError as error:
        _raise_storage_error(error)


@router.patch("/checksheets/{checksheet_id}/runs/{run_id}/readings")
def save_checksheet_readings(
    checksheet_id: str,
    run_id: str,
    request: ReadingPatchRequest,
) -> dict:
    try:
        return get_checksheet_storage().update_readings(
            checksheet_id,
            run_id,
            request.readings,
        )
    except ChecksheetStorageError as error:
        _raise_storage_error(error)


@router.post("/checksheets/{checksheet_id}/runs/{run_id}/complete")
def complete_checksheet_run(checksheet_id: str, run_id: str) -> dict:
    try:
        return get_checksheet_storage().complete_run(checksheet_id, run_id)
    except ChecksheetStorageError as error:
        _raise_storage_error(error)


@router.get("/checksheet-documents/{document_id}")
def get_checksheet_document(document_id: str) -> FileResponse:
    try:
        document = get_checksheet_storage().get_document(document_id)
    except ChecksheetStorageError as error:
        _raise_storage_error(error)

    # Use an internal preview MIME type and no Content-Disposition header.
    # This prevents download managers such as IDM from capturing the PDF.
    # The frontend already restores the real MIME type from checksheet metadata.
    return FileResponse(
        path=document.path,
        media_type="application/vnd.doc-ocr.preview",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
