"""Pydantic contracts for the internal, backend-owned checksheet system."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChecksheetBBox(BaseModel):
    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ChecksheetOrientedBox(ChecksheetBBox):
    rotation: float = 0


class ChecksheetSnapshotItem(BaseModel):
    annotation_id: str = Field(min_length=1, max_length=128)
    balloon_number: int = Field(ge=1)
    page: int = Field(ge=1)
    bbox: ChecksheetBBox
    oriented_box: ChecksheetOrientedBox | None = None
    rotation: float = 0
    label: str = Field(default="", max_length=500)
    specification: str = Field(min_length=1, max_length=2000)
    tolerance: str = Field(default="", max_length=1000)
    method: str = Field(default="", max_length=500)
    tool: str = Field(default="", max_length=500)
    dimension_type: str = Field(default="Unknown", max_length=100)
    assembly: dict[str, Any] | None = None
    engineering_parse: dict[str, Any] | None = None


class ChecksheetScanCandidate(BaseModel):
    """Unaccepted detector outcome retained with a checksheet revision."""

    candidate_id: str = Field(min_length=1, max_length=256)
    source_candidate_id: str | None = Field(default=None, max_length=256)
    page: int = Field(ge=1)
    order: int = Field(ge=0)
    state: Literal["review", "other", "ignored"]
    restore_state: Literal["review", "other"] | None = None
    text: str = Field(default="", max_length=4000)
    raw_text: str = Field(default="", max_length=4000)
    preliminary_text: str | None = Field(default=None, max_length=4000)
    confidence: float = 0
    recognized: bool = False
    reason: str = Field(default="", max_length=4000)
    rule: str | None = Field(default=None, max_length=500)
    type: str | None = Field(default=None, max_length=200)
    category: str | None = Field(default=None, max_length=200)
    subtype: str | None = Field(default=None, max_length=500)
    label: str | None = Field(default=None, max_length=1000)
    orientation: Literal["horizontal", "vertical", "rotated"] = "horizontal"
    rotation: float = 0
    recovery_attempted: bool = False
    authoritative_reread: bool = False
    assembly: dict[str, Any] | None = None
    engineering_parse: dict[str, Any] | None = None
    bbox: ChecksheetBBox
    oriented_box: ChecksheetOrientedBox | None = None
    duplicate_source_ids: list[str] = Field(default_factory=list, max_length=100)
    duplicate_count: int = Field(default=0, ge=0)
    created_at: int = Field(ge=0)
    updated_at: int = Field(ge=0)


class ChecksheetMetadata(BaseModel):
    """Drawing metadata snapshotted with one checksheet revision."""

    model_config = ConfigDict(populate_by_name=True)

    part_name: str = Field(default="", alias="partName", max_length=500)
    document_number: str = Field(
        default="",
        alias="documentNumber",
        max_length=500,
    )
    revision_number: str = Field(
        default="",
        alias="revisionNumber",
        max_length=500,
    )

    @field_validator("part_name", "document_number", "revision_number")
    @classmethod
    def strip_optional_text(cls, value: str) -> str:
        return value.strip()


class ChecksheetSnapshot(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    drawing_name: str = Field(min_length=1, max_length=255)
    source_project_id: str | None = Field(default=None, max_length=128)
    source_file_type: Literal["pdf", "image"]
    pdf_render_scale: float = Field(default=1.5, gt=0, le=10)
    metadata: ChecksheetMetadata = Field(default_factory=ChecksheetMetadata)
    scan_candidates: list[ChecksheetScanCandidate] = Field(
        default_factory=list,
        max_length=10000,
    )
    reading_columns: list[str] = Field(min_length=1, max_length=20)
    items: list[ChecksheetSnapshotItem] = Field(min_length=1, max_length=5000)

    @field_validator("name", "drawing_name")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("reading_columns")
    @classmethod
    def validate_reading_columns(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("reading column names must not be blank")
        if any(len(value) > 120 for value in normalized):
            raise ValueError("reading column names must be 120 characters or fewer")
        folded = [value.casefold() for value in normalized]
        if len(folded) != len(set(folded)):
            raise ValueError("reading column names must be unique")
        return normalized

    @field_validator("scan_candidates")
    @classmethod
    def validate_scan_candidates(
        cls,
        values: list[ChecksheetScanCandidate],
    ) -> list[ChecksheetScanCandidate]:
        candidate_ids = [value.candidate_id for value in values]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("scan candidate IDs must be unique")
        return sorted(values, key=lambda value: (value.order, value.candidate_id))

    @field_validator("items")
    @classmethod
    def validate_items(
        cls,
        values: list[ChecksheetSnapshotItem],
    ) -> list[ChecksheetSnapshotItem]:
        annotation_ids = [value.annotation_id for value in values]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("annotation IDs must be unique")
        balloon_numbers = [value.balloon_number for value in values]
        if len(balloon_numbers) != len(set(balloon_numbers)):
            raise ValueError("balloon numbers must be unique")
        if sorted(balloon_numbers) != list(range(1, len(values) + 1)):
            raise ValueError("balloon numbers must be contiguous from 1")
        return sorted(values, key=lambda value: value.balloon_number)


class ReadingPatch(BaseModel):
    annotation_id: str = Field(min_length=1, max_length=128)
    column_id: str = Field(min_length=1, max_length=128)
    value: str = Field(default="", max_length=2000)


class ReadingPatchRequest(BaseModel):
    readings: list[ReadingPatch] = Field(min_length=1, max_length=5000)


class ChecksheetUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    archived: bool | None = None

    @field_validator("name")
    @classmethod
    def strip_optional_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        return stripped


class StartRunRequest(BaseModel):
    revision_id: str | None = Field(default=None, max_length=128)


class DuplicateChecksheetRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)

    @field_validator("name")
    @classmethod
    def strip_duplicate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        return stripped
