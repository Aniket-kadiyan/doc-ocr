"""Pydantic contracts for the internal, backend-owned checksheet system."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChecksheetBBox(BaseModel):
    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ChecksheetSnapshotItem(BaseModel):
    annotation_id: str = Field(min_length=1, max_length=128)
    balloon_number: int = Field(ge=1)
    page: int = Field(ge=1)
    bbox: ChecksheetBBox
    rotation: float = 0
    label: str = Field(default="", max_length=500)
    specification: str = Field(min_length=1, max_length=2000)
    tolerance: str = Field(default="", max_length=1000)
    method: str = Field(default="", max_length=500)
    tool: str = Field(default="", max_length=500)
    dimension_type: str = Field(default="Unknown", max_length=100)


class ChecksheetSnapshot(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    drawing_name: str = Field(min_length=1, max_length=255)
    source_project_id: str | None = Field(default=None, max_length=128)
    source_file_type: Literal["pdf", "image"]
    pdf_render_scale: float = Field(default=1.5, gt=0, le=10)
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
