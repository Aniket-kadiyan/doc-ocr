"""Validation and page rendering for uploaded raster drawing files.

Browsers decode PNG and JPEG directly, but TIFF support is inconsistent and
multi-page TIFF is not exposed through ``HTMLImageElement``.  This module keeps
the original file untouched and renders only the requested TIFF frame to PNG
for the viewer, OCR, checksheet preview, and exports.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_RASTER_FORMATS = frozenset({"png", "jpeg", "tiff"})
MAX_SOURCE_BYTES = int(os.getenv("SOURCE_IMAGE_MAX_BYTES", str(256 * 1024 * 1024)))
MAX_PAGE_COUNT = int(os.getenv("SOURCE_IMAGE_MAX_PAGES", "500"))
MAX_PAGE_PIXELS = int(os.getenv("SOURCE_IMAGE_MAX_PIXELS", "120000000"))


class SourceImageError(ValueError):
    """A source image cannot be safely identified or rendered."""


@dataclass(frozen=True)
class RenderedSourcePage:
    png: bytes
    page_count: int
    width: int
    height: int
    format: str


def _signature_format(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "tiff"
    return None


def _hinted_format(filename: str, content_type: str) -> str | None:
    suffix = Path(filename or "").suffix.casefold()
    if suffix == ".png":
        return "png"
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix in {".tif", ".tiff"}:
        return "tiff"

    mime = (content_type or "").split(";", 1)[0].strip().casefold()
    return {
        "image/png": "png",
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/tiff": "tiff",
        "image/x-tiff": "tiff",
    }.get(mime)


def detect_raster_format(data: bytes, filename: str = "", content_type: str = "") -> str:
    """Identify a supported image and reject misleading names/MIME types."""

    if not data:
        raise SourceImageError("The image file is empty")
    if len(data) > MAX_SOURCE_BYTES:
        raise SourceImageError(
            f"The image is larger than the {MAX_SOURCE_BYTES // (1024 * 1024)} MB limit"
        )

    actual = _signature_format(data)
    hinted = _hinted_format(filename, content_type)
    if actual is None:
        raise SourceImageError("Only PNG, JPG/JPEG, TIF, and TIFF files are supported")
    if hinted is not None and hinted != actual:
        raise SourceImageError(
            f"The file contents are {actual.upper()}, but its name or MIME type says {hinted.upper()}"
        )
    return actual


def render_source_page(
    data: bytes,
    *,
    page: int = 1,
    filename: str = "",
    content_type: str = "",
) -> RenderedSourcePage:
    """Render one 1-based image page as PNG without modifying the original."""

    source_format = detect_raster_format(data, filename, content_type)
    if page < 1:
        raise SourceImageError("The image page must be 1 or greater")

    try:
        with Image.open(io.BytesIO(data)) as image:
            decoded_format = (image.format or "").casefold()
            if decoded_format == "jpg":
                decoded_format = "jpeg"
            if decoded_format not in SUPPORTED_RASTER_FORMATS:
                raise SourceImageError("The decoder did not recognize a supported image format")
            if decoded_format != source_format:
                raise SourceImageError("The image signature does not match the decoded format")

            page_count = int(getattr(image, "n_frames", 1) or 1)
            if page_count > MAX_PAGE_COUNT:
                raise SourceImageError(
                    f"The image contains {page_count} pages; the limit is {MAX_PAGE_COUNT}"
                )
            if page > page_count:
                raise SourceImageError(
                    f"Page {page} is not present; this image has {page_count} page(s)"
                )

            image.seek(page - 1)
            frame = ImageOps.exif_transpose(image.copy())
            width, height = frame.size
            if width <= 0 or height <= 0:
                raise SourceImageError("The selected image page has invalid dimensions")
            if width * height > MAX_PAGE_PIXELS:
                raise SourceImageError(
                    f"The selected page has {width * height:,} pixels; the limit is {MAX_PAGE_PIXELS:,}"
                )

            if frame.mode in {"RGBA", "LA"} or (
                frame.mode == "P" and "transparency" in frame.info
            ):
                rgba = frame.convert("RGBA")
                flattened = Image.new("RGB", rgba.size, "white")
                flattened.paste(rgba, mask=rgba.getchannel("A"))
                frame = flattened
            else:
                frame = frame.convert("RGB")

            output = io.BytesIO()
            frame.save(output, format="PNG", optimize=False)
            return RenderedSourcePage(
                png=output.getvalue(),
                page_count=page_count,
                width=width,
                height=height,
                format=source_format,
            )
    except SourceImageError:
        raise
    except (UnidentifiedImageError, OSError, EOFError) as error:
        raise SourceImageError(f"Could not decode the image: {error}") from error
