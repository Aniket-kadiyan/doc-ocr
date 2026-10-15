"""
Tests for the orientation decisions behind a page scan's recognition crops.

Run: PYTHONPATH=. python -m pytest test_reading_orientation.py
"""

from __future__ import annotations

from PIL import Image

from image_preprocess import (
    is_vertical_dimension,
    orientations_for_ocr,
    pad_image,
    prepare_primary_ocr_variant,
    primary_oriented,
    upscale_min_edge,
)
from ocr_pipeline import ROTATED_READING_RATIO


def _crop(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def test_padding_no_longer_decides_a_crops_reading_direction() -> None:
    # 45x82 is a vertical "0.5" beside a dimension line, comfortably past the
    # 1.35 ratio. The fixed 36px pad on every side turns it into 212x280,
    # which is not, so the crop was handed to the recogniser lying on its side
    # and read back as "5,0" where turning it gives "0.5" at full confidence.
    crop = _crop(45, 82)
    padded = upscale_min_edge(pad_image(crop))

    assert is_vertical_dimension(crop)
    assert not is_vertical_dimension(padded)

    name, _image = orientations_for_ocr(padded, aspect_source=crop)[0]
    assert name == "cw"
    assert orientations_for_ocr(padded)[0][0] == "h"

    assert primary_oriented(padded, aspect_source=crop).size != padded.size
    assert prepare_primary_ocr_variant(crop)[0] == "cw_clahe"


def test_a_plainly_horizontal_crop_is_still_read_as_it_arrived() -> None:
    crop = _crop(120, 40)

    assert prepare_primary_ocr_variant(crop)[0] == "h_clahe"
    assert orientations_for_ocr(
        upscale_min_edge(pad_image(crop)),
        aspect_source=crop,
    )[0][0] == "h"


def test_preparation_can_be_asked_to_turn_nothing() -> None:
    # The page route resolves direction by reading both ways, so it needs the
    # preparation without a guess baked into it.
    crop = _crop(45, 82)

    name, image = prepare_primary_ocr_variant(crop, orient=False)
    assert name == "h_clahe"
    assert image.size == upscale_min_edge(pad_image(crop)).size


def test_a_two_glyph_quarter_turn_callout_is_shaped_like_neither() -> None:
    # The real crop behind a quarter-turn "10": the ratio test calls it
    # horizontal, so nothing but reading it both ways can recover it.
    crop = _crop(54, 62)

    assert not is_vertical_dimension(crop)
    assert crop.height > crop.width * ROTATED_READING_RATIO
