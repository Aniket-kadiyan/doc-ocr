"""Tests for page-candidate reading-direction decisions."""

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
from ocr_pipeline import OcrPipeline, ROTATED_READING_RATIO


def _crop(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def test_padding_does_not_decide_a_crops_reading_direction() -> None:
    crop = _crop(45, 82)
    padded = upscale_min_edge(pad_image(crop))

    assert is_vertical_dimension(crop)
    assert not is_vertical_dimension(padded)
    assert orientations_for_ocr(padded, aspect_source=crop)[0][0] == "cw"
    assert orientations_for_ocr(padded)[0][0] == "h"
    assert primary_oriented(padded, aspect_source=crop).size != padded.size
    assert prepare_primary_ocr_variant(crop)[0] == "cw_clahe"


def test_preparation_can_preserve_the_received_direction() -> None:
    crop = _crop(45, 82)
    name, prepared = prepare_primary_ocr_variant(crop, orient=False)

    assert name == "h_clahe"
    assert prepared.size == upscale_min_edge(pad_image(crop)).size


def test_short_quarter_turn_callout_needs_two_direction_reads() -> None:
    crop = _crop(54, 62)

    assert not is_vertical_dimension(crop)
    assert crop.height > crop.width * ROTATED_READING_RATIO


def _batch_pipeline(reads: list[tuple[str, float]]) -> OcrPipeline:
    pipeline = object.__new__(OcrPipeline)
    pipeline._page_batch_recognition_available = True
    pipeline._text_orientation_available = True
    pipeline._text_recognizer_available = True
    pipeline.recognition_sizes = []

    def orientations(images, *, batch_size):
        return [(0, 0.99) for _image in images]

    def recognition(images, *, batch_size):
        pipeline.recognition_sizes.append(len(images))
        return reads[: len(images)]

    pipeline._predict_text_orientations = orientations
    pipeline._predict_text_recognition = recognition
    return pipeline


def test_complete_value_wins_over_non_value_from_other_direction() -> None:
    pipeline = _batch_pipeline([("o", 0.99), ("10", 0.80)])

    [result] = pipeline._recognize_page_batch(
        [_crop(54, 62)],
        batch_size=16,
    )

    assert result["text"] == "10"
    assert result["reading_rotation_cw"] == 90
    assert pipeline.recognition_sizes == [2]


def test_detector_hint_breaks_tie_between_two_valid_values() -> None:
    pipeline = _batch_pipeline([("10", 0.99), ("01", 0.70)])

    [result] = pipeline._recognize_page_batch(
        [_crop(54, 62)],
        batch_size=16,
        reading_hints=[90],
    )

    assert result["text"] == "01"
    assert result["reading_rotation_cw"] == 90


def test_horizontal_crop_costs_one_read() -> None:
    pipeline = _batch_pipeline([("24", 0.99)])

    [result] = pipeline._recognize_page_batch(
        [_crop(120, 40)],
        batch_size=16,
    )

    assert result["text"] == "24"
    assert result["reading_rotation_cw"] == 0
    assert pipeline.recognition_sizes == [1]


def test_long_vertical_prose_strip_costs_one_oriented_read() -> None:
    pipeline = _batch_pipeline([("207", 0.44)])

    [result] = pipeline._recognize_page_batch(
        [_crop(38, 566)],
        batch_size=16,
    )

    assert result["reading_rotation_cw"] == 0
    assert pipeline.recognition_sizes == [1]
