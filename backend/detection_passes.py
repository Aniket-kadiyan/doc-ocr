"""Accuracy-first detection passes and coordinate transforms.

Milestone 2 deliberately separates *where text might be* from grouping and
recognition.  Every pass operates on the same source crop, and every detected
box is mapped back into that crop's original pixel coordinate system before it
leaves this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from image_preprocess import cad_ink_to_gray

try:
    import cv2
except ImportError:  # pragma: no cover - the production OCR environment has OpenCV
    cv2 = None  # type: ignore


DETECTION_VARIANTS = (
    "source_contrast",
    "cad_contrast",
    "threshold",
    "inverted",
)
DETECTION_ROTATIONS_CW = (0, 90, 180, 270)
DETECTION_TILE_MAX_EDGE = 2000
DETECTION_TILE_OVERLAP = 160


@dataclass(frozen=True)
class DetectionPassSpec:
    """One deterministic Paddle detection pass."""

    variant: str
    rotation_cw: int
    target_long_edge: int

    @property
    def label(self) -> str:
        variant = self.variant.replace("_", " ")
        return f"{variant}, {self.rotation_cw} deg, {self.target_long_edge}px"


@dataclass(frozen=True)
class PreparedDetectionSource:
    """Deskewed source plus the transform required to restore its boxes."""

    image: Image.Image
    correction_angle: float


@dataclass(frozen=True)
class DetectionTile:
    """One overlapping work unit in a rotated detection image."""

    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def _tile_starts(length: int, max_edge: int, overlap: int) -> list[int]:
    """Return stable starts that cover one axis with the requested overlap."""

    if length <= max_edge:
        return [0]

    stride = max_edge - overlap
    starts = [0]
    while starts[-1] + max_edge < length:
        starts.append(starts[-1] + stride)
    return starts


def build_detection_tiles(
    image_or_size: Image.Image | tuple[int, int],
    *,
    max_edge: int = DETECTION_TILE_MAX_EDGE,
    overlap: int = DETECTION_TILE_OVERLAP,
) -> list[DetectionTile]:
    """Split a large pass into honest, overlapping progress work units.

    Small sections remain one unit.  Large sections use overlap so text on a
    tile boundary is still presented whole to PaddleOCR; the existing coarse
    positional deduplication removes detections repeated in the overlap.
    """

    if max_edge < 64:
        raise ValueError("Detection tile edge must be at least 64 pixels")
    if overlap < 0 or overlap >= max_edge:
        raise ValueError("Detection tile overlap must be within the tile edge")

    width, height = (
        image_or_size.size
        if isinstance(image_or_size, Image.Image)
        else image_or_size
    )
    x_starts = _tile_starts(width, max_edge, overlap)
    y_starts = _tile_starts(height, max_edge, overlap)
    return [
        DetectionTile(
            x=x,
            y=y,
            width=min(max_edge, width - x),
            height=min(max_edge, height - y),
        )
        for y in y_starts
        for x in x_starts
    ]


def detection_tile_target_edge(
    tile: DetectionTile,
    *,
    full_size: tuple[int, int],
    pass_target_edge: int,
) -> int:
    """Preserve a full-pass scale when PaddleOCR receives only one tile."""

    full_long_edge = max(full_size)
    tile_long_edge = max(tile.width, tile.height)
    scale = max(1.0, pass_target_edge / max(full_long_edge, 1))
    return max(tile_long_edge, int(round(tile_long_edge * scale)))


def offset_tile_box(
    box: dict[str, Any],
    tile: DetectionTile,
) -> dict[str, Any]:
    """Restore a tile-local detection box to the rotated pass coordinates."""

    return {
        **box,
        "x": float(box["x"]) + tile.x,
        "y": float(box["y"]) + tile.y,
    }


def detection_target_edges(image: Image.Image) -> tuple[int, ...]:
    """Return balanced and detail scales without downsampling source pixels."""

    long_edge = max(image.size)
    balanced = max(long_edge, 1100)
    # A second, substantially larger pass helps faint and very small CAD text.
    detail = max(1800, int(round(balanced * 1.55)))
    detail = min(detail, 3000)

    # Very large source images already exceed the safe detail ceiling.  Running
    # the same native-size pass twice would add time without new information.
    if detail <= balanced:
        return (balanced,)
    return (balanced, detail)


def build_detection_pass_plan(image: Image.Image) -> list[DetectionPassSpec]:
    """Build the complete, stable pass order used for progress reporting."""

    targets = detection_target_edges(image)
    return [
        DetectionPassSpec(variant, rotation, target)
        for rotation in DETECTION_ROTATIONS_CW
        for variant in DETECTION_VARIANTS
        for target in targets
    ]


def _otsu_threshold(gray: np.ndarray) -> int:
    """Small NumPy Otsu implementation so variant creation does not require cv2."""

    values = np.asarray(gray, dtype=np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = float(values.size)
    if total <= 0:
        return 127

    weighted_total = float(np.dot(np.arange(256), histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 127

    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight <= 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight <= 0:
            break
        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        between = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if between > best_variance:
            best_variance = between
            best_threshold = threshold

    return best_threshold


def build_detection_variants(image: Image.Image) -> dict[str, Image.Image]:
    """Create contrast, CAD-ink, threshold, and inverse threshold views."""

    source_gray = ImageOps.autocontrast(image.convert("L"), cutoff=0)

    # cad_ink_to_gray produces bright ink on black.  Invert it to Paddle's
    # familiar dark-text-on-light-background convention, then stretch contrast.
    cad_dark = ImageOps.invert(cad_ink_to_gray(image).convert("L"))
    cad_contrast = ImageOps.autocontrast(cad_dark, cutoff=0)

    cad_array = np.asarray(cad_contrast, dtype=np.uint8)
    threshold = _otsu_threshold(cad_array)
    binary_array = np.where(cad_array > threshold, 255, 0).astype(np.uint8)
    binary = Image.fromarray(binary_array, mode="L")

    return {
        "source_contrast": source_gray.convert("RGB"),
        "cad_contrast": cad_contrast.convert("RGB"),
        "threshold": binary.convert("RGB"),
        "inverted": ImageOps.invert(binary).convert("RGB"),
    }


def estimate_skew_correction(
    image: Image.Image,
    *,
    max_angle: float = 6.0,
    minimum_lines: int = 6,
) -> float:
    """Estimate a conservative small deskew correction from long CAD lines.

    Angles are reduced to their deviation from the nearest horizontal/vertical
    axis.  A correction is accepted only when several long lines agree tightly;
    angled part geometry alone therefore does not rotate the drawing.
    """

    if cv2 is None:
        return 0.0

    gray = np.asarray(cad_ink_to_gray(image).convert("L"), dtype=np.uint8)
    if min(gray.shape[:2]) < 24:
        return 0.0

    edges = cv2.Canny(gray, 40, 120)
    long_edge = max(gray.shape[:2])
    min_length = max(24, int(long_edge * 0.12))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(18, int(min_length * 0.35)),
        minLineLength=min_length,
        maxLineGap=max(4, int(min_length * 0.08)),
    )
    if lines is None:
        return 0.0

    deviations: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if hypot(float(x2 - x1), float(y2 - y1)) < min_length:
            continue
        angle = degrees(atan2(float(y2 - y1), float(x2 - x1)))
        deviation = ((angle + 45.0) % 90.0) - 45.0
        if abs(deviation) <= max_angle:
            deviations.append(deviation)

    if len(deviations) < minimum_lines:
        return 0.0

    correction = float(median(deviations))
    dispersion = float(median(abs(value - correction) for value in deviations))
    if dispersion > 1.25 or abs(correction) < 0.20:
        return 0.0
    return round(correction, 3)


def prepare_detection_source(image: Image.Image) -> PreparedDetectionSource:
    """Deskew at source resolution while keeping the original canvas size."""

    source = image.convert("RGB")
    correction = estimate_skew_correction(source)
    if correction == 0.0:
        return PreparedDetectionSource(source, 0.0)

    deskewed = source.rotate(
        correction,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    return PreparedDetectionSource(deskewed, correction)


def rotate_for_detection(image: Image.Image, rotation_cw: int) -> Image.Image:
    """Apply an exact quarter-turn without interpolation."""

    rotation = rotation_cw % 360
    if rotation == 0:
        return image
    if rotation == 90:
        return image.transpose(Image.Transpose.ROTATE_270)
    if rotation == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if rotation == 270:
        return image.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(f"Detection rotation must be a quarter-turn, got {rotation_cw}")


def _clip_box(
    box: dict[str, Any],
    width: float,
    height: float,
) -> dict[str, Any] | None:
    x0 = max(0.0, min(float(box["x"]), width))
    y0 = max(0.0, min(float(box["y"]), height))
    x1 = max(0.0, min(float(box["x"] + box["w"]), width))
    y1 = max(0.0, min(float(box["y"] + box["h"]), height))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return {
        **box,
        "x": round(x0, 1),
        "y": round(y0, 1),
        "w": round(x1 - x0, 1),
        "h": round(y1 - y0, 1),
    }


def map_quarter_turn_box_to_source(
    box: dict[str, Any],
    rotation_cw: int,
    source_size: tuple[int, int],
) -> dict[str, Any] | None:
    """Map a box from a quarter-turned image into its unrotated source."""

    width, height = source_size
    x = float(box["x"])
    y = float(box["y"])
    box_width = float(box["w"])
    box_height = float(box["h"])
    rotation = rotation_cw % 360

    if rotation == 0:
        mapped = {**box, "x": x, "y": y, "w": box_width, "h": box_height}
    elif rotation == 90:
        mapped = {
            **box,
            "x": y,
            "y": height - (x + box_width),
            "w": box_height,
            "h": box_width,
        }
    elif rotation == 180:
        mapped = {
            **box,
            "x": width - (x + box_width),
            "y": height - (y + box_height),
            "w": box_width,
            "h": box_height,
        }
    elif rotation == 270:
        mapped = {
            **box,
            "x": width - (y + box_height),
            "y": x,
            "w": box_height,
            "h": box_width,
        }
    else:
        raise ValueError(f"Detection rotation must be a quarter-turn, got {rotation_cw}")

    return _clip_box(mapped, float(width), float(height))


def map_deskewed_box_to_original(
    box: dict[str, Any],
    source_size: tuple[int, int],
    correction_angle: float,
) -> dict[str, Any] | None:
    """Undo the source deskew and return a clipped axis-aligned source box."""

    width, height = source_size
    if correction_angle == 0.0:
        return _clip_box(box, float(width), float(height))

    cx = width / 2.0
    cy = height / 2.0
    angle = radians(correction_angle)
    cosine = cos(angle)
    sine = sin(angle)

    x0 = float(box["x"])
    y0 = float(box["y"])
    x1 = x0 + float(box["w"])
    y1 = y0 + float(box["h"])

    def inverse(x: float, y: float) -> tuple[float, float]:
        dx = x - cx
        dy = y - cy
        return (
            cosine * dx - sine * dy + cx,
            sine * dx + cosine * dy + cy,
        )

    corners = [inverse(x0, y0), inverse(x1, y0), inverse(x0, y1), inverse(x1, y1)]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    mapped = {
        **box,
        "x": min(xs),
        "y": min(ys),
        "w": max(xs) - min(xs),
        "h": max(ys) - min(ys),
    }
    return _clip_box(mapped, float(width), float(height))


def map_detection_box_to_original(
    box: dict[str, Any],
    *,
    rotation_cw: int,
    source_size: tuple[int, int],
    correction_angle: float,
) -> dict[str, Any] | None:
    """Undo quarter-turn and deskew transforms in the correct order."""

    unrotated = map_quarter_turn_box_to_source(box, rotation_cw, source_size)
    if unrotated is None:
        return None
    return map_deskewed_box_to_original(
        unrotated,
        source_size,
        correction_angle,
    )
