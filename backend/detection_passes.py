"""Adaptive auto-balloon detection plans and coordinate transforms.

The first accuracy-first implementation exhaustively combined four image
variants, four rotations, two scales, and page tiles.  PaddleOCR 3.x ran the
complete OCR pipeline for every one of those nominal "detection" calls, which
made whole-page scans impractical.  Milestone 2B keeps the high-recall
morphology proposals but bounds learned localization to two detector-only page
passes plus a capped set of local refinements.
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
# A 180/270-degree pass cannot reveal a new text location.  The recognition
# stage resolves a crop's final reading orientation later.
DETECTION_ROTATIONS_CW = (0, 90)
DETECTION_PRIMARY_VARIANT = "source_contrast"
DETECTION_PRIMARY_MIN_EDGE = 1100
DETECTION_PRIMARY_MAX_EDGE = 2400
DETECTION_REFINEMENT_TARGET_EDGE = 1100
DETECTION_MAX_REFINEMENT_REGIONS = 24
DETECTION_FALLBACK_MAX_REFINEMENT_REGIONS = 8
DETECTION_COVERAGE_RATIO = 0.55

# These helpers remain available for local/refinement experiments, but the
# whole-page adaptive path no longer tiles every detector pass.
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


def detection_primary_target_edge(
    image_or_size: Image.Image | tuple[int, int],
) -> int:
    """Return the bounded long edge used by both primary detector passes.

    Small sections are enlarged enough for CAD digits while large drawings are
    downscaled once for localization.  Full source pixels are retained for the
    later, targeted refinement crops.
    """

    size = (
        image_or_size.size
        if isinstance(image_or_size, Image.Image)
        else image_or_size
    )
    long_edge = max(size)
    return min(
        DETECTION_PRIMARY_MAX_EDGE,
        max(DETECTION_PRIMARY_MIN_EDGE, long_edge),
    )


def build_detection_pass_plan(image: Image.Image) -> list[DetectionPassSpec]:
    """Build the two bounded primary detector-only passes."""

    target = detection_primary_target_edge(image)
    return [
        DetectionPassSpec(DETECTION_PRIMARY_VARIANT, rotation, target)
        for rotation in DETECTION_ROTATIONS_CW
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


def build_primary_detection_image(image: Image.Image) -> Image.Image:
    """Create only the view used by the primary adaptive detector.

    Building all four legacy variants over a large page consumed time and
    memory even when only one view was required.  Local fallback experiments
    can still call :func:`build_detection_variants` explicitly.
    """

    return ImageOps.autocontrast(image.convert("L"), cutoff=0).convert("RGB")


def _box_area(box: dict[str, Any]) -> float:
    return max(0.0, float(box["w"])) * max(0.0, float(box["h"]))


def _intersection_area(a: dict[str, Any], b: dict[str, Any]) -> float:
    x0 = max(float(a["x"]), float(b["x"]))
    y0 = max(float(a["y"]), float(b["y"]))
    x1 = min(float(a["x"] + a["w"]), float(b["x"] + b["w"]))
    y1 = min(float(a["y"] + a["h"]), float(b["y"] + b["h"]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _union_box(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    x0 = min(float(a["x"]), float(b["x"]))
    y0 = min(float(a["y"]), float(b["y"]))
    x1 = max(float(a["x"] + a["w"]), float(b["x"] + b["w"]))
    y1 = max(float(a["y"] + a["h"]), float(b["y"] + b["h"]))
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _boxes_touch(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (
        float(a["x"] + a["w"]) < float(b["x"])
        or float(b["x"] + b["w"]) < float(a["x"])
        or float(a["y"] + a["h"]) < float(b["y"])
        or float(b["y"] + b["h"]) < float(a["y"])
    )


def build_refinement_regions(
    morphology_boxes: list[dict[str, Any]],
    detector_boxes: list[dict[str, Any]],
    *,
    source_size: tuple[int, int],
    max_regions: int = DETECTION_MAX_REFINEMENT_REGIONS,
    coverage_ratio: float = DETECTION_COVERAGE_RATIO,
    margin_ratio: float = 0.45,
    max_region_area_ratio: float = 0.18,
) -> list[dict[str, float]]:
    """Build bounded local retries for morphology proposals not yet covered.

    Morphology boxes always remain candidates, including proposals beyond the
    retry cap.  The cap therefore bounds detector work without silently losing
    recall; local refinement only improves the learned box around a gap.
    """

    if max_regions <= 0:
        return []

    width, height = source_size
    source_area = max(float(width * height), 1.0)
    max_region_area = source_area * max_region_area_ratio
    gaps: list[dict[str, float]] = []

    for proposal in morphology_boxes:
        area = _box_area(proposal)
        if area < 1.0:
            continue
        covered = any(
            _intersection_area(proposal, detected) / area >= coverage_ratio
            for detected in detector_boxes
        )
        if covered:
            continue

        margin = max(8.0, min(float(proposal["w"]), float(proposal["h"])) * margin_ratio)
        expanded = _clip_box(
            {
                "x": float(proposal["x"]) - margin,
                "y": float(proposal["y"]) - margin,
                "w": float(proposal["w"]) + margin * 2,
                "h": float(proposal["h"]) + margin * 2,
            },
            float(width),
            float(height),
        )
        # Very large morphology components are usually drawing geometry.  They
        # are still retained as proposals, but a page-like local retry adds no
        # detail beyond the bounded primary scan.
        if expanded is not None and _box_area(expanded) <= max_region_area:
            gaps.append(
                {
                    "x": float(expanded["x"]),
                    "y": float(expanded["y"]),
                    "w": float(expanded["w"]),
                    "h": float(expanded["h"]),
                }
            )

    # Expanded neighbouring gaps are one detector call.  Repeat until no
    # transitive overlaps remain, but do not merge into a page-sized retry.
    regions = gaps
    changed = True
    while changed:
        changed = False
        merged: list[dict[str, float]] = []
        while regions:
            current = regions.pop(0)
            remainder: list[dict[str, float]] = []
            for other in regions:
                union = _union_box(current, other)
                if _boxes_touch(current, other) and _box_area(union) <= max_region_area:
                    current = union
                    changed = True
                else:
                    remainder.append(other)
            regions = remainder
            merged.append(current)
        regions = merged

    # Coarse/large gaps benefit most from refinement.  Return the capped set in
    # stable page-reading order so progress is deterministic.
    selected = sorted(regions, key=_box_area, reverse=True)[:max_regions]
    return sorted(selected, key=lambda box: (box["y"], box["x"]))


def refinement_rotation(region: dict[str, Any]) -> int:
    """Make a tall local gap horizontal for its one detector retry."""

    return 90 if float(region["h"]) > float(region["w"]) * 1.25 else 0


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
