"""Bounded post-detection recovery for whole-page OCR candidates.

The page detector is intentionally outside this module.  These helpers consume
its already-deduplicated geometry, build recovery-only crops, and resolve the
primary/recovery reads without ever invoking another detection pass.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import atan2, degrees
import re
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover - production OCR installs OpenCV
    cv2 = None  # type: ignore


RECOVERY_CONFIDENCE_THRESHOLD = 0.90
RECOVERY_ORIENTATION_THRESHOLD = 0.70
RECOVERY_MAX_CANDIDATES = 96
RECOVERY_VARIANTS_PER_CANDIDATE = 2
CONTEXT_MAX_CANDIDATES = 48

_DIGIT = re.compile(r"\d")
_SUSPICIOUS_SINGLE = re.compile(r"^[018]$")
_COMPACT_TOKEN = re.compile(r"[^A-Z0-9]+")
_CONFUSABLE_PAIRS = (
    frozenset(("8", "B")),
    frozenset(("0", "O")),
    frozenset(("1", "I")),
    frozenset(("1", "L")),
)


@dataclass(frozen=True)
class RecoveryCrop:
    """One bounded recognition-only retry for a detected page object."""

    profile: str
    image: Image.Image


def result_needs_recovery(result: Mapping[str, Any]) -> bool:
    """Return whether the primary read is unsafe to decide automatically."""

    text = str(result.get("text") or "").strip()
    confidence = float(result.get("confidence") or 0.0)
    orientation_confidence = float(
        result.get("orientation_confidence") or 0.0
    )
    return bool(
        not text
        or not _DIGIT.search(text)
        or confidence < RECOVERY_CONFIDENCE_THRESHOLD
        or result.get("confusable_corrected")
        or _SUSPICIOUS_SINGLE.fullmatch(text)
        or orientation_confidence < RECOVERY_ORIENTATION_THRESHOLD
    )


def recovery_priority(result: Mapping[str, Any]) -> int:
    """Prioritize likely missed dimensions before long, stable text labels."""

    text = str(result.get("text") or "").strip()
    confidence = float(result.get("confidence") or 0.0)
    if not text:
        return 0
    if _DIGIT.search(text) and (
        confidence < RECOVERY_CONFIDENCE_THRESHOLD
        or result.get("confusable_corrected")
        or _SUSPICIOUS_SINGLE.fullmatch(text)
    ):
        return 1
    if len(text) <= 6:
        return 2
    return 3


def select_recovery_record_indexes(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum: int = RECOVERY_MAX_CANDIDATES,
) -> list[int]:
    """Choose a deterministic, bounded subset of doubtful page candidates."""

    if maximum < 0:
        raise ValueError("Recovery candidate maximum cannot be negative")
    doubtful = [
        index
        for index, record in enumerate(records)
        if result_needs_recovery(record.get("result", {}))
    ]
    doubtful.sort(
        key=lambda index: (
            recovery_priority(records[index].get("result", {})),
            index,
        )
    )
    return doubtful[:maximum]


def _clip_bbox(
    bbox: Mapping[str, float],
    image_size: tuple[int, int],
    *,
    pad_x: float,
    pad_y: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x0 = max(0, int(float(bbox["x"]) - pad_x))
    y0 = max(0, int(float(bbox["y"]) - pad_y))
    x1 = min(
        width,
        int(float(bbox["x"]) + float(bbox["width"]) + pad_x + 0.999),
    )
    y1 = min(
        height,
        int(float(bbox["y"]) + float(bbox["height"]) + pad_y + 0.999),
    )
    return x0, y0, x1, y1


def expanded_axis_crop(
    image: Image.Image,
    bbox: Mapping[str, float],
    *,
    padding_ratio: float = 0.55,
) -> Image.Image:
    """Keep surrounding source pixels when the detector clipped a glyph."""

    short_edge = max(
        1.0,
        min(float(bbox["width"]), float(bbox["height"])),
    )
    pad = max(8.0, short_edge * padding_ratio)
    x0, y0, x1, y1 = _clip_bbox(
        bbox,
        image.size,
        pad_x=pad * 1.25,
        pad_y=pad,
    )
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (1, 1), "white")
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def _ordered_quad(
    polygon: Sequence[Sequence[float]],
) -> np.ndarray | None:
    points: list[tuple[float, float]] = []
    for point in polygon:
        if len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    if len(points) < 4:
        return None

    values = np.asarray(points, dtype=np.float32)
    if len(values) != 4:
        if cv2 is None:
            return None
        values = cv2.boxPoints(cv2.minAreaRect(values)).astype(np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = values.sum(axis=1)
    differences = np.diff(values, axis=1).reshape(-1)
    ordered[0] = values[np.argmin(sums)]  # top-left
    ordered[2] = values[np.argmax(sums)]  # bottom-right
    ordered[1] = values[np.argmin(differences)]  # top-right
    ordered[3] = values[np.argmax(differences)]  # bottom-left
    if len(np.unique(ordered, axis=0)) < 4:
        return None
    return ordered


def _expanded_quad(quad: np.ndarray, padding_ratio: float) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = quad
    horizontal = (top_right - top_left) + (bottom_right - bottom_left)
    vertical = (bottom_left - top_left) + (bottom_right - top_right)
    horizontal_norm = float(np.linalg.norm(horizontal))
    vertical_norm = float(np.linalg.norm(vertical))
    if horizontal_norm < 1e-6 or vertical_norm < 1e-6:
        return quad

    axis_x = horizontal / horizontal_norm
    axis_y = vertical / vertical_norm
    width = max(
        float(np.linalg.norm(top_right - top_left)),
        float(np.linalg.norm(bottom_right - bottom_left)),
    )
    height = max(
        float(np.linalg.norm(bottom_left - top_left)),
        float(np.linalg.norm(bottom_right - top_right)),
    )
    pad = max(4.0, min(width, height) * padding_ratio)
    half_width = width / 2.0 + pad * 1.25
    half_height = height / 2.0 + pad
    center = quad.mean(axis=0)
    return np.asarray(
        [
            center - axis_x * half_width - axis_y * half_height,
            center + axis_x * half_width - axis_y * half_height,
            center + axis_x * half_width + axis_y * half_height,
            center - axis_x * half_width + axis_y * half_height,
        ],
        dtype=np.float32,
    )


def rectify_polygon_crop(
    image: Image.Image,
    polygon: Sequence[Sequence[float]],
    bbox: Mapping[str, float],
    *,
    padding_ratio: float = 0.35,
) -> Image.Image:
    """Perspective-straighten the detector polygon with source-pixel padding."""

    quad = _ordered_quad(polygon)
    if quad is None:
        return expanded_axis_crop(image, bbox, padding_ratio=padding_ratio)
    expanded = _expanded_quad(quad, padding_ratio)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image.width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image.height - 1)
    polygon_area = 0.5 * abs(
        float(
            np.dot(expanded[:, 0], np.roll(expanded[:, 1], 1))
            - np.dot(expanded[:, 1], np.roll(expanded[:, 0], 1))
        )
    )
    if len(np.unique(expanded, axis=0)) < 4 or polygon_area < 4.0:
        return expanded_axis_crop(image, bbox, padding_ratio=padding_ratio)

    top_left, top_right, bottom_right, bottom_left = expanded
    target_width = max(
        1,
        int(
            round(
                max(
                    np.linalg.norm(top_right - top_left),
                    np.linalg.norm(bottom_right - bottom_left),
                )
            )
        ),
    )
    target_height = max(
        1,
        int(
            round(
                max(
                    np.linalg.norm(bottom_left - top_left),
                    np.linalg.norm(bottom_right - top_right),
                )
            )
        ),
    )

    if cv2 is not None:
        destination = np.asarray(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(expanded, destination)
        warped = cv2.warpPerspective(
            np.asarray(image.convert("RGB")),
            transform,
            (target_width, target_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return Image.fromarray(warped).convert("RGB")

    # Pillow-only fallback: retain the expanded pixels and deskew the dominant
    # top edge. The production route above performs the full perspective warp.
    min_x = max(0, int(np.floor(expanded[:, 0].min())))
    min_y = max(0, int(np.floor(expanded[:, 1].min())))
    max_x = min(image.width, int(np.ceil(expanded[:, 0].max())))
    max_y = min(image.height, int(np.ceil(expanded[:, 1].max())))
    if max_x <= min_x or max_y <= min_y:
        return expanded_axis_crop(image, bbox, padding_ratio=padding_ratio)
    angle = degrees(
        atan2(
            float(top_right[1] - top_left[1]),
            float(top_right[0] - top_left[0]),
        )
    )
    return image.crop((min_x, min_y, max_x, max_y)).rotate(
        -angle,
        expand=True,
        fillcolor=(255, 255, 255),
    )


def build_recovery_crops(
    image: Image.Image,
    bbox: Mapping[str, float],
    polygon: Sequence[Sequence[float]],
) -> tuple[RecoveryCrop, ...]:
    """Build exactly two complementary, recognition-only recovery inputs."""

    rectified = rectify_polygon_crop(image, polygon, bbox)
    expanded = expanded_axis_crop(image, bbox)
    return (
        RecoveryCrop("recovery_rectified", rectified),
        RecoveryCrop(
            "recovery_expanded_sharp",
            ImageOps.autocontrast(expanded.convert("L"), cutoff=0).convert("RGB"),
        ),
    )


def build_context_crop(
    image: Image.Image,
    bbox: Mapping[str, float],
) -> Image.Image:
    """Expand mainly to the left so split labels such as ``SCALE 2:1`` rejoin."""

    width = max(float(bbox["width"]), 1.0)
    height = max(float(bbox["height"]), 1.0)
    x0, y0, x1, y1 = _clip_bbox(
        bbox,
        image.size,
        pad_x=max(90.0, width * 3.5),
        pad_y=max(12.0, height * 1.25),
    )
    # The generic helper expands symmetrically. Give preceding field labels
    # extra room without turning the crop into a complete drawing row.
    x0 = max(0, int(float(bbox["x"]) - max(140.0, width * 6.0)))
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (1, 1), "white")
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def _normalized_vote(text: str) -> str:
    return "".join(text.upper().split())


def _raw_token(result: Mapping[str, Any]) -> str:
    raw = str(result.get("raw_ocr") or result.get("text") or "").upper()
    return _COMPACT_TOKEN.sub("", raw)


def _has_confusable_conflict(tokens: Sequence[str]) -> bool:
    token_set = set(tokens)
    return any(pair.issubset(token_set) for pair in _CONFUSABLE_PAIRS)


def resolve_recovery_consensus(
    primary: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    *,
    attempted: bool,
    budget_exhausted: bool = False,
) -> dict[str, Any]:
    """Choose the best read and decide whether human review is still required."""

    all_attempts = [dict(primary), *(dict(item) for item in attempts)]
    nonempty = [
        item for item in all_attempts if str(item.get("text") or "").strip()
    ]
    base = dict(primary)
    base["recovery_attempted"] = attempted
    base["recovery_attempts"] = [
        {
            "text": str(item.get("text") or ""),
            "raw_ocr": str(item.get("raw_ocr") or ""),
            "confidence": float(item.get("confidence") or 0.0),
            "ocr_profile": str(item.get("ocr_profile") or ""),
        }
        for item in attempts
    ]

    if not nonempty:
        base.update(
            text="",
            confidence=0.0,
            agreement=0.0,
            needs_review=True,
            review_reason=(
                "Recovery budget reached before this unread candidate"
                if budget_exhausted
                else "Primary and recovery recognition returned no text"
            ),
            stable_alpha=False,
            confusable_conflict=False,
        )
        return base

    raw_tokens = [token for token in map(_raw_token, nonempty) if token]
    stable_alpha = bool(
        attempted
        and len(raw_tokens) >= 2
        and len(set(raw_tokens)) == 1
        and len(raw_tokens[0]) == 1
        and raw_tokens[0].isalpha()
    )
    if stable_alpha:
        selected = max(
            nonempty,
            key=lambda item: float(item.get("confidence") or 0.0),
        )
        base.update(selected)
        base.update(
            text=raw_tokens[0],
            agreement=1.0,
            needs_review=False,
            review_reason="",
            stable_alpha=True,
            confusable_conflict=False,
            ocr_profile="recovery_consensus",
        )
        return base

    confusable_conflict = _has_confusable_conflict(raw_tokens)
    votes = Counter(
        _normalized_vote(str(item.get("text") or "")) for item in nonempty
    )
    winner, winner_count = max(
        votes.items(),
        key=lambda item: (
            item[1],
            max(
                float(candidate.get("confidence") or 0.0)
                for candidate in nonempty
                if _normalized_vote(str(candidate.get("text") or "")) == item[0]
            ),
        ),
    )
    winner_attempts = [
        item
        for item in nonempty
        if _normalized_vote(str(item.get("text") or "")) == winner
    ]
    selected = max(
        winner_attempts,
        key=lambda item: float(item.get("confidence") or 0.0),
    )
    agreement = winner_count / max(len(nonempty), 1)
    stable_consensus = bool(
        attempted
        and winner_count >= 2
        and agreement >= (2 / 3)
        and float(selected.get("confidence") or 0.0) >= 0.55
    )
    primary_was_safe = not result_needs_recovery(primary)
    needs_review = bool(
        confusable_conflict
        or budget_exhausted
        or not (stable_consensus or primary_was_safe)
    )
    if confusable_conflict:
        reason = "Numeric and alphabetic recovery reads conflict"
    elif budget_exhausted:
        reason = "Recovery budget reached before this doubtful candidate"
    elif needs_review:
        reason = "Recovery reads did not reach a stable consensus"
    else:
        reason = ""

    base.update(selected)
    base.update(
        agreement=agreement,
        needs_review=needs_review,
        review_reason=reason,
        stable_alpha=False,
        confusable_conflict=confusable_conflict,
        ocr_profile=(
            "recovery_consensus" if attempted else selected.get("ocr_profile")
        ),
    )
    return base


def reconstruct_line_context(
    record: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Join nearby same-line reads without making another model call."""

    bbox = record.get("bbox", {})
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = max(float(bbox["width"]), 1.0)
        height = max(float(bbox["height"]), 1.0)
    except (KeyError, TypeError, ValueError):
        return str(record.get("text") or "").strip()

    neighbours: list[tuple[float, str]] = []
    for other in records:
        text = str(other.get("text") or "").strip()
        other_bbox = other.get("bbox", {})
        if not text or other is record:
            continue
        try:
            other_x = float(other_bbox["x"])
            other_y = float(other_bbox["y"])
            other_width = max(float(other_bbox["width"]), 1.0)
            other_height = max(float(other_bbox["height"]), 1.0)
        except (KeyError, TypeError, ValueError):
            continue
        overlap = max(
            0.0,
            min(y + height, other_y + other_height) - max(y, other_y),
        )
        if overlap / min(height, other_height) < 0.35:
            continue
        gap = max(
            0.0,
            max(x, other_x) - min(x + width, other_x + other_width),
        )
        if gap > max(120.0, 10.0 * max(height, other_height)):
            continue
        neighbours.append((other_x, text))

    neighbours.append((x, str(record.get("text") or "").strip()))
    neighbours.sort(key=lambda item: item[0])
    return " ".join(text for _x, text in neighbours if text)
