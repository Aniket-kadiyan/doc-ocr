"""Candidate-local graphical enrichment for authoritative native PDF text.

The text layer owns every letter and digit. This module only supplies marks
that CAD commonly draws as vectors: degree rings and structured engineering
frames. It never invokes OCR and never substitutes a raster-recognized value.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import replace
import math
import re
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from feature_classifier import classify_feature
from structured_symbol_vision import (
    StructuredObjectPlan,
    plan_structured_engineering_symbols,
    structured_symbol_statistics,
)


_EXPLICIT_SURFACE_RE = re.compile(
    r"^(?:RA|RZ|RMAX|RQ)\s*\d|^N\s?(?:1[0-2]|[1-9])$",
    re.IGNORECASE,
)
_FCF_NATIVE_RE = re.compile(
    r"^(?:[Ø⌀])?(?:\d+(?:\.\d+)?|\.\d+)(?:[ⓂⓁⓅMLP])?"
    r"(?:[A-Z](?:[ⓂⓁⓅMLP])?){1,3}$",
    re.IGNORECASE,
)
_ANGLE_TOLERANCE_RE = re.compile(
    r"^(?P<base>\d+(?:\.\d+)?)\s*"
    r"(?P<body>(?:±|\+|[-−])\s*\d+(?:\.\d+)?"
    r"(?:\s*/\s*(?:\+|[-−])?\s*\d+(?:\.\d+)?)?)$"
)
_PAREN_NUMBER_RE = re.compile(r"^\(\s*(\d+(?:\.\d+)?)\s*\)$")


def _box(item: Mapping[str, Any]) -> dict[str, float]:
    raw = item.get("bbox") if isinstance(item.get("bbox"), Mapping) else item
    return {
        key: float(raw[key])  # type: ignore[index]
        for key in ("x", "y", "width", "height")
    }


def _candidate_children(outcome: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_children = outcome.get("assembly_children")
    children = [
        deepcopy(dict(child))
        for child in raw_children or ()
        if isinstance(child, Mapping) and isinstance(child.get("bbox"), Mapping)
    ]
    if children:
        return children
    return [
        {
            "candidate_id": str(outcome.get("candidate_id") or ""),
            "text": str(outcome.get("text") or ""),
            "raw_text": str(outcome.get("raw_text") or outcome.get("text") or ""),
            "bbox": dict(_box(outcome)),
            "confidence": float(outcome.get("confidence") or 0.995),
            "orientation": str(outcome.get("orientation") or "horizontal"),
            "rotation": float(outcome.get("rotation") or 0.0),
            "recognition_source": "native_pdf",
        }
    ]


def _candidate_scale(outcome: Mapping[str, Any]) -> float:
    sizes = []
    for child in _candidate_children(outcome):
        box = _box(child)
        sizes.append(max(3.0, min(box["width"], box["height"])))
    if sizes:
        return float(median(sizes))
    box = _box(outcome)
    return max(3.0, min(box["width"], box["height"]))


def _local_structured_records(
    outcome: Mapping[str, Any],
    *,
    origin_x: float,
    origin_y: float,
) -> list[dict[str, Any]]:
    children = _candidate_children(outcome)
    parent_text = re.sub(r"\s+", "", str(outcome.get("text") or ""))
    diameter_prefix = parent_text.startswith(("Ø", "⌀"))
    prefixed = False
    records: list[dict[str, Any]] = []
    for index, child in enumerate(children):
        text = str(child.get("text") or "").strip()
        if diameter_prefix and not prefixed and re.search(r"\d", text):
            if not text.startswith(("Ø", "⌀")):
                text = "Ø" + text
            prefixed = True
        box = _box(child)
        local_box = {
            "x": box["x"] - origin_x,
            "y": box["y"] - origin_y,
            "width": box["width"],
            "height": box["height"],
        }
        records.append(
            {
                **child,
                "candidate_id": str(
                    child.get("candidate_id")
                    or f"{outcome.get('candidate_id', 'PDF')}:R{index + 1:03d}"
                ),
                "bbox": local_box,
                "text": text,
                "recognized": bool(text),
                "table_excluded": False,
                "result": {
                    "text": text,
                    "confidence": float(child.get("confidence") or 0.995),
                    "recognition_source": "native_pdf",
                },
            }
        )
    return records


def _candidate_structured_plan(
    image: Image.Image,
    outcome: Mapping[str, Any],
) -> StructuredObjectPlan | None:
    box = _box(outcome)
    scale = _candidate_scale(outcome)
    # The first characteristic cell contains no text-layer run and can sit a
    # full cell beyond the native text hull. Datum cells at the far end add a
    # similar overhang, especially after a vertical frame is rotated upright.
    pad = max(10.0, 6.5 * scale)
    left = max(0, int(math.floor(box["x"] - pad)))
    top = max(0, int(math.floor(box["y"] - pad)))
    right = min(image.width, int(math.ceil(box["x"] + box["width"] + pad)))
    bottom = min(image.height, int(math.ceil(box["y"] + box["height"] + pad)))
    if right - left < 8 or bottom - top < 8:
        return None
    crop = image.crop((left, top, right, bottom))
    records = _local_structured_records(
        outcome,
        origin_x=float(left),
        origin_y=float(top),
    )
    plans = plan_structured_engineering_symbols(
        crop,
        records,
        rotations=(0, 90, 270),
    )
    feature_plans = [
        plan for plan in plans if plan.evidence.kind == "feature_control_frame"
    ]
    if not feature_plans:
        return None
    plan = max(
        feature_plans,
        key=lambda item: (
            item.evidence.complete,
            len(item.member_indexes),
            item.evidence.confidence,
        ),
    )

    def translate(box_value: Mapping[str, float]) -> dict[str, float]:
        return {
            "x": round(float(box_value["x"]) + left, 1),
            "y": round(float(box_value["y"]) + top, 1),
            "width": round(float(box_value["width"]), 1),
            "height": round(float(box_value["height"]), 1),
        }

    evidence = replace(
        plan.evidence,
        bbox=translate(plan.evidence.bbox),
        cells=tuple(translate(cell) for cell in plan.evidence.cells),
        source="frame_geometry+native_pdf",
        visual_features={
            **dict(plan.evidence.visual_features),
            "native_text_authoritative": True,
        },
    )
    return replace(plan, bbox=translate(plan.bbox), evidence=evidence)


def _structured_text_candidate(text: str) -> bool:
    compact = re.sub(r"[\s|]", "", text).upper()
    if compact.endswith(("MIN", "MAX", "TYP", "REF", "BASIC", "THRU")):
        return False
    if not _FCF_NATIVE_RE.fullmatch(compact):
        return False
    return bool(
        compact.startswith(("Ø", "⌀"))
        or re.search(r"[A-ZⓂⓁⓅMLP]\s+[A-ZⓂⓁⓅMLP]", text, re.IGNORECASE)
    )


def _update_feature_metadata(outcome: dict[str, Any]) -> None:
    feature = classify_feature(str(outcome.get("text") or ""))
    outcome.update(
        {
            "category": feature.category,
            "subtype": feature.subtype,
            "label": feature.label,
        }
    )


def _set_state(
    outcome: dict[str, Any],
    state: str,
    rule: str,
    reason: str,
) -> None:
    outcome.update(
        {
            "state": state,
            "rule": rule,
            "reason": reason,
            "page_filter_rule": rule,
            "page_filter_reason": reason,
            "needs_review": state == "review",
            "review_reason": reason if state == "review" else "",
        }
    )


def _apply_structured_plan(
    outcome: dict[str, Any], plan: StructuredObjectPlan
) -> None:
    outcome.update(
        {
            "text": plan.text,
            "bbox": dict(plan.bbox),
            "confidence": min(
                float(outcome.get("confidence") or 0.995),
                float(plan.confidence or 0.995),
            ),
            "type": "GD&T",
            "category": "GD&T",
            "subtype": plan.evidence.subtype,
            "label": plan.evidence.subtype or "Feature Control Frame",
            "engineering_symbol": plan.evidence.to_dict(),
            "assembly_rule": "native_geometry_feature_control_frame",
            "assembly_review_reason": plan.review_reason,
        }
    )
    if plan.evidence.complete:
        _set_state(
            outcome,
            "eligible",
            "native_geometry_feature_control_frame",
            "Exact PDF text confirmed by a graphical feature-control frame",
        )
    else:
        _set_state(
            outcome,
            "review",
            "native_geometry_feature_control_frame_unresolved",
            plan.review_reason
            or "Feature-control frame requires user confirmation",
        )


def _surface_plan(
    image: Image.Image, outcome: Mapping[str, Any]
) -> StructuredObjectPlan | None:
    box = _box(outcome)
    left, top = max(0, int(box["x"])), max(0, int(box["y"]))
    right = min(image.width, max(left + 2, int(math.ceil(box["x"] + box["width"]))))
    bottom = min(image.height, max(top + 2, int(math.ceil(box["y"] + box["height"]))))
    crop = image.crop((left, top, right, bottom))
    local = {
        **dict(outcome),
        "bbox": {
            "x": 0.0,
            "y": 0.0,
            "width": float(crop.width),
            "height": float(crop.height),
        },
        "result": {
            "text": str(outcome.get("text") or ""),
            "confidence": float(outcome.get("confidence") or 0.995),
            "recognition_source": "native_pdf",
        },
    }
    plans = plan_structured_engineering_symbols(crop, [local])
    return next(
        (plan for plan in plans if plan.evidence.kind == "surface_finish"),
        None,
    )


def _apply_surface_plan(
    outcome: dict[str, Any], plan: StructuredObjectPlan
) -> None:
    box = _box(outcome)
    evidence = replace(
        plan.evidence,
        bbox=dict(box),
        source="surface_text+native_pdf",
        visual_features={
            **dict(plan.evidence.visual_features),
            "native_text_authoritative": True,
        },
    )
    outcome.update(
        {
            "type": "Surface Finish",
            "category": "Surface Finish",
            "subtype": evidence.subtype,
            "label": "Surface Finish",
            "engineering_symbol": evidence.to_dict(),
            "assembly_rule": "native_surface_finish_syntax",
        }
    )
    _set_state(
        outcome,
        "eligible",
        "native_surface_finish_syntax",
        "Explicit surface-finish syntax preserved from the PDF text layer",
    )


def _oriented_candidate_crop(
    image: Image.Image, outcome: Mapping[str, Any]
) -> Image.Image | None:
    box = _box(outcome)
    oriented = outcome.get("oriented_box")
    if isinstance(oriented, Mapping):
        origin_x = float(oriented.get("x", box["x"]))
        origin_y = float(oriented.get("y", box["y"]))
        width = max(2.0, float(oriented.get("width", box["width"])))
        height = max(2.0, float(oriented.get("height", box["height"])))
        angle = math.radians(float(oriented.get("rotation", 0.0)))
    else:
        origin_x, origin_y = box["x"], box["y"]
        width, height, angle = box["width"], box["height"], 0.0

    local_x0, local_x1 = -0.10 * height, width + 1.50 * height
    local_y0, local_y1 = -0.25 * height, 1.25 * height
    target_width = max(2, int(round(local_x1 - local_x0)))
    target_height = max(2, int(round(local_y1 - local_y0)))
    cosine, sine = math.cos(angle), math.sin(angle)
    source_x = origin_x + cosine * local_x0 - sine * local_y0
    source_y = origin_y + sine * local_x0 + cosine * local_y0
    return image.convert("RGB").transform(
        (target_width, target_height),
        Image.Transform.AFFINE,
        (cosine, -sine, source_x, sine, cosine, source_y),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )


def _components(binary: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    height, width = binary.shape
    visited = np.zeros(binary.shape, dtype=bool)
    components: list[tuple[int, int, int, int, np.ndarray]] = []
    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not binary[start_y, start_x]:
                continue
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            points: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                points.append((y, x))
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        next_y, next_x = y + delta_y, x + delta_x
                        if (
                            0 <= next_y < height
                            and 0 <= next_x < width
                            and not visited[next_y, next_x]
                            and binary[next_y, next_x]
                        ):
                            visited[next_y, next_x] = True
                            queue.append((next_y, next_x))
            ys = np.asarray([point[0] for point in points])
            xs = np.asarray([point[1] for point in points])
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            components.append(
                (x0, y0, x1 - x0 + 1, y1 - y0 + 1, binary[y0 : y1 + 1, x0 : x1 + 1])
            )
    return components


def _has_degree_ring(image: Image.Image) -> bool:
    """Detect a small, enclosed superscript ring in an already levelled crop."""

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.size == 0 or min(gray.shape) < 6:
        return False
    band_height = max(4, int(round(gray.shape[0] * 0.72)))
    binary = gray[:band_height, :] < 190
    components = [
        component
        for component in _components(binary)
        if component[2] >= 2 and component[3] >= 2
        and component[2] < 0.45 * max(gray.shape[1], 1)
    ]
    if not components:
        return False
    component_heights = sorted(component[3] for component in components)
    main_height = float(component_heights[max(0, int(0.75 * (len(component_heights) - 1)))])
    main_height = max(main_height, 4.0)
    main_tops = [
        component[1]
        for component in components
        if component[3] >= 0.75 * main_height
    ]
    main_top = float(median(main_tops)) if main_tops else 0.0

    for _x, y, width, height, pixels in components:
        if (
            min(width, height) < 2
            or max(width, height) > 0.62 * main_height
            or max(width, height) / max(min(width, height), 1) > 1.8
            or y + height > main_top + 0.42 * main_height
            or y + height >= band_height
        ):
            continue
        centre_y, centre_x = height // 2, width // 2
        centre_y0, centre_y1 = max(0, centre_y - 1), min(height, centre_y + 2)
        centre_x0, centre_x1 = max(0, centre_x - 1), min(width, centre_x + 2)
        if np.all(pixels[centre_y0:centre_y1, centre_x0:centre_x1]):
            continue
        third_x, third_y = max(1, width // 3), max(1, height // 3)
        touches_sides = (
            pixels[:, :third_x].any()
            and pixels[:, -third_x:].any()
            and pixels[:third_y, :].any()
            and pixels[-third_y:, :].any()
        )
        density = float(pixels.mean())
        if touches_sides and 0.12 <= density <= 0.82:
            return True
    return False


def _angle_candidate(text: str) -> bool:
    stripped = text.strip()
    return bool(
        "°" not in stripped
        and (
            _PAREN_NUMBER_RE.fullmatch(stripped)
            or _ANGLE_TOLERANCE_RE.fullmatch(stripped)
        )
    )


def _add_degree_marks(text: str) -> str:
    stripped = text.strip()
    parenthesized = _PAREN_NUMBER_RE.fullmatch(stripped)
    if parenthesized:
        return f"({parenthesized.group(1)}°)"
    tolerance = _ANGLE_TOLERANCE_RE.fullmatch(stripped)
    if not tolerance:
        return stripped
    body = re.sub(r"\s+", "", tolerance.group("body")).replace("−", "-")
    body = re.sub(r"(\d+(?:\.\d+)?)", r"\1°", body)
    return f"{tolerance.group('base')}°{body}"


def _apply_degree_enrichment(
    image: Image.Image, outcome: dict[str, Any]
) -> None:
    text = str(outcome.get("text") or "")
    if not _angle_candidate(text):
        return
    crop = _oriented_candidate_crop(image, outcome)
    if crop is None or not _has_degree_ring(crop):
        return
    enriched = _add_degree_marks(text)
    if enriched == text:
        return
    evidence = deepcopy(dict(outcome.get("recognition_evidence") or {}))
    evidence.update(
        {
            "selected_source": "native_pdf",
            "sources": ["native_pdf"],
            "native_text": enriched,
            "ocr_text": "",
            "agreement": 1.0,
            "conflict": False,
            "geometry_symbols": ["degree"],
            "native_text_before_geometry": text,
        }
    )
    outcome.update(
        {
            "text": enriched,
            "recognition_source": "native_pdf",
            "recognition_evidence": evidence,
            "symbols_detected": {
                **dict(outcome.get("symbols_detected") or {}),
                "degree": True,
            },
        }
    )
    _update_feature_metadata(outcome)
    structurally_uncertain = bool(
        outcome.get("native_structure_ambiguous")
        or outcome.get("assembly_conflict")
        or outcome.get("source_conflict")
        or evidence.get("conflict")
    )
    if structurally_uncertain:
        _set_state(
            outcome,
            "review",
            "native_degree_geometry_requires_review",
            str(outcome.get("assembly_review_reason") or "")
            or "Degree geometry was found, but the native value structure remains uncertain",
        )
    else:
        _set_state(
            outcome,
            "eligible",
            "native_text+degree_geometry",
            "Exact PDF digits enriched by a candidate-local graphical degree mark",
        )


def _rebuild_result(
    original: Mapping[str, Any], outcomes: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    regions = [item for item in outcomes if item.get("state") == "eligible"]
    reviews = [item for item in outcomes if item.get("state") == "review"]
    rule_counts = Counter(str(item.get("rule") or "unknown") for item in outcomes)
    source_counts = Counter(
        str(item.get("recognition_source") or "native_pdf") for item in outcomes
    )
    structured = [item for item in outcomes if item.get("engineering_symbol")]
    return {
        **deepcopy(dict(original)),
        "count": len(regions),
        "detected_count": len(outcomes),
        "recognized_count": sum(bool(item.get("recognized")) for item in outcomes),
        "eligible_count": len(regions),
        "excluded_count": sum(item.get("state") == "excluded" for item in outcomes),
        "review_count": len(reviews),
        "unread_count": sum(not bool(item.get("recognized")) for item in outcomes),
        "filter_rule_counts": dict(sorted(rule_counts.items())),
        "recognition_source_counts": dict(sorted(source_counts.items())),
        "regions": regions,
        "review_candidates": reviews,
        "candidate_outcomes": list(outcomes),
        "structured_symbol_stats": structured_symbol_statistics(structured),
    }


def enrich_native_result_with_geometry(
    image: Image.Image,
    result: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Enrich native candidates without running OCR or replacing native text."""

    if not result:
        return None
    outcomes = [
        deepcopy(dict(item))
        for item in result.get("candidate_outcomes", ())
        if isinstance(item, Mapping)
    ]
    for outcome in outcomes:
        text = str(outcome.get("text") or "").strip()
        if _EXPLICIT_SURFACE_RE.search(text):
            plan = _surface_plan(image, outcome)
            if plan is not None:
                _apply_surface_plan(outcome, plan)
            continue

        if _structured_text_candidate(text):
            plan = _candidate_structured_plan(image, outcome)
            if plan is not None:
                _apply_structured_plan(outcome, plan)
            else:
                _set_state(
                    outcome,
                    "review",
                    "native_structured_symbol_unresolved",
                    "Native tolerance/datum text needs graphical frame confirmation",
                )
            continue

        _apply_degree_enrichment(image, outcome)
    return _rebuild_result(result, outcomes)

