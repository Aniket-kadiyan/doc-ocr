"""Production whole-page integration for graphical GD&T frames.

The established scan remains authoritative.  This module runs only after the
native-PDF/text-first and raster-OCR results have already been selected and
merged.  It changes nothing outside a visually detected feature-control frame.

Inside a frame, the small OCR objects that represent individual compartments
are retired to the visible Other lifecycle state and replaced by one logical
GD&T object.  Exact native-PDF text is reused before the frame OCR result, and
all absorbed evidence remains attached to the frame and its retired outcomes.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

import fcf_parse
import gdt_registry as registry
from candidate_lifecycle import final_candidate_counts, retire_candidate_outcomes
from engineering_disposition import (
    disposition_engineering_object,
    engineering_disposition_statistics,
)
from engineering_value_parser import (
    engineering_parse_statistics,
    parse_engineering_value,
)
from gdt_config import GdtConfig, get_config
from gdt_pipeline import MAX_PLAUSIBLE_TOLERANCE, scan_regions
from page_scan import bbox_overlap_fraction, overlaps_existing_value
from structured_symbol_vision import structured_symbol_statistics


OcrFn = Callable[[Image.Image], tuple[str, float]]
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d+(?:[.,]\d+)?|[.,]\d+)")


def page_scan_enabled(config: GdtConfig | None = None) -> bool:
    """Whether the production whole-page hook is explicitly enabled."""

    resolved = config or get_config()
    return bool(resolved.enabled and resolved.page_scan_enabled)


def _box(item: Mapping[str, Any]) -> dict[str, float]:
    raw = item.get("bbox") if isinstance(item.get("bbox"), Mapping) else item
    return {
        key: float(raw[key])  # type: ignore[index]
        for key in ("x", "y", "width", "height")
    }


def _centre_inside(inner: Mapping[str, Any], outer: Mapping[str, Any]) -> bool:
    try:
        inner_box, outer_box = _box(inner), _box(outer)
    except (KeyError, TypeError, ValueError):
        return False
    centre_x = inner_box["x"] + inner_box["width"] / 2.0
    centre_y = inner_box["y"] + inner_box["height"] / 2.0
    return (
        outer_box["x"] <= centre_x <= outer_box["x"] + outer_box["width"]
        and outer_box["y"] <= centre_y <= outer_box["y"] + outer_box["height"]
    )


def _candidate_id(item: Mapping[str, Any]) -> str:
    return str(item.get("candidate_id") or "").strip()


def _is_native(item: Mapping[str, Any]) -> bool:
    source = str(item.get("recognition_source") or "").casefold()
    evidence = item.get("recognition_evidence")
    sources = (
        [str(value).casefold() for value in evidence.get("sources", ())]
        if isinstance(evidence, Mapping)
        else []
    )
    return source.startswith("native_pdf") or "native_pdf" in sources


def _numbers(text: object) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for token in _NUMBER_RE.findall(str(text or "")):
        normalized = token.replace(",", ".")
        if normalized.startswith("."):
            normalized = f"0{normalized}"
        try:
            values.append(Decimal(normalized).normalize())
        except InvalidOperation:
            continue
    return tuple(values)


def _dedupe_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for source in items:
        try:
            box = _box(source)
        except (KeyError, TypeError, ValueError):
            continue
        key = (
            _candidate_id(source),
            str(source.get("text") or ""),
            round(box["x"], 2),
            round(box["y"], 2),
            round(box["width"], 2),
            round(box["height"], 2),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(deepcopy(dict(source)))
    return unique


def _native_evidence(
    frame: Mapping[str, Any],
    regions: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _dedupe_items(
        [
            item
            for item in (*regions, *reviews, *outcomes)
            if _is_native(item)
            and str(item.get("text") or "").strip()
            and _centre_inside(item, frame)
        ]
    )


def _full_native_gdt(items: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for item in items:
        parsed = item.get("engineering_parse")
        parse_kind = str(parsed.get("kind") or "") if isinstance(parsed, Mapping) else ""
        if str(item.get("category") or "").upper() == "GD&T" or parse_kind == "gdt":
            candidates.append(dict(item))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            len(_numbers(item.get("text"))),
            len(str(item.get("text") or "")),
            float(item.get("confidence") or 0.0),
        ),
    )


def _cell_sources(
    cell: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_box = cell.get("bounding_box")
    if not isinstance(raw_box, Mapping):
        return []
    return [dict(item) for item in items if _centre_inside(item, raw_box)]


def _best_cell_text(
    cell: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    *,
    numeric: bool,
) -> tuple[str, float] | None:
    sources = _cell_sources(cell, items)
    if numeric:
        sources = [item for item in sources if _numbers(item.get("text"))]
    else:
        sources = [
            item
            for item in sources
            if re.fullmatch(r"[A-Za-z]{1,2}(?:\s*[-/]\s*[A-Za-z])?", str(item.get("text") or "").strip())
        ]
    if not sources:
        return None
    source = max(
        sources,
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            len(str(item.get("text") or "")),
        ),
    )
    return str(source.get("text") or "").strip(), float(source.get("confidence") or 0.995)


def _apply_native_cells(
    frames: list[dict[str, Any]],
    native_items: Sequence[Mapping[str, Any]],
) -> bool:
    """Use exact vector text for tolerance/datum cells when geometry owns it."""

    used = False
    for frame in frames:
        compartments = [
            cell
            for cell in frame.get("compartments", ())
            if isinstance(cell, dict)
        ]
        for cell in compartments:
            index = int(cell.get("index") or 0)
            if index == 1:
                selected = _best_cell_text(cell, native_items, numeric=True)
                if selected is None:
                    continue
                text, confidence = selected
                parsed = fcf_parse.parse_tolerance_compartment(text, confidence)
                if parsed.get("value") is None:
                    continue
                frame["tolerance"] = {
                    "value": parsed["value"],
                    "state": fcf_parse.PRESENT,
                    "confidence": round(confidence, 4),
                    "raw": text,
                }
                if parsed.get("diameter_zone"):
                    frame["diameter_zone"] = {
                        "value": True,
                        "state": fcf_parse.PRESENT,
                        "confidence": round(confidence, 4),
                        "raw": text,
                    }
                if parsed.get("material_condition"):
                    frame["material_condition"] = {
                        "value": parsed["material_condition"],
                        "state": fcf_parse.PRESENT,
                        "confidence": round(confidence, 4),
                        "raw": text,
                    }
                frame["modifiers"] = list(parsed.get("modifiers") or ())
                cell["text"] = text
                cell["confidence"] = round(confidence, 4)
                used = True
            elif index >= 2:
                selected = _best_cell_text(cell, native_items, numeric=False)
                if selected is None:
                    continue
                text, confidence = selected
                datum, warnings = fcf_parse.parse_datum_compartment(
                    text,
                    confidence,
                    index - 1,
                )
                if datum is None:
                    continue
                datums = [
                    value
                    for value in frame.get("datums", ())
                    if int(value.get("precedence") or 0) != datum.precedence
                ]
                datums.append(datum.to_dict())
                frame["datums"] = sorted(
                    datums,
                    key=lambda value: int(value.get("precedence") or 0),
                )
                frame["warnings"] = list(
                    dict.fromkeys([*frame.get("warnings", ()), *warnings])
                )
                cell["text"] = text
                cell["confidence"] = round(confidence, 4)
                used = True
    return used


def _registry_glyph(symbol_id: object, fallback: object = "") -> str:
    symbol = registry.find(str(symbol_id or ""))
    return str((symbol.unicode if symbol else None) or fallback or "").strip()


def _frame_payload_text(frame: Mapping[str, Any]) -> str:
    parts: list[str] = []
    symbol = frame.get("symbol") if isinstance(frame.get("symbol"), Mapping) else {}
    characteristic = _registry_glyph(
        symbol.get("symbol_id"),
        symbol.get("normalized") or symbol.get("name"),
    )
    if characteristic:
        parts.append(characteristic)

    tolerance = frame.get("tolerance") if isinstance(frame.get("tolerance"), Mapping) else {}
    diameter = frame.get("diameter_zone") if isinstance(frame.get("diameter_zone"), Mapping) else {}
    tolerance_text = "Ø" if diameter.get("value") is True else ""
    if tolerance.get("state") == fcf_parse.PRESENT and tolerance.get("value") is not None:
        tolerance_text += str(tolerance.get("value"))
    elif tolerance.get("state") == fcf_parse.UNREADABLE:
        tolerance_text += "?"
    if tolerance_text:
        parts.append(tolerance_text)

    material = frame.get("material_condition") if isinstance(frame.get("material_condition"), Mapping) else {}
    if material.get("state") == fcf_parse.PRESENT and material.get("value"):
        glyph = _registry_glyph(material.get("value"), material.get("value"))
        if glyph:
            parts.append(glyph)
    for datum in frame.get("datums", ()):
        label = str(datum.get("label") or "").strip()
        if label:
            parts.append(label)
        modifier = datum.get("material_condition")
        if modifier:
            parts.append(_registry_glyph(modifier, modifier))
    return " ".join(parts)


def _review_reasons(frames: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for frame in frames:
        symbol = frame.get("symbol") if isinstance(frame.get("symbol"), Mapping) else {}
        if not symbol or symbol.get("status") != "confirmed":
            reasons.append("the geometric characteristic symbol is uncertain")
        tolerance = frame.get("tolerance") if isinstance(frame.get("tolerance"), Mapping) else {}
        if tolerance.get("state") != fcf_parse.PRESENT:
            reasons.append("the tolerance could not be read")
        else:
            try:
                value = float(tolerance.get("value"))
            except (TypeError, ValueError):
                reasons.append("the tolerance is not a valid number")
            else:
                if value <= 0 or value > MAX_PLAUSIBLE_TOLERANCE:
                    reasons.append("the tolerance is outside the expected GD&T review range")
        material = frame.get("material_condition") if isinstance(frame.get("material_condition"), Mapping) else {}
        if material.get("state") == fcf_parse.UNREADABLE:
            reasons.append("a modifier is present and could not be read")
    return list(dict.fromkeys(reasons))


def _engineering_symbol(
    region: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    *,
    native_used: bool,
    reasons: Sequence[str],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    symbols: list[str] = []
    structural_complete = True
    for frame in frames:
        symbol = frame.get("symbol") if isinstance(frame.get("symbol"), Mapping) else {}
        normalized = _registry_glyph(
            symbol.get("symbol_id"),
            symbol.get("normalized") or symbol.get("name"),
        )
        if normalized:
            symbols.append(normalized)
        tolerance = frame.get("tolerance") if isinstance(frame.get("tolerance"), Mapping) else {}
        structural_complete = structural_complete and bool(
            symbol.get("status") == "confirmed"
            and tolerance.get("state") == fcf_parse.PRESENT
            and tolerance.get("value") is not None
        )
        for cell in frame.get("compartments", ()):
            if not isinstance(cell, Mapping):
                continue
            cells.append(
                {
                    "index": int(cell.get("index") or 0),
                    "bbox": dict(cell.get("bounding_box") or {}),
                    "text": str(cell.get("text") or ""),
                    "confidence": float(cell.get("confidence") or 0.0),
                }
            )
    return {
        "schema_version": 1,
        "kind": "feature_control_frame",
        "bbox": dict(_box(region)),
        "confidence": float(region.get("confidence") or 0.0),
        "complete": structural_complete,
        "subtype": str(region.get("subtype") or "Feature Control Frame"),
        "completed_symbol": " / ".join(symbols),
        "source": "frame_geometry+native_pdf" if native_used else "frame_geometry+ocr",
        "cells": cells,
        "warnings": list(reasons),
        "visual_features": {"frame_count": len(frames)},
    }


def _absorbed_summary(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": _candidate_id(item),
            "text": str(item.get("text") or ""),
            "bbox": dict(_box(item)),
            "state": str(item.get("state") or ""),
            "recognition_source": str(item.get("recognition_source") or ""),
        }
        for item in items
    ]


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base or "fcf"
    suffix = 2
    while candidate in used:
        candidate = f"{base or 'fcf'}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _compatibility_disposition(item: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    state = str(item.get("state") or "excluded")
    return {
        "schema_version": 1,
        "state": "other" if state == "excluded" else state,
        "rule": str(item.get("rule") or item.get("page_filter_rule") or "existing_scan_state"),
        "reason": str(item.get("reason") or item.get("page_filter_reason") or "Existing scan state"),
        "parse_status": str(parsed.get("status") or "unparsed"),
        "parse_kind": str(parsed.get("kind") or "unknown"),
        "hard_context": state == "excluded",
    }


def _parse_for(item: Mapping[str, Any]) -> dict[str, Any]:
    existing = item.get("engineering_parse")
    if isinstance(existing, Mapping):
        return dict(existing)
    return parse_engineering_value(
        str(item.get("text") or ""),
        feature_category=str(item.get("category") or "") or None,
        engineering_symbol=(
            item.get("engineering_symbol")
            if isinstance(item.get("engineering_symbol"), Mapping)
            else {}
        ),
    ).to_dict()


def _disposition_for(item: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    existing = item.get("engineering_disposition")
    if isinstance(existing, Mapping):
        return dict(existing)
    return _compatibility_disposition(item, parsed)


def _retire_absorbed(
    outcomes: list[dict[str, Any]],
    absorbed: Sequence[Mapping[str, Any]],
    *,
    used_ids: set[str],
    frame_id: str,
) -> int:
    reason = "Absorbed into one graphical feature-control-frame object"
    rule = "feature_control_frame_absorption"
    ids = {_candidate_id(item) for item in absorbed if _candidate_id(item)}
    retired = retire_candidate_outcomes(
        outcomes,
        ids,
        reason=reason,
        rule=rule,
    )
    outcome_ids = {_candidate_id(item) for item in outcomes}
    for outcome in outcomes:
        if _candidate_id(outcome) not in retired:
            continue
        parsed = _parse_for(outcome)
        outcome["engineering_parse"] = parsed
        outcome["engineering_disposition"] = {
            "schema_version": 1,
            "state": "other",
            "rule": rule,
            "reason": reason,
            "parse_status": str(parsed.get("status") or "unparsed"),
            "parse_kind": str(parsed.get("kind") or "unknown"),
            "hard_context": False,
        }

    # A few recovered objects intentionally have no detector identity. They
    # were lifecycle-counted as synthetic. Give them an excluded identity now
    # so replacing them with the frame does not silently erase their evidence.
    created = 0
    for index, item in enumerate(absorbed, start=1):
        original_id = _candidate_id(item)
        if original_id and original_id in outcome_ids:
            continue
        identity = _unique_id(
            original_id or f"{frame_id}-absorbed-{index}",
            used_ids,
        )
        parsed = _parse_for(item)
        outcomes.append(
            {
                **deepcopy(dict(item)),
                "candidate_id": identity,
                "object_id": identity,
                "state": "excluded",
                "reason": reason,
                "rule": rule,
                "engineering_parse": parsed,
                "engineering_disposition": {
                    "schema_version": 1,
                    "state": "other",
                    "rule": rule,
                    "reason": reason,
                    "parse_status": str(parsed.get("status") or "unparsed"),
                    "parse_kind": str(parsed.get("kind") or "unknown"),
                    "hard_context": False,
                },
            }
        )
        outcome_ids.add(identity)
        created += 1
    return len(retired) + created


def _frame_object(
    source: Mapping[str, Any],
    *,
    candidate_id: str,
    native_items: Sequence[Mapping[str, Any]],
    absorbed: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    region = deepcopy(dict(source))
    frames = [deepcopy(dict(value)) for value in (region.get("gdt") or {}).get("frames", ())]
    graphical_text = str(region.get("text") or "")
    full_native = _full_native_gdt(native_items)
    native_cells_used = _apply_native_cells(frames, native_items)
    native_used = bool(full_native or native_cells_used)

    rebuilt = " / ".join(_frame_payload_text(frame) for frame in frames)
    selected_text = (
        str(full_native.get("text") or "").strip()
        if full_native is not None
        else rebuilt or graphical_text
    )
    native_texts = [
        str(item.get("text") or "").strip()
        for item in native_items
        if str(item.get("text") or "").strip()
    ]
    native_numbers = tuple(
        dict.fromkeys(
            value for text in native_texts for value in _numbers(text)
        )
    )
    graphical_numbers = _numbers(graphical_text)
    source_conflict = bool(
        native_numbers
        and graphical_numbers
        and any(value not in graphical_numbers for value in native_numbers)
    )
    reasons = _review_reasons(frames)
    if source_conflict:
        reasons.append("native PDF text and graphical GD&T OCR disagree on the numeric value")
    reasons = list(dict.fromkeys(reasons))

    symbol = _engineering_symbol(
        region,
        frames,
        native_used=native_used,
        reasons=reasons,
    )
    parsed = parse_engineering_value(
        selected_text,
        feature_category="GD&T",
        engineering_symbol=symbol,
    )
    disposition = disposition_engineering_object(
        parsed,
        recognized=True,
        authoritative=True,
        context_rule="feature_control_frame",
        context_reason=str(region.get("page_filter_reason") or "Graphical feature-control frame"),
        source_conflict=source_conflict,
        numeric_conflict=source_conflict,
        recognition_review_required=bool(reasons),
        recognition_needs_review=bool(reasons),
        recognition_review_reason="; ".join(reasons),
    )
    state = "excluded" if disposition.state == "other" else disposition.state
    source_name = "native_pdf+gdt" if native_used else "gdt"
    evidence = {
        "selected_source": "native_pdf" if native_used else "gdt",
        "sources": ["gdt", *( ["native_pdf"] if native_used else [])],
        "native_text": " | ".join(dict.fromkeys(native_texts)),
        "ocr_text": graphical_text,
        "agreement": 0.0 if source_conflict else (1.0 if native_used else 0.5),
        "conflict": source_conflict,
        "absorbed_candidates": _absorbed_summary(absorbed),
    }
    common = {
        **region,
        "candidate_id": candidate_id,
        "object_id": candidate_id,
        "assembly_id": candidate_id,
        "assembly_rule": "graphical_feature_control_frame",
        "assembly_conflict": source_conflict,
        "assembly_review_reason": "; ".join(reasons) if source_conflict else "",
        "assembly_children": _absorbed_summary(absorbed),
        "text": selected_text,
        "raw_text": selected_text,
        "preliminary_text": graphical_text,
        "recognized": True,
        "needs_review": state == "review",
        "review_reason": disposition.reason if state == "review" else "",
        "recognition_source": source_name,
        "recognition_evidence": evidence,
        "source_conflict": source_conflict,
        "numeric_conflict": source_conflict,
        "engineering_symbol": symbol,
        "engineering_parse": parsed.to_dict(),
        "engineering_disposition": disposition.to_dict(),
        "gdt": {"frames": frames},
    }
    outcome = {
        **deepcopy(common),
        "state": state,
        "reason": disposition.reason,
        "rule": disposition.rule,
    }
    return common, outcome, native_used


def _duplicate_frame(
    frame: Mapping[str, Any],
    regions: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
) -> bool:
    for item in (*regions, *reviews):
        if str(item.get("page_filter_rule") or "") != "feature_control_frame":
            continue
        try:
            if bbox_overlap_fraction(_box(frame), _box(item)) >= 0.8:
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _refresh_statistics(result: dict[str, Any]) -> None:
    regions = list(result.get("regions", ()))
    reviews = list(result.get("review_candidates", ()))
    outcomes = list(result.get("candidate_outcomes", ()))
    counts = final_candidate_counts(regions, reviews, outcomes)
    result.update(
        {
            "count": counts.eligible,
            "detected_count": counts.detected,
            "recognized_count": counts.recognized,
            "eligible_count": counts.eligible,
            "review_count": counts.review,
            "excluded_count": counts.excluded,
            "unread_count": counts.unread,
        }
    )

    represented = {_candidate_id(item) for item in outcomes if _candidate_id(item)}
    statistical_items: list[Mapping[str, Any]] = [*outcomes]
    statistical_items.extend(
        item
        for item in (*regions, *reviews)
        if not _candidate_id(item) or _candidate_id(item) not in represented
    )
    parses = [_parse_for(item) for item in statistical_items]
    dispositions = [
        _disposition_for(item, parsed)
        for item, parsed in zip(statistical_items, parses)
    ]
    result["engineering_parse_stats"] = engineering_parse_statistics(parses)
    result["engineering_disposition_stats"] = engineering_disposition_statistics(
        dispositions
    )
    structured = [
        item
        for item in statistical_items
        if isinstance(item.get("engineering_symbol"), Mapping)
    ]
    result["structured_symbol_stats"] = structured_symbol_statistics(structured)

    source_counts: Counter[str] = Counter()
    for item in statistical_items:
        if bool(item.get("recognized", True)):
            source_counts[str(item.get("recognition_source") or "ocr")] += 1
    result["recognition_source_counts"] = dict(sorted(source_counts.items()))


def augment_page_scan_result(
    image: Image.Image,
    segment_result: dict[str, Any],
    *,
    ocr_fn: OcrFn,
    page: int = 1,
    existing_value_boxes: Sequence[Mapping[str, float]] = (),
    config: GdtConfig | None = None,
) -> dict[str, Any]:
    """Add graphical FCF objects after the established whole-page scan.

    When the production flag is off, the exact input object is returned.  If
    the optional pass fails, the original scan still succeeds and carries a
    small integration error record for diagnostics.
    """

    resolved = config or get_config()
    if not page_scan_enabled(resolved):
        return segment_result

    original = deepcopy(segment_result)
    try:
        detected_frames = scan_regions(
            image,
            ocr_fn=ocr_fn,
            page=page,
            config=resolved,
        )
        result = deepcopy(segment_result)
        regions = [deepcopy(dict(item)) for item in result.get("regions", ())]
        reviews = [
            deepcopy(dict(item))
            for item in result.get("review_candidates", ())
        ]
        outcomes = [
            deepcopy(dict(item))
            for item in result.get("candidate_outcomes", ())
        ]
        result["regions"] = regions
        result["review_candidates"] = reviews
        result["candidate_outcomes"] = outcomes

        used_ids = {
            _candidate_id(item)
            for item in (*regions, *reviews, *outcomes)
            if _candidate_id(item)
        }
        published = eligible = review = absorbed_count = 0
        skipped_existing = duplicate_count = native_frame_count = 0

        for source in detected_frames:
            if overlaps_existing_value(_box(source), existing_value_boxes):
                skipped_existing += 1
                continue
            if _duplicate_frame(source, regions, reviews):
                duplicate_count += 1
                continue

            native_items = _native_evidence(source, regions, reviews, outcomes)
            absorbed = [
                deepcopy(dict(item))
                for item in (*regions, *reviews)
                if _centre_inside(item, source)
            ]
            regions[:] = [item for item in regions if not _centre_inside(item, source)]
            reviews[:] = [item for item in reviews if not _centre_inside(item, source)]

            frame_id = _unique_id(
                str(source.get("candidate_id") or "fcf"),
                used_ids,
            )
            absorbed_count += _retire_absorbed(
                outcomes,
                absorbed,
                used_ids=used_ids,
                frame_id=frame_id,
            )
            frame, outcome, native_used = _frame_object(
                source,
                candidate_id=frame_id,
                native_items=native_items,
                absorbed=absorbed,
            )
            outcomes.append(outcome)
            if outcome["state"] == "eligible":
                regions.append(frame)
                eligible += 1
            elif outcome["state"] == "review":
                reviews.append(frame)
                review += 1
            published += 1
            native_frame_count += int(native_used)

        rule_counts = Counter(
            {
                str(name): int(value)
                for name, value in dict(result.get("filter_rule_counts", {})).items()
            }
        )
        rule_counts["feature_control_frame"] += published
        result["filter_rule_counts"] = dict(sorted(rule_counts.items()))
        result["skipped_existing_count"] = int(
            result.get("skipped_existing_count", 0)
        ) + skipped_existing
        result["gdt_integration"] = {
            "schema_version": 1,
            "status": "completed",
            "enabled": True,
            "detected_frame_count": len(detected_frames),
            "published_frame_count": published,
            "eligible_frame_count": eligible,
            "review_frame_count": review,
            "absorbed_candidate_count": absorbed_count,
            "skipped_existing_count": skipped_existing,
            "duplicate_frame_count": duplicate_count,
            "native_evidence_frame_count": native_frame_count,
        }
        _refresh_statistics(result)
        return result
    except Exception as error:  # noqa: BLE001 - optional evidence cannot fail a scan
        original["gdt_integration"] = {
            "schema_version": 1,
            "status": "error",
            "enabled": True,
            "error": f"{type(error).__name__}: {error}"[:240],
        }
        return original
