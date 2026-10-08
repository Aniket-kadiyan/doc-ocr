"""Orchestration for native-PDF callouts before raster OCR.

The PDF text layer owns exact text and geometry when it is trustworthy. Raster
OCR is used only as a fallback (scan/searchable-scan), or as a supplement for a
hybrid page. Every rejected native group is retained as a candidate outcome so
the existing Review/Other lifecycle can show it rather than silently dropping
it.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import re
from typing import Any, Mapping, Sequence

from engineering_object_assembly import plan_engineering_object_assemblies
from page_scan import bbox_overlap_fraction, overlaps_existing_value
from page_value_filters import (
    PageValueCandidate,
    evaluate_scan_value,
    normalize_page_value_text,
)


def native_vector_result_is_primary(
    result: Mapping[str, Any] | None,
    *,
    evidence_profile: str | None,
) -> bool:
    """Return whether a vector PDF result should bypass page-wide OCR.

    A usable vector text layer is authoritative for the page.  Graphical
    symbol discovery may enrich individual candidates later, but must never
    turn a vector page into a full-page OCR job: doing so discards exact CAD
    text and lets lower-confidence raster reads replace it.
    """

    return bool(
        result
        and evidence_profile == "vector"
        and (
            int(result.get("eligible_count", 0))
            + int(result.get("review_count", 0))
            > 0
        )
    )


def _box(item: Mapping[str, Any]) -> dict[str, float]:
    raw = item.get("bbox") if isinstance(item.get("bbox"), Mapping) else item
    return {
        key: float(raw[key])  # type: ignore[index]
        for key in ("x", "y", "width", "height")
    }


def _centre_inside(box: Mapping[str, float], scope: Mapping[str, float]) -> bool:
    centre_x = float(box["x"]) + float(box["width"]) / 2.0
    centre_y = float(box["y"]) + float(box["height"]) / 2.0
    return (
        float(scope["x"]) <= centre_x <= float(scope["x"]) + float(scope["width"])
        and float(scope["y"]) <= centre_y <= float(scope["y"]) + float(scope["height"])
    )


def crop_text_layer_result(
    result: Mapping[str, Any],
    scope: Mapping[str, float],
) -> dict[str, Any]:
    """Select native groups inside a section and translate to section pixels."""

    origin_x, origin_y = float(scope["x"]), float(scope["y"])

    def crop_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        cropped: list[dict[str, Any]] = []
        for source in items:
            bbox = _box(source)
            if not _centre_inside(bbox, scope):
                continue
            item = deepcopy(dict(source))
            item["bbox"] = {
                "x": round(bbox["x"] - origin_x, 1),
                "y": round(bbox["y"] - origin_y, 1),
                "width": round(bbox["width"], 1),
                "height": round(bbox["height"], 1),
            }
            children = item.get("assembly_children")
            if isinstance(children, list):
                mapped_children: list[dict[str, Any]] = []
                for child in children:
                    if not isinstance(child, Mapping):
                        continue
                    mapped_child = deepcopy(dict(child))
                    child_bbox = child.get("bbox")
                    if isinstance(child_bbox, Mapping):
                        mapped_child["bbox"] = {
                            "x": round(float(child_bbox.get("x", 0.0)) - origin_x, 1),
                            "y": round(float(child_bbox.get("y", 0.0)) - origin_y, 1),
                            "width": round(float(child_bbox.get("width", 0.0)), 1),
                            "height": round(float(child_bbox.get("height", 0.0)), 1),
                        }
                    mapped_children.append(mapped_child)
                item["assembly_children"] = mapped_children
            oriented = item.get("oriented_box")
            if isinstance(oriented, Mapping):
                item["oriented_box"] = {
                    **oriented,
                    "x": round(float(oriented.get("x", 0.0)) - origin_x, 1),
                    "y": round(float(oriented.get("y", 0.0)) - origin_y, 1),
                }
            evidence = item.get("recognition_evidence")
            if isinstance(evidence, dict) and isinstance(
                evidence.get("native_bbox"), Mapping
            ):
                native_bbox = evidence["native_bbox"]
                evidence["native_bbox"] = {
                    "x": round(float(native_bbox.get("x", 0.0)) - origin_x, 1),
                    "y": round(float(native_bbox.get("y", 0.0)) - origin_y, 1),
                    "width": float(native_bbox.get("width", 0.0)),
                    "height": float(native_bbox.get("height", 0.0)),
                }
            cropped.append(item)
        return cropped

    return {
        **dict(result),
        "page_size": (
            max(1, int(round(float(scope["width"])))),
            max(1, int(round(float(scope["height"])))),
        ),
        "regions": crop_items(list(result.get("regions", ()))),
        "excluded": crop_items(list(result.get("excluded", ()))),
    }


def _native_evidence(text: str, bbox: Mapping[str, float]) -> dict[str, Any]:
    return {
        "selected_source": "native_pdf",
        "sources": ["native_pdf"],
        "native_text": text,
        "ocr_text": "",
        "agreement": 1.0,
        "conflict": False,
        "native_span_ids": [],
        "native_bbox": dict(bbox),
    }


def _common_native_region(
    source: Mapping[str, Any],
    *,
    candidate_id: str,
) -> dict[str, Any]:
    text = normalize_page_value_text(str(source.get("text") or ""))
    bbox = _box(source)
    return {
        **dict(source),
        "candidate_id": candidate_id,
        "object_id": candidate_id,
        "assembly_id": candidate_id,
        "bbox": bbox,
        "text": text,
        "raw_text": str(source.get("text") or ""),
        "confidence": float(source.get("confidence") or 0.995),
        "orientation": str(source.get("orientation") or "horizontal"),
        "rotation": float(source.get("rotation") or 0.0),
        "recognized": bool(text),
        "recognition_source": "native_pdf",
        "recognition_evidence": dict(
            source.get("recognition_evidence") or _native_evidence(text, bbox)
        ),
        "source_conflict": False,
    }


def _native_result_record(region: Mapping[str, Any]) -> dict[str, Any]:
    """Expose native provenance in the record shape used by the assembler."""

    record = deepcopy(dict(region))
    record["native_authoritative"] = True
    record["result"] = {
        "text": str(region.get("text") or ""),
        "raw_ocr": str(region.get("raw_text") or region.get("text") or ""),
        "confidence": float(region.get("confidence") or 0.995),
        "orientation": str(region.get("orientation") or "horizontal"),
        "rotation": float(region.get("rotation") or 0.0),
        "recognition_source": "native_pdf",
        "recognition_evidence": deepcopy(
            dict(region.get("recognition_evidence") or {})
        ),
        "source_conflict": False,
        "needs_review": bool(region.get("needs_review")),
    }
    return record


def _flatten_native_children(
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    for member in members:
        member_children = member.get("assembly_children")
        if isinstance(member_children, list) and member_children:
            children.extend(
                deepcopy(dict(child))
                for child in member_children
                if isinstance(child, Mapping)
            )
            continue
        children.append(
            {
                "candidate_id": str(member.get("candidate_id") or ""),
                "text": str(member.get("text") or ""),
                "raw_text": str(member.get("raw_text") or member.get("text") or ""),
                "bbox": dict(_box(member)),
                "polygon": deepcopy(list(member.get("polygon") or ())),
                "confidence": float(member.get("confidence") or 0.995),
                "orientation": str(member.get("orientation") or "horizontal"),
                "rotation": float(member.get("rotation") or 0.0),
                "recognition_source": "native_pdf",
                "recognition_evidence": deepcopy(
                    dict(member.get("recognition_evidence") or {})
                ),
                "source_conflict": False,
            }
        )
    return children


def _assemble_native_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    accepted_identity: set[int],
    page_number: int,
) -> tuple[list[dict[str, Any]], list[Any], int]:
    """Join safe native fragments before any fragment is context-filtered.

    A nominal and its adjacent tolerance must be judged as one engineering
    value. Filtering ``±1`` first loses that relationship and can mistake the
    tolerance for a drawing-frame label. The existing conservative assembly
    planner already encodes the required geometry; this adapter preserves the
    exact native text and flattens every original run into audit evidence.
    """

    common: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        item = _common_native_region(
            source,
            candidate_id=f"PDF:P{page_number}:T{index:05d}",
        )
        item["_native_text_layer_accepted"] = id(source) in accepted_identity
        common.append(item)

    planned = plan_engineering_object_assemblies(
        [_native_result_record(item) for item in common]
    )
    plans = [
        plan
        for plan in planned
        if _native_plan_is_tight(plan.member_indexes, common)
    ]
    plans_by_first = {min(plan.member_indexes): plan for plan in plans}
    absorbed = {
        member_index
        for plan in plans
        for member_index in plan.member_indexes
    }
    assembled: list[dict[str, Any]] = []
    for index, item in enumerate(common):
        plan = plans_by_first.get(index)
        if plan is None:
            if index not in absorbed:
                assembled.append(item)
            continue

        members = [common[member_index] for member_index in plan.member_indexes]
        anchor = deepcopy(common[plan.anchor_index])
        anchor.update(
            {
                "candidate_id": plan.object_id,
                "object_id": plan.object_id,
                "assembly_id": plan.object_id,
                "bbox": dict(plan.bbox),
                "polygon": [list(point) for point in plan.polygon],
                "text": plan.text,
                "raw_text": " ".join(
                    str(member.get("raw_text") or member.get("text") or "")
                    for member in members
                ).strip(),
                "confidence": plan.confidence,
                "orientation": plan.orientation,
                "rotation": plan.rotation,
                "recognized": bool(plan.text),
                "needs_review": plan.needs_review,
                "recognition_source": "native_pdf",
                "recognition_evidence": {
                    **deepcopy(plan.recognition_evidence),
                    "selected_source": "native_pdf",
                    "sources": ["native_pdf"],
                    "native_text": plan.text,
                    "ocr_text": "",
                    "agreement": 1.0,
                    "conflict": False,
                    "native_bbox": dict(plan.bbox),
                },
                "source_conflict": False,
                "assembly_rule": f"native_pre_filter+{plan.rule}",
                "assembly_conflict": plan.conflict,
                "assembly_review_reason": plan.review_reason,
                "assembly_children": _flatten_native_children(members),
                # An accepted base remains eligible for evaluation even when
                # its modifier came from the text layer's non-callout list.
                "_native_text_layer_accepted": any(
                    bool(member.get("_native_text_layer_accepted"))
                    for member in members
                ),
            }
        )
        assembled.append(anchor)

    atomic_count = sum(
        max(1, len(item.get("assembly_children") or ())) for item in common
    )
    return assembled, plans, atomic_count


def _native_plan_is_tight(
    member_indexes: Sequence[int],
    records: Sequence[Mapping[str, Any]],
) -> bool:
    """Reject geometrically legal but visibly separated native-text joins."""

    pending = set(member_indexes)
    if len(pending) < 2:
        return False
    connected = {pending.pop()}
    while pending:
        newly_connected: set[int] = set()
        for right_index in pending:
            right = _box(records[right_index])
            for left_index in connected:
                left = _box(records[left_index])
                horizontal_gap = max(
                    0.0,
                    right["x"] - (left["x"] + left["width"]),
                    left["x"] - (right["x"] + right["width"]),
                )
                vertical_gap = max(
                    0.0,
                    right["y"] - (left["y"] + left["height"]),
                    left["y"] - (right["y"] + right["height"]),
                )
                distance = math.hypot(horizontal_gap, vertical_gap)
                scale = max(
                    min(left["width"], left["height"]),
                    min(right["width"], right["height"]),
                    1.0,
                )
                if distance <= 0.9 * scale:
                    newly_connected.add(right_index)
                    break
        if not newly_connected:
            return False
        connected.update(newly_connected)
        pending.difference_update(newly_connected)
    return True


def native_text_segment_result(
    text_layer: Mapping[str, Any],
    *,
    page_number: int,
    scope_kind: str,
    table_masks: Sequence[Mapping[str, float]] = (),
    existing_value_boxes: Sequence[Mapping[str, float]] = (),
    source_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert exact text groups into the normal scan result contract."""

    page_size = tuple(text_layer.get("page_size") or (0, 0))
    accepted_sources = list(text_layer.get("regions", ()))
    excluded_sources = list(text_layer.get("excluded", ()))
    all_sources = [*accepted_sources, *excluded_sources]
    accepted_identity = {id(item) for item in accepted_sources}
    assembled_sources, assembly_plans, atomic_count = _assemble_native_sources(
        all_sources,
        accepted_identity=accepted_identity,
        page_number=page_number,
    )
    filter_candidates = [
        PageValueCandidate(
            text=normalize_page_value_text(str(item.get("text") or "")),
            bbox=_box(item),
        )
        for item in assembled_sources
    ]

    regions: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()
    skipped_existing = 0

    for common, filter_candidate in zip(assembled_sources, filter_candidates):
        if overlaps_existing_value(common["bbox"], existing_value_boxes):
            skipped_existing += 1
            continue

        decision = evaluate_scan_value(
            filter_candidate,
            scope_kind="section" if scope_kind == "section" else "page",
            table_masks=table_masks,
            page_size=(int(page_size[0]), int(page_size[1])),
            page_candidates=filter_candidates,
        )
        from_text_layer = bool(common.pop("_native_text_layer_accepted", False))
        if not from_text_layer:
            state = "excluded"
            rule = "native_text_non_callout"
            reason = "Exact PDF text did not form an engineering callout"
        elif (str(common.get("category") or "") == "GD&T"
              and decision.rule_name not in {"table_region", "sheet_frame_label",
                  "detail_view_section", "scale_information", "date",
                  "revision_history", "note_information", "document_metadata"}):
            state = "review"
            rule = "gdt_requires_visual_frame_evidence"
            reason = "GD&T text requires a matching graphical feature-control frame"
        elif not decision.accepted:
            state = "excluded"
            rule = decision.rule_name
            reason = decision.reason
        elif bool(common.get("needs_review")):
            state = "review"
            rule = "native_text_needs_review"
            reason = "Exact PDF text is numeric but missing a confirming engineering mark"
        else:
            state = "eligible"
            rule = "native_text_engineering_value"
            reason = "Accepted from the original PDF text layer"
        rule_counts[rule] += 1

        common.update(
            {
                "needs_review": state == "review",
                "page_filter_rule": rule,
                "page_filter_reason": reason,
                "review_reason": reason if state == "review" else "",
            }
        )
        if state == "eligible":
            regions.append(common)
        elif state == "review":
            reviews.append(common)

        outcomes.append(
            {
                **common,
                "state": state,
                "reason": reason,
                "rule": rule,
            }
        )

    profile = dict(source_profile or {})
    profile.update(
        {
            "text_layer_usable": bool(text_layer.get("usable")),
            "text_layer_strategy": "native_text_first",
            "text_layer_fonts": list(text_layer.get("fonts", ())),
            "text_layer_fallback_reason": text_layer.get("fallback_reason"),
        }
    )
    detected = len(outcomes)
    return {
        "count": len(regions),
        "detected_count": detected,
        "recognized_count": sum(
            1 for outcome in outcomes if bool(outcome.get("recognized"))
        ),
        "eligible_count": len(regions),
        "excluded_count": sum(1 for outcome in outcomes if outcome["state"] == "excluded"),
        "review_count": len(reviews),
        "unread_count": sum(
            1 for outcome in outcomes if not bool(outcome.get("recognized"))
        ),
        "skipped_existing_count": skipped_existing,
        "filter_rule_counts": dict(sorted(rule_counts.items())),
        "regions": regions,
        "review_candidates": reviews,
        "candidate_outcomes": outcomes,
        "source_profile": profile,
        "recognition_source_counts": {"native_pdf": detected},
        "assembly_stats": {
            "atomic_detection_count": atomic_count,
            "object_count": detected,
            "assembled_object_count": sum(
                1
                for outcome in outcomes
                if len(outcome.get("assembly_children") or ()) > 1
            ),
            "absorbed_fragment_count": max(0, atomic_count - detected),
            "conflict_count": sum(1 for plan in assembly_plans if plan.conflict),
            "rule_counts": dict(
                sorted(Counter(plan.rule for plan in assembly_plans).items())
            ),
        },
    }


def _digits(text: object) -> str:
    return "".join(character for character in str(text or "") if character.isdigit())


def _compact(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def _matching_outcome(
    candidate: Mapping[str, Any],
    native_outcomes: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any] | None:
    bbox = _box(candidate)
    best: tuple[float, dict[str, Any]] | None = None
    for native in native_outcomes:
        overlap = bbox_overlap_fraction(bbox, _box(native))
        if overlap >= threshold and (best is None or overlap > best[0]):
            best = (overlap, native)
    return best[1] if best else None


def _merge_evidence(native: dict[str, Any], ocr: Mapping[str, Any]) -> None:
    if not bool(ocr.get("recognized", True)):
        return
    native_text = str(native.get("text") or "")
    ocr_text = str(ocr.get("text") or "")
    ocr_supports_value = ocr.get("state") in {"eligible", "review"}
    digit_conflict = (
        ocr_supports_value
        and bool(_digits(native_text) and _digits(ocr_text))
        and _digits(native_text) != _digits(ocr_text)
    )
    exact = _compact(native_text) == _compact(ocr_text)
    native["recognition_source"] = "native_pdf+ocr"
    native["source_conflict"] = digit_conflict
    native["recognition_evidence"] = {
        "selected_source": "native_pdf",
        "sources": ["native_pdf", "ocr"],
        "native_text": native_text,
        "ocr_text": ocr_text,
        "agreement": 1.0 if exact else (0.5 if not digit_conflict else 0.0),
        "conflict": digit_conflict,
        "native_span_ids": list(
            (native.get("recognition_evidence") or {}).get("native_span_ids", ())
        ),
        "native_bbox": dict(_box(native)),
    }
    structured = ocr.get("engineering_symbol")
    if isinstance(structured, Mapping):
        for field in ("bbox", "oriented_box", "category", "subtype", "label", "type",
                      "assembly_id", "assembly_rule", "assembly_conflict",
                      "assembly_review_reason", "assembly_children",
                      "engineering_parse", "engineering_disposition"):
            if field in ocr:
                native[field] = deepcopy(ocr[field])
        native["engineering_symbol"] = deepcopy(dict(structured))
        # Geometry can supplement an exact native value, but raster OCR must
        # never replace that value.  The completed graphical symbol remains
        # available in engineering_symbol for display/export.
        native["text"] = native_text
        if ocr.get("state") in {"eligible", "review"}:
            native["state"] = ocr.get("state")
            native["needs_review"] = ocr.get("state") == "review"
            native["reason"] = str(ocr.get("reason") or "")
            native["rule"] = str(ocr.get("rule") or "")
            native["page_filter_rule"] = str(ocr.get("page_filter_rule") or ocr.get("rule") or "")
            native["page_filter_reason"] = str(ocr.get("page_filter_reason") or ocr.get("reason") or "")
            native["review_reason"] = str(ocr.get("reason") or "") if ocr.get("state") == "review" else ""
    if digit_conflict and native.get("state") == "eligible":
        native.update(
            {
                "state": "review",
                "needs_review": True,
                "review_reason": "Native PDF text and raster OCR disagree on the numeric value",
                "reason": "Native PDF text and raster OCR disagree on the numeric value",
                "rule": "native_ocr_numeric_conflict",
                "page_filter_rule": "native_ocr_numeric_conflict",
                "page_filter_reason": "Native PDF text and raster OCR disagree on the numeric value",
            }
        )


def merge_native_and_ocr_results(
    native: Mapping[str, Any],
    ocr: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse a hybrid page, keeping native text primary and OCR additive."""

    native_outcomes = [deepcopy(item) for item in native.get("candidate_outcomes", ())]
    ocr_outcomes = [deepcopy(item) for item in ocr.get("candidate_outcomes", ())]
    retained_ocr: list[dict[str, Any]] = []

    for outcome in ocr_outcomes:
        match = _matching_outcome(outcome, native_outcomes)
        if match is None:
            retained_ocr.append(outcome)
            continue
        if match.get("state") == "excluded" and outcome.get("state") in {
            "eligible",
            "review",
        }:
            native_outcomes.remove(match)
            retained_ocr.append(outcome)
            continue
        _merge_evidence(match, outcome)

    final_outcomes = [*native_outcomes, *retained_ocr]
    region_sources = [
        *[deepcopy(item) for item in native.get("regions", ())],
        *[deepcopy(item) for item in ocr.get("regions", ())],
        *[deepcopy(item) for item in native.get("review_candidates", ())],
        *[deepcopy(item) for item in ocr.get("review_candidates", ())],
    ]

    def region_for(outcome: Mapping[str, Any]) -> dict[str, Any]:
        candidate_id = str(outcome.get("candidate_id") or "")
        source = next(
            (
                item
                for item in region_sources
                if candidate_id and str(item.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        return {**(source or {}), **deepcopy(dict(outcome))}

    regions = [
        region_for(outcome)
        for outcome in final_outcomes
        if outcome.get("state") == "eligible"
    ]
    reviews = [
        region_for(outcome)
        for outcome in final_outcomes
        if outcome.get("state") == "review"
    ]

    represented_ids = {
        str(outcome.get("candidate_id") or "") for outcome in final_outcomes
    }
    for orphan in ocr.get("regions", ()):
        candidate_id = str(orphan.get("candidate_id") or "")
        if candidate_id and candidate_id in represented_ids:
            continue
        if _matching_outcome(orphan, native_outcomes) is None:
            regions.append(deepcopy(orphan))

    rule_counts: Counter[str] = Counter()
    for outcome in final_outcomes:
        rule_counts[str(outcome.get("rule") or "unclassified")] += 1
    source_counts: Counter[str] = Counter()
    for outcome in final_outcomes:
        if bool(outcome.get("recognized", True)):
            source_counts[
                str(outcome.get("recognition_source") or "ocr")
            ] += 1

    detected = len(final_outcomes)
    atomic_count = sum(
        max(1, len(outcome.get("assembly_children") or ()))
        for outcome in final_outcomes
    )
    return {
        **dict(ocr),
        "count": len(regions),
        "detected_count": detected,
        "recognized_count": sum(
            1 for outcome in final_outcomes if bool(outcome.get("recognized", True))
        ),
        "eligible_count": len(regions),
        "excluded_count": sum(
            1 for outcome in final_outcomes if outcome.get("state") == "excluded"
        ),
        "review_count": len(reviews),
        "unread_count": sum(
            1 for outcome in final_outcomes if not bool(outcome.get("recognized", True))
        ),
        "skipped_existing_count": int(native.get("skipped_existing_count", 0))
        + int(ocr.get("skipped_existing_count", 0)),
        "filter_rule_counts": dict(sorted(rule_counts.items())),
        "regions": regions,
        "review_candidates": reviews,
        "candidate_outcomes": final_outcomes,
        "source_profile": {
            **dict(ocr.get("source_profile") or {}),
            "text_layer_usable": True,
            "text_layer_strategy": "native_text_first_hybrid_ocr",
        },
        "recognition_source_counts": dict(sorted(source_counts.items())),
        "assembly_stats": {
            "atomic_detection_count": atomic_count,
            "object_count": detected,
            "assembled_object_count": sum(
                1
                for outcome in final_outcomes
                if len(outcome.get("assembly_children") or ()) > 1
            ),
            "absorbed_fragment_count": max(0, atomic_count - detected),
            "conflict_count": sum(
                1 for outcome in final_outcomes if outcome.get("assembly_conflict")
            ),
            "rule_counts": {},
        },
    }
