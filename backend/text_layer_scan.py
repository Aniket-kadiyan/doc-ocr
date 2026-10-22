"""Adapt PDF text-layer output to the durable scan-job result contract."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def _inside_scope(box: dict[str, Any], scope: dict[str, float]) -> bool:
    """Whether the centre of ``box`` lies in the requested scan scope."""

    try:
        centre_x = float(box["x"]) + float(box["width"]) / 2.0
        centre_y = float(box["y"]) + float(box["height"]) / 2.0
    except (KeyError, TypeError, ValueError):
        return False
    return (
        float(scope["x"]) <= centre_x <= float(scope["x"]) + float(scope["width"])
        and float(scope["y"]) <= centre_y <= float(scope["y"]) + float(scope["height"])
    )


def _candidate_outcome(
    item: dict[str, Any],
    *,
    candidate_id: str,
    state: str,
    rule: str,
    reason: str,
) -> dict[str, Any]:
    """One text-layer object in the same audit shape as OCR candidates."""

    text = str(item.get("text") or "").strip()
    outcome = {
        "candidate_id": candidate_id,
        "bbox": dict(item.get("bbox") or {}),
        "state": state,
        "text": text,
        "raw_text": str(item.get("raw_text") or text),
        "confidence": float(item.get("confidence", 1.0) or 0.0),
        "recognized": bool(text),
        "reason": reason,
        "rule": rule,
    }
    for key in (
        "type",
        "category",
        "subtype",
        "label",
        "orientation",
        "rotation",
        "oriented_box",
        "recovery_attempted",
        "authoritative_reread",
    ):
        if key in item:
            outcome[key] = item[key]
    return outcome


def scan_result_from_text_layer(
    result: dict[str, Any],
    *,
    scope_kind: str,
    scope: dict[str, float],
) -> dict[str, Any]:
    """Build an atomic scan result from native PDF text and its rejects.

    Accepted values become ordinary regions. Uncertain values become Review
    candidates, while text rejected as a measurement and classified prose
    become Other candidates. Consequently, the PDF-first path has the same
    visible accounting and persistence as the raster OCR path.
    """

    if scope_kind not in {"page", "section"}:
        raise ValueError(f"Unsupported scan scope: {scope_kind}")

    def relevant(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        copied = [dict(item) for item in items]
        if scope_kind == "page":
            return copied
        return [item for item in copied if _inside_scope(item.get("bbox") or {}, scope)]

    regions: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    rules: Counter[str] = Counter()
    sequence = 0

    for region in relevant(result.get("regions", [])):
        sequence += 1
        candidate_id = f"T{sequence:04d}"
        region["candidate_id"] = candidate_id
        if region.get("needs_review"):
            state = "review"
            rule = "pdf_text_layer_review"
            reason = str(
                region.get("review_reason")
                or "The PDF text is exact, but the engineering meaning needs review"
            )
            region["review_reason"] = reason
            reviews.append(region)
        else:
            state = "eligible"
            rule = "pdf_text_layer_value"
            reason = "Accepted from the drawing PDF's native text layer"
            regions.append(region)
        rules[rule] += 1
        outcomes.append(
            _candidate_outcome(
                region,
                candidate_id=candidate_id,
                state=state,
                rule=rule,
                reason=reason,
            )
        )

    rejected = relevant(result.get("excluded", []))
    text_blocks = relevant(result.get("text_blocks", []))
    for item in [*rejected, *text_blocks]:
        sequence += 1
        candidate_id = f"T{sequence:04d}"
        category = str(item.get("category") or "text").strip().lower()
        rule = str(item.get("rule") or f"pdf_text_layer_{category.replace(' ', '_')}")
        reason = str(
            item.get("reason")
            or "PDF text was read but does not match an inspection value"
        )
        rules[rule] += 1
        outcomes.append(
            _candidate_outcome(
                item,
                candidate_id=candidate_id,
                state="excluded",
                rule=rule,
                reason=reason,
            )
        )

    recognized_count = sum(1 for item in outcomes if item["recognized"])
    return {
        "regions": regions,
        "review_candidates": reviews,
        "candidate_outcomes": outcomes,
        "coordinate_space": "page",
        "detected_count": len(outcomes),
        "recognized_count": recognized_count,
        "eligible_count": len(regions),
        "excluded_count": len(rejected) + len(text_blocks),
        "review_count": len(reviews),
        "unread_count": len(outcomes) - recognized_count,
        "filter_rule_counts": dict(rules),
    }
