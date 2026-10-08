"""Geometry-first assembly of OCR fragments into engineering objects.

The detector is intentionally allowed to return small, atomic text boxes.  A
balloon, however, belongs to the logical callout those boxes form (for example
``4X`` + ``Ø10`` + ``THRU``), not to every box independently.  This module is
the model-free layer between recognition and final page filtering.

Assembly is deliberately conservative:

* text with incompatible writing directions never joins;
* two ordinary complete values stay separate unless an explicit ``X``/``×``
  operator makes them a chamfer/size expression;
* tables and structured GD&T frames are outside this pass; and
* every absorbed detector box is serialized as child evidence.

No item is silently deleted.  A caller replaces a group with one object and
keeps ``AssemblyPlan.children`` on that object for Review/audit persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from page_value_filters import (
    is_complete_engineering_value,
    normalize_page_value_text,
)


BBox = Mapping[str, float]

_NUMBER = r"(?:\d+(?:[.,]\d+)?|[.,]\d+)"
_MULTIPLIER_RE = re.compile(r"^\d+\s*[X×]$", re.IGNORECASE)
_TOLERANCE_RE = re.compile(
    rf"^(?:±|\+|-|−)\s*{_NUMBER}(?:\s*/\s*(?:\+|-|−)?\s*{_NUMBER})?\s*°?$"
)
_ZERO_RE = re.compile(r"^[+-]?0+(?:[.,]0+)?\s*°?$")
_CONVERSION_RE = re.compile(rf"^[\[(]\s*{_NUMBER}\s*[\])]$")
_BASE_LIKE_RE = re.compile(
    rf"^(?:(?:SR|SØ|R|Ø|⌀|M)\s*)?[+-]?{_NUMBER}(?:[A-Z]\d+)?[.]?$",
    re.IGNORECASE,
)
_PREFIXES = {"Ø", "⌀", "ø", "φ", "Φ", "R", "SR", "SØ", "M"}
_QUALIFIERS = {
    "THRU",
    "THROUGH",
    "TYP",
    "TYP.",
    "MIN",
    "MAX",
    "REF",
    "REF.",
    "BASIC",
    "DEEP",
    "EQ SP",
}
_UNITS = {"MM", "CM", "IN", "IN.", "INCH", "INCHES"}
_STRUCTURED_GDT_MARKS = set("|⌖⏥⌭⌯⌰⊥∥◎Ⓜ")


@dataclass(frozen=True)
class _Token:
    index: int
    candidate_id: str
    text: str
    role: str
    bbox: dict[str, float]
    polygon: tuple[tuple[float, float], ...]
    orientation: str
    rotation: float
    angle: float
    u0: float
    u1: float
    v0: float
    v1: float
    confidence: float
    record: Mapping[str, Any]

    @property
    def along_size(self) -> float:
        return max(self.u1 - self.u0, 1.0)

    @property
    def line_size(self) -> float:
        return max(self.v1 - self.v0, 1.0)

    @property
    def u_center(self) -> float:
        return (self.u0 + self.u1) / 2.0

    @property
    def v_center(self) -> float:
        return (self.v0 + self.v1) / 2.0


@dataclass(frozen=True)
class AssemblyPlan:
    """One multi-fragment engineering object to build in the caller."""

    member_indexes: tuple[int, ...]
    anchor_index: int
    object_id: str
    text: str
    bbox: dict[str, float]
    polygon: tuple[tuple[float, float], ...]
    orientation: str
    rotation: float
    confidence: float
    rule: str
    conflict: bool
    needs_review: bool
    review_reason: str
    children: tuple[dict[str, Any], ...]
    recognition_source: str
    recognition_evidence: dict[str, Any]


def _box(value: Mapping[str, Any]) -> dict[str, float]:
    source = value.get("bbox") if isinstance(value.get("bbox"), Mapping) else value
    return {
        key: float(source[key])  # type: ignore[index]
        for key in ("x", "y", "width", "height")
    }


def _rect_polygon(box: BBox) -> tuple[tuple[float, float], ...]:
    x, y = float(box["x"]), float(box["y"])
    width, height = float(box["width"]), float(box["height"])
    return (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )


def _record_polygon(record: Mapping[str, Any], bbox: BBox) -> tuple[tuple[float, float], ...]:
    raw = record.get("polygon")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        candidate = record.get("candidate")
        raw = getattr(candidate, "polygon", ()) if candidate is not None else ()
    points: list[tuple[float, float]] = []
    if isinstance(raw, Sequence):
        for point in raw:
            if not isinstance(point, Sequence) or len(point) < 2:
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    return tuple(points) if len(points) >= 3 else _rect_polygon(bbox)


def _normalized_angle(record: Mapping[str, Any]) -> tuple[float, str, float]:
    result = record.get("result")
    result = result if isinstance(result, Mapping) else {}
    orientation = str(
        result.get("orientation") or record.get("orientation") or "horizontal"
    )
    try:
        rotation = float(result.get("rotation") or record.get("rotation") or 0.0)
    except (TypeError, ValueError):
        rotation = 0.0
    if orientation == "vertical" and abs(rotation) < 45.0:
        rotation = 90.0
    angle = ((rotation + 180.0) % 360.0) - 180.0
    return angle, orientation, rotation


def _project(
    polygon: Sequence[tuple[float, float]], angle: float
) -> tuple[float, float, float, float]:
    theta = math.radians(angle)
    cosine, sine = math.cos(theta), math.sin(theta)
    along = [x * cosine + y * sine for x, y in polygon]
    cross = [-x * sine + y * cosine for x, y in polygon]
    return min(along), max(along), min(cross), max(cross)


def _structured_gdt(text: str) -> bool:
    return any(mark in text for mark in _STRUCTURED_GDT_MARKS)


def classify_assembly_role(text: object) -> str:
    """Classify a recognized fragment without interpreting its measurement."""

    value = normalize_page_value_text(str(text or "")).strip()
    compact = " ".join(value.upper().split())
    if not value:
        return "empty"
    if _structured_gdt(value):
        return "structured_gdt"
    if compact in {"X", "×"}:
        return "operator"
    if _MULTIPLIER_RE.fullmatch(compact):
        return "multiplier"
    if compact in _PREFIXES:
        return "prefix"
    if compact in _QUALIFIERS:
        return "qualifier"
    if compact in _UNITS or value == '"':
        return "unit"
    if _TOLERANCE_RE.fullmatch(value):
        return "zero" if _ZERO_RE.fullmatch(value) else "tolerance"
    if _ZERO_RE.fullmatch(value):
        return "zero"
    if _CONVERSION_RE.fullmatch(value):
        return "conversion"
    if is_complete_engineering_value(value):
        return "base"
    if _BASE_LIKE_RE.fullmatch(value):
        return "base_like"
    return "other"


def _token(index: int, record: Mapping[str, Any]) -> _Token | None:
    if record.get("table_excluded") or record.get("engineering_symbol"):
        return None
    text = normalize_page_value_text(str(record.get("text") or "")).strip()
    role = classify_assembly_role(text)
    if role in {"empty", "other", "structured_gdt"}:
        return None
    try:
        bbox = _box(record)
    except (KeyError, TypeError, ValueError):
        return None
    if bbox["width"] <= 0 or bbox["height"] <= 0:
        return None
    polygon = _record_polygon(record, bbox)
    angle, orientation, rotation = _normalized_angle(record)
    u0, u1, v0, v1 = _project(polygon, angle)
    result = record.get("result")
    result = result if isinstance(result, Mapping) else {}
    return _Token(
        index=index,
        candidate_id=str(record.get("candidate_id") or f"C{index + 1:04d}"),
        text=text,
        role=role,
        bbox=bbox,
        polygon=polygon,
        orientation=orientation,
        rotation=rotation,
        angle=angle,
        u0=u0,
        u1=u1,
        v0=v0,
        v1=v1,
        confidence=float(result.get("confidence") or record.get("confidence") or 0.0),
        record=record,
    )


def _same_direction(left: _Token, right: _Token) -> bool:
    delta = abs(((left.angle - right.angle + 90.0) % 180.0) - 90.0)
    return delta <= 12.0


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _same_line(left: _Token, right: _Token) -> bool:
    overlap = _overlap(left.v0, left.v1, right.v0, right.v1)
    return overlap / max(min(left.line_size, right.line_size), 1.0) >= 0.42


def _forward_gap(left: _Token, right: _Token) -> float:
    return right.u0 - left.u1


def _prefix_score(modifier: _Token, base: _Token) -> float | None:
    if not _same_direction(modifier, base) or not _same_line(modifier, base):
        return None
    gap = _forward_gap(modifier, base)
    line = max(modifier.line_size, base.line_size)
    if not -0.25 * line <= gap <= 1.6 * line:
        return None
    return abs(gap) / line + abs(modifier.v_center - base.v_center) / line


def _suffix_score(base: _Token, modifier: _Token) -> float | None:
    if not _same_direction(base, modifier) or not _same_line(base, modifier):
        return None
    gap = _forward_gap(base, modifier)
    line = max(base.line_size, modifier.line_size)
    limit = 3.0 if modifier.role == "qualifier" else 1.8
    if not -0.2 * line <= gap <= limit * line:
        return None
    return abs(gap) / line + abs(modifier.v_center - base.v_center) / line


def _tolerance_score(base: _Token, tolerance: _Token) -> float | None:
    if not _same_direction(base, tolerance):
        return None
    line = max(base.line_size, tolerance.line_size)
    if _same_line(base, tolerance):
        gap = _forward_gap(base, tolerance)
        if -0.2 * line <= gap <= 1.8 * line:
            return abs(gap) / line

    along_overlap = _overlap(base.u0, base.u1, tolerance.u0, tolerance.u1)
    along_ratio = along_overlap / max(min(base.along_size, tolerance.along_size), 1.0)
    vertical_gap = max(
        0.0,
        max(base.v0, tolerance.v0) - min(base.v1, tolerance.v1),
    )
    if along_ratio >= 0.35 and vertical_gap <= 1.0 * line:
        return 1.0 + vertical_gap / line + abs(base.u_center - tolerance.u_center) / max(
            base.along_size, tolerance.along_size, 1.0
        )

    # A smaller two-row deviation column immediately to the right of a value.
    side_gap = tolerance.u0 - base.u1
    if (
        -0.2 * line <= side_gap <= 1.5 * line
        and abs(base.v_center - tolerance.v_center) <= 1.6 * line
    ):
        return 1.5 + abs(side_gap) / line + abs(base.v_center - tolerance.v_center) / line
    return None


def _operator_side_score(operator: _Token, value: _Token, *, left: bool) -> float | None:
    if not _same_direction(operator, value) or not _same_line(operator, value):
        return None
    line = max(operator.line_size, value.line_size)
    gap = (
        operator.u0 - value.u1
        if left
        else value.u0 - operator.u1
    )
    if not -0.2 * line <= gap <= 1.4 * line:
        return None
    return abs(gap) / line + abs(operator.v_center - value.v_center) / line


def _union_box(tokens: Iterable[_Token]) -> dict[str, float]:
    items = list(tokens)
    x0 = min(token.bbox["x"] for token in items)
    y0 = min(token.bbox["y"] for token in items)
    x1 = max(token.bbox["x"] + token.bbox["width"] for token in items)
    y1 = max(token.bbox["y"] + token.bbox["height"] for token in items)
    return {
        "x": round(x0, 1),
        "y": round(y0, 1),
        "width": round(x1 - x0, 1),
        "height": round(y1 - y0, 1),
    }


def _cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (
        left[1] - origin[1]
    ) * (right[0] - origin[0])


def _convex_hull(tokens: Iterable[_Token]) -> tuple[tuple[float, float], ...]:
    points = sorted({point for token in tokens for point in token.polygon})
    if len(points) < 3:
        return _rect_polygon(_union_box(tokens))
    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    # The authoritative crop rectifier consumes four corners only.  A hull
    # with more corners is intentionally reduced to its safe axis rectangle.
    return tuple(hull) if len(hull) == 4 else _rect_polygon(_union_box(tokens))


def _child(token: _Token) -> dict[str, Any]:
    record = token.record
    result = record.get("result")
    result = result if isinstance(result, Mapping) else {}
    source = str(
        result.get("recognition_source")
        or record.get("recognition_source")
        or ("native_pdf" if record.get("native_authoritative") else "ocr")
    )
    return {
        "candidate_id": token.candidate_id,
        "text": token.text,
        "raw_text": str(result.get("raw_ocr") or result.get("text") or token.text),
        "bbox": dict(token.bbox),
        "polygon": [list(point) for point in token.polygon],
        "confidence": token.confidence,
        "orientation": token.orientation,
        "rotation": token.rotation,
        "role": token.role,
        "recognition_source": source,
        "recognition_evidence": dict(result.get("recognition_evidence") or {}),
        "source_conflict": bool(result.get("source_conflict")),
    }


def atomic_object_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the same evidence shape for a one-fragment object."""

    token = _token(0, record)
    if token is None:
        try:
            bbox = _box(record)
        except (KeyError, TypeError, ValueError):
            bbox = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
        result = record.get("result")
        result = result if isinstance(result, Mapping) else {}
        return {
            "candidate_id": str(record.get("candidate_id") or ""),
            "text": str(record.get("text") or ""),
            "raw_text": str(result.get("raw_ocr") or result.get("text") or record.get("text") or ""),
            "bbox": bbox,
            "polygon": [list(point) for point in _record_polygon(record, bbox)],
            "confidence": float(result.get("confidence") or 0.0),
            "orientation": str(result.get("orientation") or "horizontal"),
            "rotation": float(result.get("rotation") or 0.0),
            "role": classify_assembly_role(record.get("text")),
            "recognition_source": str(result.get("recognition_source") or "ocr"),
            "recognition_evidence": dict(result.get("recognition_evidence") or {}),
            "source_conflict": bool(result.get("source_conflict")),
        }
    return _child(token)


def _compose_text(tokens: Sequence[_Token]) -> str:
    operators = [token for token in tokens if token.role == "operator"]
    bases = [token for token in tokens if token.role in {"base", "base_like"}]
    if operators and len(bases) >= 2:
        ordered = sorted(tokens, key=lambda token: (token.u_center, token.v_center))
        parts: list[str] = []
        for token in ordered:
            if token.role == "prefix" and parts:
                parts[-1] += token.text
            else:
                parts.append(token.text)
        return normalize_page_value_text(" ".join(parts))

    base = min(bases, key=lambda token: token.index)
    prefixes = sorted(
        (token for token in tokens if token.role in {"multiplier", "prefix"}),
        key=lambda token: token.u_center,
    )
    base_text = base.text
    leading: list[str] = []
    for token in prefixes:
        if token.role == "prefix" and token.text.upper() in _PREFIXES:
            base_text = token.text + base_text
        else:
            leading.append(token.text)

    tolerances = [token for token in tokens if token.role in {"tolerance", "zero"}]
    tolerances.sort(key=lambda token: (token.v_center, token.u_center))
    tolerance_text = ""
    if tolerances:
        if len(tolerances) == 1:
            tolerance_text = tolerances[0].text
        else:
            tolerance_text = "/".join(token.text for token in tolerances)

    conversions = sorted(
        (token.text for token in tokens if token.role == "conversion")
    )
    suffixes = sorted(
        (
            token
            for token in tokens
            if token.role in {"unit", "qualifier"}
        ),
        key=lambda token: token.u_center,
    )
    pieces = [*leading, base_text]
    if tolerance_text:
        pieces.append(tolerance_text)
    pieces.extend(conversions)
    pieces.extend(token.text for token in suffixes)
    return normalize_page_value_text(" ".join(piece for piece in pieces if piece))


def _source_evidence(tokens: Sequence[_Token], text: str, bbox: BBox) -> tuple[str, dict[str, Any], bool]:
    sources: list[str] = []
    native_texts: list[str] = []
    ocr_texts: list[str] = []
    native_span_ids: list[str] = []
    conflicts = False
    agreements: list[float] = []
    child_evidence: list[dict[str, Any]] = []
    for token in tokens:
        result = token.record.get("result")
        result = result if isinstance(result, Mapping) else {}
        source = str(
            result.get("recognition_source")
            or ("native_pdf" if token.record.get("native_authoritative") else "ocr")
        )
        for item in source.split("+"):
            if item in {"native_pdf", "ocr"} and item not in sources:
                sources.append(item)
        evidence = dict(result.get("recognition_evidence") or {})
        child_evidence.append(evidence)
        native_value = str(evidence.get("native_text") or "")
        ocr_value = str(evidence.get("ocr_text") or "")
        if native_value:
            native_texts.append(native_value)
        elif "native_pdf" in source:
            native_texts.append(token.text)
        if ocr_value:
            ocr_texts.append(ocr_value)
        elif "ocr" in source:
            ocr_texts.append(token.text)
        for span_id in evidence.get("native_span_ids") or ():
            if str(span_id) not in native_span_ids:
                native_span_ids.append(str(span_id))
        conflicts = conflicts or bool(
            result.get("source_conflict") or evidence.get("conflict")
        )
        try:
            agreements.append(float(evidence.get("agreement")))
        except (TypeError, ValueError):
            pass
    if not sources:
        sources = ["ocr"]
    selected = "native_pdf+ocr" if set(sources) == {"native_pdf", "ocr"} else sources[0]
    evidence = {
        "selected_source": selected,
        "sources": sources,
        "native_text": " ".join(native_texts),
        "ocr_text": " ".join(ocr_texts),
        "agreement": min(agreements) if agreements else (1.0 if not conflicts else 0.0),
        "conflict": conflicts,
        "native_span_ids": native_span_ids,
        "native_bbox": dict(bbox) if native_texts else None,
        "assembled_text": text,
        "child_evidence": child_evidence,
    }
    return selected, evidence, conflicts


def plan_engineering_object_assemblies(
    records: Sequence[Mapping[str, Any]],
) -> list[AssemblyPlan]:
    """Plan only high-confidence fragment joins for recognized page records."""

    tokens = [
        token
        for index, record in enumerate(records)
        if (token := _token(index, record)) is not None
    ]
    by_index = {token.index: token for token in tokens}
    bases = [token for token in tokens if token.role in {"base", "base_like"}]
    groups: dict[int, set[int]] = {base.index: {base.index} for base in bases}
    rules: dict[int, set[str]] = {base.index: set() for base in bases}
    consumed: set[int] = set()

    # An explicit multiplication symbol is the one safe reason to combine two
    # complete values.  This covers chamfers and split thread expressions.
    for operator in (token for token in tokens if token.role == "operator"):
        left_options = [
            (score, base)
            for base in bases
            if base.index not in consumed
            and (score := _operator_side_score(operator, base, left=True)) is not None
        ]
        right_options = [
            (score, base)
            for base in bases
            if base.index not in consumed
            and (score := _operator_side_score(operator, base, left=False)) is not None
        ]
        if not left_options or not right_options:
            continue
        left = min(left_options, key=lambda item: item[0])[1]
        right = min(right_options, key=lambda item: item[0])[1]
        if left.index == right.index:
            continue
        anchor = left.index
        groups[anchor].update((operator.index, right.index))
        groups.pop(right.index, None)
        rules[anchor].add("explicit_multiplication")
        rules.pop(right.index, None)
        consumed.update((operator.index, right.index))

    root_for_base = {
        member: root
        for root, members in groups.items()
        for member in members
        if member in by_index and by_index[member].role in {"base", "base_like"}
    }

    def attach(modifier: _Token, score_fn: Any, rule: str) -> bool:
        choices: list[tuple[float, int]] = []
        for base in bases:
            root = root_for_base.get(base.index, base.index)
            if root not in groups:
                continue
            score = score_fn(base)
            if score is not None:
                choices.append((float(score), root))
        if not choices:
            return False
        _, root = min(choices, key=lambda item: (item[0], item[1]))
        groups[root].add(modifier.index)
        rules[root].add(rule)
        consumed.add(modifier.index)
        return True

    for modifier in tokens:
        if modifier.index in consumed or modifier.role not in {
            "multiplier",
            "prefix",
            "qualifier",
            "unit",
            "conversion",
            "tolerance",
        }:
            continue
        if modifier.role in {"multiplier", "prefix"}:
            attach(
                modifier,
                lambda base, item=modifier: _prefix_score(item, base),
                "inline_prefix",
            )
        elif modifier.role in {"qualifier", "unit", "conversion"}:
            attach(
                modifier,
                lambda base, item=modifier: _suffix_score(base, item),
                "inline_suffix",
            )
        else:
            attach(
                modifier,
                lambda base, item=modifier: _tolerance_score(base, item),
                "stacked_or_inline_tolerance",
            )

    # An unsigned zero is a lower deviation only when a signed deviation has
    # already established the tolerance stack.  By itself it stays atomic.
    for zero in (token for token in tokens if token.role == "zero" and token.index not in consumed):
        choices: list[tuple[float, int]] = []
        for root, members in groups.items():
            signed = [
                by_index[index]
                for index in members
                if by_index[index].role == "tolerance"
            ]
            if not signed:
                continue
            base = by_index[root]
            score = _tolerance_score(base, zero)
            if score is None:
                score = min(
                    (
                        _tolerance_score(item, zero)
                        for item in signed
                        if _tolerance_score(item, zero) is not None
                    ),
                    default=None,
                )
            if score is not None:
                choices.append((float(score), root))
        if choices:
            _, root = min(choices, key=lambda item: (item[0], item[1]))
            groups[root].add(zero.index)
            rules[root].add("stacked_zero_deviation")
            consumed.add(zero.index)

    plans: list[AssemblyPlan] = []
    for root, member_set in sorted(groups.items()):
        if len(member_set) < 2:
            continue
        members = [by_index[index] for index in sorted(member_set)]
        text = _compose_text(members)
        bbox = _union_box(members)
        polygon = _convex_hull(members)
        source, recognition_evidence, conflict = _source_evidence(members, text, bbox)
        complete = is_complete_engineering_value(text)
        child_review = any(
            bool(
                (token.record.get("result") or {}).get("needs_review")
                if isinstance(token.record.get("result"), Mapping)
                else False
            )
            for token in members
        )
        needs_review = conflict or not complete or child_review
        if conflict:
            review_reason = "Assembled fragments contain conflicting recognition evidence"
        elif not complete:
            review_reason = "Assembled fragments need confirmation as one engineering value"
        elif child_review:
            review_reason = "At least one assembled fragment remains uncertain"
        else:
            review_reason = ""
        rule = "+".join(sorted(rules[root])) or "geometric_fragment_assembly"
        anchor = by_index[root]
        plans.append(
            AssemblyPlan(
                member_indexes=tuple(sorted(member_set)),
                anchor_index=root,
                object_id=anchor.candidate_id,
                text=text,
                bbox=bbox,
                polygon=polygon,
                orientation=anchor.orientation,
                rotation=anchor.rotation,
                confidence=min(token.confidence for token in members),
                rule=rule,
                conflict=conflict,
                needs_review=needs_review,
                review_reason=review_reason,
                children=tuple(_child(token) for token in members),
                recognition_source=source,
                recognition_evidence=recognition_evidence,
            )
        )
    return plans


def assembly_statistics(
    input_count: int,
    output_count: int,
    plans: Sequence[AssemblyPlan],
) -> dict[str, Any]:
    rule_counts: dict[str, int] = {}
    for plan in plans:
        rule_counts[plan.rule] = rule_counts.get(plan.rule, 0) + 1
    return {
        "atomic_detection_count": int(input_count),
        "object_count": int(output_count),
        "assembled_object_count": len(plans),
        "absorbed_fragment_count": max(0, int(input_count) - int(output_count)),
        "conflict_count": sum(1 for plan in plans if plan.conflict),
        "rule_counts": dict(sorted(rule_counts.items())),
    }
