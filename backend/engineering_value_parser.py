"""Lossless structural parser for engineering drawing values.

M4 turns primitive detector boxes into one logical engineering object.  This
module is the next boundary: it describes the text owned by that object without
deciding whether the object should be ballooned, reviewed, or excluded.

The parser is deliberately lossless:

* ``raw_text`` is returned byte-for-byte as received;
* ``normalized_text`` records only explicit, reversible OCR-symbol cleanup;
* tokens cover every character of the normalized text, including whitespace;
* partial and unknown expressions retain their unparsed fragments;
* absent tolerance bounds remain absent -- the parser never invents values.

Disposition belongs to the page policy (M6), not to this syntax layer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Literal, Mapping


ParseStatus = Literal["complete", "partial", "unparsed", "empty"]
ParseKind = Literal[
    "linear",
    "angle",
    "chamfer",
    "thread",
    "dual_unit",
    "ratio",
    "standalone_tolerance",
    "gdt",
    "datum",
    "surface_finish",
    "identifier",
    "unknown",
]

_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_QUALIFIER = r"(?:MAX|MIN|TYP|REF|BASIC|THRU)"
_QUALIFIER_RE = re.compile(rf"{_QUALIFIER}", re.IGNORECASE)
_HAS_DIGIT_RE = re.compile(r"\d")

_TOKEN_RE = re.compile(
    rf"(?P<space>\s+)"
    rf"|(?P<number>{_NUMBER})"
    r"|(?P<diameter>Ø)"
    r"|(?P<plus_minus>±)"
    r"|(?P<degree>°)"
    r"|(?P<prime>[′'])"
    r'|(?P<double_prime>[″\"])'
    r"|(?P<operator>[+\-/:X×])"
    r"|(?P<bracket>[\[\]()])"
    r"|(?P<word>[A-Za-z]+)"
    r"|(?P<other>.)",
    re.DOTALL,
)

_DUAL_UNIT_RE = re.compile(
    r"^(?P<primary>[^\[\]]+?)\s*\[\s*(?P<secondary>[^\[\]]+?)\s*\]"
    rf"\s*(?P<qualifier>{_QUALIFIER})?\.?$",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(
    rf"^(?P<first>{_NUMBER})\s*:\s*(?P<second>{_NUMBER})$",
    re.IGNORECASE,
)
_THREAD_RE = re.compile(
    rf"^(?:(?P<quantity>\d+)\s*[X×]\s*)?"
    rf"M\s*(?P<nominal>{_NUMBER})"
    rf"(?:\s*[X×]\s*(?P<pitch>{_NUMBER}))?"
    r"(?:\s*-\s*(?P<fit>[0-9A-Z]+))?"
    rf"(?P<qualifiers>(?:\s*{_QUALIFIER})*)$",
    re.IGNORECASE,
)
_CHAMFER_RE = re.compile(
    rf"^(?P<size>{_NUMBER})\s*[X×]\s*"
    rf"(?P<angle>{_NUMBER})\s*°"
    rf"(?P<tolerance>\s*(?:±\s*{_NUMBER}\s*°?"
    rf"|\+\s*{_NUMBER}\s*°?\s*-\s*{_NUMBER}\s*°?))?"
    rf"(?P<qualifiers>(?:\s*{_QUALIFIER})*)$",
    re.IGNORECASE,
)
_STANDALONE_TOLERANCE_RE = re.compile(
    rf"^(?P<sign>±|\+|-)\s*(?P<value>{_NUMBER})\s*(?P<degree>°)?$"
)
_ANGLE_RE = re.compile(
    rf"^(?P<degrees>{_NUMBER})\s*°"
    r"(?:\s*(?P<minutes>\d{1,2})\s*[′']"
    rf"(?:\s*(?P<seconds>{_NUMBER})\s*[″\"])?"
    r"|\s*(?P<tickless_minutes>\d{1,2})(?![\d.]))?"
    rf"(?P<tolerance>\s*(?:±\s*(?:{_NUMBER}\s*°?"
    rf"|{_NUMBER}\s*°\s*\d{{1,2}}\s*[′'])"
    rf"|\+\s*{_NUMBER}\s*°?\s*-\s*{_NUMBER}\s*°?))?"
    rf"(?P<qualifiers>(?:\s*{_QUALIFIER})*)$",
    re.IGNORECASE,
)
_DIMENSION_RE = re.compile(
    rf"^(?:(?P<quantity>\d+)\s*[X×]\s*)?"
    r"(?P<prefix>SR|SØ|Ø|R)?\s*"
    rf"(?P<nominal>[+-]?{_NUMBER})"
    r"(?P<fit>[A-Z]\d{1,3})?"
    rf"(?P<tolerance>\s*(?:±\s*{_NUMBER}\s*°?"
    rf"|[+-]\s*{_NUMBER}\s*(?:/\s*(?:[+-]\s*)?{_NUMBER}"
    rf"|\s*[+-]\s*{_NUMBER})"
    rf"|[+-]\s*{_NUMBER}"
    rf"|(?:/|\bTO\b|:)\s*{_NUMBER}))?"
    r"\s*(?P<unit>MM|CM|INCHES|INCH|IN|\")?"
    rf"(?P<qualifiers>(?:\s*{_QUALIFIER})*)$",
    re.IGNORECASE,
)
_PARTIAL_DIMENSION_RE = re.compile(
    rf"^(?:(?P<quantity>\d+)\s*[X×]\s*)?"
    r"(?P<prefix>SR|SØ|Ø|R)?\s*"
    rf"(?P<nominal>[+-]?{_NUMBER})"
    r"(?P<fit>[A-Z]\d{1,3})?",
    re.IGNORECASE,
)
_INCOMPLETE_DECIMAL_TOLERANCE_RE = re.compile(
    r"^\s*(?P<sign>[+-])\s*(?P<digits>\d+)\.\s*$"
)
_IDENTIFIER_RE = re.compile(
    r"^(?=.{2,}$)(?=.*[A-Z])(?=.*\d)[A-Z0-9][A-Z0-9._/\-]*$",
    re.IGNORECASE,
)
_SURFACE_VALUE_RE = re.compile(
    rf"^(?P<parameter>RA|RZ|RMAX|RQ)\s*(?P<value>{_NUMBER})(?:\s*(?P<unit>µM|UM))?$"
    r"|^(?P<grade>N(?:1[0-2]|[1-9]))$", re.IGNORECASE,
)
_GDT_TEXT_RE = re.compile(
    r"[⏤⏥▱○◯⌭⌒⌓∥⟂⊥∠⌖◎⌯↗⌰]|\b(?:STRAIGHTNESS|FLATNESS|CIRCULARITY|"
    r"ROUNDNESS|CYLINDRICITY|PARALLELISM|PERPENDICULARITY|ANGULARITY|POSITION|"
    r"CONCENTRICITY|SYMMETRY|RUNOUT)\b", re.IGNORECASE,
)

_SYMBOL_REPLACEMENTS = (
    ("+/-", "±", "ascii_plus_minus"),
    ("+/−", "±", "ascii_plus_minus"),
    ("º", "°", "degree_variant"),
    ("˚", "°", "degree_variant"),
    ("⁰", "°", "degree_variant"),
    ("−", "-", "minus_variant"),
    ("–", "-", "minus_variant"),
    ("—", "-", "minus_variant"),
    ("φ", "Ø", "diameter_variant"),
    ("Φ", "Ø", "diameter_variant"),
    ("⌀", "Ø", "diameter_variant"),
    ("ø", "Ø", "diameter_variant"),
    ("×", "X", "multiplication_variant"),
    ("‘", "′", "prime_variant"),
    ("’", "′", "prime_variant"),
    ("`", "′", "prime_variant"),
    ("“", "″", "double_prime_variant"),
    ("”", "″", "double_prime_variant"),
)


@dataclass(frozen=True)
class EngineeringValueToken:
    kind: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class UnparsedFragment:
    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class EngineeringValueParse:
    raw_text: str
    normalized_text: str
    status: ParseStatus
    kind: ParseKind
    components: Mapping[str, Any] = field(default_factory=dict)
    tokens: tuple[EngineeringValueToken, ...] = ()
    unparsed_fragments: tuple[UnparsedFragment, ...] = ()
    warnings: tuple[str, ...] = ()
    normalization_steps: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "status": self.status,
            "kind": self.kind,
            "complete": self.complete,
            "components": dict(self.components),
            "tokens": [token.to_dict() for token in self.tokens],
            "unparsed_fragments": [
                fragment.to_dict() for fragment in self.unparsed_fragments
            ],
            "warnings": list(self.warnings),
            "normalization_steps": list(self.normalization_steps),
        }


def _normalize(text: str) -> tuple[str, tuple[str, ...]]:
    value = str(text or "")
    steps: list[str] = []
    for source, replacement, step in _SYMBOL_REPLACEMENTS:
        if source in value:
            value = value.replace(source, replacement)
            if step not in steps:
                steps.append(step)
    value, plus_minus_count = re.subn(
        r"\+\s*(?:/\s*)?[-−](?=\s*\d)",
        "±",
        value,
    )
    if plus_minus_count and "ascii_plus_minus" not in steps:
        steps.append("ascii_plus_minus")
    collapsed = " ".join(value.strip().split())
    if collapsed != value:
        steps.append("whitespace")
    return collapsed, tuple(steps)


def _tokens(text: str) -> tuple[EngineeringValueToken, ...]:
    return tuple(
        EngineeringValueToken(
            kind=str(match.lastgroup or "other"),
            text=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for match in _TOKEN_RE.finditer(text)
    )


def _qualifiers(text: str | None) -> list[str]:
    return [match.group(0).upper() for match in _QUALIFIER_RE.finditer(text or "")]


def _reference_wrapper(text: str) -> tuple[str, bool, int]:
    if len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip(), True, 1
    return text, False, 0


def _signed(sign: str, value: str) -> str:
    return f"{sign}{value}"


def _parse_tolerance(text: str | None) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if not value:
        return None

    angle_symmetric = re.fullmatch(
        rf"±\s*(?P<value>{_NUMBER}\s*°(?:\s*\d{{1,2}}\s*[′'])?)",
        value,
    )
    if angle_symmetric:
        magnitude = angle_symmetric.group("value").replace(" ", "")
        return {
            "mode": "symmetric",
            "text": value,
            "upper": f"+{magnitude}",
            "lower": f"-{magnitude}",
            "degree": True,
        }

    angle_pair = re.fullmatch(
        rf"(?P<first_sign>[+-])\s*(?P<first>{_NUMBER})\s*°?\s*"
        rf"(?P<second_sign>[+-])\s*(?P<second>{_NUMBER})\s*°?",
        value,
    )
    if angle_pair and "°" in value:
        return {
            "mode": "deviation_pair",
            "text": value,
            "upper": _signed(
                angle_pair.group("first_sign"), angle_pair.group("first")
            ),
            "lower": _signed(
                angle_pair.group("second_sign"), angle_pair.group("second")
            ),
            "degree": True,
        }

    symmetric = re.fullmatch(rf"±\s*(?P<value>{_NUMBER})\s*(?P<degree>°)?", value)
    if symmetric:
        magnitude = symmetric.group("value")
        return {
            "mode": "symmetric",
            "text": value,
            "upper": f"+{magnitude}",
            "lower": f"-{magnitude}",
            "degree": bool(symmetric.group("degree")),
        }

    pair = re.fullmatch(
        rf"(?P<first_sign>[+-])\s*(?P<first>{_NUMBER})\s*"
        rf"(?:(?P<slash>/)\s*(?P<slash_sign>[+-])?"
        rf"|(?P<second_sign>[+-]))\s*(?P<second>{_NUMBER})",
        value,
    )
    if pair:
        first = _signed(pair.group("first_sign"), pair.group("first"))
        second_sign = pair.group("slash_sign") or pair.group("second_sign") or ""
        second = f"{second_sign}{pair.group('second')}"
        return {
            "mode": "deviation_pair",
            "text": value,
            "upper": first,
            "lower": second,
            "separator": "/" if pair.group("slash") else "",
        }

    single = re.fullmatch(
        rf"(?P<sign>[+-])\s*(?P<value>{_NUMBER})\s*(?P<degree>°)?",
        value,
    )
    if single:
        sign = single.group("sign")
        signed = _signed(sign, single.group("value"))
        return {
            "mode": "single_deviation",
            "text": value,
            "upper": signed if sign == "+" else None,
            "lower": signed if sign == "-" else None,
            "degree": bool(single.group("degree")),
        }

    limits = re.fullmatch(
        rf"(?P<separator>/|TO|:)\s*(?P<limit>{_NUMBER})",
        value,
        re.IGNORECASE,
    )
    if limits:
        return {
            "mode": "limits",
            "text": value,
            "second_limit": limits.group("limit"),
            "separator": limits.group("separator").upper(),
        }
    return None


def _result(
    *,
    raw_text: str,
    normalized_text: str,
    status: ParseStatus,
    kind: ParseKind,
    components: Mapping[str, Any] | None = None,
    unparsed_fragments: Iterable[UnparsedFragment] = (),
    warnings: Iterable[str] = (),
    normalization_steps: tuple[str, ...] = (),
) -> EngineeringValueParse:
    return EngineeringValueParse(
        raw_text=raw_text,
        normalized_text=normalized_text,
        status=status,
        kind=kind,
        components=dict(components or {}),
        tokens=_tokens(normalized_text),
        unparsed_fragments=tuple(unparsed_fragments),
        warnings=tuple(dict.fromkeys(warnings)),
        normalization_steps=normalization_steps,
    )


def parse_engineering_value(
    text: str,
    *,
    feature_category: str | None = None,
    engineering_symbol: Mapping[str, Any] | None = None,
    _allow_dual: bool = True,
) -> EngineeringValueParse:
    """Parse one assembled object without changing or disposing of it."""

    raw = str(text or "")
    normalized, normalization_steps = _normalize(raw)
    if not normalized:
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="empty",
            kind="unknown",
            normalization_steps=normalization_steps,
        )

    value, reference, wrapper_offset = _reference_wrapper(normalized)
    base_components: dict[str, Any] = {"reference": reference}

    structured = engineering_symbol or {}
    structured_kind = str(structured.get("kind") or "")
    structured_complete = bool(structured.get("complete"))
    if structured_kind == "feature_control_frame":
        complete = structured_complete and bool(_HAS_DIGIT_RE.search(value))
        warnings = list(structured.get("warnings") or ())
        if not complete and "unresolved_gdt_characteristic" not in warnings:
            warnings.append("unresolved_gdt_characteristic")
        return _result(raw_text=raw, normalized_text=normalized,
            status="complete" if complete else "partial", kind="gdt",
            components={**base_components, "characteristic": structured.get("subtype"),
                        "symbol": structured.get("completed_symbol"),
                        "cells": list(structured.get("cells") or ())},
            unparsed_fragments=() if complete else
                (UnparsedFragment(value, wrapper_offset, wrapper_offset + len(value)),),
            warnings=warnings, normalization_steps=normalization_steps)
    if structured_kind == "datum":
        complete = structured_complete and bool(structured.get("subtype") or value)
        return _result(raw_text=raw, normalized_text=normalized,
            status="complete" if complete else "partial", kind="datum",
            components={**base_components, "datum": structured.get("subtype") or value},
            unparsed_fragments=() if complete else
                (UnparsedFragment(value, wrapper_offset, wrapper_offset + len(value)),),
            warnings=() if complete else ("incomplete_datum_evidence",),
            normalization_steps=normalization_steps)
    if structured_kind == "surface_finish":
        complete = structured_complete and bool(_HAS_DIGIT_RE.search(value))
        match = _SURFACE_VALUE_RE.fullmatch(value)
        return _result(raw_text=raw, normalized_text=normalized,
            status="complete" if complete else "partial", kind="surface_finish",
            components={**base_components,
                "parameter": (match.group("parameter").upper() if match and match.group("parameter")
                              else structured.get("subtype")),
                "value": match.group("value") if match else value,
                "unit": match.group("unit").upper() if match and match.group("unit") else None,
                "grade": match.group("grade").upper() if match and match.group("grade") else None},
            unparsed_fragments=() if complete else
                (UnparsedFragment(value, wrapper_offset, wrapper_offset + len(value)),),
            warnings=() if complete else ("incomplete_surface_finish_evidence",),
            normalization_steps=normalization_steps)

    surface = _SURFACE_VALUE_RE.fullmatch(value)
    if surface:
        return _result(raw_text=raw, normalized_text=normalized, status="complete",
            kind="surface_finish", components={**base_components,
                "parameter": surface.group("parameter").upper() if surface.group("parameter") else None,
                "value": surface.group("value"),
                "unit": surface.group("unit").upper() if surface.group("unit") else None,
                "grade": surface.group("grade").upper() if surface.group("grade") else None},
            normalization_steps=normalization_steps)
    if (feature_category == "GD&T"
            or (_GDT_TEXT_RE.search(value) and _HAS_DIGIT_RE.search(value))
            or (value.count("|") >= 2 and _HAS_DIGIT_RE.search(value))):
        return _result(raw_text=raw, normalized_text=normalized, status="partial", kind="gdt",
            components=base_components,
            unparsed_fragments=(UnparsedFragment(value, wrapper_offset, wrapper_offset + len(value)),),
            warnings=("gdt_requires_visual_frame_evidence",),
            normalization_steps=normalization_steps)

    if _allow_dual:
        dual = _DUAL_UNIT_RE.fullmatch(value)
        if dual:
            primary = parse_engineering_value(dual.group("primary"), _allow_dual=False)
            secondary = parse_engineering_value(
                dual.group("secondary"), _allow_dual=False
            )
            qualifiers = _qualifiers(dual.group("qualifier"))
            complete = primary.complete and secondary.complete
            warnings: list[str] = []
            if not complete:
                warnings.append("dual_unit_component_incomplete")
            return _result(
                raw_text=raw,
                normalized_text=normalized,
                status="complete" if complete else "partial",
                kind="dual_unit",
                components={
                    **base_components,
                    "primary": primary.to_dict(),
                    "secondary": secondary.to_dict(),
                    "qualifiers": qualifiers,
                },
                unparsed_fragments=(
                    ()
                    if complete
                    else (UnparsedFragment(value, wrapper_offset, wrapper_offset + len(value)),)
                ),
                warnings=warnings,
                normalization_steps=normalization_steps,
            )

    ratio = _RATIO_RE.fullmatch(value)
    if ratio:
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="complete",
            kind="ratio",
            components={
                **base_components,
                "first": ratio.group("first"),
                "second": ratio.group("second"),
            },
            normalization_steps=normalization_steps,
        )

    thread = _THREAD_RE.fullmatch(value)
    if thread:
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="complete",
            kind="thread",
            components={
                **base_components,
                "quantity": thread.group("quantity"),
                "prefix": "M",
                "nominal": thread.group("nominal"),
                "pitch": thread.group("pitch"),
                "fit": thread.group("fit"),
                "qualifiers": _qualifiers(thread.group("qualifiers")),
            },
            normalization_steps=normalization_steps,
        )

    chamfer = _CHAMFER_RE.fullmatch(value)
    if chamfer:
        tolerance = _parse_tolerance(chamfer.group("tolerance"))
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="complete",
            kind="chamfer",
            components={
                **base_components,
                "size": chamfer.group("size"),
                "angle_degrees": chamfer.group("angle"),
                "tolerance": tolerance,
                "qualifiers": _qualifiers(chamfer.group("qualifiers")),
            },
            normalization_steps=normalization_steps,
        )

    standalone = _STANDALONE_TOLERANCE_RE.fullmatch(value)
    if standalone:
        tolerance = _parse_tolerance(value)
        warnings = (
            ("single_bound_tolerance",)
            if standalone.group("sign") in {"+", "-"}
            else ()
        )
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="complete",
            kind="standalone_tolerance",
            components={**base_components, "tolerance": tolerance},
            warnings=warnings,
            normalization_steps=normalization_steps,
        )

    angle = _ANGLE_RE.fullmatch(value)
    if angle:
        tolerance = _parse_tolerance(angle.group("tolerance"))
        minutes = angle.group("minutes") or angle.group("tickless_minutes")
        warnings: list[str] = []
        if angle.group("tickless_minutes"):
            warnings.append("tickless_angle_minutes")
        if tolerance and tolerance.get("mode") == "single_deviation":
            warnings.append("single_bound_tolerance")
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="complete",
            kind="angle",
            components={
                **base_components,
                "degrees": angle.group("degrees"),
                "minutes": minutes,
                "seconds": angle.group("seconds"),
                "tolerance": tolerance,
                "qualifiers": _qualifiers(angle.group("qualifiers")),
            },
            warnings=warnings,
            normalization_steps=normalization_steps,
        )

    dimension = _DIMENSION_RE.fullmatch(value)
    if dimension:
        tolerance = _parse_tolerance(dimension.group("tolerance"))
        warnings: list[str] = []
        if tolerance and tolerance.get("mode") == "single_deviation":
            warnings.append("single_bound_tolerance")
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="complete",
            kind="linear",
            components={
                **base_components,
                "quantity": dimension.group("quantity"),
                "prefix": (dimension.group("prefix") or "").upper() or None,
                "nominal": dimension.group("nominal"),
                "fit": (dimension.group("fit") or "").upper() or None,
                "tolerance": tolerance,
                "unit": (dimension.group("unit") or "").upper() or None,
                "qualifiers": _qualifiers(dimension.group("qualifiers")),
            },
            warnings=warnings,
            normalization_steps=normalization_steps,
        )

    # Keep useful structure from malformed OCR without pretending the whole
    # expression was valid.  ``R0.6-0.`` is the important example: its radius
    # and nominal are known, its lower deviation is explicitly incomplete, and
    # the original dot remains present in both text fields and token evidence.
    partial = _PARTIAL_DIMENSION_RE.match(value)
    if partial:
        remainder = value[partial.end() :]
        components: dict[str, Any] = {
            **base_components,
            "quantity": partial.group("quantity"),
            "prefix": (partial.group("prefix") or "").upper() or None,
            "nominal": partial.group("nominal"),
            "fit": (partial.group("fit") or "").upper() or None,
        }
        warnings = ["unparsed_fragment"]
        incomplete = _INCOMPLETE_DECIMAL_TOLERANCE_RE.fullmatch(remainder)
        fragments: tuple[UnparsedFragment, ...]
        if incomplete:
            sign = incomplete.group("sign")
            signed = f"{sign}{incomplete.group('digits')}."
            components["tolerance"] = {
                "mode": "single_deviation",
                "text": remainder.strip(),
                "upper": signed if sign == "+" else None,
                "lower": signed if sign == "-" else None,
            }
            warnings = ["incomplete_tolerance_decimal"]
            fragments = ()
        else:
            start = wrapper_offset + partial.end()
            fragments = (
                UnparsedFragment(remainder, start, start + len(remainder)),
            ) if remainder else ()
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="partial",
            kind="linear",
            components=components,
            unparsed_fragments=fragments,
            warnings=warnings,
            normalization_steps=normalization_steps,
        )

    if _IDENTIFIER_RE.fullmatch(value):
        return _result(
            raw_text=raw,
            normalized_text=normalized,
            status="complete",
            kind="identifier",
            components={**base_components, "identifier": value},
            warnings=("identifier_requires_context",),
            normalization_steps=normalization_steps,
        )

    status: ParseStatus = "partial" if _HAS_DIGIT_RE.search(value) else "unparsed"
    return _result(
        raw_text=raw,
        normalized_text=normalized,
        status=status,
        kind="unknown",
        components=base_components,
        unparsed_fragments=(
            UnparsedFragment(value, wrapper_offset, wrapper_offset + len(value)),
        ),
        warnings=("unparsed_engineering_text",),
        normalization_steps=normalization_steps,
    )


def engineering_parse_statistics(
    parses: Iterable[EngineeringValueParse | Mapping[str, Any]],
) -> dict[str, Any]:
    """Return deterministic parser telemetry without affecting disposition."""

    status_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    count = 0
    for item in parses:
        count += 1
        if isinstance(item, EngineeringValueParse):
            status = item.status
            kind = item.kind
            warnings = item.warnings
        else:
            status = str(item.get("status") or "unparsed")
            kind = str(item.get("kind") or "unknown")
            warnings = tuple(str(value) for value in item.get("warnings") or ())
        status_counts[status] += 1
        kind_counts[kind] += 1
        warning_counts.update(warnings)
    return {
        "schema_version": 1,
        "object_count": count,
        "complete_count": status_counts["complete"],
        "partial_count": status_counts["partial"],
        "unparsed_count": status_counts["unparsed"],
        "empty_count": status_counts["empty"],
        "status_counts": dict(sorted(status_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }
