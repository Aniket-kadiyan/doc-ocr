"""
Which sections of the GDTOCR spec the rule engine applies.

The spec (``Aniket/GDTOCR.docx``) numbers 26 detection rules. Some are marked in
yellow in that document as out of scope, and one more needs geometry the
pipeline does not measure yet. Rather than deleting the vocabulary for those —
which would make it hard to turn them back on, and would lose the record of why
they are off — every section is listed here with a flag and a reason.

`feature_classifier` consults this table before applying a rule, so switching a
section back on is a one-line change here plus whatever detection it needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import feature_dictionary as fd

#: Reasons a section is not applied, kept short so they read well in a listing.
OUT_OF_SCOPE = "marked out of scope in the spec"
NEEDS_GEOMETRY = "needs dimension-line geometry the pipeline does not measure"


@dataclass(frozen=True)
class SpecRule:
    """One numbered detection rule from the spec."""

    section: int
    name: str
    category: str | None
    enabled: bool
    reason: str = ""


SPEC_RULES: tuple[SpecRule, ...] = (
    SpecRule(1, "General Notes", fd.CAT_NOTE, True),
    SpecRule(2, "Overall Dimensions", fd.CAT_LINEAR, False, NEEDS_GEOMETRY),
    SpecRule(3, "Linear Dimensions", fd.CAT_LINEAR, True),
    SpecRule(4, "Diameter", fd.CAT_DIAMETER, True),
    SpecRule(5, "Radius", fd.CAT_RADIUS, True),
    SpecRule(6, "Chamfer", fd.CAT_CHAMFER, True),
    SpecRule(7, "Fillet", fd.CAT_RADIUS, False, OUT_OF_SCOPE),
    SpecRule(8, "Angle", fd.CAT_ANGLE, True),
    SpecRule(9, "Taper", fd.CAT_TAPER, True),
    SpecRule(10, "Thread", fd.CAT_THREAD, True),
    SpecRule(11, "THRU", fd.CAT_HOLE, True),
    SpecRule(12, "Counterbore", fd.CAT_HOLE, False, OUT_OF_SCOPE),
    SpecRule(13, "Countersink", fd.CAT_HOLE, True),
    SpecRule(14, "Spotface", fd.CAT_HOLE, True),
    SpecRule(15, "Depth", fd.CAT_HOLE, False, OUT_OF_SCOPE),
    SpecRule(16, "GD&T", fd.CAT_GDT, False, OUT_OF_SCOPE),
    SpecRule(17, "Datum", fd.CAT_DATUM, True),
    SpecRule(18, "Surface Finish", fd.CAT_SURFACE, True),
    SpecRule(19, "Weld", fd.CAT_WELD, True),
    SpecRule(20, "Material", fd.CAT_MATERIAL, True),
    SpecRule(21, "Heat Treatment", fd.CAT_HEAT, True),
    SpecRule(22, "Coating", fd.CAT_COATING, True),
    SpecRule(23, "Deburring", fd.CAT_NOTE, True),
    SpecRule(24, "Quantity Prefix", None, False, OUT_OF_SCOPE),
    SpecRule(25, "Reference Dimension", fd.CAT_REFERENCE, False, OUT_OF_SCOPE),
    SpecRule(26, "Basic Dimension", fd.CAT_BASIC, True),
)

_BY_SECTION = {rule.section: rule for rule in SPEC_RULES}

# Section numbers, named so the classifier reads as the spec does.
GENERAL_NOTES = 1
OVERALL_DIMENSIONS = 2
LINEAR = 3
DIAMETER = 4
RADIUS = 5
CHAMFER = 6
FILLET = 7
ANGLE = 8
TAPER = 9
THREAD = 10
THRU = 11
COUNTERBORE = 12
COUNTERSINK = 13
SPOTFACE = 14
DEPTH = 15
GDT = 16
DATUM = 17
SURFACE_FINISH = 18
WELD = 19
MATERIAL = 20
HEAT_TREATMENT = 21
COATING = 22
DEBURRING = 23
QUANTITY_PREFIX = 24
REFERENCE = 25
BASIC = 26


def rule(section: int) -> SpecRule:
    return _BY_SECTION[section]


def is_enabled(section: int) -> bool:
    """True when the classifier should apply this section's rule."""
    return _BY_SECTION[section].enabled


def enabled_rules() -> tuple[SpecRule, ...]:
    return tuple(r for r in SPEC_RULES if r.enabled)


def disabled_rules() -> tuple[SpecRule, ...]:
    return tuple(r for r in SPEC_RULES if not r.enabled)


#: Hole modifier keywords belonging to sections that are switched off, so the
#: classifier can filter them out of the shared HOLE_MODIFIERS table.
_SECTION_KEYWORDS: dict[int, frozenset[str]] = {
    COUNTERBORE: frozenset({"CBORE", "COUNTERBORE", "C'BORE"}),
    DEPTH: frozenset({"DEEP", "DP", "DEPTH"}),
    COUNTERSINK: frozenset({"CSK", "COUNTERSINK", "C'SINK"}),
    SPOTFACE: frozenset({"SF", "SPOTFACE", "SPOT FACE"}),
    THRU: frozenset({"THRU", "THROUGH"}),
}


def active_hole_modifiers() -> frozenset[str]:
    """Hole keywords from the sections that are switched on."""
    blocked: set[str] = set()
    for section, words in _SECTION_KEYWORDS.items():
        if not is_enabled(section):
            blocked |= set(words)
    return frozenset(w for w in fd.HOLE_MODIFIERS if w not in blocked)


def active_hole_subtypes() -> list[tuple[frozenset[str], str]]:
    """Hole subtypes for the sections that are switched on, in spec order."""
    blocked_names = {
        "Counterbore": COUNTERBORE,
        "Countersink": COUNTERSINK,
        "Spotface": SPOTFACE,
        "Through Hole": THRU,
        "Blind Hole": DEPTH,
    }
    return [
        (words, name)
        for words, name in fd.HOLE_SUBTYPES
        if is_enabled(blocked_names.get(name, THRU)) or name not in blocked_names
    ]


def describe() -> str:
    """A listing of the rules and their state, for logs and the API."""
    lines = ["GDTOCR rule engine — section coverage"]
    for r in SPEC_RULES:
        mark = "on " if r.enabled else "off"
        tail = f"  ({r.reason})" if r.reason else ""
        lines.append(f"  {r.section:>2}. [{mark}] {r.name}{tail}")
    return "\n".join(lines)
