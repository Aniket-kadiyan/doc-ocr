"""
The GD&T symbol registry — one authoritative table of every symbol the
recognizer can name.

Why this is a module of its own
-------------------------------
``feature_dictionary.py`` already carries a *text* vocabulary for the GDTOCR
rule engine: upper-cased keywords and a handful of glyph→name pairs used to
label a value that OCR already read. That table answers "what does this string
mean". It cannot answer "is this 12×14 patch of ink a circularity symbol", and
it was never meant to — spec section 16 (GD&T) is switched off in
``feature_rules.py`` precisely because a text rule cannot carry that weight.

This registry is the other half: it describes each symbol as a *thing on paper*
— how it is drawn, what OCR turns it into when it fails, what it is confused
with, and how a detector should go after it. ``gdt_detect`` and ``fcf_parse``
consume it. Nothing here imports cv2 or numpy, so the whole table stays
importable (and testable) on a machine with no vision stack.

Deliberate boundaries
---------------------
A symbol and an *interpretation* are not the same thing, and the registry keeps
them apart with :data:`KIND_CONCEPT`. Virtual condition and bonus tolerance are
real and important, but neither is drawn: they are computed from a size
tolerance plus a material-condition modifier. Marking them as concepts means
the detector can never "find" one, which is the correct behaviour — reporting a
bonus tolerance as a detected glyph would be a fabrication.

Likewise a control that a standard has retired is recorded, not deleted, and
carries the revision that retired it. Concentricity and symmetry still appear
on ISO drawings and on every ASME drawing printed before 2018. Dropping them
would make the recognizer silently blind to a legacy sheet; labelling them
"valid under ASME Y14.5-2018" would be wrong. :class:`StandardsNote` lets an
entry say both things at once.

Unicode is recorded where it exists and is never treated as proof of anything.
A CAD export may encode Ø as the letter ``P`` of the AIGDT font (see
``pdf_text_layer``), a scan encodes it as pixels, and a PDF producer may encode
circularity as a plain ``O``. ``ocr_variants`` is the record of what we have
actually seen come back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

# ── Categories ───────────────────────────────────────────────────────────────
# The group a symbol belongs to. These are the registry's own grouping, chosen
# to match how a drawing uses the symbol rather than how a catalogue lists it.
CAT_GEOMETRIC = "geometric_tolerance"
CAT_DATUM = "datum"
CAT_MATERIAL_CONDITION = "material_condition"
CAT_DIMENSIONING = "dimensioning"
CAT_MANUFACTURING = "manufacturing"
CAT_FRAME = "frame_element"
CAT_RULE = "rule_or_principle"

CATEGORIES: tuple[str, ...] = (
    CAT_GEOMETRIC,
    CAT_DATUM,
    CAT_MATERIAL_CONDITION,
    CAT_DIMENSIONING,
    CAT_MANUFACTURING,
    CAT_FRAME,
    CAT_RULE,
)

# ── Kinds ────────────────────────────────────────────────────────────────────
#: Drawn on its own and meaningful on its own (Ø, a datum triangle).
KIND_STANDALONE = "standalone"
#: Only meaningful attached to something else (Ⓜ qualifies a tolerance).
KIND_MODIFIER = "modifier"
#: Several marks that are read as one annotation (a datum feature symbol is a
#: triangle plus a boxed letter plus a leader).
KIND_COMPOSITE = "composite"
#: A container whose *compartments* carry the meaning (the feature control
#: frame itself).
KIND_FRAME = "frame"
#: Real engineering meaning, no glyph of its own — inferred or computed. The
#: detector must never emit one of these as a detection.
KIND_CONCEPT = "concept"
#: Written as words or letters rather than drawn (a restrained-condition note).
KIND_TEXTUAL = "textual"

KINDS: tuple[str, ...] = (
    KIND_STANDALONE,
    KIND_MODIFIER,
    KIND_COMPOSITE,
    KIND_FRAME,
    KIND_CONCEPT,
    KIND_TEXTUAL,
)

#: Kinds a vision detector is allowed to report. Anything else is derived.
DETECTABLE_KINDS: frozenset[str] = frozenset(
    {KIND_STANDALONE, KIND_MODIFIER, KIND_COMPOSITE, KIND_FRAME}
)

# ── Recognition strategies ───────────────────────────────────────────────────
# What a detector should actually try for this symbol. Several per entry, in
# the order they are worth attempting.
S_TEMPLATE = "template_match"          # procedural template, multi-scale
S_CONTOUR = "contour_shape"            # contour/topology features
S_OCR_GLYPH = "ocr_glyph"              # OCR returns the glyph or a variant
S_OCR_TEXT = "ocr_text"                # OCR returns a word/abbreviation
S_CIRCLED_LETTER = "circled_letter"    # ring + one letter inside
S_BOXED_LETTER = "boxed_letter"        # rectangle + one letter inside
S_FRAME_SEGMENT = "frame_compartment"  # position inside a feature control frame
S_LEADER = "leader_geometry"           # mark sitting on a leader/witness line
S_RECTANGLE = "rectangle_detect"       # the frame/box outline itself
S_DERIVED = "derived"                  # computed, never seen
S_TEXT_NOTE = "text_note"              # found by reading a note

STRATEGIES: tuple[str, ...] = (
    S_TEMPLATE,
    S_CONTOUR,
    S_OCR_GLYPH,
    S_OCR_TEXT,
    S_CIRCLED_LETTER,
    S_BOXED_LETTER,
    S_FRAME_SEGMENT,
    S_LEADER,
    S_RECTANGLE,
    S_DERIVED,
    S_TEXT_NOTE,
)

# ── Value parsing ────────────────────────────────────────────────────────────
#: What sort of value, if any, is read alongside the symbol.
V_NONE = "none"
V_LENGTH = "length"            # a linear size in the drawing's own unit
V_ANGLE = "angle"              # degrees (optionally minutes/seconds)
V_RATIO = "ratio"              # 1:5, 0.5:1 — taper and slope
V_COUNT = "count"              # 4X — how many identical features
V_LETTER = "letter"            # a datum letter, not a number
V_TOLERANCE = "tolerance"      # a geometric tolerance zone width
V_COMPOSITE = "composite"      # several values (counterbore: Ø then depth)


@dataclass(frozen=True)
class ValueSpec:
    """How to read the number (or letter) that travels with a symbol.

    ``unit`` is deliberately absent. A drawing's unit comes from the drawing —
    its title block, a units note, or a dual-unit bracket — never from the
    symbol, and never from an assumption that engineering means millimetres.
    Consumers carry the unit as ``drawing_unit`` until something on the sheet
    says otherwise.
    """

    kind: str = V_NONE
    #: True when the value may legitimately be absent (an all-around symbol
    #: carries no number of its own).
    optional: bool = True
    #: A regex fragment documenting the expected shape, for the parser and for
    #: anyone reading the registry. Not compiled here.
    pattern: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class StandardsNote:
    """Which standards define a symbol, and what happened to it since.

    ``retired_in`` records a revision that removed the control *without*
    implying the symbol is meaningless: a drawing released under an earlier
    revision, or under ISO, still uses it legitimately.
    """

    #: Standards that define or defined the symbol.
    standards: tuple[str, ...] = ()
    #: Revision that removed it, e.g. "ASME Y14.5-2018".
    retired_in: str | None = None
    #: Standards where it remains current.
    current_in: tuple[str, ...] = ()
    #: Present in only one standards family, e.g. ISO-only or ASME-only.
    exclusive_to: str | None = None
    note: str | None = None

    @property
    def is_legacy(self) -> bool:
        return self.retired_in is not None


@dataclass(frozen=True)
class GdtSymbol:
    """One entry of the registry."""

    id: str
    name: str
    category: str
    kind: str
    description: str
    standards: StandardsNote
    #: The Unicode code point normally used, when one exists. Absent for
    #: symbols with no Unicode representation (a datum target, a parting line).
    unicode: str | None = None
    #: Human names and shop abbreviations, upper-cased on lookup.
    aliases: tuple[str, ...] = ()
    #: Strings OCR has actually returned for this glyph, including wrong ones.
    #: Used for normalization, never as sole evidence.
    ocr_variants: tuple[str, ...] = ()
    #: How the glyph is drawn, in words. This is the specification the
    #: procedural template in ``gdt_templates`` implements.
    visual: str = ""
    #: Name of the procedural template that draws this symbol, when one exists.
    template: str | None = None
    #: Symbols this one is genuinely mistaken for, by id or by plain name for
    #: non-GD&T confusions ("the letter O").
    confusable_with: tuple[str, ...] = ()
    #: Ordered recognition strategies.
    strategy: tuple[str, ...] = ()
    value: ValueSpec = field(default_factory=ValueSpec)
    #: Appears inside a feature control frame's first compartment.
    frame_characteristic: bool = False

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: unknown kind {self.kind!r}")
        for s in self.strategy:
            if s not in STRATEGIES:
                raise ValueError(f"{self.id}: unknown strategy {s!r}")
        if self.kind in DETECTABLE_KINDS and not self.strategy:
            raise ValueError(f"{self.id}: detectable but names no strategy")
        if self.kind == KIND_CONCEPT and self.strategy not in ((), (S_DERIVED,)):
            raise ValueError(f"{self.id}: a concept may only be derived")

    @property
    def detectable(self) -> bool:
        """True when a vision pass may legitimately report this symbol."""
        return self.kind in DETECTABLE_KINDS

    @property
    def is_legacy(self) -> bool:
        return self.standards.is_legacy

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "kind": self.kind,
            "unicode": self.unicode,
            "aliases": list(self.aliases),
            "ocr_variants": list(self.ocr_variants),
            "description": self.description,
            "visual": self.visual,
            "template": self.template,
            "confusable_with": list(self.confusable_with),
            "strategy": list(self.strategy),
            "detectable": self.detectable,
            "frame_characteristic": self.frame_characteristic,
            "value": {
                "kind": self.value.kind,
                "optional": self.value.optional,
                "pattern": self.value.pattern,
                "note": self.value.note,
            },
            "standards": {
                "standards": list(self.standards.standards),
                "retired_in": self.standards.retired_in,
                "current_in": list(self.standards.current_in),
                "exclusive_to": self.standards.exclusive_to,
                "is_legacy": self.standards.is_legacy,
                "note": self.standards.note,
            },
        }


# Shorthand so the table below reads as a table.
_ASME = "ASME Y14.5"
_ASME_2009 = "ASME Y14.5-2009"
_ASME_2018 = "ASME Y14.5-2018"
_ISO_1101 = "ISO 1101"
_ISO_GPS = "ISO GPS"
_ISO_5459 = "ISO 5459"      # datums
_ISO_286 = "ISO 286"        # envelope / fits
_ISO_2768 = "ISO 2768"
_ASME_Y14_8 = "ASME Y14.8"  # castings and forgings (parting line)

_TOL = ValueSpec(
    kind=V_TOLERANCE,
    optional=False,
    pattern=r"\d+(?:\.\d+)?",
    note="Zone width from the frame's tolerance compartment; unit is the drawing's.",
)


# ─────────────────────────────────────────────────────────────────────────────
# A. Geometric tolerance controls
# ─────────────────────────────────────────────────────────────────────────────
_GEOMETRIC: tuple[GdtSymbol, ...] = (
    GdtSymbol(
        id="straightness",
        name="Straightness",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⏤",  # ⏤
        aliases=("STRAIGHTNESS", "STRAIGHT"),
        ocr_variants=("-", "—", "–", "_", "一", "|"),
        description=(
            "How straight a line element or an axis must be. Takes no datum: "
            "a form control is self-referencing."
        ),
        visual="One horizontal stroke spanning the compartment.",
        template="straightness",
        confusable_with=("a minus sign", "a dimension line", "flatness"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="flatness",
        name="Flatness",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⏥",  # ⏥
        aliases=("FLATNESS", "FLAT"),
        ocr_variants=("/7", "□", "O", "7", "Z", "◇"),
        description=(
            "How flat a surface must be — the zone between two parallel "
            "planes. Self-contained; no datum is referenced."
        ),
        visual="A parallelogram (a leaning rectangle), wider than it is tall.",
        template="flatness",
        confusable_with=("straightness", "parallelism", "a basic-dimension box"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="circularity",
        name="Circularity",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="○",  # ○
        aliases=("CIRCULARITY", "ROUNDNESS", "ROUND"),
        ocr_variants=("O", "0", "o", "()", "Q", "C", "◯", "〇"),
        description=(
            "How round one cross-section must be, measured slice by slice. "
            "A form control: no datum."
        ),
        visual="A single open circle, unslashed, filling most of the compartment.",
        template="circularity",
        confusable_with=(
            "diameter",
            "concentricity",
            "the letter O",
            "the digit 0",
            "an ordinary drawn hole",
        ),
        strategy=(S_FRAME_SEGMENT, S_CONTOUR, S_TEMPLATE, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="cylindricity",
        name="Cylindricity",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⌭",  # ⌭
        aliases=("CYLINDRICITY", "CYLINDRICAL"),
        ocr_variants=("/O/", "O", "Ø", "101", "(O)", "@"),
        description=(
            "The whole cylindrical surface at once — roundness, straightness "
            "and taper together. A form control: no datum."
        ),
        visual="A circle crossed by two parallel slanted strokes, one each side.",
        template="cylindricity",
        confusable_with=("circularity", "diameter", "total_runout"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="profile_of_a_line",
        name="Profile of a Line",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⌒",  # ⌒
        aliases=("PROFILE OF A LINE", "LINE PROFILE", "PROFILE LINE"),
        ocr_variants=("^", "~", "n", "A", "("),
        description=(
            "A two-dimensional zone along a curved line or cross-section. "
            "May or may not reference a datum."
        ),
        visual="An open upward arc, its ends not closed.",
        template="profile_of_a_line",
        confusable_with=("arc_length", "profile_of_a_surface"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101),
            note=(
                "Shares its glyph with the arc-length symbol. Only position "
                "tells them apart: inside a frame's first compartment it is a "
                "profile control; above a dimension value it is an arc length."
            ),
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="profile_of_a_surface",
        name="Profile of a Surface",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⌓",  # ⌓
        aliases=("PROFILE OF A SURFACE", "SURFACE PROFILE", "PROFILE SURFACE"),
        ocr_variants=("^", "A", "n", "⌒", "O"),
        description=(
            "A three-dimensional zone around a surface of any shape. One of "
            "the broadest controls available; often datum-referenced."
        ),
        visual="An upward arc closed by a horizontal stroke across its ends.",
        template="profile_of_a_surface",
        confusable_with=("profile_of_a_line", "arc_length"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="angularity",
        name="Angularity",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="∠",  # ∠
        aliases=("ANGULARITY", "ANGULAR"),
        ocr_variants=("<", "/", "L", "Z", "4", "∧"),
        description=(
            "A surface or axis held at an exact angle to a datum. Requires a "
            "basic angle dimension elsewhere on the drawing."
        ),
        visual="An acute angle: one horizontal stroke and one rising stroke "
               "meeting at the left.",
        template="angularity",
        confusable_with=("the letter Z", "a less-than sign", "perpendicularity"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="perpendicularity",
        name="Perpendicularity",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⟂",  # ⟂
        aliases=("PERPENDICULARITY", "PERPENDICULAR", "SQUARENESS", "NORMALITY"),
        ocr_variants=("⊥", "_|_", "L", "T", "1", "上", "+"),
        description=(
            "A surface or axis at exactly 90° to a datum. Always datum-"
            "referenced — perpendicular to nothing is meaningless."
        ),
        visual="An upright stroke standing on the middle of a horizontal stroke.",
        template="perpendicularity",
        confusable_with=(
            "the letter T",
            "the letter L",
            "ordinary right-angle drawing geometry",
            "a witness line meeting a dimension line",
        ),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101),
            note=(
                "The most over-reported control on a drawing: right angles are "
                "everywhere in the geometry. Requires frame context."
            ),
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="parallelism",
        name="Parallelism",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="∥",  # ∥
        aliases=("PARALLELISM", "PARALLEL"),
        ocr_variants=("//", "||", "ll", "11", "I I", "/", "〃"),
        description=(
            "A surface or axis running parallel to a datum. Zero degrees is "
            "implied, so no angle dimension is needed."
        ),
        visual="Two parallel slanted strokes.",
        template="parallelism",
        confusable_with=("a double slash", "the digits 11", "hatching", "flatness"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="position",
        name="Position",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⌖",  # ⌖
        aliases=("POSITION", "TRUE POSITION", "POSITIONAL TOLERANCE", "TP"),
        ocr_variants=("+", "⊕", "(+)", "⌀", "o", "×", "X", "①"),
        description=(
            "Where a feature must sit, relative to a datum reference frame. "
            "Normally a diameter-shaped zone, and the control that most often "
            "carries MMC and so a bonus tolerance."
        ),
        visual="A circle with a cross through it, the cross arms extending past "
               "the ring.",
        template="position",
        confusable_with=("circularity", "a plus sign", "a centre mark", "concentricity"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101),
            note=(
                "Must be read from inside a feature control frame. A nearby "
                "coordinate dimension pair is not evidence of a position "
                "control, however suggestive."
            ),
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="concentricity",
        name="Concentricity",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="◎",  # ◎
        aliases=("CONCENTRICITY", "CONCENTRIC", "COAXIALITY", "ECCENTRICITY"),
        ocr_variants=("O", "0", "@", "(O)", "©", "⊙", "◉"),
        description=(
            "Whether the median points of a feature share a datum axis. Not "
            "the same as runout, and considerably harder to inspect."
        ),
        visual="Two concentric circles, the inner one roughly half the outer.",
        template="concentricity",
        confusable_with=("circularity", "symmetry", "position", "the letter O"),
        strategy=(S_FRAME_SEGMENT, S_CONTOUR, S_TEMPLATE, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME_2009, _ISO_1101),
            retired_in=_ASME_2018,
            current_in=(_ISO_1101,),
            note=(
                "Removed from ASME Y14.5-2018, which directs the use of "
                "position or runout instead. Still current in ISO, and still "
                "correct on any drawing released to an earlier ASME revision — "
                "so recognition must not be gated on the revision."
            ),
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="symmetry",
        name="Symmetry",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⌯",  # ⌯
        aliases=("SYMMETRY", "SYMMETRICAL", "SYMMETRIC"),
        ocr_variants=("=", "≡", "三", "E", "#"),
        description=(
            "Whether the median points of a feature are symmetric about a "
            "datum plane."
        ),
        visual="Three horizontal strokes, the middle one longest, flanked by "
               "two short vertical strokes.",
        template="symmetry",
        confusable_with=("an equals sign", "concentricity", "straightness"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME_2009, _ISO_1101),
            retired_in=_ASME_2018,
            current_in=(_ISO_1101,),
            note=(
                "Removed from ASME Y14.5-2018 alongside concentricity. "
                "Retained here for ISO and for legacy ASME sheets."
            ),
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="circular_runout",
        name="Circular Runout",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="↗",  # ↗
        aliases=("CIRCULAR RUNOUT", "RUNOUT", "RADIAL RUNOUT", "TIR"),
        ocr_variants=("/", "↑", "7", "1", "→", "∕"),
        description=(
            "Surface variation relative to a datum axis as the part turns, "
            "taken one cross-section at a time."
        ),
        visual="A single arrow slanting up to the right.",
        template="circular_runout",
        confusable_with=("total_runout", "a slash", "a leader arrow"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101),
            note="Single vs double arrow is the only difference from total runout.",
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
    GdtSymbol(
        id="total_runout",
        name="Total Runout",
        category=CAT_GEOMETRIC,
        kind=KIND_STANDALONE,
        unicode="⌰",  # ⌰
        aliases=("TOTAL RUNOUT",),
        ocr_variants=("//", "↑↑", "⇈", "11", "∕∕"),
        description=(
            "The entire surface relative to a datum axis in one sweep, rather "
            "than section by section."
        ),
        visual="Two parallel arrows slanting up to the right.",
        template="total_runout",
        confusable_with=("circular_runout", "parallelism", "a double slash"),
        strategy=(S_FRAME_SEGMENT, S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=_TOL,
        frame_characteristic=True,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# B. Datum and reference symbols
# ─────────────────────────────────────────────────────────────────────────────
_DATUM: tuple[GdtSymbol, ...] = (
    GdtSymbol(
        id="datum_feature_symbol",
        name="Datum Feature Symbol",
        category=CAT_DATUM,
        kind=KIND_COMPOSITE,
        unicode=None,
        aliases=("DATUM FEATURE", "DATUM",),
        ocr_variants=("[A]", "(A)", "A", "|A|", "-A-"),
        description=(
            "Labels a feature as the datum a tolerance is measured from. The "
            "whole annotation is a triangle on the feature, a short leader, "
            "and the datum letter in a box."
        ),
        visual="A filled or open triangle touching the feature, joined by a "
               "short stem to a square box holding one capital letter.",
        template="datum_triangle",
        confusable_with=("datum_target_symbol", "a section arrow", "a weld symbol"),
        strategy=(S_TEMPLATE, S_BOXED_LETTER, S_CONTOUR, S_LEADER, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_5459), current_in=(_ASME_2018, _ISO_5459),
            note=(
                "The older -A- form (letter between dashes) appears on ASME "
                "drawings predating Y14.5M-1994 and is still read here."
            ),
        ),
        value=ValueSpec(
            kind=V_LETTER,
            optional=False,
            pattern=r"[A-Z]{1,2}",
            note="I, O and Q are not used as datum letters in ASME Y14.5.",
        ),
    ),
    GdtSymbol(
        id="datum_triangle",
        name="Datum Triangle",
        category=CAT_DATUM,
        kind=KIND_STANDALONE,
        unicode="▲",  # ▲
        aliases=("DATUM TRIANGLE",),
        ocr_variants=("▲", "△", "▼", "▽", "A", "4", "<"),
        description=(
            "The triangle alone, where it is found without a readable letter "
            "box. Reported as a datum marker whose letter is unknown rather "
            "than guessed."
        ),
        visual="A small filled (ASME) or open (older ISO) triangle sitting on "
               "a surface, extension line, or dimension line.",
        template="datum_triangle",
        confusable_with=("a weld symbol", "a section-view arrow", "an arrowhead"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_LEADER),
        standards=StandardsNote(standards=(_ASME, _ISO_5459)),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="datum_reference",
        name="Datum Reference",
        category=CAT_DATUM,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("DATUM REFERENCE", "DATUM REF"),
        description=(
            "A datum letter cited inside a feature control frame. Its "
            "precedence — primary, secondary, tertiary — is given by which "
            "compartment it sits in, left to right, not by any mark on it."
        ),
        visual="One capital letter in its own compartment of a feature control "
               "frame, optionally followed by a material-condition modifier.",
        confusable_with=("datum_feature_symbol", "an ordinary letter in a note"),
        strategy=(S_FRAME_SEGMENT, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101),
            note=(
                "Precedence is positional. A letter found on its own, outside "
                "a frame, is not a datum reference and must not be given one."
            ),
        ),
        value=ValueSpec(kind=V_LETTER, optional=False, pattern=r"[A-Z]{1,2}"),
    ),
    GdtSymbol(
        id="datum_target_symbol",
        name="Datum Target Symbol",
        category=CAT_DATUM,
        kind=KIND_COMPOSITE,
        unicode=None,
        aliases=("DATUM TARGET",),
        ocr_variants=("A1", "(A1)", "O", "A-1"),
        description=(
            "Establishes a datum from specific points, lines or areas instead "
            "of a whole surface — how a casting or forging is located."
        ),
        visual="A circle divided by a horizontal line: the target size above "
               "(when the target is an area), the datum letter and target "
               "number below.",
        template="datum_target",
        confusable_with=("circularity", "concentricity", "a balloon", "all_around"),
        strategy=(S_CONTOUR, S_TEMPLATE, S_CIRCLED_LETTER, S_OCR_TEXT, S_LEADER),
        standards=StandardsNote(
            standards=(_ASME, _ISO_5459), current_in=(_ASME_2018, _ISO_5459)
        ),
        value=ValueSpec(
            kind=V_COMPOSITE,
            pattern=r"[A-Z]\d+",
            note=(
                "Lower half is datum letter plus target number (A1, B2). Upper "
                "half carries a target size only for a target area."
            ),
        ),
    ),
    GdtSymbol(
        id="datum_target_point",
        name="Datum Target Point",
        category=CAT_DATUM,
        kind=KIND_STANDALONE,
        unicode=None,
        aliases=("TARGET POINT",),
        ocr_variants=("X", "x", "+", "*"),
        description="A datum target that contacts the part at a single point.",
        visual="A cross (an X) drawn at the contact point, tied to a datum "
               "target symbol by a leader.",
        template="datum_target_point",
        confusable_with=("position", "a centre mark", "the letter X", "a multiplier ×"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_LEADER),
        standards=StandardsNote(standards=(_ASME, _ISO_5459)),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="datum_target_line",
        name="Datum Target Line",
        category=CAT_DATUM,
        kind=KIND_STANDALONE,
        unicode=None,
        aliases=("TARGET LINE",),
        description="A datum target that contacts the part along a line.",
        visual="A phantom line along the contact, or an X on an edge view, "
               "tied to a datum target symbol.",
        confusable_with=("datum_target_point", "a centre line"),
        strategy=(S_LEADER, S_CONTOUR),
        standards=StandardsNote(standards=(_ASME, _ISO_5459)),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="datum_target_area",
        name="Datum Target Area",
        category=CAT_DATUM,
        kind=KIND_STANDALONE,
        unicode=None,
        aliases=("TARGET AREA",),
        description=(
            "A datum target that contacts the part over an area, whose size is "
            "given in the upper half of the target symbol."
        ),
        visual="A section-hatched (crosshatched) outline — usually circular — "
               "bounded by a phantom line.",
        confusable_with=("a section view", "ordinary hatching"),
        strategy=(S_CONTOUR, S_LEADER),
        standards=StandardsNote(standards=(_ASME, _ISO_5459)),
        value=ValueSpec(
            kind=V_LENGTH,
            pattern=r"Ø?\d+(?:\.\d+)?",
            note="Target area size, read from the upper half of the target symbol.",
        ),
    ),
    GdtSymbol(
        id="datum_reference_frame",
        name="Datum Reference Frame",
        category=CAT_DATUM,
        kind=KIND_CONCEPT,
        unicode=None,
        aliases=("DATUM REFERENCE FRAME", "DRF"),
        description=(
            "The coordinate system the datums establish together. Not drawn: "
            "it is implied by the datum references and their precedence, and "
            "is therefore assembled from detections rather than detected."
        ),
        confusable_with=(),
        strategy=(S_DERIVED,),
        standards=StandardsNote(
            standards=(_ASME, _ISO_5459),
            note="Derived from the ordered datum references of a frame.",
        ),
        value=ValueSpec(kind=V_NONE),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# C. Material-condition modifiers and related concepts
# ─────────────────────────────────────────────────────────────────────────────
_MATERIAL: tuple[GdtSymbol, ...] = (
    GdtSymbol(
        id="mmc",
        name="Maximum Material Condition",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode="Ⓜ",  # Ⓜ
        aliases=("MMC", "MAXIMUM MATERIAL CONDITION", "MAX MATERIAL"),
        ocr_variants=("(M)", "M", "@", "(W)", "Ⓜ", "M)"),
        description=(
            "The tolerance applies when the feature holds the most material — "
            "largest shaft, smallest hole. Unlocks bonus tolerance as the "
            "feature departs from it."
        ),
        visual="A capital M inside a circle.",
        template="circled_M",
        confusable_with=("lmc", "the letter M", "an at-sign"),
        strategy=(S_FRAME_SEGMENT, S_CIRCLED_LETTER, S_TEMPLATE, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101),
            note=(
                "Must be associated with the compartment it follows — a "
                "tolerance value or a datum reference — never with the frame "
                "as a whole."
            ),
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="lmc",
        name="Least Material Condition",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode="Ⓛ",  # Ⓛ
        aliases=("LMC", "LEAST MATERIAL CONDITION", "MIN MATERIAL"),
        ocr_variants=("(L)", "L", "(1)", "Ⓛ", "L)"),
        description=(
            "The tolerance applies when the feature holds the least material — "
            "smallest shaft, largest hole. Used where wall thickness or "
            "material retention matters."
        ),
        visual="A capital L inside a circle.",
        template="circled_L",
        confusable_with=("mmc", "the letter L", "the digit 1"),
        strategy=(S_FRAME_SEGMENT, S_CIRCLED_LETTER, S_TEMPLATE, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="rfs",
        name="Regardless of Feature Size",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_CONCEPT,
        unicode=None,
        aliases=("RFS", "REGARDLESS OF FEATURE SIZE"),
        ocr_variants=("(S)", "Ⓢ"),
        description=(
            "The default when no material-condition modifier is present: the "
            "tolerance applies at any actual size, and there is no bonus "
            "tolerance. Modern ASME and ISO give it no symbol — its absence is "
            "the statement."
        ),
        confusable_with=(),
        strategy=(S_DERIVED,),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101),
            note=(
                "The circled S was used in ASME Y14.5M-1982 and earlier and is "
                "obsolete; an Ⓢ found on an old sheet is read as an explicit "
                "RFS statement. Absence of any modifier is recorded as RFS by "
                "default, which is an interpretation, not a detection."
            ),
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="independency",
        name="Independency",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode="Ⓘ",  # Ⓘ
        aliases=("INDEPENDENCY",),
        ocr_variants=("(I)", "I", "(1)", "Ⓘ", "①"),
        description=(
            "Removes the Rule #1 envelope requirement from a feature of size, "
            "so form and size are controlled independently."
        ),
        visual="A capital I inside a circle.",
        template="circled_I",
        confusable_with=("the digit 1", "the letter I", "envelope_requirement"),
        strategy=(S_CIRCLED_LETTER, S_TEMPLATE, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME_2018,),
            exclusive_to="ASME",
            current_in=(_ASME_2018,),
            note=(
                "ASME-only, introduced in Y14.5-2018. Finding it on a drawing "
                "stated to be ISO is a contradiction worth reporting, not "
                "silently normalising."
            ),
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="envelope_requirement",
        name="Envelope Requirement",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode="Ⓔ",  # Ⓔ
        aliases=("ENVELOPE REQUIREMENT", "ENVELOPE", "TAYLOR PRINCIPLE"),
        ocr_variants=("(E)", "E", "Ⓔ", "(F)"),
        description=(
            "Requires the feature's actual surface to stay inside a perfect-"
            "form envelope at MMC — what ASME Rule #1 imposes by default and "
            "ISO requires to be stated."
        ),
        visual="A capital E inside a circle.",
        template="circled_E",
        confusable_with=("the letter E", "independency"),
        strategy=(S_CIRCLED_LETTER, S_TEMPLATE, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ISO_GPS, _ISO_286),
            exclusive_to="ISO",
            current_in=(_ISO_GPS,),
            note=(
                "ISO-only. The ASME equivalent is Rule #1, which is a default "
                "rather than a mark on the drawing."
            ),
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="projected_tolerance_zone",
        name="Projected Tolerance Zone",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode="Ⓟ",  # Ⓟ
        aliases=("PROJECTED TOLERANCE ZONE", "PROJECTED TOLERANCE"),
        ocr_variants=("(P)", "P", "Ⓟ", "(R)"),
        description=(
            "Projects the tolerance zone out of the feature by a stated "
            "height — typically above a threaded or press-fit hole, where the "
            "mating fastener's attitude is what matters."
        ),
        visual="A capital P inside a circle, followed by the projection height.",
        template="circled_P",
        confusable_with=("the letter P", "diameter in the AIGDT font"),
        strategy=(S_FRAME_SEGMENT, S_CIRCLED_LETTER, S_TEMPLATE, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=ValueSpec(
            kind=V_LENGTH,
            optional=False,
            pattern=r"\d+(?:\.\d+)?",
            note=(
                "The projection height follows the Ⓟ inside the frame. AIGDT "
                "encodes Ø as the letter P, so a bare P next to a number is "
                "ambiguous between this and a diameter — resolve from frame "
                "context before deciding."
            ),
        ),
    ),
    GdtSymbol(
        id="free_state",
        name="Free State",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode="Ⓕ",  # Ⓕ
        aliases=("FREE STATE",),
        ocr_variants=("(F)", "F", "Ⓕ", "(E)"),
        description=(
            "The tolerance applies with the part unrestrained, carrying only "
            "its own weight — for parts that relax out of shape when released."
        ),
        visual="A capital F inside a circle.",
        template="circled_F",
        confusable_with=("the letter F", "envelope_requirement"),
        strategy=(S_FRAME_SEGMENT, S_CIRCLED_LETTER, S_TEMPLATE, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="restrained_condition",
        name="Restrained Condition",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_TEXTUAL,
        unicode=None,
        aliases=("RESTRAINED", "RESTRAINED CONDITION", "RESTRAINT NOTE"),
        description=(
            "States that the part is to be measured while held in a defined "
            "restraint. Written as a note, not drawn as a symbol."
        ),
        confusable_with=(),
        strategy=(S_TEXT_NOTE,),
        standards=StandardsNote(
            standards=(_ASME,),
            note="Carried by drawing notes; there is no restrained-condition glyph.",
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="tangent_plane",
        name="Tangent Plane",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode="Ⓣ",  # Ⓣ
        aliases=("TANGENT PLANE",),
        ocr_variants=("(T)", "T", "Ⓣ", "①"),
        description=(
            "Applies the tolerance to the plane tangent to the surface's high "
            "points rather than to every point of the surface, so local "
            "irregularities are ignored."
        ),
        visual="A capital T inside a circle.",
        template="circled_T",
        confusable_with=("the letter T", "perpendicularity"),
        strategy=(S_FRAME_SEGMENT, S_CIRCLED_LETTER, S_TEMPLATE, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="continuous_feature",
        name="Continuous Feature",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("CONTINUOUS FEATURE", "CF"),
        ocr_variants=("CF", "(CF)", "<CF>", "C F", "GF"),
        description=(
            "Treats surfaces interrupted by a groove, slot or step as one "
            "continuous feature under a single tolerance."
        ),
        visual="The letters CF inside a circle (ASME) or a boxed CF.",
        template="circled_CF",
        confusable_with=("an ordinary CF abbreviation in a note",),
        strategy=(S_CIRCLED_LETTER, S_OCR_TEXT, S_TEMPLATE),
        standards=StandardsNote(
            standards=(_ASME_2009, _ASME_2018), current_in=(_ASME_2018,),
            note="Two letters rather than one, so the circled-letter pass must "
                 "allow a wider ring.",
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="unequally_disposed_profile",
        name="Unequally Disposed Profile",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode="Ⓤ",  # Ⓤ
        aliases=("UNEQUALLY DISPOSED PROFILE", "UNEQUAL PROFILE", "UZ"),
        ocr_variants=("(U)", "U", "Ⓤ", "UZ", "(V)"),
        description=(
            "Shifts a profile tolerance zone to one side of the true profile "
            "by a stated amount instead of splitting it evenly."
        ),
        visual="A capital U inside a circle, followed by the outward portion "
               "of the zone. ISO writes UZ with a signed value.",
        template="circled_U",
        confusable_with=("the letter U", "the letter V"),
        strategy=(S_FRAME_SEGMENT, S_CIRCLED_LETTER, S_TEMPLATE, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME_2009, _ASME_2018, _ISO_1101),
            current_in=(_ASME_2018, _ISO_1101),
            note="ASME writes Ⓤ with the outward value; ISO writes UZ with a "
                 "signed offset. Both forms are read.",
        ),
        value=ValueSpec(
            kind=V_LENGTH,
            optional=False,
            pattern=r"[+-]?\d+(?:\.\d+)?",
            note="Outward portion of the zone (ASME) or signed offset (ISO UZ).",
        ),
    ),
    GdtSymbol(
        id="statistical_tolerance",
        name="Statistical Tolerance",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("STATISTICAL TOLERANCE", "ST"),
        ocr_variants=("(ST)", "ST", "<ST>", "5T"),
        description=(
            "Marks a tolerance that is only valid when the process is held in "
            "statistical control."
        ),
        visual="The letters ST inside a hexagon or circle.",
        template="circled_ST",
        confusable_with=("an ordinary ST abbreviation",),
        strategy=(S_CIRCLED_LETTER, S_OCR_TEXT),
        standards=StandardsNote(standards=(_ASME,), current_in=(_ASME_2018,)),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="virtual_condition",
        name="Virtual Condition",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_CONCEPT,
        unicode=None,
        aliases=("VIRTUAL CONDITION", "VC"),
        description=(
            "The worst-case boundary a feature's size and geometric tolerance "
            "produce together. Computed, never drawn — so it is reported as a "
            "derived value with its inputs named, or not at all."
        ),
        confusable_with=(),
        strategy=(S_DERIVED,),
        standards=StandardsNote(
            standards=(_ASME, _ISO_GPS),
            note="Derived from the size limits and the geometric tolerance; "
                 "requires both to be known.",
        ),
        value=ValueSpec(
            kind=V_LENGTH,
            note="Computable only when size limits and the geometric tolerance "
                 "are both read with confidence.",
        ),
    ),
    GdtSymbol(
        id="bonus_tolerance",
        name="Bonus Tolerance",
        category=CAT_MATERIAL_CONDITION,
        kind=KIND_CONCEPT,
        unicode=None,
        aliases=("BONUS TOLERANCE",),
        description=(
            "The extra positional tolerance a feature earns by departing from "
            "MMC. A consequence of the Ⓜ modifier plus the actual size — not "
            "a symbol, and never reported as one."
        ),
        confusable_with=(),
        strategy=(S_DERIVED,),
        standards=StandardsNote(
            standards=(_ASME,),
            note="Requires an MMC modifier and an as-measured size; neither is "
                 "available from the drawing alone.",
        ),
        value=ValueSpec(kind=V_LENGTH, note="Needs an as-measured feature size."),
    ),
    GdtSymbol(
        id="envelope_principle_rule_1",
        name="Rule #1 — Envelope Principle",
        category=CAT_RULE,
        kind=KIND_CONCEPT,
        unicode=None,
        aliases=("RULE 1", "RULE #1", "ENVELOPE PRINCIPLE", "TAYLOR PRINCIPLE"),
        description=(
            "The ASME default that a feature of size must have perfect form at "
            "MMC. A rule the standard imposes, not a mark on the sheet; "
            "Independency removes it and the ISO Ⓔ states it explicitly."
        ),
        confusable_with=(),
        strategy=(S_DERIVED,),
        standards=StandardsNote(
            standards=(_ASME,),
            exclusive_to="ASME",
            note=(
                "Applies by default to ASME drawings only. Asserting it over "
                "an ISO drawing needs an explicit basis such as an Ⓔ."
            ),
        ),
        value=ValueSpec(kind=V_NONE),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# D. Supporting dimensioning and manufacturing symbols
# ─────────────────────────────────────────────────────────────────────────────
_SUPPORTING: tuple[GdtSymbol, ...] = (
    GdtSymbol(
        id="feature_control_frame",
        name="Feature Control Frame",
        category=CAT_FRAME,
        kind=KIND_FRAME,
        unicode=None,
        aliases=("FEATURE CONTROL FRAME", "FCF", "CONTROL FRAME", "TOLERANCE FRAME"),
        ocr_variants=("|", "[]", "| |"),
        description=(
            "The box that houses a geometric tolerance. Read left to right: "
            "the characteristic symbol, then the tolerance (with any zone and "
            "material-condition modifiers), then the datum references in "
            "precedence order."
        ),
        visual="A rectangle divided by vertical lines into two or more "
               "compartments, the first holding a geometric characteristic "
               "symbol. Stacked rectangles form a composite frame.",
        template=None,
        confusable_with=("a basic_dimension box", "a title-block cell", "a table"),
        strategy=(S_RECTANGLE, S_CONTOUR, S_FRAME_SEGMENT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101)
        ),
        value=ValueSpec(kind=V_COMPOSITE, note="Parsed per compartment by fcf_parse."),
    ),
    GdtSymbol(
        id="basic_dimension",
        name="Basic Dimension",
        category=CAT_DIMENSIONING,
        kind=KIND_COMPOSITE,
        unicode=None,
        aliases=("BASIC DIMENSION", "BASIC", "THEORETICALLY EXACT", "TED"),
        ocr_variants=("[50]", "(50)", "□50", "|50|"),
        description=(
            "A theoretically exact dimension with no tolerance of its own — "
            "the geometric tolerances locate the feature instead."
        ),
        visual="The dimension value enclosed in a plain rectangle.",
        template=None,
        confusable_with=("feature_control_frame", "a reference dimension in brackets"),
        strategy=(S_RECTANGLE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101),
            note="ISO 1101 calls this a theoretically exact dimension (TED).",
        ),
        value=ValueSpec(
            kind=V_LENGTH, optional=False, pattern=r"\d+(?:\.\d+)?",
            note="A boxed value carries no tolerance; do not synthesise ±0.",
        ),
    ),
    GdtSymbol(
        id="diameter",
        name="Diameter",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode="⌀",  # ⌀
        aliases=("DIAMETER", "DIA", "Ø", "PHI"),
        ocr_variants=("Ø", "ø", "φ", "Φ", "∅", "O", "0", "Q", "©", "¢", "P", "@", "8"),
        description=(
            "The dimension is a diameter. Inside a feature control frame the "
            "same mark says the tolerance zone is cylindrical rather than "
            "parallel-plane."
        ),
        visual="A circle crossed by a single diagonal slash.",
        template="diameter",
        confusable_with=(
            "circularity",
            "the letter O",
            "the digit 0",
            "degree",
            "an ordinary drawn hole",
        ),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_GLYPH, S_FRAME_SEGMENT),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101, "ISO 129"),
            current_in=(_ASME_2018, _ISO_1101),
            note=(
                "AutoCAD's AIGDT font encodes Ø as the letter P; "
                "pdf_text_layer already maps that. The existing phi_detector "
                "owns the pixel decision for a dimension prefix."
            ),
        ),
        value=ValueSpec(
            kind=V_LENGTH, pattern=r"\d+(?:\.\d+)?",
            note="Prefixes the value it qualifies.",
        ),
    ),
    GdtSymbol(
        id="square_feature",
        name="Square",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode="□",  # □
        aliases=("SQUARE", "SQ"),
        ocr_variants=("□", "[]", "O", "0", "D", "口"),
        description=(
            "The feature has a square cross-section, dimensioned once rather "
            "than on both sides."
        ),
        visual="A small open square preceding the value.",
        template="square_feature",
        confusable_with=("basic_dimension", "flatness", "the digit 0"),
        strategy=(S_CONTOUR, S_TEMPLATE, S_OCR_GLYPH),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,),
            note=(
                "A square around a value is a basic dimension; a square "
                "*before* a value is a square cross-section. Position decides."
            ),
        ),
        value=ValueSpec(kind=V_LENGTH, pattern=r"\d+(?:\.\d+)?"),
    ),
    GdtSymbol(
        id="radius",
        name="Radius",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("RADIUS", "R", "RAD"),
        ocr_variants=("R", "r", "P", "B", "12", "A"),
        description=(
            "The dimension is a radius. The surface must lie between two arcs "
            "offset by the tolerance."
        ),
        visual="The capital letter R immediately before the value.",
        template=None,
        confusable_with=(
            "the letter R in ordinary text",
            "controlled_radius",
            "spherical_radius",
            "Ra surface roughness",
            "Rc hardness",
        ),
        strategy=(S_OCR_TEXT, S_LEADER),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,),
            note=(
                "A letter, not a glyph, so text context carries the whole "
                "decision: R must be followed by a number and must not be the "
                "start of RA/RZ (roughness) or RC (hardness)."
            ),
        ),
        value=ValueSpec(
            kind=V_LENGTH, optional=False, pattern=r"R\s*\d+(?:\.\d+)?",
            note="R must be immediately followed by a number to count.",
        ),
    ),
    GdtSymbol(
        id="controlled_radius",
        name="Controlled Radius",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("CONTROLLED RADIUS", "CR"),
        ocr_variants=("CR", "C R", "GR", "OR"),
        description=(
            "A radius whose surface must be a fair curve — no flats and no "
            "reversals inside the tolerance zone."
        ),
        visual="The letters CR before the value.",
        confusable_with=("radius", "an ordinary CR abbreviation"),
        strategy=(S_OCR_TEXT,),
        standards=StandardsNote(
            standards=(_ASME,), exclusive_to="ASME", current_in=(_ASME_2018,),
            note="ASME-only; ISO states the equivalent requirement in a note.",
        ),
        value=ValueSpec(kind=V_LENGTH, optional=False, pattern=r"CR\s*\d+(?:\.\d+)?"),
    ),
    GdtSymbol(
        id="spherical_radius",
        name="Spherical Radius",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("SPHERICAL RADIUS", "SR", "S R"),
        ocr_variants=("SR", "S R", "5R", "SB"),
        description="The radius of a spherical surface.",
        visual="The letters SR before the value.",
        confusable_with=("radius", "spherical_diameter"),
        strategy=(S_OCR_TEXT,),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,)
        ),
        value=ValueSpec(kind=V_LENGTH, optional=False, pattern=r"SR\s*\d+(?:\.\d+)?"),
    ),
    GdtSymbol(
        id="spherical_diameter",
        name="Spherical Diameter",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("SPHERICAL DIAMETER", "SØ", "S DIA"),
        ocr_variants=("SØ", "SO", "S0", "S⌀", "5Ø", "SP"),
        description="The diameter of a full spherical feature.",
        visual="The letter S followed by the diameter symbol, before the value.",
        template="diameter",
        confusable_with=("diameter", "spherical_radius"),
        strategy=(S_OCR_TEXT, S_TEMPLATE, S_CONTOUR),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,),
            note="The S prefix is what separates it from a plain diameter.",
        ),
        value=ValueSpec(kind=V_LENGTH, optional=False, pattern=r"S\s*[Ø⌀]\s*\d+(?:\.\d+)?"),
    ),
    GdtSymbol(
        id="depth",
        name="Depth",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode="↧",  # ↧
        aliases=("DEPTH", "DEEP", "DP", "T"),
        ocr_variants=("↓", "V", "T", "⌷", "l", "丁", "⌵"),
        description=(
            "How deep a hole, slot or counterbore goes, where the depth is not "
            "visible in the view."
        ),
        visual="A downward arrow under a short horizontal bar.",
        template="depth",
        confusable_with=("countersink", "a down arrow", "the letter V", "the letter T"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,),
            note="OCR renders the depth glyph very inconsistently; the shop "
                 "abbreviations DEEP and DP are more reliable when present.",
        ),
        value=ValueSpec(kind=V_LENGTH, optional=False, pattern=r"\d+(?:\.\d+)?"),
    ),
    GdtSymbol(
        id="counterbore",
        name="Counterbore",
        category=CAT_MANUFACTURING,
        kind=KIND_MODIFIER,
        unicode="⌴",  # ⌴
        aliases=("COUNTERBORE", "CBORE", "C'BORE", "CB"),
        ocr_variants=("⌴", "LJ", "U", "u", "凵", "[]"),
        description=(
            "A flat-bottomed enlargement above a through hole, so a fastener "
            "head sits below the surface."
        ),
        visual="A squared-off U — two uprights joined by a flat base, open at "
               "the top.",
        template="counterbore",
        confusable_with=("spotface", "the letter U", "depth"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,)
        ),
        value=ValueSpec(
            kind=V_COMPOSITE,
            optional=False,
            pattern=r"Ø?\d+(?:\.\d+)?",
            note="A counterbore callout carries its own diameter and usually a "
                 "depth; both belong to the counterbore, not to the pilot hole.",
        ),
    ),
    GdtSymbol(
        id="spotface",
        name="Spotface",
        category=CAT_MANUFACTURING,
        kind=KIND_MODIFIER,
        unicode="⌴",  # ⌴ with an SF annotation
        aliases=("SPOTFACE", "SPOT FACE", "SF"),
        ocr_variants=("SF", "⌴SF", "S F", "5F"),
        description=(
            "A very shallow counterbore that cleans up a cast or rough surface "
            "to give a flat bearing face."
        ),
        visual="The counterbore glyph annotated SF, or the letters SF alone.",
        template="counterbore",
        confusable_with=("counterbore", "a surface-finish Ra callout"),
        strategy=(S_OCR_TEXT, S_TEMPLATE, S_CONTOUR),
        standards=StandardsNote(
            standards=(_ASME,), current_in=(_ASME_2018,),
            note="Drawn with the counterbore glyph, so only the SF annotation "
                 "distinguishes it.",
        ),
        value=ValueSpec(kind=V_LENGTH, pattern=r"Ø?\d+(?:\.\d+)?"),
    ),
    GdtSymbol(
        id="countersink",
        name="Countersink",
        category=CAT_MANUFACTURING,
        kind=KIND_MODIFIER,
        unicode="⌵",  # ⌵
        aliases=("COUNTERSINK", "CSK", "C'SINK", "CS"),
        ocr_variants=("⌵", "V", "v", "\\/", "Y", "X"),
        description=(
            "A conical enlargement at a hole mouth, given as an included angle "
            "and a major diameter."
        ),
        visual="A wide shallow V, open at the top.",
        template="countersink",
        confusable_with=("depth", "the letter V", "a chamfer note"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,)
        ),
        value=ValueSpec(
            kind=V_COMPOSITE,
            optional=False,
            pattern=r"Ø?\d+(?:\.\d+)?|\d+\s*°",
            note="Carries a major diameter and an included angle (often 90° or "
                 "82°); the angle must not be read as a separate dimension.",
        ),
    ),
    GdtSymbol(
        id="dimension_origin",
        name="Dimension Origin",
        category=CAT_DIMENSIONING,
        kind=KIND_STANDALONE,
        unicode=None,
        aliases=("DIMENSION ORIGIN", "ORIGIN"),
        ocr_variants=("O", "o", "0", "•"),
        description=(
            "Says which of the two surfaces a dimension starts from, when that "
            "choice changes how the part is inspected."
        ),
        visual="A small open circle replacing the arrowhead at the origin end "
               "of the dimension line.",
        template="dimension_origin",
        confusable_with=("circularity", "a leader arrow", "the letter O"),
        strategy=(S_LEADER, S_CONTOUR, S_TEMPLATE),
        standards=StandardsNote(
            standards=(_ASME,), current_in=(_ASME_2018,),
            note="Only distinguishable from an arrowhead by looking at the "
                 "dimension line it terminates.",
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="arc_length",
        name="Arc Length",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode="⌒",  # ⌒
        aliases=("ARC LENGTH", "ARC"),
        ocr_variants=("^", "~", "(", "n", "⌒"),
        description=(
            "The dimension is measured along the arc, not across it as a chord."
        ),
        visual="An arc drawn above the dimension value.",
        template="profile_of_a_line",
        confusable_with=("profile_of_a_line", "profile_of_a_surface"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_GLYPH),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,),
            note=(
                "Identical glyph to profile of a line. Outside a feature "
                "control frame and sitting above a number, it is an arc "
                "length; inside a frame's first compartment it is a profile "
                "control. Position is the only discriminator."
            ),
        ),
        value=ValueSpec(kind=V_LENGTH, optional=False, pattern=r"\d+(?:\.\d+)?"),
    ),
    GdtSymbol(
        id="conical_taper",
        name="Conical Taper",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode="⌲",  # ⌲
        aliases=("CONICAL TAPER", "TAPER"),
        ocr_variants=("<", "⊲", "◁", "C", "4"),
        description=(
            "The rate at which a cone's diameter changes along its length, as "
            "a ratio."
        ),
        visual="A narrow horizontal cone pointing right — two converging "
               "strokes meeting at a point.",
        template="conical_taper",
        confusable_with=("slope", "angularity", "a less-than sign"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, "ISO 3040"), current_in=(_ASME_2018,)
        ),
        value=ValueSpec(
            kind=V_RATIO, optional=False, pattern=r"\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?",
            note="A ratio such as 1:5. Not a length and not an angle.",
        ),
    ),
    GdtSymbol(
        id="slope",
        name="Slope",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode="⌳",  # ⌳
        aliases=("SLOPE",),
        ocr_variants=("<", "◣", "∠", "L", "4"),
        description=(
            "The rate of change of a flat tapered surface — what conical taper "
            "is for round features."
        ),
        visual="A right triangle lying on its long side, pointing right.",
        template="slope",
        confusable_with=("conical_taper", "angularity"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME, "ISO 3040"), current_in=(_ASME_2018,)
        ),
        value=ValueSpec(
            kind=V_RATIO, optional=False, pattern=r"\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?"
        ),
    ),
    GdtSymbol(
        id="multiple_identical_features",
        name="Multiple Identical Features",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("NUMBER OF PLACES", "PLACES", "TIMES", "MULTIPLE FEATURES", "X"),
        ocr_variants=("4X", "4x", "4×", "4 X", "AX", "4*"),
        description=(
            "One callout covers several identical features: the count, the "
            "letter X, then the dimension."
        ),
        visual="A count followed by X, preceding the dimension.",
        confusable_with=("a multiplication sign in a chamfer note", "a thread pitch"),
        strategy=(S_OCR_TEXT,),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,),
            note="'2X 45°' is a chamfer, not two features — the X in a chamfer "
                 "note separates length from angle. Order decides.",
        ),
        value=ValueSpec(
            kind=V_COUNT, optional=False, pattern=r"(\d+)\s*[Xx×]",
            note="The count precedes the X.",
        ),
    ),
    GdtSymbol(
        id="all_around",
        name="All Around",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("ALL AROUND", "ALL-AROUND"),
        ocr_variants=("O", "0", "o", "Q"),
        description=(
            "The profile tolerance applies continuously around the whole "
            "outline of the view."
        ),
        visual="A small circle at the kink of the leader line.",
        template="all_around",
        confusable_with=("all_over", "circularity", "a leader kink", "a balloon"),
        strategy=(S_LEADER, S_CONTOUR, S_TEMPLATE),
        standards=StandardsNote(
            standards=(_ASME, _ISO_1101), current_in=(_ASME_2018, _ISO_1101),
            note="Only the leader it sits on distinguishes it from any other "
                 "small circle on the sheet.",
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="all_over",
        name="All Over",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode=None,
        aliases=("ALL OVER", "ALL-OVER"),
        ocr_variants=("O", "@", "◎", "(O)"),
        description=(
            "The profile tolerance applies over every surface of the part, not "
            "just the outline shown."
        ),
        visual="A double circle at the kink of the leader line.",
        template="all_over",
        confusable_with=("all_around", "concentricity", "datum_target_symbol"),
        strategy=(S_LEADER, S_CONTOUR, S_TEMPLATE),
        standards=StandardsNote(
            standards=(_ASME_2009, _ASME_2018, _ISO_1101),
            current_in=(_ASME_2018, _ISO_1101),
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="between",
        name="Between",
        category=CAT_DIMENSIONING,
        kind=KIND_MODIFIER,
        unicode="↔",  # ↔
        aliases=("BETWEEN",),
        ocr_variants=("<->", "↔", "A-B", "<>", "-"),
        description=(
            "Limits a profile tolerance to the stretch of surface between two "
            "named points."
        ),
        visual="A double-headed arrow, with the two bounding point letters "
               "either side.",
        template="between",
        confusable_with=("a dimension line", "straightness", "a leader"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_GLYPH, S_OCR_TEXT),
        standards=StandardsNote(
            standards=(_ASME_2009, _ASME_2018, _ISO_1101),
            current_in=(_ASME_2018, _ISO_1101),
            note="ASME Y14.5-1994 used a two-headed arrow between letters; "
                 "2009 onward places it inside the frame.",
        ),
        value=ValueSpec(
            kind=V_LETTER, pattern=r"[A-Z]\s*↔?\s*[A-Z]",
            note="The two bounding point identifiers.",
        ),
    ),
    GdtSymbol(
        id="parting_line",
        name="Parting Line",
        category=CAT_MANUFACTURING,
        kind=KIND_STANDALONE,
        unicode=None,
        aliases=("PARTING LINE", "PARTING SURFACE", "PL", "MOLD LINE", "DIE LINE"),
        ocr_variants=("PL", "P/L", "><", "-|-"),
        description=(
            "Where the halves of a mould or die meet on a cast or forged part."
        ),
        visual="Two arrowheads pointing away from each other on a short line "
               "crossing the part outline.",
        template="parting_line",
        confusable_with=("between", "a dimension line", "a section line"),
        strategy=(S_TEMPLATE, S_CONTOUR, S_OCR_TEXT, S_LEADER),
        standards=StandardsNote(
            standards=(_ASME_Y14_8,),
            note="Defined by ASME Y14.8 for castings and forgings rather than "
                 "by Y14.5.",
        ),
        value=ValueSpec(kind=V_NONE),
    ),
    GdtSymbol(
        id="leader_arrow",
        name="Leader and Arrow",
        category=CAT_FRAME,
        kind=KIND_COMPOSITE,
        unicode=None,
        aliases=("LEADER", "LEADER LINE", "ARROW", "CALLOUT LEADER"),
        description=(
            "Ties an annotation to the feature it controls. Not a GD&T symbol "
            "in itself, but what makes a frame's attachment knowable — and so "
            "what the detector needs to associate a frame with a feature."
        ),
        visual="A straight or kinked line from the annotation to the feature, "
               "ending in an arrowhead, a dot, or nothing.",
        confusable_with=("a dimension line", "an extension line", "a centre line"),
        strategy=(S_LEADER, S_CONTOUR),
        standards=StandardsNote(
            standards=(_ASME, "ISO 129"), current_in=(_ASME_2018,)
        ),
        value=ValueSpec(kind=V_NONE),
    ),
)


ALL_SYMBOLS: tuple[GdtSymbol, ...] = (
    _GEOMETRIC + _DATUM + _MATERIAL + _SUPPORTING
)

# ── Indexes ──────────────────────────────────────────────────────────────────


def _build_by_id() -> Mapping[str, GdtSymbol]:
    out: dict[str, GdtSymbol] = {}
    for sym in ALL_SYMBOLS:
        if sym.id in out:
            raise ValueError(f"duplicate symbol id {sym.id!r}")
        out[sym.id] = sym
    return MappingProxyType(out)


BY_ID: Mapping[str, GdtSymbol] = _build_by_id()


def _build_by_name() -> Mapping[str, GdtSymbol]:
    """Canonical name and every alias, upper-cased.

    An alias shared by two symbols is ambiguous by nature and is dropped
    rather than silently resolved to whichever entry was declared last.
    """
    # Keyed by id, so an entry whose canonical name repeats one of its own
    # aliases ("Circularity" / alias "CIRCULARITY") does not read as a clash
    # with itself. Only two *different* symbols claiming one name is ambiguous.
    seen: dict[str, dict[str, GdtSymbol]] = {}
    for sym in ALL_SYMBOLS:
        for key in (sym.name, sym.id.replace("_", " "), *sym.aliases):
            seen.setdefault(key.upper(), {})[sym.id] = sym
    return MappingProxyType(
        {
            key: next(iter(syms.values()))
            for key, syms in seen.items()
            if len(syms) == 1
        }
    )


BY_NAME: Mapping[str, GdtSymbol] = _build_by_name()

#: Aliases that more than one symbol claims. Resolving one needs context, so
#: lookups return nothing and the caller is told why.
AMBIGUOUS_NAMES: frozenset[str] = frozenset(
    key
    for key in {
        k.upper()
        for sym in ALL_SYMBOLS
        for k in (sym.name, sym.id.replace("_", " "), *sym.aliases)
    }
    if key not in BY_NAME
)


def _build_by_unicode() -> Mapping[str, tuple[GdtSymbol, ...]]:
    """Code point → the symbols drawn with it.

    A tuple, not a single entry: ⌒ is both profile-of-a-line and arc length,
    and ⌴ is both counterbore and spotface. Collapsing those to one answer is
    the mistake this index exists to prevent.
    """
    out: dict[str, list[GdtSymbol]] = {}
    for sym in ALL_SYMBOLS:
        if sym.unicode:
            out.setdefault(sym.unicode, []).append(sym)
    return MappingProxyType({k: tuple(v) for k, v in out.items()})


BY_UNICODE: Mapping[str, tuple[GdtSymbol, ...]] = _build_by_unicode()


def _build_ocr_variants() -> Mapping[str, tuple[GdtSymbol, ...]]:
    """An OCR string → every symbol known to produce it.

    Almost every entry here is multi-valued, which is the point: "O" is
    produced by circularity, diameter, concentricity, all-around and the
    letter O. A variant table is for *generating candidates*, never for
    deciding between them.
    """
    out: dict[str, list[GdtSymbol]] = {}
    for sym in ALL_SYMBOLS:
        for variant in sym.ocr_variants:
            key = variant.strip().upper()
            if key:
                out.setdefault(key, []).append(sym)
    return MappingProxyType({k: tuple(v) for k, v in out.items()})


OCR_VARIANTS: Mapping[str, tuple[GdtSymbol, ...]] = _build_ocr_variants()

#: The characteristic symbols that may open a feature control frame.
FRAME_CHARACTERISTICS: tuple[GdtSymbol, ...] = tuple(
    s for s in ALL_SYMBOLS if s.frame_characteristic
)

#: Symbols a vision pass may report.
DETECTABLE: tuple[GdtSymbol, ...] = tuple(s for s in ALL_SYMBOLS if s.detectable)

#: Concepts and textual notes — recorded, never "detected".
NON_VISUAL: tuple[GdtSymbol, ...] = tuple(s for s in ALL_SYMBOLS if not s.detectable)


# ── Lookup helpers ───────────────────────────────────────────────────────────


def get(symbol_id: str) -> GdtSymbol:
    """Look up by id. Raises :class:`KeyError` on an unknown id."""
    return BY_ID[symbol_id]


def find(symbol_id: str) -> GdtSymbol | None:
    return BY_ID.get(symbol_id)


def by_category(category: str) -> tuple[GdtSymbol, ...]:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    return tuple(s for s in ALL_SYMBOLS if s.category == category)


def by_name(name: str) -> GdtSymbol | None:
    """Resolve a canonical name, alias or abbreviation.

    Returns ``None`` for an unknown *and* for a genuinely ambiguous name; use
    :func:`candidates_for_name` when the distinction matters.
    """
    return BY_NAME.get((name or "").strip().upper())


def candidates_for_name(name: str) -> tuple[GdtSymbol, ...]:
    """Every symbol a name could mean, including the ambiguous cases."""
    key = (name or "").strip().upper()
    hit = BY_NAME.get(key)
    if hit is not None:
        return (hit,)
    return tuple(
        sym
        for sym in ALL_SYMBOLS
        if key in {sym.name.upper(), sym.id.replace("_", " ").upper()}
        or key in {a.upper() for a in sym.aliases}
    )


def candidates_for_ocr(text: str) -> tuple[GdtSymbol, ...]:
    """Symbols that could have produced this OCR output.

    Tries the literal string, then its Unicode code points. Order is the
    registry's declaration order, which is *not* a ranking — the caller ranks.
    """
    raw = (text or "").strip()
    if not raw:
        return ()
    hits: list[GdtSymbol] = []
    seen: set[str] = set()

    def add(items: Iterable[GdtSymbol]) -> None:
        for sym in items:
            if sym.id not in seen:
                seen.add(sym.id)
                hits.append(sym)

    add(OCR_VARIANTS.get(raw.upper(), ()))
    for char in raw:
        add(BY_UNICODE.get(char, ()))
    named = BY_NAME.get(raw.upper())
    if named is not None:
        add((named,))
    return tuple(hits)


def confusion_set(symbol_id: str) -> tuple[str, ...]:
    """Ids this symbol is confused with, plus the ids that name it in return.

    Confusability is symmetric in practice but easy to record on only one side,
    so the lookup closes it rather than trusting the table to be complete.
    """
    sym = BY_ID[symbol_id]
    out: list[str] = [c for c in sym.confusable_with if c in BY_ID]
    for other in ALL_SYMBOLS:
        if other.id == symbol_id:
            continue
        if symbol_id in other.confusable_with and other.id not in out:
            out.append(other.id)
    return tuple(out)


def legacy_symbols() -> tuple[GdtSymbol, ...]:
    """Controls a standard revision has retired."""
    return tuple(s for s in ALL_SYMBOLS if s.is_legacy)


def exclusive_to(family: str) -> tuple[GdtSymbol, ...]:
    """Symbols defined by only one standards family ("ASME" or "ISO")."""
    key = (family or "").strip().upper()
    return tuple(
        s for s in ALL_SYMBOLS if (s.standards.exclusive_to or "").upper() == key
    )


def describe() -> str:
    """A listing of the registry, for logs and for the API."""
    lines = [f"GD&T symbol registry — {len(ALL_SYMBOLS)} entries"]
    for category in CATEGORIES:
        members = by_category(category)
        if not members:
            continue
        lines.append(f"  {category} ({len(members)})")
        for sym in members:
            marks = []
            if not sym.detectable:
                marks.append(sym.kind)
            if sym.is_legacy:
                marks.append(f"retired in {sym.standards.retired_in}")
            if sym.standards.exclusive_to:
                marks.append(f"{sym.standards.exclusive_to}-only")
            tail = f"  [{'; '.join(marks)}]" if marks else ""
            glyph = f" {sym.unicode}" if sym.unicode else ""
            lines.append(f"    {sym.id:<34}{glyph:<3} {sym.name}{tail}")
    return "\n".join(lines)


def to_dict() -> dict[str, object]:
    """The whole registry as plain data, for the API and the frontend."""
    return {
        "version": 1,
        "categories": list(CATEGORIES),
        "kinds": list(KINDS),
        "strategies": list(STRATEGIES),
        "count": len(ALL_SYMBOLS),
        "symbols": [s.to_dict() for s in ALL_SYMBOLS],
    }
