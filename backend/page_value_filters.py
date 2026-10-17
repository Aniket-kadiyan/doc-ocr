"""Editable whole-page balloon eligibility rules.

This file is deliberately independent from detection, OCR, and annotation
code.  To change filtering, edit the enabled rules or their Python predicates
here and restart the backend.

Policy order:

1. Reject table-region content whenever the caller supplies page table masks.
2. For section scans, apply only a light complete-value syntax gate.
3. Normalize harmless OCR symbol and crop-boundary noise.
4. For whole-page scans, reject an enabled never-balloon rule.
5. Reject whole-page text with no numeric component.
6. Review isolated confusable digits and incomplete/mixed numeric text.
7. Accept complete engineering-value syntax.

Units and drawing geometry are not required.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Literal, Mapping, Sequence


BBox = Mapping[str, float]
ScanScopeKind = Literal["page", "section"]


@dataclass(frozen=True)
class PageValueCandidate:
    """One recognized whole-page candidate available to the filter."""

    text: str
    bbox: BBox
    # Context is used only for exclusion decisions. It never replaces the
    # value returned to the annotation layer.
    context_text: str = ""


@dataclass(frozen=True)
class PageValueFilterDecision:
    """Stable decision metadata returned to the page pipeline."""

    accepted: bool
    rule_name: str
    reason: str


PageRulePredicate = Callable[
    [PageValueCandidate, Sequence[PageValueCandidate]],
    bool,
]


@dataclass(frozen=True)
class PageValueFilterRule:
    """One independently switchable never-balloon rule."""

    name: str
    enabled: bool
    reason: str
    predicate: PageRulePredicate


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


DETAIL_VIEW_SECTION = _compile(r"\b(?:DETAIL|VIEW|SECTION)\b")
SCALE_LABEL = _compile(r"\bSCALE\b")
REVISION_LABEL = _compile(
    r"\b(?:REV(?:ISION)?|MODIFICATIONS?|RELEASED|"
    r"CHANGE(?:\s+(?:NO|NUMBER))?)\b"
)
NOTE_LABEL = _compile(r"\bNOTES?\b")
DOCUMENT_METADATA_LABEL = _compile(
    r"\b(?:DRAWING|DWG|DOCUMENT|DOC|PART)\s*"
    r"(?:NO\.?|NUMBER|#)\b"
    r"|\bSHEET(?:\s*(?:NO\.?|NUMBER|#))?\b"
)
DATE_VALUE = _compile(
    r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-]"
    r"(?:0?[1-9]|1[0-2])[./-](?:\d{2}|\d{4})(?!\d)"
    r"|(?<!\d)(?:19|20)\d{2}[./-](?:0?[1-9]|1[0-2])"
    r"[./-](?:0?[1-9]|[12]\d|3[01])(?!\d)"
)
NUMERIC_COMPONENT = re.compile(r"\d")
SCALE_RATIO_VALUE = _compile(r"^\s*\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\s*$")
# Deliberately case-SENSITIVE, unlike every other pattern in this file. A
# part number, a material code or a standard reference is set in capitals on
# a drawing; nothing is printed as "listed2". Compiled with IGNORECASE this
# accepted any three characters carrying one letter and one digit, which is
# what a misread word looks like — "e11" out of "ties", "Roper0" out of
# "proper", "mus1" out of "must" — and eleven of the hundred and thirteen
# values published from the benchmark sheet reached the drawing through this
# pattern alone, every one of them a word the recogniser had mangled.
COMPACT_IDENTIFIER_VALUE = re.compile(
    r"^(?=.{3,}$)(?=.*[A-Z])(?=.*\d)[A-Z0-9][A-Z0-9./_-]*$"
)
CONTEXT_IDENTIFIER_VALUE = _compile(
    r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9][A-Z0-9./_-]{5,}$"
)
SINGLE_CHARACTER_VALUE = _compile(r"^[A-Z0-9]$")
# A plain run of digits this long, with no decimal point and no symbol, is a
# document number rather than a measurement: nothing on a drawing is
# dimensioned to six significant figures. Matching only makes the value
# eligible for the nearby-label test — it still needs a DWG NO or PART NO
# printed beside it before anything is excluded.
BARE_IDENTIFIER_NUMBER = _compile(r"^\d{6,}$")
SUSPICIOUS_SINGLE_VALUE = _compile(r"^[018]$")

_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
# Degrees, then optional minutes and seconds. The minute mark is often lost:
# it is a small tick and the recogniser drops it, so "60°±0°30" arrives where
# the drawing reads "60°±0°30'". A bare 1-2 digit integer straight after the
# degree symbol is unambiguously minutes, so it is accepted without the prime —
# guarded against a following digit or decimal point so "30.0°" and a longer
# run are not swallowed.
_ANGLE_MAGNITUDE = (
    rf"{_NUMBER}\s*°(?:\s*{_NUMBER}\s*['′]"
    rf"(?:\s*{_NUMBER}\s*[\"″])?"
    rf"|\s*\d{{1,2}}(?![\d.]))?"
)
_ANGLE_VALUE = _compile(
    rf"^{_ANGLE_MAGNITUDE}"
    rf"(?:\s*±\s*(?:{_ANGLE_MAGNITUDE}|{_NUMBER}\s*°?)"
    rf"|\s*\+\s*(?:{_ANGLE_MAGNITUDE}|{_NUMBER}\s*°?)"
    rf"\s*-\s*(?:{_ANGLE_MAGNITUDE}|{_NUMBER}\s*°?))?"
    r"(?:\s*(?:MAX|MIN|TYP|REF|BASIC)\.?)?$"
)
# An ISO fit class: the tolerance-zone letter and its grade, printed straight
# after the size ("Ø4 H7", "Ø18h6", "20 JS9"). Grades run 01 to 18, which is
# what keeps a misread word with a long number behind it out.
_FIT_CLASS = r"[A-Za-z]{1,2}\s*(?:0?[1-9]|1[0-8])(?![\d.])"
# "THRU" on its own, and the "THRU ALL" that CAD packages write for a hole
# that passes through every wall it meets. Without ALL every one of them was
# rejected as an incomplete value.
_THRU = r"THRU(?:\s*ALL)?"
_LINEAR_VALUE = _compile(
    rf"^(?:\d+\s*[X×]\s*)?"
    rf"(?:SR|SØ|R|Ø)?\s*[+-]?{_NUMBER}"
    rf"(?:\s*{_FIT_CLASS})?"
    rf"(?:\s*±\s*{_NUMBER}"
    rf"|\s*\+\s*{_NUMBER}\s*/?\s*-\s*{_NUMBER}"
    # Two stacked deviations, printed one above the other on the drawing and
    # read back on one line ("Ø33 -0.05 -0.1"). Both may carry the same sign:
    # a shaft or hole limit often has upper AND lower below nominal. Listed
    # before the single-deviation branch so the longer form wins.
    #
    # The second sign is optional because a limit at nominal is printed as a
    # bare "0" ("Ø33 +0.012 / 0" — the commonest hole fit on a sheet). The
    # FIRST sign stays required, which is what keeps two fused unrelated
    # numbers from passing as a nominal and its deviation.
    rf"|\s*[+-]\s*{_NUMBER}\s*/?\s*[+-]?\s*{_NUMBER}"
    rf"|\s*[+-]\s*{_NUMBER}"
    rf"|\s*(?:/|:|\bTO\b)\s*{_NUMBER})?"
    r"(?:\s*(?:MM|CM|IN|INCH|INCHES|\"))?"
    rf"(?:\s*(?:MAX|MIN|TYP|REF|BASIC|{_THRU})\.?)?$"
)
# A chamfer: the leg length and the angle it cuts, written "1 × 45°" and
# often prefixed CHF or C. Neither grammar above covers it. The linear one
# allows a leading count ("4× Ø5") but then wants a plain number, and the
# angle one allows no prefix at all, so a correctly read chamfer failed the
# completeness test and every one on the sheet went to review instead of
# being ballooned. Both orders are standard, and the pattern is tight enough
# to be safe: it needs the × and the degree mark.
_CHAMFER_VALUE = _compile(
    rf"^(?:CHF|C)?\s*(?:"
    rf"{_NUMBER}\s*[X×]\s*{_ANGLE_MAGNITUDE}"
    rf"|{_ANGLE_MAGNITUDE}\s*[X×]\s*{_NUMBER}"
    rf")(?:\s*(?:MAX|MIN|TYP|REF|BASIC)\.?)?$"
)
# The same chamfer in the short form JIS and ISO drawings use, where the 45
# degrees is implied and only the leg is printed: "C3", "C0.5", "CHF2". The
# pattern above needs both the multiplication sign and the degree mark, so a
# perfectly read "C3" matched no grammar at all and was dropped from a page
# scan as "not one complete engineering value". The letter must carry a
# number, which is what keeps a "C" zone label or a datum letter out.
_SHORT_CHAMFER_VALUE = _compile(
    rf"^(?:CHF|C)\s*{_NUMBER}(?:\s*(?:MM|MAX|MIN|TYP|REF)\.?)?$"
)
_THREAD_VALUE = _compile(
    rf"^M\s*\d+(?:\.\d+)?"
    rf"(?:\s*[X×]\s*{_NUMBER})?"
    # A metric thread may carry a pitch class and a tolerance class ("6H",
    # "6g", "6H-6g"), so the hyphenated field repeats.
    r"(?:\s*-\s*[0-9A-Z]+)*"
    rf"(?:\s*(?:{_THRU}|TYP|REF)\.?)?$"
)
_STANDALONE_TOLERANCE = _compile(
    # A deviation pair printed one above the other and read back on one line
    # with no nominal in the crop ("+0.012 0", "+0.05 -0.05"). The first
    # member must carry an explicit sign, which is what stops a pair of plain
    # numbers from passing as a tolerance.
    rf"^(?:±|\+|-)\s*{_NUMBER}\s*°?"
    rf"(?:\s*[+-]?\s*{_NUMBER}\s*°?)?$"
)
# ── Hole callouts ───────────────────────────────────────────────────────────
#
# A hole is not dimensioned like a length. It is written as a feature term —
# a diameter or a thread — followed by the machining instructions that apply
# to it, in the compact glyph notation of ISO 129 / ASME Y14.5:
#
#     Ø2.50 ↧13                 2.5 diameter, 13 deep
#     M3x0.5 - 6H ↧3            tapped M3, thread 3 deep
#     2 x Ø2.50 THRU ALL        two holes, through every wall
#     Ø4 H7 ↧10                 reamed to an H7 fit, 10 deep
#     Ø6.50 ⌴Ø11 ↧4.40          6.5 drill, counterbored Ø11 by 4.4 deep
#     Ø3.50 ⌵Ø7 X 90°           3.5 drill, countersunk to Ø7 at 90°
#     ↧0.90                     a depth on its own, pointing at the feature
#
# None of it matched any grammar above, so on a sheet that dimensions its
# holes this way every hole callout was rejected as "not one complete
# engineering value" and went to the review queue instead of being ballooned.
# That is most of the dimensions on a typical machined part.
#
# The shape is: an optional count, one feature term, then one or more
# modifiers. At least one modifier is required — a feature term on its own is
# already a plain linear or thread value and is matched as one above.
_HOLE_COUNT = rf"(?:\d+\s*[X×]\s*)?"
_DIAMETER_TERM = rf"(?:SØ|Ø)?\s*{_NUMBER}(?:\s*{_FIT_CLASS})?"
_THREAD_TERM = (
    rf"M\s*\d+(?:\.\d+)?"
    rf"(?:\s*[X×]\s*{_NUMBER})?"
    r"(?:\s*-\s*[0-9A-Za-z]+)*"
)
_DEPTH_TERM = rf"↧\s*{_NUMBER}"
_COUNTERBORE_TERM = (
    rf"⌴\s*(?:Ø\s*)?{_NUMBER}"
    rf"(?:\s*{_DEPTH_TERM})?"
)
_COUNTERSINK_TERM = (
    rf"⌵\s*(?:Ø\s*)?{_NUMBER}"
    rf"(?:\s*[X×]\s*{_ANGLE_MAGNITUDE})?"
)
# Where the feature is cut, and the shop words for the glyphs above. "NEAR
# SIDE" / "FAR SIDE" is how a drawing says which face a blind feature enters.
_HOLE_WORD = (
    rf"(?:{_THRU}|DEEP|DP|DEPTH|CBORE|C'BORE|CSK|C'SINK|SF|SPOTFACE"
    r"|TYP|REF|MIN|MAX|BASIC|TAP|TAPPED|DRILL|REAM"
    r"|(?:NEAR|FAR)\s*SIDE|BOTH\s*SIDES)"
)
_HOLE_MODIFIER = (
    rf"(?:{_DEPTH_TERM}|{_COUNTERBORE_TERM}|{_COUNTERSINK_TERM}"
    rf"|[X×]\s*{_ANGLE_MAGNITUDE}|{_HOLE_WORD}\.?)"
)
_HOLE_SEPARATOR = r"\s*[,/]?\s*"
_ONE_HOLE_CALLOUT = (
    rf"{_HOLE_COUNT}(?:{_THREAD_TERM}|{_DIAMETER_TERM})"
    rf"(?:{_HOLE_SEPARATOR}{_HOLE_MODIFIER})+"
)
# A hole is usually specified on two printed lines — the drill on one, the
# thread or counterbore on the next — and a crop that covers both comes back
# as a single string. Two callouts are allowed for that reason; more than two
# is a clustering error, not a callout.
_HOLE_CALLOUT_VALUE = _compile(
    rf"^{_ONE_HOLE_CALLOUT}(?:{_HOLE_SEPARATOR}{_ONE_HOLE_CALLOUT})?$"
)
# A geometric tolerance and the datums it is taken from, as it is printed
# inside a feature control frame: "Ø0.02 A", "Ø0.1 A B". The authoritative
# pass reads these correctly and the grammar then threw them out, which sent
# a value the fast pass had read as "0.02A" — and which only passed at all by
# accident, through the compact-identifier escape hatch — to review instead.
#
# The Ø is required. Without it "25 A" is a length beside a section label as
# easily as it is a tolerance, and fused reads of exactly that shape are
# common. Datum letters are the ISO set: I, O and Q are never used, because
# they read as 1 and 0. Case-sensitive, unlike most of this file: a datum is
# printed as a capital, and "0.5 m" is not one.
_DATUM_LETTER = r"[A-HJ-NP-Z]"
_DIAMETRAL_TOLERANCE_WITH_DATUM = re.compile(
    rf"^Ø\s*{_NUMBER}"
    r"(?:\s*[ⓂⓁⓈ])?"
    rf"(?:\s*-?\s*{_DATUM_LETTER}\b){{1,3}}$"
)
# A depth with no feature term in the crop. It is a complete dimension on its
# own: the leader says which feature it belongs to.
_STANDALONE_DEPTH_VALUE = _compile(
    rf"^{_DEPTH_TERM}(?:\s*(?:MIN|MAX|TYP|REF)\.?)?$"
)


# Leader and extension lines touch the crop edge and read as a stray dash or
# bar, so "-" and "~" join the punctuation already treated as boundary noise.
# normalize_page_value_text only strips these when what remains is itself a
# complete value, so a real signed value like "-0.5" is untouched.
# A leader arrowhead touching the crop reads as an arrow glyph. Arrows are
# never part of a value so they are stripped from the front; a leading "-" is
# deliberately NOT stripped, because it is the sign of a real value.
# U+21A7 is carved out of the arrow range: by this point _normalize_symbols
# has folded the depth glyph onto it, and "↧0.90" is a whole callout — the
# depth IS the dimension there. Stripping it published a bare "0.90" and
# threw away the only thing that said what the number measures.
_BOUNDARY_NOISE_START = re.compile(
    r"^[?¦|;,\u2190-\u21A6\u21A8-\u21FF\u25B6\u25C0]+\s*"
)
# ")" joins the trailing noise because a reference dimension's opening
# bracket is routinely read as something else — "(20.86)" comes back
# "Ø20.86)" — and the orphaned bracket then failed every grammar. Stripping
# is still only kept when what remains is itself a complete value, so a
# balanced "(25)" and a dual-unit "1.00[25.40]" are untouched: removing their
# closing bracket leaves something incomplete and the strip is discarded.
_BOUNDARY_NOISE_END = re.compile(r"\s*[?¦|;,)\-\u2010-\u2015~]+$")
_KEYWORD_TOKEN = re.compile(r"[A-Z0-9]+")
_KEYWORD_CONFUSABLES = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "3": "E",
        "4": "A",
        "5": "S",
        "7": "T",
    }
)


def _normalize_symbols(text: str) -> str:
    normalized = str(text or "")
    normalized = re.sub(r"\+\s*/\s*[-−]", "±", normalized)
    normalized = re.sub(r"\+\s*[-−](?=\s*\d)", "±", normalized)
    normalized = (
        normalized.replace("º", "°")
        .replace("˚", "°")
        .replace("⁰", "°")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("‘", "′")
        .replace("’", "′")
        .replace("`", "′")
        .replace("“", "″")
        .replace("”", "″")
        .replace("ø", "Ø")
        .replace("φ", "Ø")
        .replace("ϕ", "Ø")
        .replace("Φ", "Ø")
        .replace("⌀", "Ø")
        # U+2205 EMPTY SET is what PaddleOCR actually emits for a drawing's Ø:
        # it is in the recogniser's charset and U+2300 DIAMETER SIGN is not.
        # Only the LEADING one was being folded (symbol_normalize), so a
        # counted callout — "2 x ∅2.50 ↧8.50" — kept a diameter glyph that no
        # grammar here recognises and was rejected whole.
        .replace("∅", "Ø")
    )
    normalized = canonical_hole_glyphs(normalized)
    return " ".join(normalized.strip().split())


# The ISO depth glyph (↧ U+21A7) is not in PaddleOCR's charset. Measured
# against CAD-rendered callouts, the recogniser returns U+2193 DOWNWARDS ARROW
# for it on every read it does not drop entirely — "Ø2.50 ↧13" comes back
# "2.50↓13" — so the down-arrow family is folded onto the canonical glyph.
#
# Only an arrow that leads straight into a number is folded. That is the one
# shape a leader arrowhead intruding on a crop does not have: an arrowhead is
# a filled triangle (handled separately as boundary noise) and carries no
# number behind it, while a depth callout is always the glyph and its depth.
_DEPTH_GLYPH_SOURCES = "↧↓⇓⇩↡⤓⍗⌄"
_COUNTERBORE_GLYPH_SOURCES = "⌴⊔"
_COUNTERSINK_GLYPH_SOURCES = "⌵∨"
DEPTH_GLYPH = "↧"
COUNTERBORE_GLYPH = "⌴"
COUNTERSINK_GLYPH = "⌵"
_DEPTH_GLYPH_RE = re.compile(rf"[{_DEPTH_GLYPH_SOURCES}]\s*(?=[\d.])")
_COUNTERBORE_GLYPH_RE = re.compile(
    rf"[{_COUNTERBORE_GLYPH_SOURCES}]\s*(?=[Ø\d.])"
)
_COUNTERSINK_GLYPH_RE = re.compile(
    rf"[{_COUNTERSINK_GLYPH_SOURCES}]\s*(?=[Ø\d.])"
)


def canonical_hole_glyphs(text: str) -> str:
    """Fold the hole-callout glyphs OCR renders inconsistently.

    Public so the OCR cleanup layers can apply it as early as possible: a
    candidate scored and voted on while it still carries a raw down-arrow is
    scored as if it carried a stray letter.
    """

    out = _DEPTH_GLYPH_RE.sub(DEPTH_GLYPH, text)
    out = _COUNTERBORE_GLYPH_RE.sub(COUNTERBORE_GLYPH, out)
    return _COUNTERSINK_GLYPH_RE.sub(COUNTERSINK_GLYPH, out)


def _unwrapped_value(text: str) -> str:
    if len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


# A dual-dimensioned callout: the primary value followed by its converted
# equivalent in brackets ("2.25[57.15]", "Ø0.33[8.38]", "1.13[28.70]REF.").
# Drawings that carry INCHES[MILLIMETERS] in the title block write nearly every
# dimension this way, and without this rule the grammar below rejects all of
# them as incomplete.
_DUAL_UNIT_VALUE = _compile(
    r"^(?P<primary>[^\[\]]+?)"
    r"\s*\[\s*(?P<secondary>[^\[\]]+?)\s*\]"
    r"\s*(?P<suffix>REF|BASIC|MAX|MIN|TYP|THRU)?\.?$"
)


def _is_single_value(value: str) -> bool:
    """One value on its own, by the grammar rules below."""
    return bool(
        _ANGLE_VALUE.fullmatch(value)
        or _LINEAR_VALUE.fullmatch(value)
        or _THREAD_VALUE.fullmatch(value)
        or _STANDALONE_TOLERANCE.fullmatch(value)
        or _SHORT_CHAMFER_VALUE.fullmatch(value)
        or _HOLE_CALLOUT_VALUE.fullmatch(value)
        or _STANDALONE_DEPTH_VALUE.fullmatch(value)
    )


def _is_dual_unit_value(value: str) -> bool:
    """
    A primary value with its bracketed unit conversion.

    Both halves have to be complete values in their own right, which is what
    keeps two callouts fused by a clustering error ("1.00[25.40]0.25[6.35]")
    from passing: their combined text is not primary + one bracket + suffix.
    """
    match = _DUAL_UNIT_VALUE.fullmatch(value)
    if not match:
        return False
    primary = match.group("primary").strip()
    secondary = match.group("secondary").strip()
    if not primary or not secondary:
        return False
    return _is_single_value(primary) and _is_single_value(secondary)


def is_complete_engineering_value(text: str) -> bool:
    """Recognize a complete value without extracting digits from prose."""

    value = _unwrapped_value(text)
    if not value or not NUMERIC_COMPONENT.search(value):
        return False
    if SCALE_RATIO_VALUE.fullmatch(value):
        return True
    if _ANGLE_VALUE.fullmatch(value):
        return True
    if _LINEAR_VALUE.fullmatch(value):
        return True
    if _CHAMFER_VALUE.fullmatch(value):
        return True
    if _SHORT_CHAMFER_VALUE.fullmatch(value):
        return True
    if _THREAD_VALUE.fullmatch(value):
        return True
    if _STANDALONE_TOLERANCE.fullmatch(value):
        return True
    if _HOLE_CALLOUT_VALUE.fullmatch(value):
        return True
    if _STANDALONE_DEPTH_VALUE.fullmatch(value):
        return True
    if _DIAMETRAL_TOLERANCE_WITH_DATUM.fullmatch(value):
        return True
    if _is_dual_unit_value(value):
        return True
    return bool(COMPACT_IDENTIFIER_VALUE.fullmatch(value))


# The depth glyph is a bar, a stem and an arrowhead, so a recogniser that
# cannot place it reaches for the letter it looks most like. Measured over
# CAD-rendered callouts at 26/34/44px, the substitution it makes is "T"
# ("M3x0.5-6H \u21a74.50" comes back "M3x0.5-6H T4.50").
#
# Folding a T onto the glyph on sight would be reckless — a part number
# carries capitals next to digits. Two guards keep it honest:
#
#   * the letter must stand apart. A fit class is set tight against the size
#     it qualifies ("Ø50T7", "ISD-600T500"); a depth glyph has a space in
#     front of it, which is where the recogniser puts the substitution.
#   * the repaired string must parse as a whole hole callout. That is what
#     stops it anywhere else: "PART T5" repairs to "PART↧5", which is not a
#     callout, so the repair is discarded and the text is left alone.
_DEPTH_LETTER_RE = re.compile(r"(?<=[\dA-Za-z])\s+[T⊤Ŧ]\s*(?=[\d.])")


def _repaired_depth_letter(value: str) -> str | None:
    """A hole callout whose depth glyph was read as a letter, or None."""

    if not _DEPTH_LETTER_RE.search(value):
        return None
    repaired = _DEPTH_LETTER_RE.sub(DEPTH_GLYPH, value)
    if repaired == value or not _HOLE_CALLOUT_VALUE.fullmatch(repaired):
        return None
    return repaired


def normalize_page_value_text(text: str) -> str:
    """Normalize safe OCR variants while preserving the technical value.

    Crop-boundary punctuation is removed only when the remaining complete
    string satisfies the engineering grammar.  This prevents the filter from
    extracting a convenient number from a note or title-block sentence.
    """

    normalized = _normalize_symbols(text)
    stripped = _BOUNDARY_NOISE_START.sub("", normalized)
    stripped = _BOUNDARY_NOISE_END.sub("", stripped).strip()
    if stripped != normalized and is_complete_engineering_value(stripped):
        return stripped
    repaired = _repaired_depth_letter(normalized)
    if repaired is not None:
        return repaired
    return normalized


def _canonical_keyword_text(text: str) -> str:
    """Canonicalize OCR-confusable word tokens for exclusion matching only."""

    tokens = []
    for token in _KEYWORD_TOKEN.findall(str(text or "").upper()):
        tokens.append(
            token.translate(_KEYWORD_CONFUSABLES)
            if any(character.isalpha() for character in token)
            else token
        )
    return " ".join(tokens)


def _contains_direct_exclusion_keyword(text: str) -> bool:
    keyword_text = _canonical_keyword_text(text)
    return any(
        pattern.search(keyword_text)
        for pattern in (
            DETAIL_VIEW_SECTION,
            SCALE_LABEL,
            REVISION_LABEL,
            NOTE_LABEL,
            DOCUMENT_METADATA_LABEL,
        )
    )


def _candidate_search_text(candidate: PageValueCandidate) -> str:
    return " ".join(
        part
        for part in (candidate.text.strip(), candidate.context_text.strip())
        if part
    )


def _candidate_keyword_text(candidate: PageValueCandidate) -> str:
    return _canonical_keyword_text(_candidate_search_text(candidate))


def _is_compact_identifier(text: str) -> bool:
    return bool(
        COMPACT_IDENTIFIER_VALUE.fullmatch(text)
        and not _ANGLE_VALUE.fullmatch(_unwrapped_value(text))
        and not _LINEAR_VALUE.fullmatch(_unwrapped_value(text))
        and not _THREAD_VALUE.fullmatch(_unwrapped_value(text))
    )


def _is_context_exclusion_prone(text: str) -> bool:
    """Limit nearby labels to values likely to be metadata or callouts."""

    normalized = normalize_page_value_text(text)
    return bool(
        SCALE_RATIO_VALUE.fullmatch(normalized)
        or _is_compact_identifier(normalized)
        or SINGLE_CHARACTER_VALUE.fullmatch(normalized)
        or BARE_IDENTIFIER_NUMBER.fullmatch(normalized)
        or not is_complete_engineering_value(normalized)
    )


def needs_expanded_filter_context(text: str, bbox: BBox) -> bool:
    """Limit extra context OCR to values prone to metadata false positives."""

    normalized = normalize_page_value_text(text)
    if not normalized:
        return False
    if _contains_direct_exclusion_keyword(normalized):
        return False
    height = max(float(bbox.get("height", 0.0)), 1.0)
    width = max(float(bbox.get("width", 0.0)), 1.0)
    return bool(
        SCALE_RATIO_VALUE.fullmatch(normalized)
        or DATE_VALUE.search(normalized)
        or CONTEXT_IDENTIFIER_VALUE.fullmatch(normalized)
        or (len(normalized) <= 2 and width / height >= 8.0)
    )


# Longest a token may be and still be read as a piece of a view caption.
# A caption the detector split comes back as "C", "A-A", "(3x)", "1:1" or
# "B(3x)"; five characters covers all of them.
_CAPTION_FRAGMENT_MAX_CHARS = 5
# ...or the caption's ratio, however it is spaced and whatever it is glued to.
# "SCALE 1 : 1" comes back as the single token "Scale1:1", which the SCALE
# word pattern cannot see (it needs a word boundary) and which is too long to
# be a fragment, so without this it reached the review queue as a value.
_CAPTION_RATIO = re.compile(r"(?<!\d)\d{1,3}\s*:\s*\d{1,3}(?!\d)")


def _is_caption_fragment(text: str) -> bool:
    """Whether a token could be part of a split view or scale caption."""

    compact = "".join(str(text or "").split())
    if len(compact) <= _CAPTION_FRAGMENT_MAX_CHARS:
        return True
    return bool(_CAPTION_RATIO.search(compact))


def _matches_text_or_nearby_label(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
    pattern: re.Pattern[str],
    *,
    nearby_only_caption_fragments: bool = False,
) -> bool:
    if pattern.search(_canonical_keyword_text(candidate.text)):
        return True
    # A view caption is a heading over a whole region, not a field label
    # beside its value, and the nearby-label radius is sized for the latter:
    # ~240px at 250dpi. So every damaged dimension inside a detail view was
    # being deleted by the caption above it — "Ø20.6 -0.1" read as
    # "20.6~0.1" and "Ø20.86 +0.01" read as "20.86=0.01" both vanished
    # under "Detail C", with no balloon and no review box, because this rule
    # is a hard exclusion. Only something short enough to be a piece of the
    # caption itself may be taken by proximity to one.
    # The candidate's reconstructed line context is the same kind of evidence
    # as a neighbouring box, and it reaches further: "R212.5" was excluded
    # because the context rebuilt for it swept up "SECTION VIEW B-B" from the
    # line above. It goes behind the same guard.
    if nearby_only_caption_fragments and not _is_caption_fragment(candidate.text):
        return False
    if pattern.search(_candidate_keyword_text(candidate)):
        return True
    if not _is_context_exclusion_prone(candidate.text):
        return False
    return any(
        other is not candidate
        and pattern.search(_canonical_keyword_text(other.text))
        and _is_nearby_label(candidate.bbox, other.bbox)
        for other in page_candidates
    )


def _axis_overlap(
    first_start: float,
    first_length: float,
    second_start: float,
    second_length: float,
) -> float:
    end = min(first_start + first_length, second_start + second_length)
    start = max(first_start, second_start)
    return max(0.0, end - start)


def _axis_gap(
    first_start: float,
    first_length: float,
    second_start: float,
    second_length: float,
) -> float:
    first_end = first_start + first_length
    second_end = second_start + second_length
    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0.0


def _is_nearby_label(value_box: BBox, label_box: BBox) -> bool:
    """Conservatively associate a split metadata label and value.

    Labels in drawing title blocks normally sit on the same row or immediately
    above their value.  The limits scale with detected text height so the rule
    behaves consistently across render resolutions.  A distant matching word
    elsewhere on the page cannot exclude a technical value.
    """

    vx = float(value_box["x"])
    vy = float(value_box["y"])
    vw = max(float(value_box["width"]), 1.0)
    vh = max(float(value_box["height"]), 1.0)
    lx = float(label_box["x"])
    ly = float(label_box["y"])
    lw = max(float(label_box["width"]), 1.0)
    lh = max(float(label_box["height"]), 1.0)
    text_height = max(vh, lh)

    vertical_overlap = _axis_overlap(vy, vh, ly, lh)
    horizontal_overlap = _axis_overlap(vx, vw, lx, lw)
    horizontal_gap = _axis_gap(vx, vw, lx, lw)
    vertical_gap = _axis_gap(vy, vh, ly, lh)

    same_row = (
        vertical_overlap / min(vh, lh) >= 0.35
        and horizontal_gap <= max(80.0, 8.0 * text_height)
    )
    stacked_field = (
        horizontal_overlap / min(vw, lw) >= 0.20
        and vertical_gap <= max(50.0, 4.0 * text_height)
    )
    return same_row or stacked_field


def _detail_view_section(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        DETAIL_VIEW_SECTION,
        nearby_only_caption_fragments=True,
    )


def _scale_information(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        SCALE_LABEL,
        # "SCALE 5:1" is printed under the detail caption it belongs to, so
        # it covers the same view and needs the same guard.
        nearby_only_caption_fragments=True,
    )


def _date_value(
    candidate: PageValueCandidate,
    _page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return bool(DATE_VALUE.search(_candidate_search_text(candidate)))


def _revision_history(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        REVISION_LABEL,
    )


def _note_information(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        NOTE_LABEL,
    )


def _document_metadata(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    return _matches_text_or_nearby_label(
        candidate,
        page_candidates,
        DOCUMENT_METADATA_LABEL,
    )


# What makes one box a line of type: much wider than tall, and carrying a
# run of smaller boxes along it. The fragment count is what separates a line
# from a callout the detector split in two — a letter-spaced heading returns
# twenty-five pieces, a fused pair of dimensions returns two.
TEXT_LINE_MIN_ASPECT = 8.0
TEXT_LINE_MIN_FRAGMENTS = 6
TEXT_LINE_MAX_FRAGMENT_SHARE = 0.5

# What puts a box on that line: nearly all of its width inside the line's,
# and its middle within the line's band. Measured with slack above and below
# rather than by containment, because the line box is tight to the x-height
# and an ascender or a descender stands outside it.
TEXT_LINE_MIN_HORIZONTAL_COVER = 0.9
TEXT_LINE_BASELINE_SLACK = 0.6

# A pathological page of long overlapping boxes would otherwise make this
# quadratic in the page's candidate count, per candidate.
TEXT_LINE_MAX_LINES_CONSIDERED = 4


def _bbox_area(box: BBox) -> float:
    return max(float(box["width"]) * float(box["height"]), 1.0)


def _is_line_shaped(box: BBox) -> bool:
    height = float(box["height"])
    return height > 0 and float(box["width"]) / height >= TEXT_LINE_MIN_ASPECT


def _sits_on_line(fragment: BBox, line: BBox) -> bool:
    """Whether a box is one of the characters printed along a line."""

    if _bbox_area(fragment) > _bbox_area(line) * TEXT_LINE_MAX_FRAGMENT_SHARE:
        return False
    fragment_left = float(fragment["x"])
    fragment_right = fragment_left + float(fragment["width"])
    line_left = float(line["x"])
    line_right = line_left + float(line["width"])
    covered = max(0.0, min(fragment_right, line_right) - max(fragment_left, line_left))
    if covered / max(float(fragment["width"]), 1.0) < TEXT_LINE_MIN_HORIZONTAL_COVER:
        return False
    middle = float(fragment["y"]) + float(fragment["height"]) / 2
    slack = TEXT_LINE_BASELINE_SLACK * float(line["height"])
    return (
        float(line["y"]) - slack
        <= middle
        <= float(line["y"]) + float(line["height"]) + slack
    )


def _text_line_fragment(
    candidate: PageValueCandidate,
    page_candidates: Sequence[PageValueCandidate],
) -> bool:
    """Reject a piece of a line of type that another pass detected whole.

    A whole-page scan inspects each panel upright and turned a quarter turn.
    A printed line is one box to the upright pass and, to the turned one,
    nothing it can join into a line at all — so it comes back as a box per
    letter or per syllable, and a drawing office that letter-spaces its
    headings ("G u a r a n t e e d  M e c h a n i c a l  P r o p e r t i e s")
    gives the turned pass nothing else to do. On the benchmark sheet one
    panel returned five boxes upright over that block and thirty-seven
    turned. Those pieces are then read on their own, at a size no recogniser
    can do anything with, and whatever digit comes back is published:
    "proper" came back "Roper0", "listed" came back "listed2", one letter of
    "properties" came back "031". Deduplication cannot help, because it
    refuses on principle to let a large box suppress a small one — a
    dimension string must never be swallowed by the note above it.

    The test is deliberately blind to what anything reads as. The obvious
    rule — find the box some pass read as a sentence — does not work, and was
    measured not to: the preliminary pass reads a crop whole, and handed a
    657-pixel line of letter-spaced hairline type it returns an empty string.
    Every line on the benchmark sheet came back blank, so there was no
    sentence anywhere to recognise. The geometry survives that: a long thin
    box with two dozen small boxes strung along it is a line of type whether
    or not anyone managed to read it.
    """

    lines = 0
    for other in page_candidates:
        if other is candidate:
            continue
        if not _is_line_shaped(other.bbox):
            continue
        if not _sits_on_line(candidate.bbox, other.bbox):
            continue
        lines += 1
        fragments = sum(
            1
            for item in page_candidates
            if item is not other and _sits_on_line(item.bbox, other.bbox)
        )
        if fragments >= TEXT_LINE_MIN_FRAGMENTS:
            return True
        if lines >= TEXT_LINE_MAX_LINES_CONSIDERED:
            break
    return False


# Edit this ordered tuple to change whole-page exclusions.  Earlier enabled
# rules win, making behavior deterministic and easy to audit in result counts.
NEVER_BALLOON_RULES: tuple[PageValueFilterRule, ...] = (
    PageValueFilterRule(
        name="detail_view_section",
        enabled=True,
        reason="Detail, view, or section label",
        predicate=_detail_view_section,
    ),
    PageValueFilterRule(
        name="scale_information",
        enabled=True,
        reason="Drawing scale information",
        predicate=_scale_information,
    ),
    PageValueFilterRule(
        name="date",
        enabled=True,
        reason="Date value",
        predicate=_date_value,
    ),
    PageValueFilterRule(
        name="revision_history",
        enabled=True,
        reason="Revision or change-history entry",
        predicate=_revision_history,
    ),
    PageValueFilterRule(
        name="note_information",
        enabled=True,
        reason="Drawing note or note-associated value",
        predicate=_note_information,
    ),
    PageValueFilterRule(
        name="document_metadata",
        enabled=True,
        reason="Drawing, document, part, or sheet metadata",
        predicate=_document_metadata,
    ),
    PageValueFilterRule(
        name="text_line_fragment",
        enabled=True,
        reason="Piece of a line of type detected whole by another pass",
        predicate=_text_line_fragment,
    ),
)

REQUIRE_NUMERIC_COMPONENT = True

# Table policy. Whole-page scans supply masks; raw section scans deliberately
# do not. Manual Draw Value OCR never uses this policy.
EXCLUDE_TABLE_REGIONS = True
TABLE_OVERLAP_REJECTION_RATIO = 0.50
TABLE_CENTER_REJECTION = True


def _bbox_intersection_area(left: BBox, right: BBox) -> float:
    x0 = max(float(left["x"]), float(right["x"]))
    y0 = max(float(left["y"]), float(right["y"]))
    x1 = min(
        float(left["x"]) + float(left["width"]),
        float(right["x"]) + float(right["width"]),
    )
    y1 = min(
        float(left["y"]) + float(left["height"]),
        float(right["y"]) + float(right["height"]),
    )
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def candidate_is_in_table(
    bbox: BBox,
    table_masks: Sequence[BBox],
    *,
    overlap_rejection_ratio: float = TABLE_OVERLAP_REJECTION_RATIO,
    center_rejection: bool = TABLE_CENTER_REJECTION,
) -> bool:
    """Reject a candidate, not its complete mixed-content scan section."""

    width = max(float(bbox["width"]), 1.0)
    height = max(float(bbox["height"]), 1.0)
    area = width * height
    center_x = float(bbox["x"]) + width / 2
    center_y = float(bbox["y"]) + height / 2
    for mask in table_masks:
        center_inside = (
            float(mask["x"]) <= center_x <= float(mask["x"]) + float(mask["width"])
            and float(mask["y"]) <= center_y <= float(mask["y"]) + float(mask["height"])
        )
        if center_rejection and center_inside:
            return True
        if _bbox_intersection_area(bbox, mask) / area >= overlap_rejection_ratio:
            return True
    return False


# A drawing sheet's frame carries zone labels — single digits along the top and
# bottom, letters up the sides — plus trim and fold marks. They are not
# dimensions, but they read as clean short values and get ballooned. On a real
# sheet every one sat within 4.3% of an edge while the nearest true dimension
# was at 15.4%, so a conservative band with a length guard separates them
# cleanly: a zone label is one or two characters, so nothing like "12.0" or
# "R1.4" can be caught by this even if a drawing puts one near the border.
_FRAME_BAND_FRACTION = 0.055
_ZONE_LABEL_MAX_CHARS = 2


def is_sheet_frame_label(bbox: BBox, page_size: tuple[int, int] | None, text: str) -> bool:
    """True for a short token sitting in the sheet's border band."""
    if not page_size:
        return False
    width, height = page_size
    if width <= 0 or height <= 0:
        return False
    compact = "".join(str(text or "").split())
    if not compact or len(compact) > _ZONE_LABEL_MAX_CHARS:
        return False
    centre_x = float(bbox["x"]) + float(bbox["width"]) / 2.0
    centre_y = float(bbox["y"]) + float(bbox["height"]) / 2.0
    edge_distance = min(
        centre_x / width,
        centre_y / height,
        (width - centre_x) / width,
        (height - centre_y) / height,
    )
    return edge_distance <= _FRAME_BAND_FRACTION


def evaluate_scan_value(
    candidate: PageValueCandidate,
    *,
    scope_kind: ScanScopeKind,
    table_masks: Sequence[BBox] = (),
    page_size: tuple[int, int] | None = None,
    page_candidates: Sequence[PageValueCandidate] = (),
    rules: Sequence[PageValueFilterRule] = NEVER_BALLOON_RULES,
    require_numeric_component: bool = REQUIRE_NUMERIC_COMPONENT,
) -> PageValueFilterDecision:
    """Apply the global table rule, then the scope-specific value policy."""

    normalized = PageValueCandidate(
        text=normalize_page_value_text(candidate.text),
        bbox=candidate.bbox,
        context_text=_normalize_symbols(candidate.context_text),
    )
    if EXCLUDE_TABLE_REGIONS and candidate_is_in_table(
        normalized.bbox,
        table_masks,
    ):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="table_region",
            reason="Candidate lies inside a detected table region",
        )

    if is_sheet_frame_label(normalized.bbox, page_size, normalized.text):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="sheet_frame_label",
            reason="Short token inside the drawing frame's zone-label band",
        )

    if scope_kind == "section":
        if not NUMERIC_COMPONENT.search(normalized.text):
            return PageValueFilterDecision(
                accepted=False,
                rule_name="section_no_numeric_component",
                reason="Section text has no numeric component",
            )
        if not is_complete_engineering_value(normalized.text):
            return PageValueFilterDecision(
                accepted=False,
                rule_name="section_invalid_engineering_value",
                reason="Section text is not one complete engineering value",
            )
        return PageValueFilterDecision(
            accepted=True,
            rule_name="section_engineering_value",
            reason="Matches the light section engineering-value syntax gate",
        )

    context = tuple(
        normalized if item is candidate else item
        for item in page_candidates
    )
    for rule in rules:
        if rule.enabled and rule.predicate(normalized, context):
            return PageValueFilterDecision(
                accepted=False,
                rule_name=rule.name,
                reason=rule.reason,
            )

    if require_numeric_component and not NUMERIC_COMPONENT.search(normalized.text):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="no_numeric_component",
            reason="Recognized text has no numeric component",
        )

    if not require_numeric_component and not NUMERIC_COMPONENT.search(
        normalized.text
    ):
        return PageValueFilterDecision(
            accepted=True,
            rule_name="numeric_requirement_disabled",
            reason="Numeric-component requirement is disabled",
        )

    if SUSPICIOUS_SINGLE_VALUE.fullmatch(normalized.text):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="ambiguous_single_character",
            reason="Isolated 0, 1, or 8 may be an OCR-confused drawing label",
        )

    if not is_complete_engineering_value(normalized.text):
        return PageValueFilterDecision(
            accepted=False,
            rule_name="invalid_engineering_value",
            reason=(
                "Numeric text does not form one complete engineering value"
            ),
        )

    return PageValueFilterDecision(
        accepted=True,
        rule_name="engineering_value",
        reason="Matches complete engineering-value syntax",
    )


def evaluate_page_value(
    candidate: PageValueCandidate,
    *,
    page_candidates: Sequence[PageValueCandidate] = (),
    table_masks: Sequence[BBox] = (),
    page_size: tuple[int, int] | None = None,
    rules: Sequence[PageValueFilterRule] = NEVER_BALLOON_RULES,
    require_numeric_component: bool = REQUIRE_NUMERIC_COMPONENT,
) -> PageValueFilterDecision:
    """Evaluate one recognized value using the editable page policy."""

    return evaluate_scan_value(
        candidate,
        scope_kind="page",
        table_masks=table_masks,
        page_size=page_size,
        page_candidates=page_candidates,
        rules=rules,
        require_numeric_component=require_numeric_component,
    )
