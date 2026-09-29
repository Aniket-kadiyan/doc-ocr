"""
Feature dictionary for the GD&T ballooning rule engine.

Standardized symbols, abbreviations, keywords and regex patterns drawn from the
GDTOCR spec (ISO / ASME Y14.5 and common CAD abbreviations). The classifier in
``feature_classifier.py`` consumes these tables; keeping the vocabulary here
(instead of inline in the engine) makes it cheap to extend toward the
200-300 term dictionary the spec recommends.

Every keyword is upper-cased and matched against upper-cased OCR text so callers
never worry about case. Regexes are pre-compiled and case-insensitive.
"""

from __future__ import annotations

import re

# ── Category labels ──────────────────────────────────────────────────────────
# These strings are the single source of truth shared with the frontend
# DimensionType union (src/types/annotation.ts). Keep the two in sync.
CAT_DIAMETER = "Diameter"
CAT_RADIUS = "Radius"
CAT_CHAMFER = "Chamfer"
CAT_ANGLE = "Angle"
CAT_LINEAR = "Linear"
CAT_TOLERANCE = "Tolerance"
CAT_THREAD = "Thread"
CAT_HOLE = "Hole"
CAT_GDT = "GD&T"
CAT_DATUM = "Datum"
CAT_SURFACE = "Surface Finish"
CAT_WELD = "Weld"
CAT_MATERIAL = "Material"
CAT_HEAT = "Heat Treatment"
CAT_COATING = "Coating"
CAT_TAPER = "Taper"
CAT_NOTE = "General Note"
CAT_REFERENCE = "Reference"
# Title-block fields pulled by the user's configured keywords (DWG NO., REV…).
CAT_TITLE_BLOCK = "Title Block"
CAT_BASIC = "Basic"
CAT_UNKNOWN = "Unknown"

# ── Unicode symbols ──────────────────────────────────────────────────────────
DIAMETER_SYMBOLS = "Øø⌀φΦ∅"
COUNTERBORE_SYMBOL = "⌴"
COUNTERSINK_SYMBOL = "⌵"
DEPTH_SYMBOL = "⌵⌷↧⌇"  # depth glyph is OCR'd inconsistently; treat loosely
DATUM_TRIANGLE = "▽▼△▲"
WELD_SYMBOLS = "△▽◁▷⏊"

# ── GD&T (Feature Control Frame) characteristic symbols ──────────────────────
# Maps the geometric-characteristic glyph -> human name. When OCR mangles the
# glyph we fall back to the keyword table below.
GDT_SYMBOLS: dict[str, str] = {
    "⏤": "Straightness",
    "⏥": "Flatness",
    "○": "Circularity",
    "◯": "Circularity",
    "⌭": "Cylindricity",
    "⌒": "Profile of a Line",
    "⌓": "Profile of a Surface",
    "∥": "Parallelism",
    "//": "Parallelism",  # how PaddleOCR renders the ∥ frame glyph
    "⟂": "Perpendicularity",
    "∠": "Angularity",
    "⌖": "Position",
    "◎": "Concentricity",
    "⌯": "Symmetry",
    "↗": "Runout",
    "⌰": "Total Runout",
}

GDT_KEYWORDS: dict[str, str] = {
    "STRAIGHTNESS": "Straightness",
    "FLATNESS": "Flatness",
    "CIRCULARITY": "Circularity",
    "ROUNDNESS": "Circularity",
    "CYLINDRICITY": "Cylindricity",
    "PROFILE": "Profile",
    "PARALLELISM": "Parallelism",
    "PERPENDICULARITY": "Perpendicularity",
    "SQUARENESS": "Perpendicularity",
    "ANGULARITY": "Angularity",
    "POSITION": "Position",
    "TRUE POSITION": "Position",
    "CONCENTRICITY": "Concentricity",
    "SYMMETRY": "Symmetry",
    "RUNOUT": "Runout",
    "TOTAL RUNOUT": "Total Runout",
    "ECCENTRICITY": "Concentricity",
}

# Material-condition modifiers seen inside feature control frames.
FCF_MODIFIERS = frozenset({"Ⓜ", "Ⓛ", "Ⓢ", "(M)", "(L)", "(S)", "MMC", "LMC", "RFS"})

# ── Keyword tables per category ──────────────────────────────────────────────
HOLE_MODIFIERS = frozenset(
    {"THRU", "THROUGH", "CBORE", "COUNTERBORE", "C'BORE", "CSK", "COUNTERSINK",
     "C'SINK", "SF", "SPOTFACE", "SPOT FACE", "DEEP", "DP", "DEPTH", "DRILL",
     "REAM", "BORE", "TAP", "TAPPED"}
)

# Distinct hole subtypes for labelling.
HOLE_SUBTYPES: list[tuple[frozenset[str], str]] = [
    (frozenset({"CBORE", "COUNTERBORE", "C'BORE"}), "Counterbore"),
    (frozenset({"CSK", "COUNTERSINK", "C'SINK"}), "Countersink"),
    (frozenset({"SF", "SPOTFACE", "SPOT FACE"}), "Spotface"),
    (frozenset({"THRU", "THROUGH"}), "Through Hole"),
    (frozenset({"DEEP", "DP", "DEPTH"}), "Blind Hole"),
    (frozenset({"TAP", "TAPPED"}), "Tapped Hole"),
]

NOTE_KEYWORDS = frozenset(
    {"NOTE", "NOTES", "REMOVE", "BURR", "BURRS", "EDGE", "EDGES", "UNLESS",
     "SPECIFIED", "OTHERWISE", "ALL DIMENSIONS", "DO NOT SCALE", "BREAK",
     "SHARP", "TYP", "TYPICAL", "REF ONLY", "SCALE", "DETAIL", "SECTION",
     "GENERAL", "TOLERANCE UNLESS"}
)

MATERIAL_KEYWORDS = frozenset(
    {"MATERIAL", "MAT'L", "MATL", "ASTM", "AISI", "SAE", "EN8", "EN9", "EN24",
     "AL6061", "AL7075", "S45C", "SS304", "SS316", "STEEL", "ALUMINIUM",
     "ALUMINUM", "BRASS", "BRONZE", "CAST IRON", "SUJ2", "FORGED", "FORGING",
     "TUBE", "RING", "BILLET", "STOCK"}
)

HEAT_KEYWORDS = frozenset(
    {"HEAT TREAT", "HEAT TREATMENT", "H/T", "HT", "HARDNESS", "HARDEN",
     "HARDENED", "TEMPER", "NITRIDING", "NITRIDE", "ANNEAL", "ANNEALING",
     "CARBURIZE", "CARBURISE", "CASE DEPTH", "QUENCH", "NORMALIZE", "HRC",
     "RC", "HB", "HV", "RA58", "AFTER H/T"}
)

COATING_KEYWORDS = frozenset(
    {"ZINC", "BLACK OXIDE", "ANODIZE", "ANODISE", "ANODIZED", "CHROME",
     "CHROMIUM", "NICKEL", "GALVANIZED", "GALVANISED", "PAINT", "POWDER COAT",
     "POWDER COATING", "PHOSPHATE", "PASSIVATE", "PLATE", "PLATING", "COATING",
     "ELECTROLESS"}
)

SURFACE_KEYWORDS = frozenset(
    {"RA", "RZ", "RMAX", "RQ", "SURFACE FINISH", "SURFACE ROUGHNESS", "SURFACE",
     "ROUGHNESS", "FINISH", "LAY", "MICRON", "µM", "UM"}
)

WELD_KEYWORDS = frozenset(
    {"WELD", "FILLET WELD", "FIELD", "ALL AROUND", "TAIL", "AWS", "ISO2553",
     "ISO 2553", "PLUG", "SPOT WELD", "SEAM", "BACK", "GROOVE", "BEVEL WELD"}
)

CHAMFER_KEYWORDS = frozenset({"CHAMFER", "CHAM", "CHFR"})
FILLET_KEYWORDS = frozenset({"FILLET"})
TAPER_KEYWORDS = frozenset({"TAPER", "TAPERED"})

# Datum feature symbols are single boxed capital letters (A, B, C…) with a
# datum triangle; the triangle detection happens in vision, keyword here.
DATUM_KEYWORDS = frozenset({"DATUM"})

# ── Regex patterns ───────────────────────────────────────────────────────────
RE_THREAD = re.compile(
    r"\b(M\d+(?:\s*[×xX]\s*\d+(?:\.\d+)?)?"          # metric  M6, M10x1.25
    r"|\d+[/-]\d+\s*-\s*\d+\s*(?:UNC|UNF|UNEF)"       # unified fractional
    r"|\d+\s*-\s*\d+\s*(?:UNC|UNF|UNEF)"
    r"|(?:UNC|UNF|UNEF|NPT|NPTF|BSP|BSPT|BSPP|BSW)"   # thread series
    r"|G\s*\d+(?:/\d+)?"                              # G1/4
    r"|Rc\s*\d+(?:/\d+)?"
    r"|PT\s*\d+)\b",
    re.IGNORECASE,
)
RE_QUANTITY = re.compile(r"^\s*(\d+)\s*[×xX]\s*")     # 4X , 2 x
RE_CHAMFER = re.compile(r"\d+(?:\.\d+)?\s*[×xX]\s*(?:45|30|60)\s*°?")
RE_REFERENCE = re.compile(r"^\(.*\)$|(?<![A-Z])REF(?:\.|ERENCE)?\b", re.IGNORECASE)
RE_BASIC = re.compile(r"^\s*[\[□▭].+[\]□▭]\s*$")      # boxed dimension
RE_SURFACE_VALUE = re.compile(r"\b(?:Ra|Rz|Rmax|Rq)\s*\d", re.IGNORECASE)
RE_SURFACE_N = re.compile(r"\bN\s?(?:1[0-2]|[1-9])\b")   # ISO N grades N1..N12
RE_TAPER_RATIO = re.compile(r"\b\d+\s*(?::|IN)\s*\d+\b", re.IGNORECASE)
RE_LEADING_NUMBER = re.compile(r"^\s*[ØøRr]?\s*[\+\-]?\d")
RE_ANGLE = re.compile(r"\d+\s*°|\d+\s*DEG\b", re.IGNORECASE)
RE_TOLERANCE = re.compile(r"±|\+/-|\+\s*0?\.\d+\s*/\s*-")

# A "large text block" (general note) heuristic: many words, few digits.
def looks_like_note_block(text: str) -> bool:
    words = text.split()
    if len(words) < 4:
        return False
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    digit_chars = sum(1 for c in text if c.isdigit())
    return alpha_words >= 4 and digit_chars <= len(text) * 0.25
