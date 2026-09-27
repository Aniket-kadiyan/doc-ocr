"""
GD&T ballooning rule engine (GDTOCR spec).

Classifies one OCR object into a feature category using three sources of
information — OCR text, detected symbols, and (optional) geometry — then applies
a priority-ordered decision engine and produces a balloon label.

    Feature = OCR + Symbols + Geometry + Context

The priority order follows the spec's decision engine:

    Feature Control Frame -> GD&T
    Datum Symbol          -> Datum
    Weld Symbol           -> Weld
    Surface Texture       -> Surface Finish
    Thread Pattern        -> Thread
    Hole Modifiers        -> Hole Feature
    Diameter/Radius/
      Chamfer/Angle       -> Geometric Dimension
    Material/Heat/Coating -> Manufacturing Notes
    Large Text Blocks     -> General Notes
    Remaining Numeric     -> Overall or Linear (geometry decides)

The engine is intentionally pure/text-first so it is unit-testable without cv2
or a running model. ``symbols`` and ``geometry`` refine — never gate — the
result: a strong textual signal wins even when vision is silent, which is what
keeps false negatives down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import feature_dictionary as fd
import feature_rules as rules


@dataclass
class Feature:
    """The classified entity the balloon is built from."""

    text: str
    category: str = fd.CAT_UNKNOWN
    subtype: str | None = None
    label: str = ""
    symbols: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    quantity: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "subtype": self.subtype,
            "label": self.label,
            "symbols": self.symbols,
            "keywords": self.keywords,
            "quantity": self.quantity,
        }


def _kw_match(upper: str, kw: str) -> bool:
    """Whole-token keyword match.

    A plain ``kw in upper`` substring test produces false positives like
    "TAIL" ⊂ "DETAIL" or "RA" ⊂ "DRAWN"/"RADIUS". Require that the keyword is
    not embedded inside a longer alphanumeric run.
    """
    return re.search(
        r"(?<![A-Z0-9])" + re.escape(kw) + r"(?![A-Z0-9])", upper
    ) is not None


def _found_keywords(upper: str, table) -> list[str]:
    """Return keywords from ``table`` that appear as whole tokens."""
    return [kw for kw in table if _kw_match(upper, kw)]


def _has_datum_reference(upper: str) -> bool:
    """A lone/boxed capital letter used as a datum feature label (A, B, C…).

    Only fires on a short token so a stray 'A' inside a note never matches.
    """
    if fd.DATUM_KEYWORDS.intersection(upper.split()):
        return True
    return bool(re.fullmatch(r"[\[\(]?[A-CE-Z][\]\)]?", upper.strip()))


def _detect_gdt(text: str, upper: str, symbols: set[str]) -> tuple[str, str | None] | None:
    """Feature control frames. Off by default — spec section 16 is out of scope."""
    if not rules.is_enabled(rules.GDT):
        return None
    """Return (category, subtype) if this is a Feature Control Frame / GD&T."""
    for glyph, name in fd.GDT_SYMBOLS.items():
        if glyph in text:
            return fd.CAT_GDT, name
    # A pipe-delimited frame like "| Ø0.1 | A | B |" is a strong FCF signal.
    pipe_frame = text.count("|") >= 2 and re.search(r"\d", text)
    for kw, name in fd.GDT_KEYWORDS.items():
        if _kw_match(upper, kw):
            return fd.CAT_GDT, name
    if pipe_frame and any(g in text for g in fd.DIAMETER_SYMBOLS):
        return fd.CAT_GDT, None
    return None


def _detect_thread(upper: str) -> str | None:
    m = fd.RE_THREAD.search(upper)
    return m.group(0).strip() if m else None


def _hole_subtype(kws: list[str]) -> str | None:
    present = {k.upper() for k in kws}
    for keyset, name in rules.active_hole_subtypes():
        if present & keyset:
            return name
    return None


def _quantity(text: str) -> int | None:
    """Leading ``4X``. Off by default — spec section 24 is out of scope."""
    if not rules.is_enabled(rules.QUANTITY_PREFIX):
        return None
    m = fd.RE_QUANTITY.match(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _label_for(feat: Feature) -> str:
    """Human-readable balloon label from category + subtype + modifiers."""
    cat, sub = feat.category, feat.subtype
    qty = f"{feat.quantity}X " if feat.quantity else ""

    if cat == fd.CAT_HOLE:
        return f"{qty}{sub or 'Hole'}"
    if cat == fd.CAT_DIAMETER:
        return f"{qty}Diameter"
    if cat == fd.CAT_RADIUS:
        return "Radius"
    if cat == fd.CAT_CHAMFER:
        return "Chamfer"
    if cat == fd.CAT_ANGLE:
        return "Angle"
    if cat == fd.CAT_TAPER:
        return "Taper"
    if cat == fd.CAT_THREAD:
        return f"{qty}Thread {sub}".strip() if sub else f"{qty}Thread".strip()
    if cat == fd.CAT_GDT:
        return f"GD&T {sub}" if sub else "Feature Control Frame"
    if cat == fd.CAT_DATUM:
        return f"Datum {sub}" if sub else "Datum"
    if cat == fd.CAT_SURFACE:
        return "Surface Finish"
    if cat == fd.CAT_WELD:
        return "Weld"
    if cat == fd.CAT_MATERIAL:
        return "Material"
    if cat == fd.CAT_HEAT:
        return "Heat Treatment"
    if cat == fd.CAT_COATING:
        return "Coating"
    if cat == fd.CAT_NOTE:
        return "General Note"
    if cat == fd.CAT_REFERENCE:
        return "Reference Dimension"
    if cat == fd.CAT_BASIC:
        return "Basic Dimension"
    if cat == fd.CAT_TOLERANCE:
        return "Toleranced Dimension"
    if cat == fd.CAT_LINEAR:
        return "Linear Dimension"
    return "Feature"


def classify_feature(
    text: str,
    symbols: dict[str, Any] | None = None,
    geometry: dict[str, Any] | None = None,
) -> Feature:
    """Classify one OCR object into a :class:`Feature`.

    ``symbols`` is the DetectedSymbols dict (diameter/radius/degree/plus_minus).
    ``geometry`` optionally carries {has_leader, points_to_circle, span_ratio}.
    """
    raw = (text or "").strip()
    feat = Feature(text=raw)
    if not raw:
        return feat

    upper = raw.upper()
    sym = symbols or {}
    geo = geometry or {}
    detected_symbols: list[str] = []

    # Text-driven, not vision-bool driven. The compose step (dimension_compose)
    # already injects Ø / ° into the OCR text ONLY when there is corroborating
    # evidence, so keying off the composed text — rather than the brittle raw
    # vision flags — is what keeps false positives (a template-match ring
    # turning "45°" or a MATERIAL note block into a "Diameter") in check.
    # Vision flags are still recorded in ``symbols`` for transparency.
    has_dia = any(g in raw for g in fd.DIAMETER_SYMBOLS)
    has_degree = "°" in raw or bool(fd.RE_ANGLE.search(upper))
    has_pm = bool(fd.RE_TOLERANCE.search(raw))
    if has_dia:
        detected_symbols.append("diameter")
    if has_pm:
        detected_symbols.append("plus_minus")
    if has_degree:
        detected_symbols.append("degree")
    # Note when vision disagreed with the text, useful for debugging/telemetry.
    vision_only = [
        k for k in ("diameter", "degree", "plus_minus")
        if sym.get(k) and k not in detected_symbols
    ]
    if vision_only:
        detected_symbols.extend(f"vision:{k}" for k in vision_only)

    qty = _quantity(raw)
    feat.quantity = qty

    # ── Priority 1: Feature Control Frame / GD&T ────────────────────────────
    gdt = _detect_gdt(raw, upper, set(detected_symbols))
    if gdt is not None:
        feat.category, feat.subtype = gdt
        feat.symbols = detected_symbols
        feat.label = _label_for(feat)
        return feat

    # ── Priority 2: Datum ───────────────────────────────────────────────────
    if any(t in raw for t in fd.DATUM_TRIANGLE) or (
        _has_datum_reference(upper) and not fd.RE_LEADING_NUMBER.match(raw)
    ):
        feat.category = fd.CAT_DATUM
        m = re.search(r"[A-CE-Z]", upper)
        feat.subtype = m.group(0) if m else None
        feat.label = _label_for(feat)
        return feat

    # ── Priority 3: Weld ─────────────────────────────────────────────────────
    weld_kws = _found_keywords(upper, fd.WELD_KEYWORDS)
    if weld_kws or any(t in raw for t in fd.WELD_SYMBOLS):
        feat.category = fd.CAT_WELD
        feat.keywords = weld_kws
        feat.label = _label_for(feat)
        return feat

    # ── Priority 4: Surface Finish ──────────────────────────────────────────
    if fd.RE_SURFACE_VALUE.search(raw) or fd.RE_SURFACE_N.search(upper) or (
        _found_keywords(upper, fd.SURFACE_KEYWORDS)
        and any(k in upper for k in ("RA", "RZ", "RMAX", "ROUGHNESS", "FINISH"))
    ):
        feat.category = fd.CAT_SURFACE
        feat.keywords = _found_keywords(upper, fd.SURFACE_KEYWORDS)
        feat.label = _label_for(feat)
        return feat

    # ── Priority 5: Thread ──────────────────────────────────────────────────
    thread = _detect_thread(upper)
    # "Rc" is ambiguous: a taper-pipe thread series AND Rockwell-C hardness.
    # When hardness context is present, drop an Rc-only match so heat treatment
    # (priority 8) can claim it.
    if thread and thread.upper().startswith("RC") and (
        "HARDNESS" in upper or "HRC" in upper
    ):
        thread = None
    if thread:
        feat.category = fd.CAT_THREAD
        feat.subtype = thread
        feat.symbols = detected_symbols
        feat.label = _label_for(feat)
        return feat

    # ── Priority 6: Hole Feature (modifiers) ────────────────────────────────
    hole_kws = _found_keywords(upper, rules.active_hole_modifiers())
    if hole_kws:
        feat.category = fd.CAT_HOLE
        feat.keywords = hole_kws
        feat.subtype = _hole_subtype(hole_kws)
        feat.symbols = detected_symbols
        feat.label = _label_for(feat)
        return feat

    # ── Priority 7: Geometric dimensions ────────────────────────────────────
    # Chamfer before Angle (a chamfer is an angle with the NxAngle pattern).
    if fd.RE_CHAMFER.search(raw) or _found_keywords(upper, fd.CHAMFER_KEYWORDS):
        feat.category = fd.CAT_CHAMFER
        feat.symbols = detected_symbols
        feat.label = _label_for(feat)
        return feat

    if _found_keywords(upper, fd.TAPER_KEYWORDS) or (
        fd.RE_TAPER_RATIO.search(upper) and "TAPER" in upper
    ):
        feat.category = fd.CAT_TAPER
        feat.label = _label_for(feat)
        return feat

    if has_dia:
        feat.category = fd.CAT_DIAMETER
        feat.symbols = detected_symbols
        feat.label = _label_for(feat)
        return feat

    if (
        re.match(r"^\s*R\s*\d", upper)
        or sym.get("radius")
        or _kw_match(upper, "RADIUS")
        or (
            rules.is_enabled(rules.FILLET)
            and _found_keywords(upper, fd.FILLET_KEYWORDS)
        )
    ):
        feat.category = fd.CAT_RADIUS
        feat.subtype = (
            "Fillet"
            if rules.is_enabled(rules.FILLET)
            and _found_keywords(upper, fd.FILLET_KEYWORDS)
            else None
        )
        feat.symbols = detected_symbols
        feat.label = "Fillet Radius" if feat.subtype else _label_for(feat)
        return feat

    if has_degree:
        feat.category = fd.CAT_ANGLE
        feat.symbols = detected_symbols
        feat.label = _label_for(feat)
        return feat

    # ── Priority 8: Manufacturing notes (material / heat / coating) ─────────
    for table, cat in (
        (fd.HEAT_KEYWORDS, fd.CAT_HEAT),
        (fd.COATING_KEYWORDS, fd.CAT_COATING),
        (fd.MATERIAL_KEYWORDS, fd.CAT_MATERIAL),
    ):
        kws = _found_keywords(upper, table)
        if kws:
            feat.category = cat
            feat.keywords = kws
            feat.label = _label_for(feat)
            return feat

    # ── Priority 9: Reference / Basic dimensions ────────────────────────────
    if fd.RE_BASIC.match(raw):
        feat.category = fd.CAT_BASIC
        feat.label = _label_for(feat)
        return feat
    if rules.is_enabled(rules.REFERENCE) and fd.RE_REFERENCE.search(raw):
        feat.category = fd.CAT_REFERENCE
        feat.label = _label_for(feat)
        return feat

    # ── Priority 10: General notes ──────────────────────────────────────────
    if _found_keywords(upper, fd.NOTE_KEYWORDS) or fd.looks_like_note_block(raw):
        feat.category = fd.CAT_NOTE
        feat.keywords = _found_keywords(upper, fd.NOTE_KEYWORDS)
        feat.label = _label_for(feat)
        return feat

    # ── Priority 11: Remaining numeric -> Linear / Tolerance ────────────────
    if fd.RE_LEADING_NUMBER.match(raw) or re.search(r"\d", raw):
        span = geo.get("span_ratio", 0.0)
        feat.category = fd.CAT_TOLERANCE if has_pm else fd.CAT_LINEAR
        feat.symbols = detected_symbols
        if span and span >= 0.8:
            feat.subtype = "Overall"
            feat.label = "Overall Dimension"
        else:
            feat.label = _label_for(feat)
        return feat

    # Non-numeric leftover text.
    feat.category = fd.CAT_NOTE
    feat.label = _label_for(feat)
    return feat
