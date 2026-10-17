"""
Stage 4 — use the drawing around a mark to settle what the mark is.

The rule this module is built on
---------------------------------
**Context may promote, demote, or qualify a reading. It may never overrule
contradictory visual evidence.** A circle inside a frame's first compartment
is circularity because that is the only thing it can be there — context is
*resolving an ambiguity*, not overriding the pixels. But if the pixels say
there is a slash through that circle, the answer is that something is wrong
with the frame, not that the slash is not there.

So every function here either:

* picks between candidates the detector already judged *equally* plausible, or
* attaches a value or a relationship, or
* adds a warning,

and none of them replaces a confident detection with a different symbol. Where
context and vision genuinely disagree, :func:`reconcile` records the conflict
and lowers confidence rather than silently choosing a side.

The cases that actually matter
------------------------------
These are the confusions that occur on real sheets, and each is handled by the
one piece of context that settles it:

* ``⌒`` is profile-of-a-line inside a frame and arc length above a dimension.
* ``⌴`` is a counterbore unless an ``SF`` annotation makes it a spotface.
* A circle is circularity in a frame, a diameter before a value, all-around on
  a leader kink, and a drilled hole everywhere else.
* ``R`` is a radius before a number, and part of a word in ``RA``/``Rc``.
* A capital letter is a datum reference inside a frame and a datum feature
  only when a triangle is attached to it.
* A material-condition modifier belongs to the compartment it follows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import gdt_registry as reg
from gdt_detect import (
    STATUS_AMBIGUOUS,
    STATUS_CONFIRMED,
    STATUS_UNKNOWN,
    Candidate,
    Detection,
    iou,
)


# ── Text blocks the context pass reads ───────────────────────────────────────


@dataclass(frozen=True)
class TextBox:
    """A piece of OCR text with where it sits, in page coordinates."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)


def _distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Gap between two boxes — zero when they touch or overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0.0, max(bx - (ax + aw), ax - (bx + bw)))
    dy = max(0.0, max(by - (ay + ah), ay - (by + bh)))
    return (dx * dx + dy * dy) ** 0.5


def _is_right_of(
    symbol: tuple[float, float, float, float],
    text: tuple[float, float, float, float],
    *,
    row_tolerance: float = 0.6,
) -> bool:
    """True when ``text`` sits to the right of ``symbol`` on the same line."""
    sx, sy, sw, sh = symbol
    tx, ty, tw, th = text
    same_row = abs((ty + th / 2) - (sy + sh / 2)) <= sh * row_tolerance
    return same_row and tx >= sx + sw * 0.4


# ── Value parsing ────────────────────────────────────────────────────────────

_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_RATIO = re.compile(r"(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)")
_ANGLE = re.compile(r"(\d+(?:[.,]\d+)?)\s*°")
_COUNT = re.compile(r"^\s*(\d+)\s*[Xx×]")
_RADIUS = re.compile(r"\b(C?R|SR|S?[Ø⌀])\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_value(symbol_id: str, text: str) -> dict[str, Any] | None:
    """Read the value a symbol carries, per its registry :class:`ValueSpec`.

    Returns ``None`` when the symbol takes no value or none was found. The
    unit is always reported as ``drawing_unit``: nothing here can know whether
    the sheet is in millimetres or inches, and the brief is explicit that one
    must not be assumed.
    """
    sym = reg.find(symbol_id)
    if sym is None or sym.value.kind == reg.V_NONE:
        return None
    raw = (text or "").strip()
    if not raw:
        return None

    kind = sym.value.kind

    if kind == reg.V_RATIO:
        match = _RATIO.search(raw)
        if not match:
            return None
        left, right = _to_float(match.group(1)), _to_float(match.group(2))
        if left is None or right is None or right == 0:
            return None
        return {
            "kind": "ratio",
            "numerator": left,
            "denominator": right,
            "value": left / right,
            "raw": match.group(0),
        }

    if kind == reg.V_ANGLE:
        match = _ANGLE.search(raw)
        if not match:
            return None
        return {
            "kind": "angle",
            "value": _to_float(match.group(1)),
            "unit": "degree",
            "raw": match.group(0),
        }

    if kind == reg.V_COUNT:
        match = _COUNT.match(raw)
        if not match:
            return None
        return {
            "kind": "count",
            "value": int(match.group(1)),
            "unit": "features",
            "raw": match.group(0),
        }

    if kind == reg.V_LETTER:
        letters = re.findall(r"\b([A-Z]{1,2})\b", raw.upper())
        if not letters:
            return None
        return {"kind": "letter", "value": letters[0], "raw": letters[0]}

    # Lengths and tolerances. A prefixed form (R12, Ø8, SR4) is preferred over
    # a bare number, because the prefix proves the number belongs to this
    # symbol rather than being the next dimension along.
    prefixed = _RADIUS.search(raw)
    if prefixed is not None:
        value = _to_float(prefixed.group(2))
        if value is not None:
            return {
                "kind": "length",
                "value": value,
                "unit": "drawing_unit",
                "raw": prefixed.group(0),
                "prefixed": True,
            }
    match = _NUMBER.search(raw)
    if not match:
        return None
    return {
        "kind": "tolerance" if kind == reg.V_TOLERANCE else "length",
        "value": _to_float(match.group(0)),
        "unit": "drawing_unit",
        "raw": match.group(0),
        "prefixed": False,
    }


# ── Associating text with symbols ────────────────────────────────────────────


def attach_text(
    detections: Sequence[Detection],
    texts: Sequence[TextBox],
    *,
    max_gap_factor: float = 2.0,
) -> None:
    """Attach the nearest trailing text to each detection, and parse its value.

    Mutates the detections. Attachment prefers text to the *right* on the same
    line, which is how a drawing reads: ``Ø25``, ``R12``, ``⌴ Ø10``. A symbol
    with nothing beside it keeps ``associated_text`` of ``None`` rather than
    being given the nearest number on the sheet, which would be a fabrication.
    """
    for det in detections:
        if det.symbol_id is None:
            continue
        sym = reg.get(det.symbol_id)
        if sym.value.kind == reg.V_NONE:
            continue
        _, _, _, dh = det.bbox
        limit = max(dh, 1.0) * max_gap_factor

        best: TextBox | None = None
        best_gap = float("inf")
        for box in texts:
            if not box.text.strip():
                continue
            gap = _distance(det.bbox, box.bbox)
            if gap > limit:
                continue
            # Same-line text to the right wins over anything else nearby.
            weighted = gap * (0.5 if _is_right_of(det.bbox, box.bbox) else 1.5)
            if weighted < best_gap:
                best, best_gap = box, weighted

        if best is None:
            if not sym.value.optional:
                det.warnings.append(
                    f"{sym.name} normally carries a value, but no text was "
                    "found next to it"
                )
            continue

        det.associated_text = best.text.strip()
        value = parse_value(det.symbol_id, best.text)
        if value is not None:
            det.value = value
        elif not sym.value.optional:
            det.warnings.append(
                f"text {best.text.strip()!r} was found beside {sym.name} but "
                "carries no value of the expected form"
            )


# ── Resolving shared-glyph ambiguity by position ─────────────────────────────


def resolve_by_position(
    detection: Detection,
    *,
    in_frame: bool,
    frame_compartment: int | None = None,
    on_leader: bool = False,
    nearby_text: str = "",
) -> None:
    """Settle a detection whose glyph is shared between symbols.

    Only touches detections the detector already flagged ambiguous between
    specific alternatives. A confirmed reading is left alone — that is the
    "context must not override visual evidence" rule in code.
    """
    if detection.symbol_id is None:
        return

    # The guard that makes the module's rule real. Only a detection the
    # detector could not settle is open to context. Without this, a rule fires
    # whenever its symbol appears anywhere in the alternatives — which is how
    # a confident profile-of-a-line reading (shape 0.87, runner-up 0.59) was
    # being rewritten to circularity by the "a circle in compartment 0 is
    # circularity" rule, on the strength of circularity sitting fourth in its
    # alternatives. Context resolves ties; it does not overrule the pixels.
    if detection.status != STATUS_AMBIGUOUS:
        return

    candidates = {detection.symbol_id, *(a.symbol_id for a in detection.alternatives)}
    upper = (nearby_text or "").upper()

    def choose(symbol_id: str, reason: str) -> None:
        if symbol_id == detection.symbol_id:
            detection.evidence["context"] = reason
            if detection.status == STATUS_AMBIGUOUS:
                detection.status = STATUS_CONFIRMED
            return
        # Promote an alternative: keep the old best as an alternative so the
        # swap is auditable rather than invisible.
        previous = Candidate(
            symbol_id=detection.symbol_id,
            score=detection.confidence,
            method=detection.method,
            evidence={"demoted_by_context": reason},
        )
        sym = reg.get(symbol_id)
        detection.alternatives = [
            a for a in detection.alternatives if a.symbol_id != symbol_id
        ] + [previous]
        detection.symbol_id = symbol_id
        detection.name = sym.name
        detection.normalized = sym.unicode
        detection.evidence["context"] = reason
        detection.status = STATUS_CONFIRMED

    # ⌒ — profile of a line inside a frame, arc length above a dimension.
    if {"profile_of_a_line", "arc_length"} & candidates:
        if in_frame and frame_compartment == 0:
            choose(
                "profile_of_a_line",
                "the arc glyph sits in a frame's first compartment, where only "
                "a geometric characteristic can appear",
            )
        elif not in_frame:
            choose(
                "arc_length",
                "the arc glyph sits outside any feature control frame, above a "
                "dimension, which is an arc-length mark",
            )

    # ⌴ — counterbore unless annotated SF.
    if {"counterbore", "spotface"} & candidates:
        if re.search(r"(?<![A-Z])SF(?![A-Z])", upper):
            choose("spotface", "an 'SF' annotation accompanies the counterbore glyph")
        elif upper:
            choose(
                "counterbore",
                "the counterbore glyph carries no 'SF' annotation",
            )

    # A circle: circularity in a frame, diameter before a value, all-around on
    # a leader kink.
    if {"circularity", "diameter", "all_around"} & candidates:
        if in_frame and frame_compartment == 0:
            choose(
                "circularity",
                "a plain circle in a frame's first compartment is circularity",
            )
        elif in_frame and frame_compartment == 1:
            choose(
                "diameter",
                "a circle in the tolerance compartment marks a diametral zone",
            )
        elif on_leader:
            choose(
                "all_around",
                "the circle sits at a leader kink, which is the all-around mark",
            )
        elif re.match(r"^\s*\d", nearby_text or ""):
            choose(
                "diameter",
                "the circle immediately precedes a numeric value",
            )

    # Ø vs SØ — the S prefix is the whole difference.
    if {"diameter", "spherical_diameter"} & candidates:
        if re.search(r"(?<![A-Z])S\s*[Ø⌀]", upper):
            choose("spherical_diameter", "the diameter glyph carries an 'S' prefix")
        elif upper:
            choose("diameter", "no 'S' prefix accompanies the diameter glyph")

    # A triangle is a datum feature only when a letter box goes with it.
    if {"datum_triangle", "datum_feature_symbol"} & candidates:
        if re.search(r"\b[A-Z]\b", upper):
            choose(
                "datum_feature_symbol",
                "the triangle is accompanied by a datum letter",
            )
        else:
            choose(
                "datum_triangle",
                "a triangle with no readable datum letter is reported as a "
                "datum marker of unknown letter",
            )


# ── Radius: a letter, not a glyph ────────────────────────────────────────────

#: Words that open with R and are not a radius. Checked before an R is read as
#: one, because "RA 0.8" and "Rc 58" are both common and both look like "R" +
#: number to a naive matcher.
_R_FALSE_FRIENDS = re.compile(
    r"^(?:RA|RZ|RQ|RMAX|RC|REF|REV|REQ)\b", re.IGNORECASE
)


def radius_from_text(text: str) -> dict[str, Any] | None:
    """Read a radius callout from text, refusing the look-alikes.

    Radius has no glyph — it is the letter R — so this is the one symbol whose
    recognition is necessarily textual, and the whole job is not matching
    surface-roughness ``Ra``, hardness ``Rc``, or ``REF``.
    """
    raw = (text or "").strip()
    if not raw or _R_FALSE_FRIENDS.match(raw):
        return None
    match = re.match(
        r"^\s*(S?C?R)\s*(\d+(?:[.,]\d+)?)", raw, re.IGNORECASE
    )
    if not match:
        return None
    prefix = match.group(1).upper()
    symbol_id = {
        "R": "radius",
        "CR": "controlled_radius",
        "SR": "spherical_radius",
        "SCR": "controlled_radius",
    }.get(prefix, "radius")
    value = _to_float(match.group(2))
    if value is None:
        return None
    return {
        "symbol_id": symbol_id,
        "value": value,
        "unit": "drawing_unit",
        "raw": match.group(0).strip(),
    }


# ── Datum association ────────────────────────────────────────────────────────


def associate_datums(
    frames: Sequence[Any],
    datum_features: Sequence[Detection],
) -> list[dict[str, Any]]:
    """Link each datum reference in a frame to the datum feature it names.

    Returns one record per reference, including the ones that resolve to
    nothing. A frame citing datum ``B`` on a drawing with no ``B`` datum
    feature is a real finding — more useful reported than quietly dropped —
    and it is reported as unresolved rather than matched to the nearest
    letter.
    """
    by_letter: dict[str, Detection] = {}
    for det in datum_features:
        if det.symbol_id not in {"datum_feature_symbol", "datum_triangle"}:
            continue
        letter = None
        if det.value and det.value.get("kind") == "letter":
            letter = str(det.value["value"]).upper()
        elif det.associated_text:
            match = re.search(r"\b([A-Z])\b", det.associated_text.upper())
            letter = match.group(1) if match else None
        if letter:
            by_letter.setdefault(letter, det)

    records: list[dict[str, Any]] = []
    for frame in frames:
        for ref in getattr(frame, "datums", []):
            feature = by_letter.get(ref.label.upper())
            records.append(
                {
                    "frame_id": getattr(frame, "id", None),
                    "label": ref.label,
                    "precedence": ref.precedence,
                    "material_condition": ref.material_condition,
                    "resolved": feature is not None,
                    "datum_feature_bbox": (
                        {
                            "x": round(feature.bbox[0], 2),
                            "y": round(feature.bbox[1], 2),
                            "width": round(feature.bbox[2], 2),
                            "height": round(feature.bbox[3], 2),
                        }
                        if feature is not None
                        else None
                    ),
                    "note": (
                        None
                        if feature is not None
                        else (
                            f"no datum feature symbol labelled {ref.label!r} was "
                            "found on this page"
                        )
                    ),
                }
            )
    return records


# ── Reconciling context against vision ───────────────────────────────────────


def reconcile(detection: Detection, *, expected: Iterable[str], reason: str) -> None:
    """Record a disagreement between what position implies and what was seen.

    Called where a symbol turns up somewhere it should not be — a
    perpendicularity glyph outside any frame, an ASME-only modifier on a sheet
    declared ISO. The detection is *kept*: the drawing may be unusual, or
    wrong, and either is worth surfacing. What changes is that the conflict is
    recorded and the confidence is cut, so nothing downstream treats the
    reading as settled.
    """
    expected_ids = set(expected)
    if detection.symbol_id is None or detection.symbol_id in expected_ids:
        return
    detection.warnings.append(reason)
    detection.evidence["context_conflict"] = {
        "expected": sorted(expected_ids),
        "found": detection.symbol_id,
        "reason": reason,
    }
    detection.confidence = round(detection.confidence * 0.6, 4)
    if detection.status == STATUS_CONFIRMED:
        detection.status = STATUS_AMBIGUOUS


#: Controls that are meaningless outside a feature control frame. Finding one
#: loose on a sheet is far more likely to be drawing geometry — a right angle,
#: a pair of parallel lines — than a stray control.
FRAME_ONLY = frozenset(s.id for s in reg.FRAME_CHARACTERISTICS)


def demote_loose_characteristics(detections: Sequence[Detection]) -> None:
    """Flag geometric characteristics found outside any frame.

    This is the single biggest source of false positives in the whole system:
    a drawing is full of right angles, parallel lines and circles, and every
    one of them correlates with perpendicularity, parallelism or circularity.
    Rather than drop them — occasionally a frame is missed and the symbol is
    real — they are demoted to ambiguous and told why.
    """
    for det in detections:
        if det.symbol_id in FRAME_ONLY and det.frame_id is None:
            det.warnings.append(
                f"{det.name} was found outside any feature control frame; a "
                "geometric characteristic is only meaningful inside one, so "
                "this may be ordinary drawing geometry"
            )
            det.confidence = round(det.confidence * 0.5, 4)
            if det.status == STATUS_CONFIRMED:
                det.status = STATUS_AMBIGUOUS


# ── Standards consistency ────────────────────────────────────────────────────


def check_standards_consistency(
    detections: Sequence[Detection],
    *,
    declared_standard: str | None = None,
) -> list[dict[str, Any]]:
    """Report symbols that do not belong to the drawing's declared standard.

    Does **not** rewrite anything, and does not declare the drawing compliant
    or non-compliant — that judgement needs the whole drawing, its revision
    and its notes, none of which OCR establishes. It reports observations.
    """
    findings: list[dict[str, Any]] = []
    declared = (declared_standard or "").strip().upper()
    for det in detections:
        sym = det.symbol
        if sym is None:
            continue
        exclusive = (sym.standards.exclusive_to or "").upper()
        if exclusive and declared and exclusive not in declared:
            findings.append(
                {
                    "symbol_id": sym.id,
                    "name": sym.name,
                    "issue": "standard_mismatch",
                    "detail": (
                        f"{sym.name} is defined by {sym.standards.exclusive_to} "
                        f"only, but the drawing is declared {declared_standard}"
                    ),
                    "confidence": det.confidence,
                }
            )
        if sym.is_legacy:
            findings.append(
                {
                    "symbol_id": sym.id,
                    "name": sym.name,
                    "issue": "retired_control",
                    "detail": (
                        f"{sym.name} was removed in {sym.standards.retired_in}. "
                        "It remains valid on drawings released to an earlier "
                        "revision or to "
                        f"{', '.join(sym.standards.current_in) or 'ISO'}; the "
                        "revision this drawing was released to is not known "
                        "from the symbol alone."
                    ),
                    "confidence": det.confidence,
                }
            )
    return findings
