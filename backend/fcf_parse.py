"""
Phase 3 — find feature control frames and read their compartments.

A feature control frame is the one place on a drawing where GD&T is
unambiguous. Outside it a circle is just a circle; inside the first
compartment a circle is circularity. So the frame is not merely another thing
to detect — it is the context that makes the rest of the recognition safe, and
it is what lets a position symbol be reported as a position control rather
than guessed at from nearby coordinate dimensions.

Structure, not a string
-----------------------
The output is a :class:`FeatureControlFrame`, not flattened text. Flattening
``⌖ Ø0.5 Ⓜ A B C`` into one string throws away exactly what an inspector
needs: which modifier attached to the tolerance and which to a datum, and
which datum is primary. Those are different facts and they are kept apart.

Three different kinds of "missing"
----------------------------------
The brief asks for the distinction and it is a real one:

* **absent** — the frame genuinely has no such field. A flatness callout has
  no datum, and that is not a defect.
* **unreadable** — the compartment is there and something is in it, but it
  could not be read.
* **not_applicable** — the field cannot apply to this control at all.

A ``None`` tolerance could mean any of the three, which is why
:class:`FieldState` exists and why ``to_dict`` reports it alongside the value.

OCR is injected
---------------
``ocr_fn`` is a callable the caller supplies. The parser never imports
PaddleOCR, so the whole frame-segmentation path — the part most likely to
break — is testable on synthetic images with no model loaded and no GPU.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

import gdt_registry as reg
from gdt_config import GdtConfig, get_config
from gdt_detect import (
    STATUS_AMBIGUOUS,
    STATUS_CONFIRMED,
    STATUS_UNKNOWN,
    Detection,
    Transform,
    IDENTITY,
    binarize,
    classify_crop,
    to_gray,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


#: Signature of the OCR hook: an image in, ``(text, confidence)`` out.
OcrFn = Callable[[Image.Image], "tuple[str, float]"]

#: Pixels trimmed from each side of a compartment crop. Only a hair is needed
#: because :func:`erase_rules` has already removed the frame's lines; a wide
#: inset was tried first and could not be tuned — narrow enough to keep a
#: modifier drawn flush to a divider was too narrow to clear a border.
CELL_INSET_MIN_PX = 2

#: Length of the morphological kernel that isolates a ruled line, as a
#: fraction of the page's shorter edge. Chosen by measurement on a real
#: drawing across render scales; see the note in :func:`find_frames`.
RULE_LEN_FRAC = 0.02


# ── Field state ──────────────────────────────────────────────────────────────

#: The field was read.
PRESENT = "present"
#: The frame has no such field, legitimately.
ABSENT = "absent"
#: Something is there but could not be read.
UNREADABLE = "unreadable"
#: The field cannot apply to this control.
NOT_APPLICABLE = "not_applicable"


@dataclass
class Field:
    """One value plus an honest account of why it is missing when it is."""

    value: Any = None
    state: str = ABSENT
    confidence: float = 0.0
    raw: str | None = None

    @classmethod
    def present(cls, value: Any, confidence: float, raw: str | None = None) -> "Field":
        return cls(value=value, state=PRESENT, confidence=confidence, raw=raw)

    @classmethod
    def unreadable(cls, raw: str | None = None) -> "Field":
        return cls(value=None, state=UNREADABLE, confidence=0.0, raw=raw)

    @classmethod
    def not_applicable(cls) -> "Field":
        return cls(value=None, state=NOT_APPLICABLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "state": self.state,
            "confidence": round(float(self.confidence), 4),
            "raw": self.raw,
        }


@dataclass
class DatumRef:
    """One datum reference and the precedence its position gives it."""

    label: str
    precedence: int
    material_condition: str | None = None
    confidence: float = 0.0
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "precedence": self.precedence,
            "material_condition": self.material_condition,
            "confidence": round(float(self.confidence), 4),
            "raw": self.raw,
        }


@dataclass
class Compartment:
    """One cell of a frame, before interpretation."""

    index: int
    bbox: tuple[float, float, float, float]
    text: str = ""
    confidence: float = 0.0
    detection: Detection | None = None

    def to_dict(self) -> dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "index": self.index,
            "bounding_box": {
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "width": round(float(w), 2),
                "height": round(float(h), 2),
            },
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "detection": self.detection.to_dict() if self.detection else None,
        }


@dataclass
class FeatureControlFrame:
    """One frame — or one segment of a composite frame — read structurally."""

    id: str
    bbox: tuple[float, float, float, float]
    page: int = 1
    #: The geometric characteristic, as a Detection so its alternatives and
    #: warnings travel with it.
    symbol: Detection | None = None
    tolerance: Field = field(default_factory=Field)
    #: True when the tolerance zone is diametral (a Ø before the value).
    diameter_zone: Field = field(default_factory=Field)
    #: Material condition applied to the *tolerance*, not to a datum.
    material_condition: Field = field(default_factory=Field)
    #: Other modifiers found in the tolerance compartment (Ⓟ, Ⓕ, Ⓣ, Ⓤ…).
    modifiers: list[str] = field(default_factory=list)
    datums: list[DatumRef] = field(default_factory=list)
    compartments: list[Compartment] = field(default_factory=list)
    #: Index of this segment within a composite frame, 0 for a single frame.
    segment_index: int = 0
    #: Id shared by every segment of one composite frame.
    composite_id: str | None = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    #: Where the leader attaches, when one was found.
    leader_point: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        x, y, w, h = self.bbox
        out: dict[str, Any] = {
            "type": "feature_control_frame",
            "id": self.id,
            "page": self.page,
            "bounding_box": {
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "width": round(float(w), 2),
                "height": round(float(h), 2),
            },
            "symbol": self.symbol.to_dict() if self.symbol else None,
            "tolerance": self.tolerance.to_dict(),
            "diameter_zone": self.diameter_zone.to_dict(),
            "material_condition": self.material_condition.to_dict(),
            "modifiers": list(self.modifiers),
            "datums": [d.to_dict() for d in self.datums],
            "compartments": [c.to_dict() for c in self.compartments],
            "segment_index": self.segment_index,
            "composite_id": self.composite_id,
            "confidence": round(float(self.confidence), 4),
            "warnings": list(self.warnings),
        }
        if self.leader_point:
            out["leader_point"] = {
                "x": round(self.leader_point[0], 2),
                "y": round(self.leader_point[1], 2),
            }
        # The unit is the drawing's. Nothing here knows whether the sheet is in
        # millimetres or inches, and inventing one would be worse than saying so.
        out["tolerance_unit"] = "drawing_unit"
        return out


# ── Frame detection ──────────────────────────────────────────────────────────


def _line_mask(binary: np.ndarray, horizontal: bool, min_len: int) -> np.ndarray:
    """Isolate runs of ink at least ``min_len`` long in one direction."""
    if cv2 is None:
        return np.zeros_like(binary)
    size = max(3, min_len)
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
        if horizontal
        else cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    )
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cv2.dilate(eroded, kernel, iterations=1)


#: The first compartment holds the characteristic symbol and is drawn close
#: to square (ASME Y14.5 sizes it to the text height). A ruled box whose first
#: cell is far wider than tall is a table row or a boxed value, not a frame.
SYMBOL_CELL_ASPECT = (0.4, 2.2)

#: A frame has at least a symbol and a tolerance compartment, and at most a
#: symbol, a tolerance and a handful of datum references.
FRAME_CELL_COUNT = (2, 8)

#: Segments of a composite frame share a rule; two boxes further apart than
#: this are neighbours, not a stack.
MAX_RULE_PX = 12

#: Shortest a frame can be, as a fraction of the page's shorter edge. A frame
#: is twice the drawing's text height, and the smallest text a sheet carries
#: is about a hundredth of its short edge; a ruled box much shorter than that
#: is a table's header line or a pair of dimension lines.
FRAME_MIN_HEIGHT_FRAC = 0.012

#: Fraction of a candidate's height a column of ink must cover at each end
#: for the box to have sides. Coverage, not continuity: a scanned side rule
#: loses rows to its corners and to JPEG ringing and is still a side.
SIDE_SPAN = 0.7


def _rules_in_component(
    member: np.ndarray, x: int, y: int
) -> list[tuple[int, int, int, int]]:
    """The horizontal rules one connected component of the rule mask holds.

    A component is not a rule. Text resting on a rule survives the horizontal
    erosion where a stroke is widest and joins the component; on a page read
    enlarged, the digits between two rules join them into one component the
    height of the whole frame, and the rules of a composite stack arrive as
    one blob. The rules are the dense rows: each run of rows at least half as
    dense as the densest is one rule, with its own extent and thickness.

    ``member`` is the component's mask over its bounding box, whose top-left
    on the page is ``(x, y)``. Returns ``(x, extent, row, thickness)`` tuples.
    """
    rows = member.sum(axis=1)
    peak = int(rows.max()) if rows.size else 0
    if peak <= 0:
        return []
    dense = rows >= max(1.0, peak * 0.5)
    rules: list[tuple[int, int, int, int]] = []
    start: int | None = None
    for index, is_dense in enumerate([*dense.tolist(), False]):
        if is_dense and start is None:
            start = index
        elif not is_dense and start is not None:
            columns = np.nonzero(member[start:index].any(axis=0))[0]
            rules.append(
                (
                    x + int(columns[0]),
                    int(columns[-1] - columns[0] + 1),
                    y + (start + index - 1) // 2,
                    index - start,
                )
            )
            start = None
    return rules


def _same_extent(a: Sequence[int], b: Sequence[int]) -> bool:
    ax, _ay, aw, _ah = a
    bx, _by, bw, _bh = b
    return min(ax + aw, bx + bw) - max(ax, bx) >= 0.8 * max(aw, bw)


def _stacked(a: Sequence[int], b: Sequence[int], *, gap: int) -> bool:
    """True when two boxes share an x-extent and meet at a shared rule.

    The extents must nearly coincide, not merely overlap: the segments of a
    composite frame are drawn to about one width, and a looser test let a
    boxed value sitting under a frame join it. The vertical overlap must be
    small too — a shared rule, not a shared body — or a frame and the taller
    box its top rule makes with a dimension line below merge into one.
    """
    if not _same_extent(a, b):
        return False
    _ax, ay, _aw, ah = a
    _bx, by, _bw, bh = b
    overlap = min(ay + ah, by + bh) - max(ay, by)
    return -gap <= overlap <= gap


def _redundant(box: Sequence[int], other: Sequence[int], *, gap: int) -> bool:
    """True when ``box`` is ``other`` plus rows that are not a frame.

    Two rules pair with every rule between them, so a frame's top rule also
    pairs with the dimension line under the frame, and a composite stack's
    outer rules pair with each other. Those boxes share an extent and an edge
    with a shorter box that is the real segment, and give way to it.
    """
    if not _same_extent(box, other):
        return False
    _x, y, _w, h = box
    _ox, oy, _ow, oh = other
    if oh >= h:
        return False
    return abs(y - oy) <= gap or abs((y + h) - (oy + oh)) <= gap


def find_frames(
    image: Image.Image,
    *,
    transform: Transform = IDENTITY,
    config: GdtConfig | None = None,
    reference_edge: int | None = None,
) -> list[tuple[float, float, float, float]]:
    """Locate rectangles that could be feature control frames.

    Found from their ruled edges rather than from text: a frame is a box with
    a continuous top and bottom rule, which is what separates it from a run of
    text that happens to be boxy. Over-produces by design — a candidate that
    turns out to hold no characteristic symbol is dropped later, but a frame
    never proposed can never be read.

    A frame is assembled from its rules, not from a contour. Two horizontal
    rules one frame-height apart with the same extent, closed by a column of
    ink at each end and divided by at least one more, are a frame whatever
    else touches them. That last clause is the point. Nearly every frame on a
    drawing is joined to the part by a leader, and the leader's horizontal
    run is a rule as far as morphology is concerned, so the outer contour of a
    frame was the outer contour of the whole part outline — measured on a
    real sheet, the contour approach found 2 frames of 14 at 2x. Compartments
    found as holes of the ruling were tried next and lost every frame whose
    tolerance digits touched the bottom rule, which on a scan they do. Rule
    pairs find 13 of 14 at 1x and 2x on the same sheet.

    ``reference_edge`` is the shorter edge of the *page* this image is a
    tile of. The rule length and the smallest frame are fractions of the
    page, not of the tile: judged from a 1400px tile of a 2000px page they
    came out small enough to let a 21px box through as a frame.

    Returns boxes in the caller's coordinate space.
    """
    cfg = config or get_config()
    if cv2 is None:
        return []
    gray = to_gray(image)
    # No despeckle: its floor is 2% of the largest component, which on a page
    # is the part outline, and that erased whole frames (see ``binarize``).
    # The rule morphology below ignores specks on its own.
    binary = binarize(gray, clean=False)
    h, w = binary.shape[:2]
    if h < cfg.frame_min_height_px or w < cfg.frame_min_width_px:
        return []

    # The kernel that isolates a "rule" has to scale with the render, not be
    # an absolute pixel count. At 12px it selects long lines on a 1300px-wide
    # page and selects essentially everything on a 5000px one, so the grid
    # becomes solid ink and every frame merges into one blob: measured on a
    # real drawing, whole-page detection found 2 frames at 1x and 0 at 2x and
    # 4x. Scaling with the page recovers them. The floor keeps small crops —
    # a single frame handed in on its own — behaving exactly as before.
    edge = reference_edge or min(h, w)
    rule_len = max(cfg.frame_min_width_px // 2, 12, int(edge * RULE_LEN_FRAC))
    min_height = max(cfg.frame_min_height_px, int(edge * FRAME_MIN_HEIGHT_FRAC))
    horiz = _line_mask(binary, True, rule_len)
    # Bridge the small breaks a scanned rule carries, so it stays one rule.
    horiz = cv2.morphologyEx(
        horiz, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    )
    # On a page a frame is a small thing; on a crop handed in alone it may be
    # most of the image. The ceiling only bounds how far apart two rules may
    # be and still be tried as a pair — the compartment and aspect gates do
    # the rejecting — so it is generous: a third of the shorter edge, or a few
    # minimum frames on a crop too small for that to mean anything.
    max_height = max(int(min(h, w) * 0.3), min(h, 6 * cfg.frame_min_height_px))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    # (x, extent, row, thickness) of every rule long enough to be a frame edge.
    rules: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, bw, bh, _area = (int(v) for v in stats[index])
        if bw < cfg.frame_min_width_px:
            continue
        member = labels[y : y + bh, x : x + bw] == index
        rules.extend(
            rule
            for rule in _rules_in_component(member, x, y)
            if rule[1] >= cfg.frame_min_width_px
        )
    rules.sort(key=lambda r: (r[2], r[0]))
    # Two segments of a stack share one rule, so the boxes built on it
    # overlap by its thickness; on a page read enlarged that is more than a
    # fixed pixel count, so the gap is taken from the rules themselves.
    thicknesses = sorted(r[3] for r in rules)
    typical = thicknesses[len(thicknesses) // 2] if thicknesses else 1
    gap = max(MAX_RULE_PX, 3 * typical)
    strong = _frame_segments(binary, rules, min_height, max_height, gap, config=cfg)
    merged = _merge_segments(strong, gap=gap)
    return [
        transform.apply((float(x), float(y), float(bw), float(bh)))
        for x, y, bw, bh in merged
    ]


def _frame_segments(
    binary: np.ndarray,
    rules: Sequence[tuple[int, int, int, int]],
    min_height: int,
    max_height: int,
    gap: int = MAX_RULE_PX,
    *,
    config: GdtConfig | None = None,
) -> list[tuple[int, int, int, int]]:
    """Every pair of rules that encloses one frame segment's worth of structure.

    ``rules`` are ``(x, extent, row, thickness)`` sorted by row. Returns boxes
    including their rules, one per segment; a composite stack yields one per
    segment plus, often, one spanning the stack.
    """
    cfg = config or get_config()
    strong: list[tuple[int, int, int, int]] = []
    # Segments with a compartment structure but a side rule the scan lost.
    # Accepted below only when stacked on a segment that has its sides.
    weak: list[tuple[int, int, int, int]] = []
    for j, (bx, bw, brow, bt) in enumerate(rules):
        for i in range(j - 1, -1, -1):
            ax, aw, arow, at = rules[i]
            y0 = arow - at // 2
            y1 = brow + (bt - bt // 2)
            if y1 - y0 > max_height:
                break  # sorted by row: everything earlier is further away
            if y1 - y0 < min_height:
                continue
            tol = max(3, 2 * max(at, bt))
            a_covers_b = ax <= bx + tol and ax + aw >= bx + bw - tol
            b_covers_a = bx <= ax + tol and bx + bw >= ax + aw - tol
            if not (a_covers_b or b_covers_a):
                continue
            # The frame's extent is the shorter rule's: the longer one may
            # also be a dimension line the frame was drawn against.
            x0, x1 = (bx, bx + bw) if a_covers_b else (ax, ax + aw)
            if x1 - x0 < cfg.frame_min_width_px:
                continue
            crop = binary[y0:y1, x0:x1]
            cells = _compartment_cells(crop)
            if not (FRAME_CELL_COUNT[0] <= len(cells) <= FRAME_CELL_COUNT[1]):
                continue
            inner_h = (y1 - y0) - at - bt
            if inner_h < max(4, cfg.frame_min_height_px - 4):
                continue  # two rules with no room between them for a glyph
            if not (SYMBOL_CELL_ASPECT[0] <= cells[0][2] / inner_h <= SYMBOL_CELL_ASPECT[1]):
                continue
            # A pair of rules two frame-heights apart spanning a stack is a
            # composite's outer rules with a segment's worth of structure
            # inside; the pairs of adjacent rules describe it better. Judged
            # per segment, so the aspect gate sees one frame, not the stack.
            segment_count = max(1, _count_segments(crop))
            aspect = ((x1 - x0) / max(y1 - y0, 1)) * segment_count
            if not (cfg.frame_min_aspect <= aspect <= cfg.frame_max_aspect):
                continue
            coverage = (crop > 0).mean(axis=0)
            end = max(1, min(tol, coverage.size // 4))
            box = (x0, y0, x1 - x0, y1 - y0)
            if coverage[:end].max() >= SIDE_SPAN and coverage[-end:].max() >= SIDE_SPAN:
                strong.append(box)
            else:
                weak.append(box)

    # A segment of a composite stack whose own side rule broke in the scan is
    # still bounded by the sides of the segments above and below it.
    for box in weak:
        if any(_stacked(box, other, gap=gap) for other in strong):
            strong.append(box)
    return strong


def _merge_segments(
    segments: Sequence[tuple[int, int, int, int]],
    *,
    gap: int = MAX_RULE_PX,
) -> list[tuple[int, int, int, int]]:
    """Boxes sharing rules become one: composite stacks and duplicates.

    Every box here has passed its gates on its own, so the union is not
    re-judged — a stack is wider than it is tall only segment by segment.
    A box that is a shorter box plus extra rows (see :func:`_redundant`)
    gives way first, so a stack's outer pair does not swallow its segments.
    A box lying *inside* another with a different extent is kept: the inner
    one may be the frame and the outer the part outline around it, and the
    parser turns away whichever opens with no characteristic.
    """
    boxes = sorted(segments, key=lambda s: (s[1], s[0]))
    boxes = [
        box
        for box in boxes
        if not any(
            other is not box and _redundant(box, other, gap=gap)
            for other in boxes
        )
    ]
    merged: list[list[int]] = []
    for box in boxes:
        for other in merged:
            if _stacked(box, other, gap=gap):
                x, y, bw, bh = box
                px, py, pw, ph = other
                nx, ny = min(px, x), min(py, y)
                other[:] = [nx, ny, max(px + pw, x + bw) - nx, max(py + ph, y + bh) - ny]
                break
        else:
            merged.append(list(box))
    return [(x, y, bw, bh) for x, y, bw, bh in merged]


def _inner_rule_rows(binary: np.ndarray) -> list[int]:
    """Rows carrying a full-width rule, excluding the top and bottom borders.

    Shared by segment counting and segment splitting so the two cannot
    disagree about how many segments a rectangle holds.
    """
    if cv2 is None:
        return []
    h, w = binary.shape[:2]
    if h < 8 or w < 8:
        return []
    horizontal = _line_mask(binary, True, max(6, int(w * 0.8)))
    rows = np.nonzero((horizontal > 0).mean(axis=1) >= 0.6)[0]
    if rows.size == 0:
        return []
    rules: list[int] = []
    start = previous = rows[0]
    for row in rows[1:]:
        if row - previous > 2:
            rules.append(int((start + previous) // 2))
            start = row
        previous = row
    rules.append(int((start + previous) // 2))
    edge = max(2, int(h * 0.08))
    return [r for r in rules if edge < r < h - edge]


def _count_segments(binary: np.ndarray) -> int:
    """How many stacked frame segments a rectangle holds (1 when single)."""
    h = binary.shape[0]
    bounds = [0, *_inner_rule_rows(binary), h]
    floor = max(6, int(h * 0.15))
    return sum(1 for a, b in zip(bounds, bounds[1:]) if (b - a) >= floor) or 1


def erase_rules(frame_image: Image.Image) -> Image.Image:
    """Return the frame with its ruled lines painted out, content untouched.

    Cropping *around* the rules cannot be made to work: an inset wide enough
    to clear a border at one frame size clips a modifier drawn flush to a
    divider at another, and both failures are silent. Erasing the rules
    instead leaves every compartment's content whole and lets the crops be
    taken tight.

    The length thresholds are what keep this safe. A horizontal rule is
    erased only if it spans at least 80% of the whole frame, so the
    straightness glyph — a horizontal stroke filling its own compartment,
    roughly a third of the frame — is never mistaken for one. The vertical
    rules erased are exactly the borders and dividers
    :func:`_compartment_structure` found, so the stem of a ⟂, which runs the
    height of a scanned frame, is not erased as if it were a divider.

    The erased band is grown with the rule's own thickness: a rule's
    antialiased halo is below one threshold and above the next, and what one
    binarization left behind the glyph crop's own binarization picked up as
    ink at the corner of the cell, which then set the glyph's bounding box.
    """
    if cv2 is None:
        return frame_image
    gray = to_gray(frame_image)
    binary = binarize(gray)
    h, w = binary.shape[:2]
    if h < 6 or w < 6:
        return frame_image

    rules = _line_mask(binary, True, max(8, int(w * 0.8)))
    structure = _compartment_structure(binary)
    thickness = max(
        [width for _, width in structure.vertical_rules]
        + [int(round(float((rules > 0).sum(axis=1).max()) / max(w, 1) * 0)) + 1]
    )
    for centre, width in structure.vertical_rules:
        half = max(1, width // 2)
        rules[:, max(0, centre - half) : min(w, centre + half + 1)] = 255
    if not rules.any():
        return frame_image
    # Grow by the rule thickness so antialiased rule edges go with it.
    grow = max(3, (thickness + 2) | 1)
    rules = cv2.dilate(
        rules, cv2.getStructuringElement(cv2.MORPH_RECT, (grow, grow)), iterations=1
    )
    arr = np.array(frame_image.convert("RGB"))
    arr[rules > 0] = (255, 255, 255)
    return Image.fromarray(arr)


def split_segments(frame_image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Split a detected rectangle into its stacked composite segments.

    The segments of a composite frame share their side rules, so the edge
    morphology that finds frames returns the whole stack as one rectangle —
    they cannot be separated at detection time. They are separated here, at
    the full-width horizontal rules *between* the top and bottom borders.

    Returns one band per segment, in reading order. A single frame yields one
    band covering itself.
    """
    if cv2 is None:
        return []
    gray = to_gray(frame_image)
    binary = binarize(gray)
    h, w = binary.shape[:2]
    if h < 8 or w < 8:
        return [(0, 0, w, h)]

    inner = _inner_rule_rows(binary)
    if not inner:
        return [(0, 0, w, h)]

    bounds = [0, *inner, h]
    bands: list[tuple[int, int, int, int]] = []
    for top, bottom in zip(bounds, bounds[1:]):
        bh = bottom - top
        # A band too short to hold a glyph is a double rule, not a segment.
        if bh < max(6, int(h * 0.15)):
            continue
        bands.append((0, top, w, bh))
    return bands or [(0, 0, w, h)]


def split_compartments(
    frame_image: Image.Image, *, config: GdtConfig | None = None
) -> list[tuple[int, int, int, int]]:
    """Split a frame into compartments at its internal vertical rules.

    Works from the rules themselves — a column of ink running most of the
    frame's height — rather than from gaps between characters, because the gap
    between "Ø0" and "5" is as wide as some real compartment gaps.
    """
    if cv2 is None:
        return []
    return _compartment_cells(binarize(to_gray(frame_image)))


#: Narrowest a compartment can be, as a fraction of the frame's height. ASME
#: Y14.5 draws the symbol and datum compartments about as wide as the frame
#: is tall; nothing in a frame is a thin column. A column of ink that would
#: leave a narrower cell beside it is a glyph stroke — the stem of a ⟂ runs
#: nearly the full height of a scanned frame and was taken for the divider
#: after the symbol, which cut the symbol in half.
MIN_CELL_WIDTH_FRAC = 0.7

#: Width of the ink-free margin either side of a thin vertical run that lets
#: it count as a divider, as a fraction of the frame height.
DIVIDER_MARGIN_FRAC = 0.15

#: Most ink a clear margin may hold. Not zero: the slanted side of a flatness
#: glyph reaches a tenth of the way into the margin beside its divider.
DIVIDER_MARGIN_INK = 0.15

#: A column inked over this fraction of the frame's full height, rules
#: included, joins the top rule to the bottom one: a divider, not a stroke.
DIVIDER_FULL_SPAN = 0.95


@dataclass(frozen=True)
class FrameStructure:
    """Where a frame's rules are, in its own crop's pixels."""

    #: ``(x, y, width, height)`` per compartment, between the dividers.
    cells: tuple[tuple[int, int, int, int], ...]
    #: ``(centre, width)`` of each vertical rule, borders included.
    vertical_rules: tuple[tuple[int, int], ...]


def _compartment_structure(binary: np.ndarray) -> FrameStructure:
    """The compartments and vertical rules of an already-binarized frame.

    Shared by :func:`split_compartments`, :func:`erase_rules` and
    :func:`find_frames`, so the finder, the eraser and the parser cannot
    disagree about what a divider is.
    """
    h, w = binary.shape[:2]
    if h < 6 or w < 12:
        return FrameStructure((), ())

    # A divider is a *ruled line*, found with the same morphology that found
    # the frame. Thresholding raw column ink does not work: the upright of a
    # "0" or a "5" fills most of a short compartment's height too, and a
    # tolerance of "0.05" was being split into four compartments.
    vertical = _line_mask(binary, False, max(4, int(h * 0.8)))
    column_ink = (vertical > 0).mean(axis=0)
    divider_cols = np.nonzero(column_ink >= 0.6)[0]

    # Group adjacent columns into single dividers, keeping each run's width.
    runs: list[tuple[int, int]] = []  # (centre, width)
    if divider_cols.size:
        run_start = divider_cols[0]
        previous = divider_cols[0]
        for col in divider_cols[1:]:
            if col - previous > 2:
                runs.append((int((run_start + previous) // 2), int(previous - run_start + 1)))
                run_start = col
            previous = col
        runs.append((int((run_start + previous) // 2), int(previous - run_start + 1)))

    # A divider is a drawn rule and is as thick as the frame's own borders. An
    # upright digit stroke can survive the morphology — the "1" of a "0.1"
    # tolerance did, and split that compartment in two — but it comes back
    # much thinner than a rule. Judge each run against the frame's border
    # thickness, which is the first and last run.
    if runs:
        border_width = max(runs[0][1], runs[-1][1])
        floor = max(1, int(border_width * 0.6))
        # A divider one weight lighter than a scanned border — 2px against
        # borders of 5 and 6 at 3x, on a real flatness frame — fails the
        # thickness test and is still a divider. Two things a digit stroke
        # is not tell it apart: a divider runs from the top rule to the
        # bottom rule without a break, and nothing is drawn against it — a
        # compartment pads its contents, a digit does not pad its neighbour.
        # A thin run that does either stays.
        margin = max(2, int(h * DIVIDER_MARGIN_FRAC))
        band = binary[int(h * 0.2) : max(int(h * 0.2) + 1, int(h * 0.8)), :]
        full_span = (binary > 0).mean(axis=0)

        def thin_divider(centre: int, width: int) -> bool:
            half = max(1, width // 2) + 1
            if full_span[max(0, centre - half) : centre + half + 1].max() >= DIVIDER_FULL_SPAN:
                return True
            left = band[:, max(0, centre - half - margin) : max(0, centre - half)]
            right = band[:, centre + half + 1 : centre + half + 1 + margin]
            if left.size == 0 or right.size == 0:
                return False
            return (left > 0).mean() <= DIVIDER_MARGIN_INK and (right > 0).mean() <= DIVIDER_MARGIN_INK

        runs = [r for r in runs if r[1] >= floor or thin_divider(*r)]

    # Separate the frame's own left and right edges from the dividers.
    edge = max(2, int(w * 0.04))
    borders = [r for r in runs if not (edge < r[0] < w - edge)]
    inner = [r for r in runs if edge < r[0] < w - edge]

    # A divider leaves a compartment of plausible width on its left. One that
    # would not is a stroke inside a compartment, and is skipped so that the
    # compartment runs on to the next real divider.
    min_cell = max(4, int(h * MIN_CELL_WIDTH_FRAC))
    dividers: list[tuple[int, int]] = []
    left = 0
    for run in inner:
        if run[0] - left < min_cell:
            continue
        dividers.append(run)
        left = run[0]

    bounds = [0, *(c for c, _ in dividers), w]
    cells: list[tuple[int, int, int, int]] = []
    for lo, hi in zip(bounds, bounds[1:]):
        cw = hi - lo
        if cw < max(4, int(h * 0.3)):
            continue
        cells.append((lo, 0, cw, h))

    # The segments of a composite frame share one detected rectangle, so a
    # shorter lower segment inherits the width of the longest one and picks up
    # empty cells past its own right edge. Those are not compartments, and
    # left in place they would be read as unreadable datum references.
    inset = max(1, int(h * 0.12))
    while cells:
        cx, cy, cw, ch = cells[-1]
        interior = binary[cy + inset : cy + ch - inset, cx + inset : cx + cw - inset]
        if interior.size and (interior > 0).mean() >= 0.01:
            break
        cells.pop()
    return FrameStructure(tuple(cells), tuple(borders + dividers))


def _compartment_cells(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """:func:`split_compartments` on an already-binarized frame."""
    return list(_compartment_structure(binary).cells)


# ── Reading a compartment ────────────────────────────────────────────────────

#: Letters used as datum references. ASME Y14.5 excludes I, O and Q because
#: they read as 1 and 0; a letter outside the set is reported with a warning
#: rather than silently accepted.
_DATUM_LETTERS = re.compile(r"^[A-Z]{1,2}$")
_RESERVED_DATUM_LETTERS = frozenset({"I", "O", "Q"})

#: The circled modifiers, by the text OCR tends to return for them.
_MODIFIER_TEXT: dict[str, str] = {
    "M": "mmc",
    "(M)": "mmc",
    "Ⓜ": "mmc",
    "L": "lmc",
    "(L)": "lmc",
    "Ⓛ": "lmc",
    "P": "projected_tolerance_zone",
    "(P)": "projected_tolerance_zone",
    "Ⓟ": "projected_tolerance_zone",
    "F": "free_state",
    "(F)": "free_state",
    "Ⓕ": "free_state",
    "T": "tangent_plane",
    "(T)": "tangent_plane",
    "Ⓣ": "tangent_plane",
    "U": "unequally_disposed_profile",
    "(U)": "unequally_disposed_profile",
    "Ⓤ": "unequally_disposed_profile",
    "S": "rfs",
    "(S)": "rfs",
    "Ⓢ": "rfs",
    "E": "envelope_requirement",
    "(E)": "envelope_requirement",
    "Ⓔ": "envelope_requirement",
    "I": "independency",
    "Ⓘ": "independency",
}

_DIAMETER_GLYPHS = "Ø⌀φΦ∅ø"
_TOLERANCE_VALUE = re.compile(r"(\d+(?:[.,]\d+)?)")


def _modifiers_in(text: str) -> list[str]:
    """Modifier ids mentioned in a compartment's text, in order of appearance."""
    found: list[tuple[int, str]] = []
    upper = (text or "").upper()
    for token, symbol_id in _MODIFIER_TEXT.items():
        # A bare letter only counts when it is parenthesised or circled — a
        # lone "M" in a tolerance compartment is more often a stray mark, and
        # a lone "P" is how AIGDT writes Ø.
        if len(token) == 1 and token.isalpha():
            continue
        index = upper.find(token.upper())
        if index >= 0:
            found.append((index, symbol_id))
    out: list[str] = []
    for _, symbol_id in sorted(found):
        if symbol_id not in out:
            out.append(symbol_id)
    return out


#: An integer with a leading zero is a decimal whose point OCR dropped: "03"
#: is 0.3, never three. A geometric tolerance of 3 would not be written with
#: a leading zero, and one of 0.3 routinely loses its point at scan size.
_LOST_POINT = re.compile(r"^0(\d+)$")


def parse_tolerance_compartment(text: str, confidence: float) -> dict[str, Any]:
    """Read the tolerance compartment: zone shape, value, and modifiers."""
    raw = (text or "").strip()
    has_diameter = any(g in raw for g in _DIAMETER_GLYPHS)
    modifiers = _modifiers_in(raw)

    # Strip the modifier tokens before looking for the number, so the "0.5" of
    # a projected zone height does not get read as the tolerance.
    stripped = re.sub(r"[()][A-Z][)]?", " ", raw.upper())
    match = _TOLERANCE_VALUE.search(stripped)
    value = None
    if match:
        number = match.group(1).replace(",", ".")
        lost = _LOST_POINT.match(number)
        if lost:
            number = f"0.{lost.group(1)}"
        try:
            value = float(number)
        except ValueError:
            value = None

    material = None
    for candidate in ("mmc", "lmc", "rfs"):
        if candidate in modifiers:
            material = candidate
            break

    return {
        "value": value,
        "diameter_zone": has_diameter,
        "modifiers": [m for m in modifiers if m not in {"mmc", "lmc", "rfs"}],
        "material_condition": material,
        "confidence": confidence,
        "raw": raw,
    }


#: A compartment crop is padded and upscaled to at least this tall before OCR.
#: A datum compartment is one capital letter in a box barely 30px high, and
#: PaddleOCR reads those badly — "A" came back as "Q", "B" as "3". Giving the
#: recogniser a larger image of the same letter is the cheapest fix available
#: and needs no model change.
OCR_MIN_HEIGHT = 96
OCR_PAD_PX = 8


def _for_ocr(cell: Image.Image) -> Image.Image:
    """Pad and upscale a compartment so the recogniser has something to read."""
    padded = Image.new(
        "RGB",
        (cell.width + 2 * OCR_PAD_PX, cell.height + 2 * OCR_PAD_PX),
        (255, 255, 255),
    )
    padded.paste(cell.convert("RGB"), (OCR_PAD_PX, OCR_PAD_PX))
    if padded.height >= OCR_MIN_HEIGHT:
        return padded
    factor = OCR_MIN_HEIGHT / max(padded.height, 1)
    return padded.resize(
        (max(1, int(padded.width * factor)), OCR_MIN_HEIGHT),
        Image.Resampling.LANCZOS,
    )


#: A modifier follows the value, so an unidentified ring counts as a possible
#: modifier only in this trailing fraction of the tolerance compartment.
MODIFIER_TRAILING_FRAC = 0.4

#: Modifiers that can appear in a tolerance compartment, as symbol ids.
_COMPARTMENT_MODIFIERS = (
    "mmc",
    "lmc",
    "rfs",
    "projected_tolerance_zone",
    "free_state",
    "tangent_plane",
    "unequally_disposed_profile",
    "statistical_tolerance",
    "continuous_feature",
)


def _find_diameter_zone_visually(
    cell: Image.Image, *, config: GdtConfig | None = None
) -> tuple[float | None, bool]:
    """Look for the ⌀ zone marker in a tolerance compartment, by shape.

    Returns ``(confidence, unresolved)``. Only a confirmed read sets the
    confidence: a wrongly invented ⌀ turns a parallel-plane tolerance zone
    into a cylindrical one, which is a different acceptance boundary.

    ``unresolved`` reports that something ring-shaped sits where a zone marker
    goes but was not identified. It is returned for callers that want it and
    is deliberately **not** used to mark the field unreadable: the same test
    fires on the digit zero of a plain "0.5", so acting on it marks the
    majority of frames unknown. See the note at the call site.

    The ⌀ sits at the *left* of the compartment, before the number, so the
    search is limited to that end. Scanning the whole cell would let the "0"
    of "0.30" answer, and a zero is a ring too.
    """
    from gdt_detect import find_glyph_boxes

    cfg = config or get_config()
    if cell.width < 6 or cell.height < 6:
        return None, False
    # A zone marker occupies roughly the first third of the compartment.
    lead = cell.crop((0, 0, max(4, int(cell.width * 0.38)), cell.height))
    page_edge = max(cell.height * 12, 256)
    best: float | None = None
    unresolved = False
    for x, y, w, h in find_glyph_boxes(lead, config=cfg, page_edge=page_edge):
        crop = lead.crop((x, y, x + w, y + h))
        try:
            # Only "diameter", not also "spherical_diameter": the two share
            # one template, which makes the shared-glyph rule force an
            # ambiguous status that can never confirm. The distinction does
            # not matter here — either way the zone is diametral.
            det = classify_crop(crop, allowed=("diameter",), config=cfg)
        except Exception:  # noqa: BLE001
            continue
        if det.symbol_id and det.status == STATUS_CONFIRMED:
            if best is None or det.confidence > best:
                best = det.confidence
            continue
        family = (det.evidence.get("topology") or {}).get("family", "")
        if family.startswith("ring"):
            # A ring as wide as it is tall, at the head of the compartment,
            # before the number. The only other ring that stands there is
            # the digit 0, and a 0 is taller than it is wide. On a scanned
            # sheet the ⌀ is a smudged ring with a hint of a slash that no
            # template clears the bar on; every one of ten position frames
            # read without its diametral zone until this was allowed.
            if LEAD_RING_ASPECT[0] <= w / max(h, 1) <= LEAD_RING_ASPECT[1]:
                if best is None:
                    best = LEAD_RING_CONFIDENCE
                continue
            unresolved = True
    return best, unresolved


#: A ⌀ at the head of a tolerance compartment is a ring about as wide as it
#: is tall; the digit 0 that could stand there instead is narrower than this.
LEAD_RING_ASPECT = (0.8, 1.3)

#: Confidence given to a ⌀ identified by position and shape alone.
LEAD_RING_CONFIDENCE = 0.5


@dataclass
class ModifierFind:
    """What a visual sweep of a compartment turned up."""

    #: Confidently identified modifier, if any.
    symbol_id: str | None = None
    confidence: float = 0.0
    #: True when a ring-shaped mark was found that could not be identified.
    unresolved: bool = False
    #: Best guesses for an unresolved mark, for the operator to arbitrate.
    candidates: list[str] = field(default_factory=list)
    note: str | None = None


def _find_modifier_visually(
    cell: Image.Image, *, config: GdtConfig | None = None
) -> ModifierFind:
    """Look for a circled modifier in a compartment by shape, not by text.

    Reports three outcomes, not two. "No modifier here" and "a modifier I
    could not read" are different facts: the first means the tolerance applies
    regardless of feature size, the second means nobody knows yet. Collapsing
    them would silently turn an unreadable Ⓜ into an RFS callout and delete a
    bonus tolerance an inspector is entitled to.

    A modifier at a typical frame height is around 20px with a 10px letter
    inside it, which is at the edge of what any method resolves — so the
    unresolved outcome is the common one on a sheet rendered at scanning
    resolution, not an edge case.
    """
    from gdt_detect import find_glyph_boxes

    cfg = config or get_config()
    found = ModifierFind()
    # An ambiguous reading that no rival contends is not an ambiguity; it is a
    # confident shape that fell short of the accept bar. Held back for the
    # trailing zone, where position can settle it (see below).
    unrivalled: tuple[str, float] | None = None
    # A modifier legitimately fills most of a compartment, so the proposal
    # step is told to measure its size cap against a notional page rather than
    # against this small crop.
    page_edge = max(cell.height * 12, 256)
    for x, y, w, h in find_glyph_boxes(cell, config=cfg, page_edge=page_edge):
        crop = cell.crop((x, y, x + w, y + h))
        try:
            det = classify_crop(crop, allowed=_COMPARTMENT_MODIFIERS, config=cfg)
        except Exception:  # noqa: BLE001
            continue
        if det.symbol_id is not None and det.status == STATUS_CONFIRMED:
            if det.confidence > found.confidence:
                found.symbol_id = det.symbol_id
                found.confidence = det.confidence
            continue
        # Nothing confident. Was it at least ring-shaped — i.e. is there
        # something here that a modifier would explain? Only where a
        # modifier stands: after the value, in the trailing part of the
        # compartment. The digit 0 is a ring too, and "0.3" opens with one;
        # judged anywhere in the cell, every tolerance under 1 reported a
        # modifier it could not read, and every frame went to review.
        topology = det.evidence.get("topology") or {}
        family = topology.get("family", "")
        trailing = (x + w / 2) >= cell.width * (1.0 - MODIFIER_TRAILING_FRAC)
        # Position resolves what the confidence band left open. Only a
        # modifier stands after the tolerance value, so a mark there whose
        # best reading IS a modifier and which nothing else contends is that
        # modifier. This is the registry's own rule -- context may settle an
        # ambiguity the detector left open, never overrule it -- and without
        # it a Ⓜ drawn at frame height scored 0.60 against an accept bar of
        # 0.62, with an empty rival list, and every frame carrying one came
        # back "unreadable". The confidence is reported as measured, not
        # raised, so a reviewer sees how firm the reading actually was.
        if (
            trailing
            and det.symbol_id in _COMPARTMENT_MODIFIERS
            and det.status == STATUS_AMBIGUOUS
            and not [
                a
                for a in det.alternatives
                if a.score >= det.confidence - cfg.rival_margin
            ]
            and det.confidence > (unrivalled[1] if unrivalled else 0.0)
        ):
            unrivalled = (det.symbol_id, det.confidence)
            continue
        if family.startswith("ring") and trailing and not found.unresolved:
            ranked = [det.symbol_id, *(a.symbol_id for a in det.alternatives)]
            found.unresolved = True
            found.candidates = [
                s for s in ranked if s in _COMPARTMENT_MODIFIERS
            ][:4]
            found.note = (
                f"a ring-shaped mark {w}×{h}px sits in the tolerance "
                "compartment but is too small to identify; it may be a "
                "material-condition modifier"
            )
    if found.symbol_id is None and unrivalled is not None:
        found.symbol_id, found.confidence = unrivalled
        found.unresolved = False
        found.candidates = []
        found.note = None
    return found


def parse_datum_compartment(
    text: str, confidence: float, precedence: int
) -> tuple[DatumRef | None, list[str]]:
    """Read one datum compartment into a reference plus its modifier."""
    raw = (text or "").strip()
    warnings: list[str] = []
    if not raw:
        return None, warnings

    modifiers = _modifiers_in(raw)
    material = next((m for m in modifiers if m in {"mmc", "lmc", "rfs"}), None)

    # The letter is what is left once modifiers and punctuation are removed.
    # The parenthesised modifier has to go *as a group*: stripping brackets
    # first turns "B(M)" into "BM", which then reads as the compound datum
    # "B-M" — a different datum, silently.
    without_modifiers = re.sub(r"\(\s*[A-Za-z]{1,2}\s*\)", " ", raw.upper())
    without_modifiers = re.sub(r"[ⓂⓁⓈⓅⒻⓉⓊⒺⒾ]", " ", without_modifiers)
    letters = re.sub(r"[()\[\]\s]", "", without_modifiers)
    letters = re.sub(r"[^A-Z]", "", letters)
    if not letters:
        return None, warnings

    # A compound datum is written A-B; keep both.
    if not _DATUM_LETTERS.match(letters):
        if len(letters) > 2:
            warnings.append(
                f"datum compartment {precedence} read as {letters!r}, which is "
                "not a datum letter"
            )
            return None, warnings
    label = letters[:2] if len(letters) == 2 else letters[:1]

    if label in _RESERVED_DATUM_LETTERS:
        warnings.append(
            f"{label!r} is not used as a datum letter in ASME Y14.5 (I, O and Q "
            "are reserved); it may be a misread"
        )

    return (
        DatumRef(
            label=label,
            precedence=precedence,
            material_condition=material,
            confidence=confidence,
            raw=raw,
        ),
        warnings,
    )


# ── Parsing a whole frame ────────────────────────────────────────────────────

#: Controls that cannot reference a datum. Reporting "no datum" for one of
#: these is ``not_applicable``, not ``absent`` — the distinction matters to
#: anyone checking a drawing for completeness.
_FORM_CONTROLS = frozenset(
    {"straightness", "flatness", "circularity", "cylindricity"}
)


def parse_frame(
    frame_image: Image.Image,
    *,
    bbox: tuple[float, float, float, float],
    ocr_fn: OcrFn | None = None,
    transform: Transform = IDENTITY,
    page: int = 1,
    segment_index: int = 0,
    composite_id: str | None = None,
    config: GdtConfig | None = None,
) -> FeatureControlFrame:
    """Read one frame image into a structured :class:`FeatureControlFrame`."""
    cfg = config or get_config()
    frame = FeatureControlFrame(
        id=uuid.uuid4().hex[:12],
        bbox=bbox,
        page=page,
        segment_index=segment_index,
        composite_id=composite_id,
    )

    cells = split_compartments(frame_image, config=cfg)
    if len(cells) < 2:
        # A frame has a characteristic and a tolerance at the least. A box
        # with one compartment is a boxed basic dimension or a datum feature
        # symbol, and reading it as a frame would name its contents a symbol.
        frame.warnings.append(
            "frame has no readable compartments"
            if not cells
            else "a single compartment cannot be a feature control frame"
        )
        frame.tolerance = Field.unreadable()
        return frame

    fw, fh = frame_image.size
    scale_x = fw / max(bbox[2], 1e-6)
    scale_y = fh / max(bbox[3], 1e-6)

    # Compartment boxes run rule-to-rule, so a naive crop carries the frame's
    # own lines into the glyph. That is fatal downstream: classification
    # tight-crops to the ink, and with a border included the "glyph" becomes
    # glyph-plus-two-lines and matches nothing — every frame was dropped for
    # having no readable characteristic. The rules are erased instead of
    # cropped around, so a modifier drawn flush to a divider survives.
    clean_image = erase_rules(frame_image)
    inset = CELL_INSET_MIN_PX

    def cell_crop(cx: int, cy: int, cw: int, ch: int) -> Image.Image:
        left = cx + inset
        top = cy + inset
        right = max(left + 1, cx + cw - inset)
        bottom = max(top + 1, cy + ch - inset)
        return clean_image.crop((left, top, right, bottom))

    def read_cell(index: int) -> None:
        cx, cy, cw, ch = cells[index]
        crop = cell_crop(cx, cy, cw, ch)
        text, conf = ("", 0.0)
        if ocr_fn is not None:
            try:
                text, conf = ocr_fn(_for_ocr(crop))
            except Exception as exc:  # noqa: BLE001 - one bad cell must not
                # kill the frame; the compartment is reported unreadable.
                frame.warnings.append(f"OCR failed on compartment {index}: {exc}")
        cell = frame.compartments[index]
        cell.text = (text or "").strip()
        cell.confidence = float(conf or 0.0)

    for index, (cx, cy, cw, ch) in enumerate(cells):
        frame.compartments.append(
            Compartment(
                index=index,
                bbox=(
                    bbox[0] + cx / scale_x,
                    bbox[1] + cy / scale_y,
                    cw / scale_x,
                    ch / scale_y,
                ),
            )
        )

    # ── compartment 0: the geometric characteristic ──────────────────────────
    # Read first, and alone. The finder over-produces on purpose — every
    # boxed basic dimension and ruled table row on the sheet is a candidate —
    # and OCR is the expensive step, so a box that opens with no
    # characteristic is turned away before its other compartments are read.
    read_cell(0)
    first = frame.compartments[0]
    symbol_crop = cell_crop(*cells[0])
    # OCR of a glyph is noise unless it returned the glyph itself. PaddleOCR
    # reads a ⌖ as "0", "4", "2" or "中", and a bare "0" corroborates the
    # ring-shaped symbols: measured on a real sheet, two position frames came
    # back as circularity, *confirmed*, on the strength of that one
    # character lifting a plain circle past the cross it was drawn with.
    glyph_text = first.text if any(ch in reg.BY_UNICODE for ch in first.text) else None
    # Only the 14 characteristics may open a frame. This is the single most
    # valuable piece of context in the whole system: it is why a circle here
    # is circularity and not a drilled hole.
    frame.symbol = classify_crop(
        symbol_crop,
        ocr_text=glyph_text,
        allowed=[s.id for s in reg.FRAME_CHARACTERISTICS],
        transform=Transform(
            offset_x=first.bbox[0],
            offset_y=first.bbox[1],
            scale_x=scale_x,
            scale_y=scale_y,
        ),
        page=page,
        config=cfg,
    )
    first.detection = frame.symbol

    if frame.symbol is None or frame.symbol.status == STATUS_UNKNOWN:
        frame.warnings.append(
            "no geometric characteristic symbol could be read from the first "
            "compartment — this may not be a feature control frame"
        )
        frame.tolerance = Field.unreadable()
        return frame

    for index in range(1, len(cells)):
        read_cell(index)

    # ── compartment 1: the tolerance ─────────────────────────────────────────
    if len(frame.compartments) < 2:
        frame.tolerance = Field.unreadable()
        frame.warnings.append("frame has no tolerance compartment")
    else:
        tol_cell = frame.compartments[1]
        parsed = parse_tolerance_compartment(tol_cell.text, tol_cell.confidence)
        # A material-condition modifier is a glyph, and OCR does not read
        # glyphs — PaddleOCR returned "0.5€" for a tolerance carrying an Ⓜ,
        # so a text-only parse silently dropped the modifier that changes the
        # whole meaning of the callout. Look for it in the pixels too.
        unresolved_modifier: ModifierFind | None = None
        # The ⌀ that makes a tolerance zone cylindrical is a glyph, and OCR
        # does not read glyphs. On a real drawing rendered at note size the
        # compartment comes back as "0.30" with the ⌀ silently dropped, and a
        # text-only parse then reports a parallel-plane zone for a callout
        # that specifies a diametral one — a different control. Look for it in
        # the pixels, the same way the material-condition modifier is found.
        tol_crop = cell_crop(*cells[1])
        modifier_crop = tol_crop
        if not parsed["diameter_zone"]:
            zone, _ = _find_diameter_zone_visually(tol_crop, config=cfg)
            if zone is not None:
                parsed["diameter_zone"] = True
                # The ⌀ is a ring too; searched again for modifiers it would
                # be reported as one that could not be read.
                modifier_crop = tol_crop.crop(
                    (int(tol_crop.width * 0.38), 0, tol_crop.width, tol_crop.height)
                )

        if parsed["material_condition"] is None:
            visual = _find_modifier_visually(modifier_crop, config=cfg)
            if visual.symbol_id is not None:
                if visual.symbol_id in {"mmc", "lmc", "rfs"}:
                    parsed["material_condition"] = visual.symbol_id
                    parsed["confidence"] = max(
                        parsed["confidence"], visual.confidence
                    )
                elif visual.symbol_id not in parsed["modifiers"]:
                    parsed["modifiers"].append(visual.symbol_id)
            elif visual.unresolved:
                unresolved_modifier = visual
        if parsed["value"] is None:
            frame.tolerance = Field.unreadable(raw=parsed["raw"] or None)
            if tol_cell.text:
                frame.warnings.append(
                    f"tolerance compartment read as {tol_cell.text!r}, which "
                    "carries no number"
                )
        else:
            frame.tolerance = Field.present(
                parsed["value"], parsed["confidence"], parsed["raw"]
            )
        # Reported two-state, not three. An "unreadable" state was tried: the
        # ring test that would flag an unresolvable ⌀ also fires on the digit
        # zero of a plain "0.5", so every frame without a zone marker — the
        # majority — came back unknown. A false "unreadable" on most frames is
        # worse than a confident False on the few whose marker is too small to
        # see, so the limitation is documented instead of half-guarded.
        frame.diameter_zone = Field.present(
            parsed["diameter_zone"], parsed["confidence"], parsed["raw"]
        )
        frame.modifiers = parsed["modifiers"]
        if parsed["material_condition"]:
            frame.material_condition = Field.present(
                parsed["material_condition"], parsed["confidence"], parsed["raw"]
            )
        elif unresolved_modifier is not None:
            # Something modifier-shaped is there and could not be read. Saying
            # "absent" here would be read downstream as RFS, which is a
            # different callout with a different acceptance limit.
            frame.material_condition = Field.unreadable(raw=parsed["raw"] or None)
            frame.warnings.append(
                (unresolved_modifier.note or "an unidentified modifier is present")
                + (
                    f" (closest matches: "
                    f"{', '.join(unresolved_modifier.candidates)})"
                    if unresolved_modifier.candidates
                    else ""
                )
            )
        else:
            # No modifier means RFS by default — but that is an interpretation
            # of the standard, not something read off the page, so it is not
            # recorded as a detected value.
            frame.material_condition = Field(
                value=None, state=ABSENT, raw=parsed["raw"]
            )

    # ── remaining compartments: datum references, in precedence order ────────
    characteristic = frame.symbol.symbol_id if frame.symbol else None
    for precedence, cell in enumerate(frame.compartments[2:], start=1):
        ref, warnings = parse_datum_compartment(
            cell.text, cell.confidence, precedence
        )
        frame.warnings.extend(warnings)
        if ref is not None:
            frame.datums.append(ref)
        elif cell.text:
            frame.warnings.append(
                f"datum compartment {precedence} read as {cell.text!r} and "
                "could not be resolved to a datum letter"
            )

    if not frame.datums and characteristic in _FORM_CONTROLS:
        # A form control referencing no datum is correct, not missing.
        frame.warnings.append(
            f"{reg.get(characteristic).name} is a form control and takes no "
            "datum reference"
        )

    frame.confidence = _frame_confidence(frame)
    return frame


def _frame_confidence(frame: FeatureControlFrame) -> float:
    """Overall confidence: the weakest link, not the average.

    A frame whose symbol is certain and whose tolerance is unreadable is not
    "70% read" — it is a frame you cannot act on. Averaging would hide that.
    """
    parts: list[float] = []
    if frame.symbol is not None:
        parts.append(frame.symbol.confidence if frame.symbol.symbol_id else 0.0)
    else:
        parts.append(0.0)
    if frame.tolerance.state == PRESENT:
        parts.append(frame.tolerance.confidence)
    elif frame.tolerance.state == UNREADABLE:
        parts.append(0.0)
    if frame.datums:
        parts.append(min(d.confidence for d in frame.datums))
    return round(min(parts), 4) if parts else 0.0


# ── Composite frames ─────────────────────────────────────────────────────────


def group_composite(
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    tolerance_frac: float = 0.25,
) -> list[list[int]]:
    """Group frame boxes that stack into one composite frame.

    A composite frame is drawn as segments sharing a left edge and a width,
    stacked with no gap. Grouping them matters because the segments of a
    composite control are read together — the lower segment refines the
    upper — and reporting them as unrelated frames loses that.

    Returns lists of indexes into ``boxes``, in reading order (top to bottom).
    """
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    groups: list[list[int]] = []
    for index in order:
        x, y, w, h = boxes[index]
        placed = False
        for group in groups:
            px, py, pw, ph = boxes[group[-1]]
            shares_left = abs(px - x) <= pw * tolerance_frac
            shares_width = abs(pw - w) <= max(pw, w) * tolerance_frac
            stacked = 0 <= (y - (py + ph)) <= ph * tolerance_frac
            if shares_left and shares_width and stacked:
                group.append(index)
                placed = True
                break
        if not placed:
            groups.append([index])
    return groups


def parse_frames(
    image: Image.Image,
    *,
    ocr_fn: OcrFn | None = None,
    transform: Transform = IDENTITY,
    page: int = 1,
    config: GdtConfig | None = None,
    reference_edge: int | None = None,
) -> list[FeatureControlFrame]:
    """Find and read every feature control frame in a region.

    Frames that turn out not to open with a geometric characteristic are
    dropped: a title-block cell and a basic-dimension box are both rectangles,
    and neither is a feature control frame. The drop is the point at which a
    false positive is cheapest to remove.
    """
    cfg = config or get_config()
    detected = find_frames(
        image, transform=IDENTITY, config=cfg, reference_edge=reference_edge
    )
    if not detected:
        return []
    min_height = max(
        cfg.frame_min_height_px,
        int((reference_edge or min(image.size)) * FRAME_MIN_HEIGHT_FRAC),
    )

    # A composite frame is detected as one tall rectangle because its segments
    # share side rules. Split each detection into its bands before grouping,
    # so a two-segment position control becomes two frames that know they
    # belong together rather than one frame with twice the compartments.
    boxes: list[tuple[float, float, float, float]] = []
    for x, y, w, h in detected:
        crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
        for bx, by, bw, bh in split_segments(crop):
            # A band too short to be a frame is a double rule or a table's
            # header line. Parsed anyway, its sliver of ink classified as
            # straightness — a horizontal stroke — with real confidence. The
            # finder's structural gates are applied again here because a
            # band of a merged box is not the segment the finder judged.
            if bh < min_height or bw < cfg.frame_min_width_px:
                continue
            band = binarize(to_gray(crop.crop((bx, by, bx + bw, by + bh))))
            cells = _compartment_cells(band)
            if not (FRAME_CELL_COUNT[0] <= len(cells) <= FRAME_CELL_COUNT[1]):
                continue
            if not (SYMBOL_CELL_ASPECT[0] <= cells[0][2] / max(bh, 1) <= SYMBOL_CELL_ASPECT[1]):
                continue
            boxes.append((x + bx, y + by, float(bw), float(bh)))

    groups = group_composite(boxes)
    frames: list[FeatureControlFrame] = []
    for group in groups:
        composite_id = uuid.uuid4().hex[:12] if len(group) > 1 else None
        for segment_index, box_index in enumerate(group):
            x, y, w, h = boxes[box_index]
            crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
            try:
                frame = parse_frame(
                    crop,
                    bbox=transform.apply((x, y, w, h)),
                    ocr_fn=ocr_fn,
                    transform=transform,
                    page=page,
                    segment_index=segment_index,
                    composite_id=composite_id,
                    config=cfg,
                )
            except Exception as exc:  # noqa: BLE001 - one bad frame must not
                # take the page down with it; the brief asks for isolation.
                frames.append(
                    FeatureControlFrame(
                        id=uuid.uuid4().hex[:12],
                        bbox=transform.apply((x, y, w, h)),
                        page=page,
                        segment_index=segment_index,
                        composite_id=composite_id,
                        warnings=[f"frame could not be parsed: {exc}"],
                    )
                )
                continue
            if frame.symbol is None or frame.symbol.symbol_id is None:
                continue
            frames.append(frame)

    return drop_nested_frames(frames)


def drop_nested_frames(frames: Sequence[FeatureControlFrame]) -> list[FeatureControlFrame]:
    """A frame does not contain a frame.

    The finder keeps nested boxes because the inner one may be the frame and
    the outer the part outline around it; the outline is turned away by the
    parser for opening with no characteristic, so whatever is still nested
    inside a *parsed* frame is a box its own contents made — the legs and
    crossbar of a datum "A" under the top rule pair into one — and gives way
    to the frame. Applied again over a whole page, because a tile can hold
    the small box without the frame around it.
    """

    def inside(inner: FeatureControlFrame, outer: FeatureControlFrame) -> bool:
        ix, iy, iw, ih = inner.bbox
        ox, oy, ow, oh = outer.bbox
        if ow * oh <= iw * ih:
            return False
        overlap_w = max(0.0, min(ix + iw, ox + ow) - max(ix, ox))
        overlap_h = max(0.0, min(iy + ih, oy + oh) - max(iy, oy))
        return overlap_w * overlap_h >= 0.9 * iw * ih

    return [
        frame
        for frame in frames
        if not any(other is not frame and inside(frame, other) for other in frames)
    ]


# ── Schema validation ────────────────────────────────────────────────────────

_VALID_STATES = frozenset({PRESENT, ABSENT, UNREADABLE, NOT_APPLICABLE})


def validate(payload: dict[str, Any]) -> list[str]:
    """Check a serialized frame against the shape consumers rely on.

    Returns a list of problems, empty when the payload is well formed. Used by
    the tests and by the API before a frame is published, so a malformed frame
    is caught here rather than in the browser.
    """
    problems: list[str] = []

    if payload.get("type") != "feature_control_frame":
        problems.append("type must be 'feature_control_frame'")

    box = payload.get("bounding_box")
    if not isinstance(box, dict):
        problems.append("bounding_box is missing")
    else:
        for key in ("x", "y", "width", "height"):
            if not isinstance(box.get(key), (int, float)):
                problems.append(f"bounding_box.{key} must be a number")
        if isinstance(box.get("width"), (int, float)) and box["width"] <= 0:
            problems.append("bounding_box.width must be positive")
        if isinstance(box.get("height"), (int, float)) and box["height"] <= 0:
            problems.append("bounding_box.height must be positive")

    if not isinstance(payload.get("page"), int) or payload["page"] < 1:
        problems.append("page must be a positive integer")

    for name in ("tolerance", "diameter_zone", "material_condition"):
        field_payload = payload.get(name)
        if not isinstance(field_payload, dict):
            problems.append(f"{name} must be an object")
            continue
        if field_payload.get("state") not in _VALID_STATES:
            problems.append(
                f"{name}.state must be one of {sorted(_VALID_STATES)}, "
                f"got {field_payload.get('state')!r}"
            )
        if field_payload.get("state") == PRESENT and field_payload.get("value") is None:
            problems.append(f"{name} is marked present but carries no value")

    datums = payload.get("datums")
    if not isinstance(datums, list):
        problems.append("datums must be a list")
    else:
        precedences = [d.get("precedence") for d in datums if isinstance(d, dict)]
        if precedences != sorted(precedences):
            problems.append("datum precedence must increase in reading order")
        if len(set(precedences)) != len(precedences):
            problems.append("two datums share a precedence")
        for datum in datums:
            if not isinstance(datum, dict):
                problems.append("each datum must be an object")
                continue
            if not datum.get("label"):
                problems.append("a datum reference carries no label")

    if payload.get("tolerance_unit") != "drawing_unit":
        problems.append(
            "tolerance_unit must be 'drawing_unit' — the unit comes from the "
            "drawing, not from the symbol"
        )

    symbol = payload.get("symbol")
    if symbol is not None:
        if not isinstance(symbol, dict):
            problems.append("symbol must be an object or null")
        elif symbol.get("symbol_id") and symbol["symbol_id"] not in reg.BY_ID:
            problems.append(f"unknown symbol id {symbol['symbol_id']!r}")

    return problems
