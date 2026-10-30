"""
Stage 3 — find and name GD&T glyphs in a drawing region.

The strategy is hybrid on purpose. No single method is good enough here:

* **OCR alone** loses. PaddleOCR is trained on text; a ⌖ comes back as ``+``,
  a ⌭ as ``101``, a ⏥ as ``/7``. Worse, it returns those confidently, so its
  score cannot be used to reject them.
* **Template matching alone** over-fires. A drawing is full of circles and
  right angles, and ``matchTemplate`` will happily score a bolt hole as
  circularity.
* **Contour topology alone** cannot read a letter.

So each is used for what it is good at. **Shape comparison at an aligned
scale** does the naming: the glyph is cropped to its own ink and compared
against templates normalised identically, scoring the whole outline and the
glyph's interior separately. OCR text, where there is any, corroborates
through the registry's variant table. Context (``gdt_context``) arbitrates
what position alone can settle.

Topology — ring, slash, nested ring, stroke figure — is measured and reported,
but it does **not** move a score. It was built as a prior first, and measuring
it showed why that was wrong: hole-counting misreads concentricity's inner
ring and all-over's at 40px, so a confident wrong family penalised correct
template matches out of first place. It survives as evidence, which is what it
is actually good for.

Confidence is a blend, not a template score
-------------------------------------------
A ``cv2.matchTemplate`` peak is a similarity, not a probability, and treating
one as the other is how a detector ends up reporting 0.97 for a hole. The
confidence published here combines the shape score with OCR corroboration and
then *subtracts* for a close rival. A glyph that scored well but has an
equally good alternative is reported as ambiguous with both candidates
attached, which is the behaviour the brief requires — and in practice is what
happens for the pairs that are genuinely one glyph, such as counterbore and
spotface or profile-of-a-line and arc length.

Everything is reported in the coordinates the caller asks for, via
:class:`Transform`, so a detection found in an upscaled tile still lands on the
original page.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

import gdt_registry as reg
import gdt_templates as tpl
from gdt_config import GdtConfig, get_config
from image_preprocess import cad_ink_to_gray

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


# ── Detection record ─────────────────────────────────────────────────────────

#: Published with a named symbol the detector stands behind.
STATUS_CONFIRMED = "confirmed"
#: Something is there and it is probably one of ``alternatives``, but the
#: evidence does not separate them.
STATUS_AMBIGUOUS = "ambiguous"
#: A symbol-shaped mark that matches nothing in the registry well enough.
STATUS_UNKNOWN = "unknown"

#: Detection methods, reported so a result can be traced to how it was found.
M_TEMPLATE = "template_match"
M_TOPOLOGY = "contour_topology"
M_CIRCLED = "circled_letter"
M_OCR = "ocr_normalized"
M_FRAME = "frame_compartment"
M_TEXT = "text_pattern"


@dataclass
class Candidate:
    """One possible reading of a glyph, with the evidence behind it."""

    symbol_id: str
    score: float
    method: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        sym = reg.find(self.symbol_id)
        return sym.name if sym else self.symbol_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "name": self.name,
            "confidence": round(self.score, 4),
            "method": self.method,
            "evidence": self.evidence,
        }


@dataclass
class Detection:
    """One recognized (or explicitly unrecognized) mark on the drawing."""

    #: ``None`` when nothing in the registry explains the mark.
    symbol_id: str | None
    name: str
    status: str
    confidence: float
    #: (x, y, width, height) in the coordinate space the caller asked for.
    bbox: tuple[float, float, float, float]
    page: int = 1
    method: str = M_TEMPLATE
    raw_ocr: str | None = None
    #: The canonical Unicode glyph, when the registry has one.
    normalized: str | None = None
    alternatives: list[Candidate] = field(default_factory=list)
    #: Text or numeric value read alongside the symbol.
    associated_text: str | None = None
    value: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    #: Free-form trace of how the decision was reached.
    evidence: dict[str, Any] = field(default_factory=dict)
    #: Set when this detection belongs to a parsed feature control frame.
    frame_id: str | None = None

    @property
    def symbol(self) -> reg.GdtSymbol | None:
        return reg.find(self.symbol_id) if self.symbol_id else None

    def to_dict(self) -> dict[str, Any]:
        sym = self.symbol
        x, y, w, h = self.bbox
        out: dict[str, Any] = {
            "symbol_id": self.symbol_id,
            "name": self.name,
            "status": self.status,
            "confidence": round(float(self.confidence), 4),
            "bounding_box": {
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "width": round(float(w), 2),
                "height": round(float(h), 2),
            },
            "page": self.page,
            "method": self.method,
            "raw_ocr": self.raw_ocr,
            "normalized": self.normalized,
            "alternatives": [c.to_dict() for c in self.alternatives],
            "associated_text": self.associated_text,
            "value": self.value,
            "warnings": list(self.warnings),
            "evidence": self.evidence,
        }
        if self.frame_id:
            out["frame_id"] = self.frame_id
        if sym is not None:
            out["category"] = sym.category
            out["kind"] = sym.kind
            out["description"] = sym.description
            out["standards"] = sym.to_dict()["standards"]
        return out


# ── Coordinate transform ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Transform:
    """Maps a box found in a working image back to original page coordinates.

    A crop is padded, upscaled and tiled on the way in; without carrying the
    inverse of that, every bounding box published would be wrong in a way no
    test on the crop itself would catch.
    """

    offset_x: float = 0.0
    offset_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def apply(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x, y, w, h = box
        return (
            self.offset_x + x / self.scale_x,
            self.offset_y + y / self.scale_y,
            w / self.scale_x,
            h / self.scale_y,
        )

    def then(self, other: "Transform") -> "Transform":
        """Compose: ``self`` applied first, then ``other``."""
        return Transform(
            offset_x=other.offset_x + self.offset_x / other.scale_x,
            offset_y=other.offset_y + self.offset_y / other.scale_y,
            scale_x=self.scale_x * other.scale_x,
            scale_y=self.scale_y * other.scale_y,
        )


IDENTITY = Transform()


# ── Image helpers ────────────────────────────────────────────────────────────


def to_gray(image: Image.Image) -> np.ndarray:
    """CAD ink as a single channel, reusing the project's own conversion."""
    return np.array(cad_ink_to_gray(image).convert("L"))


#: A connected component smaller than this fraction of the largest one is
#: scanner speckle, not part of the glyph. Generous enough to keep the dot of
#: a small inner ring, tight enough to clear sensor noise.
SPECKLE_FRACTION = 0.02

#: Shortest side at which an image is a region with its own illumination
#: rather than a crop of one glyph or one frame; see :func:`binarize`.
ADAPTIVE_MIN_PX = 400

#: The adaptive block is the shorter side divided by this. An eighth hollowed
#: out the strokes of a frame read at 3x — the block was narrower than the
#: stroke — and a third keeps the whole-page behaviour while leaving a
#: 10px stroke solid.
ADAPTIVE_BLOCK_DIV = 3


def despeckle(binary: np.ndarray, *, max_floor: float | None = None) -> np.ndarray:
    """Drop isolated specks while leaving thin strokes intact.

    Size-based rather than morphological on purpose. An opening with any
    kernel big enough to clear noise also erases a one-pixel straightness
    stroke or the slash of a Ø — exactly the thin features the brief warns
    against destroying. Removing whole components by area leaves every stroke
    that is actually connected to the glyph untouched, however thin it is.
    """
    if cv2 is None:
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return binary
    floor = max(2.0, float(areas.max()) * SPECKLE_FRACTION)
    if max_floor is not None:
        floor = min(floor, float(max_floor))
    # One lookup over the label image. Clearing each speck with its own
    # ``labels == i`` mask was components x pixels: 9s on a 250dpi page.
    keep = stats[:, cv2.CC_STAT_AREA] >= floor
    keep[0] = True
    out = binary.copy()
    out[~keep[labels]] = 0
    return out


#: On a clean crop, ink is anything at least this share of the darkest ink
#: (and at least this absolute level, out of 255).
HAIRLINE_INK_FRACTION = 0.25
HAIRLINE_MIN_INK = 32

#: A piece of ink the median kept at least this share of is left as it is.
HAIRLINE_KEPT_WHOLE = 0.9

#: A piece of ink the median erased is given back to a glyph read unless it
#: spans this share of the crop's width or height: then it is a leader or a
#: border running through the crop, not part of the glyph.
HAIRLINE_SPAN_FRACTION = 0.9


def _restore_hairline_marks(filtered: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """
    Give a glyph read back the hairline marks the median erased.

    Per piece of ink, not per crop: a vector frame's tolerance cell holds
    bold digits beside a hairline Ⓤ, and judged as a whole the median kept
    "most" of the ink while erasing the ring entirely. A piece the median did
    not keep whole comes back — unless it runs across the crop, which is what
    a leader or border does, and what the median removing kept a real drawn
    ◎ boxed to its own ink (given back, it was boxed to the whole crop and
    scored 0.006 against its own template).
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (raw > 0).astype(np.uint8), connectivity=8
    )
    h, w = raw.shape[:2]
    out = filtered.copy()
    for index in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[index])
        if area < 4:
            continue
        if bw >= HAIRLINE_SPAN_FRACTION * w or bh >= HAIRLINE_SPAN_FRACTION * h:
            continue
        # Judged inside the piece's own box: a whole-image mask per piece cost
        # 67 s on a 24 MP sheet.
        window = (slice(y, y + bh), slice(x, x + bw))
        member = labels[window] == index
        # Restored unless the median kept it nearly whole: a ring the median
        # cut into arcs is still a ring cut into arcs if it kept 60% of it.
        if (filtered[window][member] > 0).mean() >= HAIRLINE_KEPT_WHOLE:
            continue
        out[window][member] = 255
    return out

#: Share of ink pixels with no inked neighbour at all above which an image is
#: treated as grainy. A clean render or a decent scan has almost none — every
#: pixel of a stroke, however thin, touches the next one — while sensor noise
#: is mostly lone pixels.
GRAIN_ISOLATED_FRACTION = 0.02


#: A clean render: of the pixels that are not ink, at least this share are
#: blank paper (within a couple of grey levels of it). A scan, a blurred copy
#: or a noisy one has grey everywhere and does not qualify.
CLEAN_BACKGROUND_FRACTION = 0.95
#: The median must have removed at least this share of the ink for the image
#: to be one drawn in hairlines.
HAIRLINE_REMOVED_FRACTION = 0.05


def has_hairlines(gray: np.ndarray, filtered: np.ndarray, raw: np.ndarray) -> bool:
    """
    Whether this is a clean render whose strokes the median filter erased.

    Both conditions are needed. Only a clean render (a vector PDF, a digital
    image) can be trusted for every grey pixel being ink; on a scan, a blurred
    or a noisy copy, giving the median's removals back restored noise and
    halos and broke frames on the synthetic benchmark. And only where the
    median actually removed a share of the ink is there anything to give back
    — a blurred render's strokes are thick and come through it whole.
    """
    if cv2 is None or gray.size == 0:
        return False
    ink = raw > 0
    total = int(ink.sum())
    if total == 0:
        return False
    # Paper is what lies away from every mark, however faint: a hairline's
    # antialiased grey is below Otsu's threshold, and counted as background it
    # made a clean render look dirty.
    marks = (gray >= HAIRLINE_MIN_INK).astype(np.uint8)
    near = cv2.dilate(marks, np.ones((5, 5), np.uint8)) > 0
    background = gray[~near]
    if background.size and float((background <= 2).mean()) < CLEAN_BACKGROUND_FRACTION:
        return False
    removed = ink & (filtered == 0)
    if int(removed.sum()) >= HAIRLINE_REMOVED_FRACTION * total:
        return True
    # Or one mark lost most of itself, however little of the whole ink that
    # is: a hairline Ⓤ beside bold digits is a small share of the cell.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    h, w = ink.shape[:2]
    for index in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[index])
        if area < 8 or bw >= HAIRLINE_SPAN_FRACTION * w or bh >= HAIRLINE_SPAN_FRACTION * h:
            continue
        member = labels[y : y + bh, x : x + bw] == index
        if float(removed[y : y + bh, x : x + bw][member].mean()) >= 0.5:
            return True
    return False


def is_grainy(gray: np.ndarray) -> bool:
    """Whether a white-on-black crop carries salt-and-pepper grain."""
    if cv2 is None or gray.size == 0:
        return False
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total = int(ink.sum())
    if total == 0:
        return False
    neighbours = cv2.filter2D(ink.astype(np.uint8), -1, np.ones((3, 3), np.uint8)) - ink
    isolated = int(((ink == 1) & (neighbours == 0)).sum())
    return isolated > GRAIN_ISOLATED_FRACTION * total


def binarize(gray: np.ndarray, *, clean: bool = True) -> np.ndarray:
    """Ink as white-on-black, which is what the contour functions expect.

    Otsu over a whole page washes out a faint symbol next to a heavy border,
    so an adaptive threshold is used on regions large enough to have local
    illumination variation, and Otsu on the small crops where adaptive mostly
    amplifies grain.

    "Small" is anything up to a frame or a glyph, at any render scale. The
    switch sat at 120px, which is a page tile and a glyph crop at 1x but only
    a glyph crop at 3x, and there the adaptive block — an eighth of the crop —
    is narrower than a stroke and hollows it out: every frame on a page
    enlarged for reading lost its symbol, and the divider between its
    tolerance and its datums with it.

    The median pre-filter and the despeckle afterwards are what make this
    survive a scan. Without them, Gaussian noise at sigma 25 put ink in every
    corner of the crop, the tight-crop in :func:`normalize_glyph` then spanned
    the whole region, and recognition fell from 39/39 to 3/39 — the glyph was
    still perfectly visible, it just no longer had a bounding box.
    """
    if cv2 is None:
        return (gray < 128).astype(np.uint8) * 255
    h, w = gray.shape[:2]
    # Median preserves edges and thin strokes where a Gaussian would smear
    # them, and it is what actually removes salt-and-pepper scanner grain.
    work = cv2.medianBlur(gray, 3) if min(h, w) >= 8 else gray

    def threshold(image: np.ndarray) -> np.ndarray:
        if min(h, w) >= ADAPTIVE_MIN_PX:
            block = max(11, (min(h, w) // ADAPTIVE_BLOCK_DIV) | 1)
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -8
            )
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    out = threshold(work)
    # But the median also erases any stroke one pixel wide: a pixel on a
    # hairline has three bright pixels in its 3x3 neighbourhood, so its median
    # is background. A vector PDF renders frame rules and symbol outlines as
    # exactly such hairlines; a frame lost its top rule and was never proposed
    # (the profile frame "⌓ | 0.2 Ⓤ" on 3015-100-2232F). So a caller reading
    # STRUCTURE (``clean=False``: rules, frames) gets back the ink the median
    # took, where there is no grain to remove. A caller classifying a glyph
    # does not: there the hairlines are the drawing's leaders and borders
    # crossing the crop, and the median removing them is what keeps the
    # glyph's own box — given back, a real drawn ◎ was boxed to the whole
    # crop and scored 0.006 against its own template.
    if work is not gray and has_hairlines(gray, out, threshold(gray)):
        raw = threshold(gray)
        if min(h, w) < ADAPTIVE_MIN_PX:
            # A clean crop has a truly blank background, so any grey is ink:
            # an antialiased hairline is half intensity, and Otsu's threshold,
            # set by whatever bold ink shares the crop, cut a hairline Ⓤ into
            # fragments beside a heavy "1". A fixed share of the darkest ink
            # keeps it whole.
            floor = max(HAIRLINE_MIN_INK, int(HAIRLINE_INK_FRACTION * int(gray.max())))
            raw = cv2.bitwise_or(raw, ((gray >= floor) * 255).astype(np.uint8))
        if not clean:
            out = cv2.bitwise_or(out, raw)
        else:
            out = _restore_hairline_marks(out, raw)
    # The despeckle floor is relative to the largest component, which is right
    # for a glyph crop and wrong for a whole page: there the largest component
    # is the part outline, and 2% of *that* is bigger than a frame's border or
    # a tolerance digit — measured on a real drawing at 2x it erased every
    # feature control frame not joined to the outline by a leader. A caller
    # reading a page for its rules passes ``clean=False``. Capping the floor
    # here instead was tried and let scanner noise through to the glyph pass:
    # the false-positive rate on the noisy benchmark went from 13% to 98%.
    return despeckle(out) if clean else out


def _upscale(gray: np.ndarray, factor: float) -> tuple[np.ndarray, float]:
    if cv2 is None or factor <= 1.0:
        return gray, 1.0
    h, w = gray.shape[:2]
    out = cv2.resize(
        gray, (max(1, int(w * factor)), max(1, int(h * factor))),
        interpolation=cv2.INTER_CUBIC,
    )
    return out, factor


# ── Topology: what family is this mark? ──────────────────────────────────────

FAM_RING = "ring"                 # a closed round outline
FAM_RING_SLASH = "ring_slashed"   # ring crossed by one diagonal
FAM_RING_CROSS = "ring_crossed"   # ring crossed by a full cross
FAM_RING_INNER = "ring_nested"    # ring containing another ring
FAM_RING_GLYPH = "ring_glyph"     # ring containing something that is not a ring
FAM_STROKES = "strokes"           # an open stroke figure
FAM_TRIANGLE = "triangle"
FAM_RECT = "rectangle"
FAM_EMPTY = "empty"


@dataclass
class Topology:
    """Shape facts measured from the ink, independent of any template."""

    family: str = FAM_EMPTY
    ink_fraction: float = 0.0
    circularity: float = 0.0
    components: int = 0
    holes: int = 0
    #: Strokes crossing the centre, by orientation.
    has_vertical: bool = False
    has_horizontal: bool = False
    has_diagonal: bool = False
    #: A cross whose arms extend past the outer ring (position, not a hole).
    arms_extend: bool = False
    aspect: float = 1.0
    #: Fraction of the glyph box the largest component fills.
    fill: float = 0.0
    solid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "ink_fraction": round(self.ink_fraction, 4),
            "circularity": round(self.circularity, 4),
            "components": self.components,
            "holes": self.holes,
            "has_vertical": self.has_vertical,
            "has_horizontal": self.has_horizontal,
            "has_diagonal": self.has_diagonal,
            "arms_extend": self.arms_extend,
            "aspect": round(self.aspect, 3),
            "fill": round(self.fill, 4),
            "solid": self.solid,
        }


def _line_orientations(binary: np.ndarray) -> tuple[bool, bool, bool]:
    """Which stroke directions cross the glyph, via a probabilistic Hough."""
    if cv2 is None:
        return (False, False, False)
    h, w = binary.shape[:2]
    span = max(h, w)
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180,
        threshold=max(8, span // 4),
        minLineLength=max(6, int(span * 0.42)),
        maxLineGap=max(2, span // 10),
    )
    vert = horiz = diag = False
    if lines is None:
        return (vert, horiz, diag)
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
        if angle < 18 or angle > 162:
            horiz = True
        elif 72 < angle < 108:
            vert = True
        else:
            diag = True
    return (vert, horiz, diag)


def analyze_topology(gray: np.ndarray) -> Topology:
    """Measure the shape facts that decide which family a mark belongs to."""
    topo = Topology()
    if cv2 is None or gray.size == 0:
        return topo
    binary = binarize(gray)
    h, w = binary.shape[:2]
    topo.ink_fraction = float((binary > 0).mean())
    topo.aspect = w / max(h, 1)
    if topo.ink_fraction < 0.005:
        return topo

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return topo

    outer = [
        (i, c)
        for i, c in enumerate(contours)
        if hierarchy is None or hierarchy[0][i][3] == -1
    ]
    topo.components = len(outer)
    if not outer:
        return topo

    idx, biggest = max(outer, key=lambda pair: cv2.contourArea(pair[1]))
    area = cv2.contourArea(biggest)
    peri = cv2.arcLength(biggest, True)
    bx, by, bw, bh = cv2.boundingRect(biggest)
    topo.fill = area / max(bw * bh, 1)
    if peri > 0:
        topo.circularity = float(4 * np.pi * area / (peri * peri))

    # Children of the biggest contour = enclosed holes.
    if hierarchy is not None:
        topo.holes = sum(1 for row in hierarchy[0] if row[3] == idx)

    topo.has_vertical, topo.has_horizontal, topo.has_diagonal = _line_orientations(
        binary
    )

    approx = cv2.approxPolyDP(biggest, 0.04 * peri, True)
    corners = len(approx)

    # A filled blob has no hole and fills its own bounding box.
    topo.solid = topo.holes == 0 and topo.fill > 0.72

    if topo.holes >= 1 and topo.circularity >= 0.45:
        # Round outline with something enclosed. What is inside decides.
        inner_ring = False
        if hierarchy is not None:
            for i, row in enumerate(hierarchy[0]):
                if row[3] != idx:
                    continue
                ic = contours[i]
                ia = cv2.contourArea(ic)
                ip = cv2.arcLength(ic, True)
                if ip > 0 and ia > 0:
                    icirc = 4 * np.pi * ia / (ip * ip)
                    # A concentric inner ring is round and a decent fraction of
                    # the outer. A sliver left by a slash is neither.
                    if icirc >= 0.6 and ia >= area * 0.06:
                        inner_ring = True
        if inner_ring:
            topo.family = FAM_RING_INNER
        elif topo.has_vertical and topo.has_horizontal:
            topo.family = FAM_RING_CROSS
        elif topo.has_diagonal:
            topo.family = FAM_RING_SLASH
        else:
            topo.family = FAM_RING_GLYPH
    elif topo.circularity >= 0.55 and topo.holes == 0 and not topo.solid:
        topo.family = FAM_RING
    elif topo.solid and corners == 3:
        topo.family = FAM_TRIANGLE
    elif corners == 4 and topo.fill > 0.6:
        topo.family = FAM_RECT
    elif topo.solid and topo.circularity >= 0.6:
        topo.family = FAM_RING
    else:
        topo.family = FAM_STROKES

    # Did the cross arms run past the ring? Position's do; a crosshair drawn
    # inside a circle (a centre mark) stops at it.
    if topo.family in {FAM_RING_CROSS, FAM_RING_SLASH}:
        margin = max(2, int(min(bw, bh) * 0.12))
        edges = (
            binary[: margin, :].any(),
            binary[-margin :, :].any(),
            binary[:, : margin].any(),
            binary[:, -margin :].any(),
        )
        topo.arms_extend = sum(bool(e) for e in edges) >= 3

    return topo



#: Which symbols each topology family makes *likely*. This is a prior, not a
#: gate. An earlier version used it as a hard filter and the result was that a
#: glyph whose topology was misread could never be recognized at all, however
#: well it matched its own template. Topology now multiplies a score; the
#: template still gets to speak for every symbol.
FAMILY_CANDIDATES: dict[str, tuple[str, ...]] = {
    FAM_RING: (
        "circularity",
        "all_around",
        "dimension_origin",
        "datum_target_symbol",
    ),
    FAM_RING_SLASH: ("diameter", "cylindricity", "spherical_diameter"),
    FAM_RING_CROSS: ("position", "datum_target_symbol"),
    FAM_RING_INNER: ("concentricity", "all_over", "datum_target_symbol"),
    FAM_RING_GLYPH: (
        "mmc",
        "lmc",
        "independency",
        "envelope_requirement",
        "projected_tolerance_zone",
        "free_state",
        "tangent_plane",
        "unequally_disposed_profile",
        "continuous_feature",
        "statistical_tolerance",
        "rfs",
        "datum_target_symbol",
    ),
    FAM_TRIANGLE: ("datum_triangle", "datum_feature_symbol"),
    FAM_RECT: ("feature_control_frame", "basic_dimension", "square_feature"),
    FAM_STROKES: (
        "straightness",
        "flatness",
        "parallelism",
        "perpendicularity",
        "angularity",
        "symmetry",
        "circular_runout",
        "total_runout",
        "profile_of_a_line",
        "profile_of_a_surface",
        "counterbore",
        "countersink",
        "depth",
        "conical_taper",
        "slope",
        "between",
        "datum_target_point",
        "parting_line",
    ),
}

#: Score multiplier for a symbol the topology agrees with, and for one it does
#: not. The penalty is mild on purpose — topology on a 12px scanned glyph is
#: the least reliable signal in the stack.
FAMILY_BONUS = 1.08
FAMILY_PENALTY = 0.82


def family_prior(topology: Topology, symbol_id: str) -> float:
    """How much the measured topology favours this symbol.

    Retained for diagnostics and for callers that want it, but **not applied
    to scores** — see the note on :func:`score_candidates`. Measuring it was
    worthwhile: it is what showed the hole-counting to be unreliable on small
    glyphs in the first place.
    """
    expected = FAMILY_CANDIDATES.get(topology.family, ())
    if not expected:
        return 1.0
    return FAMILY_BONUS if symbol_id in expected else FAMILY_PENALTY


# ── Template matching ────────────────────────────────────────────────────────
#
# Classification compares shapes at an *aligned* scale: the glyph is cropped to
# its own ink, padded to a square so its aspect survives, and resized to a
# fixed box — and the templates are normalised exactly the same way.
#
# This is not a detail. Matching a template over a padded crop leaves the
# correlation free to choose a scale and an offset, and it will happily settle
# on a smaller template that fits the glyph's ring and lets the rest fall on
# blank paper. Measured over the 41 templates, sliding-window matching named 23
# correctly and sent every circled-letter modifier to "all over"; the same
# templates compared at aligned scale named 40, and stayed at 40 through a 5px
# Gaussian blur and through Gaussian noise at sigma 25.

#: Widest a proposed glyph may be relative to its height (and the inverse).
#: Set from the registry's own geometry: the widest templated symbol is
#: straightness, a single stroke across its box.
GLYPH_MAX_ASPECT = 12.0

#: Side of the normalised comparison box.
NORM_PX = 64

#: How far in from the edge the "interior" of a glyph starts, as a fraction of
#: the normalised box. 0.26 clears a typical ring's annulus.
INTERIOR_INSET = 0.26

#: Weight given to interior agreement versus whole-shape agreement. Measured
#: across 0.0 / 0.3 / 0.45 / 0.6: 0.3 was the best of them, and 0.6 started
#: costing whole-shape accuracy (circularity began matching a square).
INTERIOR_WEIGHT = 0.30

#: Ink fraction below which an interior counts as empty. Two empty interiors
#: are agreement, not a failed comparison — see compare_normalized.
INTERIOR_MIN_INK = 0.02

#: An interior at least this full of ink has no structure left to correlate;
#: see :func:`compare_normalized`.
INTERIOR_SOLID_INK = 0.7


def normalize_glyph(gray: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Tight-crop to the ink, square it, and resize to :data:`NORM_PX`.

    Returns ``(normalised, ink_box)`` where ``ink_box`` is where the ink sat in
    ``gray`` — the caller needs it to publish a bounding box that is the glyph
    rather than whatever region it was handed.
    """
    if cv2 is None or gray.size == 0:
        return None, None
    binary = binarize(gray)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None, None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    roi = gray[y0 : y1 + 1, x0 : x1 + 1]
    h, w = roi.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    square[oy : oy + h, ox : ox + w] = roi
    norm = cv2.resize(square, (NORM_PX, NORM_PX), interpolation=cv2.INTER_AREA)
    return norm, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


_CANON: dict[str, np.ndarray] = {}


def canonical_template(name: str) -> np.ndarray | None:
    """A template normalised the same way a glyph is, memoised."""
    hit = _CANON.get(name)
    if hit is None:
        if not tpl.has_template(name):
            return None
        norm, _ = normalize_glyph(tpl.render(name, NORM_PX))
        if norm is None:
            return None
        _CANON[name] = norm
        hit = norm
    return hit


def compare_normalized(norm: np.ndarray, template_name: str) -> float:
    """Agreement between a normalised glyph and a normalised template.

    The whole shape and the glyph's *interior* are scored separately and
    blended. Almost every confusion left after scale alignment is a pair that
    agrees on its outline and differs inside it — circularity against a datum
    target, all-around against all-over, one circled letter against another —
    and in all of those the outline is most of the ink, so a whole-shape
    correlation barely moves between them. Scoring the interior as well pulls
    those pairs apart: over the 41 templates across clean, blurred and noisy
    renders it lifted top-1 from 118 to 120 of 123 and widened the mean margin
    to the runner-up from 0.264 to 0.289.
    """
    if cv2 is None:
        return 0.0
    canon = canonical_template(template_name)
    if canon is None:
        return 0.0
    full = float(cv2.minMaxLoc(cv2.matchTemplate(norm, canon, cv2.TM_CCOEFF_NORMED))[1])
    inset = int(NORM_PX * INTERIOR_INSET)
    g_in = norm[inset : NORM_PX - inset, inset : NORM_PX - inset]
    c_in = canon[inset : NORM_PX - inset, inset : NORM_PX - inset]
    if g_in.size == 0 or c_in.size == 0:
        return full

    # Two empty interiors agree perfectly, but a normalised correlation
    # between two near-constant patches has no variance to work with and
    # comes back 0.0 — not 1.0. Blending that in penalised every hollow
    # glyph by the full interior weight: a real circularity ○ matched its own
    # template at 0.768 on the outline and was reported at 0.538, under the
    # accept bar, so a plain circle in a frame never confirmed.
    #
    # Empty-vs-empty is therefore treated as "nothing to compare" and the
    # outline score stands. Empty-vs-inked stays a genuine disagreement.
    g_ink = float((g_in > 127).mean())
    c_ink = float((c_in > 127).mean())
    if g_ink < INTERIOR_MIN_INK and c_ink < INTERIOR_MIN_INK:
        return full
    # The same failure the other way up. The position template's interior is
    # 82% ink — a cross through a ring leaves almost no paper in the middle —
    # and a scanned ⌖ fills its middle with a soft blob. Correlating two
    # near-solid patches measures their noise: on a real drawing every
    # position frame's outline scored 0.73–0.82 and its interior 0.03–0.30,
    # and the blend landed at 0.52–0.64, under the accept bar. Where either
    # interior is mostly ink, how much ink the two carry is the comparison —
    # a solid middle against an empty one is still a clear disagreement.
    if c_ink < INTERIOR_MIN_INK:
        # An empty template interior against a glyph with *some* ink in the
        # middle. A correlation with a constant patch is 0 whatever the glyph
        # holds, which scored a circle whose blur leaked 2.3% of ink past the
        # inset the same as a circle with a second ring inside it. The less
        # ink in the glyph's middle, the better it agrees with an empty one.
        inner = 1.0 - min(1.0, g_ink / INTERIOR_SOLID_INK)
    elif c_ink >= INTERIOR_SOLID_INK:
        # Keyed on the *template's* interior only. A glyph whose middle has
        # filled in is not evidence that its template's has: the circled
        # modifiers at 18px are rings full of blurred letter, and judged by
        # ink alone every one of them read as the same modifier. A ratio,
        # not a difference: a mostly empty middle against a solid one is a
        # near-miss by difference (0.08 against 0.8 scores 0.28) and the
        # disagreement it is by ratio (0.1).
        inner = min(g_ink, c_ink) / max(g_ink, c_ink)
    else:
        inner = float(cv2.minMaxLoc(cv2.matchTemplate(g_in, c_in, cv2.TM_CCOEFF_NORMED))[1])
    return (1.0 - INTERIOR_WEIGHT) * full + INTERIOR_WEIGHT * inner


def match_template(
    gray: np.ndarray, template_name: str, *, scales: Sequence[int] | None = None
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Sliding-window multi-scale peak — for *locating* a glyph, not naming one.

    Kept for the frame pass, which needs to know where in a compartment a mark
    sits. Classification goes through :func:`compare_normalized`.
    """
    if cv2 is None or gray.size == 0 or not tpl.has_template(template_name):
        return 0.0, None
    h, w = gray.shape[:2]
    sizes = tuple(scales) if scales else tpl.scales_for(min(h, w))
    best, best_box = 0.0, None
    for size in sizes:
        if size > h or size > w:
            continue
        template = tpl.render_cached(template_name, size)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        if peak > best:
            best = float(peak)
            best_box = (int(loc[0]), int(loc[1]), size, size)
    return best, best_box


#: Symbols that share one template, so a template score cannot separate them.
#: Counterbore and spotface are drawn identically — only an "SF" annotation
#: tells them apart — and profile-of-a-line is the arc-length glyph.
TEMPLATE_SHARERS: dict[str, tuple[str, ...]] = {}
for _sym in reg.ALL_SYMBOLS:
    if _sym.template:
        TEMPLATE_SHARERS.setdefault(_sym.template, ())
        TEMPLATE_SHARERS[_sym.template] += (_sym.id,)


def score_candidates(
    gray: np.ndarray,
    symbol_ids: Iterable[str] | None = None,
    *,
    topology: Topology | None = None,
    config: GdtConfig | None = None,
) -> list[Candidate]:
    """Rank symbols by normalised shape agreement, best first.

    ``symbol_ids`` of ``None`` means "every symbol with a template" — the
    normal case. Passing a set narrows the field for a caller that knows the
    position, such as a frame's first compartment.

    Topology is recorded as evidence but deliberately does **not** scale the
    score. It was tried as a prior and made things worse: hole-counting on a
    40px glyph misread concentricity's inner ring and all-over's, so the
    "wrong" family was confidently applied and a correct template match got
    penalised out of first place. The shape comparison is the more reliable
    signal of the two, and it is left to speak for itself.
    """
    cfg = config or get_config()
    norm, _ = normalize_glyph(gray)
    if norm is None:
        return []

    wanted = set(symbol_ids) if symbol_ids is not None else None
    out: list[Candidate] = []
    for template_name, sharers in TEMPLATE_SHARERS.items():
        ids = [s for s in sharers if wanted is None or s in wanted]
        if not ids:
            continue
        raw = compare_normalized(norm, template_name)
        if raw <= 0:
            continue
        for symbol_id in ids:
            out.append(
                Candidate(
                    symbol_id=symbol_id,
                    score=min(0.99, raw),
                    method=M_TEMPLATE,
                    evidence={
                        "template": template_name,
                        "shape_score": round(raw, 4),
                        **(
                            {
                                "topology_family": topology.family,
                                "topology_agrees": symbol_id
                                in FAMILY_CANDIDATES.get(topology.family, ()),
                            }
                            if topology
                            else {}
                        ),
                        **(
                            {"shares_template_with": [s for s in ids if s != symbol_id]}
                            if len(ids) > 1
                            else {}
                        ),
                    },
                )
            )
    out.sort(key=lambda c: -c.score)
    return out


# ── Circled-letter modifiers ─────────────────────────────────────────────────

#: The modifier each circled letter denotes. Two letters for CF/ST.
CIRCLED_LETTERS: dict[str, str] = {
    "M": "mmc",
    "L": "lmc",
    "I": "independency",
    "E": "envelope_requirement",
    "P": "projected_tolerance_zone",
    "F": "free_state",
    "T": "tangent_plane",
    "U": "unequally_disposed_profile",
    "S": "rfs",
    "CF": "continuous_feature",
    "ST": "statistical_tolerance",
}

#: Below this normalised interior height the letter inside a ring cannot be
#: resolved. Measured: at an 18px overall glyph the circled modifiers drop from
#: 11/11 correct to 0/11, all of them landing on ring-shaped neighbours. Rather
#: than publish one of those, the detector reports the whole circled-modifier
#: family as alternatives and says why.
MIN_INTERIOR_PX = 11


def ring_interior_px(gray: np.ndarray) -> int | None:
    """Size of whatever sits inside a ring, in normalised pixels.

    Used to decide whether a circled modifier is actually readable. Returns
    ``None`` when there is no ring-like glyph or nothing inside one.
    """
    norm, _ = normalize_glyph(gray)
    if norm is None:
        return None
    inset = int(NORM_PX * INTERIOR_INSET)
    interior = norm[inset : NORM_PX - inset, inset : NORM_PX - inset]
    _, box = normalize_glyph(interior)
    if box is None:
        return None
    return int(max(box[2], box[3]))



# ── OCR normalization ────────────────────────────────────────────────────────

#: Characters that are never a GD&T glyph on their own, however they are
#: scored. Without this the variant table turns every stray "O" on a drawing
#: into a circularity candidate.
_BARE_LETTER = re.compile(r"^[A-Za-z0-9]$")


def normalize_ocr(text: str) -> tuple[str | None, list[Candidate]]:
    """Map an OCR string onto registry candidates.

    Returns the canonical glyph when exactly one symbol explains the text, and
    the ranked candidate list either way. Scores here are deliberately modest:
    OCR text is corroboration, and a bare letter is weaker still, because the
    variant table is full of single characters that are usually just letters.
    """
    raw = (text or "").strip()
    if not raw:
        return None, []

    hits = reg.candidates_for_ocr(raw)
    if not hits:
        return None, []

    bare = bool(_BARE_LETTER.match(raw))
    # An exact Unicode glyph is strong; a mangled multi-character variant is
    # middling; a single ambiguous letter is weak.
    exact_glyph = any(ch in reg.BY_UNICODE for ch in raw)
    base = 0.86 if exact_glyph else (0.3 if bare else 0.6)
    # Spread across however many symbols share this spelling: "O" explained by
    # nine symbols is nine weak candidates, not nine strong ones.
    spread = base / max(1, len(hits)) ** 0.5

    out = [
        Candidate(
            symbol_id=sym.id,
            score=round(spread, 4),
            method=M_OCR,
            evidence={
                "raw": raw,
                "shared_by": len(hits),
                "exact_glyph": exact_glyph,
                "bare_letter": bare,
            },
        )
        for sym in hits
    ]
    canonical = hits[0].unicode if len(hits) == 1 else None
    return canonical, out


# ── Combining evidence ───────────────────────────────────────────────────────


def combine(
    candidate_sets: Sequence[Sequence[Candidate]],
    *,
    config: GdtConfig | None = None,
) -> list[Candidate]:
    """Fuse candidate lists from several methods into one ranking.

    Agreement between methods is what earns confidence. A symbol found by both
    the template pass and OCR scores above either on its own, but the bonus is
    capped well below certainty — two methods that share a blind spot agree
    just as readily as two that are right.
    """
    merged: dict[str, Candidate] = {}
    for candidates in candidate_sets:
        for cand in candidates:
            existing = merged.get(cand.symbol_id)
            if existing is None:
                merged[cand.symbol_id] = Candidate(
                    symbol_id=cand.symbol_id,
                    score=cand.score,
                    method=cand.method,
                    evidence=dict(cand.evidence),
                )
                continue
            # Agreement: take the stronger, lift it by a fraction of the weaker.
            strong, weak = max(existing.score, cand.score), min(
                existing.score, cand.score
            )
            existing.score = min(0.99, strong + weak * 0.22)
            existing.method = (
                existing.method
                if existing.method == cand.method
                else f"{existing.method}+{cand.method}"
            )
            existing.evidence.update(cand.evidence)
    out = sorted(merged.values(), key=lambda c: -c.score)
    return out


def decide(
    candidates: Sequence[Candidate],
    *,
    config: GdtConfig | None = None,
) -> tuple[str, float, list[Candidate]]:
    """Turn a ranking into a status, a confidence and the alternatives.

    The rival margin is the part that matters: a top score that clears the
    accept bar is still reported ambiguous when the runner-up is within
    ``rival_margin``, because "it is one of these two" is the honest answer and
    picking the higher one by a hair would not be.
    """
    cfg = config or get_config()
    if not candidates:
        return STATUS_UNKNOWN, 0.0, []

    ranked = sorted(candidates, key=lambda c: -c.score)
    best = ranked[0]
    rivals = [c for c in ranked[1:] if c.score >= cfg.ambiguous_confidence]

    if best.score < cfg.ambiguous_confidence:
        return STATUS_UNKNOWN, best.score, ranked[:4]

    close = bool(rivals and (best.score - rivals[0].score) < cfg.rival_margin)
    if best.score >= cfg.accept_confidence and not close:
        return STATUS_CONFIRMED, best.score, ranked[1:4]
    return STATUS_AMBIGUOUS, best.score, ranked[1:4]


# ── Classifying one crop ─────────────────────────────────────────────────────


def classify_crop(
    image: Image.Image,
    *,
    ocr_text: str | None = None,
    allowed: Iterable[str] | None = None,
    transform: Transform = IDENTITY,
    page: int = 1,
    config: GdtConfig | None = None,
) -> Detection:
    """Name the glyph in one tight crop.

    ``allowed`` narrows the field to the symbols a position permits — a caller
    that knows the crop is a frame's first compartment passes the frame
    characteristics, which is what keeps perpendicularity from being "found"
    in ordinary right-angle geometry elsewhere on the sheet.

    The bounding box published is the glyph's own ink, not the crop handed in,
    mapped back through ``transform``.
    """
    cfg = config or get_config()
    gray = to_gray(image)
    h, w = gray.shape[:2]
    warnings: list[str] = []

    topo = analyze_topology(gray)
    norm, ink_box = normalize_glyph(gray)

    if norm is None or ink_box is None or topo.ink_fraction < 0.005:
        return Detection(
            symbol_id=None,
            name="Unknown",
            status=STATUS_UNKNOWN,
            confidence=0.0,
            bbox=transform.apply((0.0, 0.0, float(w), float(h))),
            page=page,
            method=M_TOPOLOGY,
            raw_ocr=ocr_text,
            warnings=["no ink in region"],
            evidence={"topology": topo.to_dict()},
        )

    box = transform.apply(
        (float(ink_box[0]), float(ink_box[1]), float(ink_box[2]), float(ink_box[3]))
    )
    glyph_px = min(ink_box[2], ink_box[3])
    if glyph_px < cfg.min_glyph_px:
        warnings.append(
            f"glyph is {glyph_px}px on its short side, below the "
            f"{cfg.min_glyph_px}px floor — too small to separate confusables"
        )

    sets: list[list[Candidate]] = []

    shape_hits = score_candidates(gray, allowed, topology=topo, config=cfg)
    sets.append([c for c in shape_hits if c.score >= cfg.template_min_score])

    canonical, ocr_hits = normalize_ocr(ocr_text or "")
    if allowed is not None:
        allow = set(allowed)
        ocr_hits = [c for c in ocr_hits if c.symbol_id in allow]
    sets.append(ocr_hits)

    merged = combine(sets, config=cfg)
    status, confidence, alternatives = decide(merged, config=cfg)

    if not merged or status == STATUS_UNKNOWN:
        return Detection(
            symbol_id=None,
            name="Unknown",
            status=STATUS_UNKNOWN,
            confidence=round(confidence, 4),
            bbox=box,
            page=page,
            method=M_TOPOLOGY,
            raw_ocr=ocr_text,
            alternatives=alternatives,
            warnings=warnings + ["no registry symbol explains this mark well enough"],
            evidence={"topology": topo.to_dict()},
        )

    best = merged[0]
    sym = reg.get(best.symbol_id)

    # A circled modifier is only as readable as the letter inside it. Below
    # the resolution floor the ring still matches but the letter does not, so
    # the result is downgraded rather than published as a specific modifier.
    if best.symbol_id in CIRCLED_LETTERS.values():
        interior_px = ring_interior_px(gray)
        if interior_px is not None and interior_px < MIN_INTERIOR_PX:
            warnings.append(
                f"the letter inside the ring is {interior_px}px — below the "
                f"{MIN_INTERIOR_PX}px needed to tell the circled modifiers "
                "apart; reporting the family rather than choosing one"
            )
            if status == STATUS_CONFIRMED:
                status = STATUS_AMBIGUOUS

    # Symbols drawn with the same glyph cannot be separated by shape. Say so
    # rather than letting declaration order pick one.
    # The explanation is attached whenever the glyph is shared, not only when
    # it causes a downgrade. Gating the warning on a confirmed status dropped
    # it in precisely the case it explains best: when both twins tie, the
    # result was already ambiguous and so arrived with no reason given.
    shared = best.evidence.get("shares_template_with") or []
    if shared:
        warnings.append(
            f"{sym.name} is drawn with the same glyph as "
            f"{', '.join(reg.get(s).name for s in shared if reg.find(s))}; "
            "shape alone cannot separate them"
        )
        status = STATUS_AMBIGUOUS

    if sym.is_legacy:
        warnings.append(
            f"{sym.name} was retired in {sym.standards.retired_in}; still "
            f"current in {', '.join(sym.standards.current_in) or 'earlier revisions'}"
        )
    if sym.standards.exclusive_to:
        warnings.append(f"{sym.name} is defined by {sym.standards.exclusive_to} only")
    if not sym.detectable:
        # Should be unreachable: concepts carry no template and no OCR variant
        # that resolves alone. Guarded anyway, because publishing a derived
        # concept as a sighting is exactly the fabrication to avoid.
        warnings.append(
            f"{sym.name} is a {sym.kind}, not a drawn symbol — not reported as a sighting"
        )
        status = STATUS_UNKNOWN

    return Detection(
        symbol_id=sym.id,
        name=sym.name,
        status=status,
        confidence=round(confidence, 4),
        bbox=box,
        page=page,
        method=best.method,
        raw_ocr=ocr_text,
        normalized=sym.unicode or canonical,
        alternatives=alternatives,
        warnings=warnings,
        evidence={"topology": topo.to_dict(), **best.evidence},
    )



# ── Finding glyph-sized marks in a region ────────────────────────────────────


def find_glyph_boxes(
    image: Image.Image,
    *,
    config: GdtConfig | None = None,
    page_edge: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Connected components that are the right size and shape to be a symbol.

    This is a *proposal* step and is meant to over-produce: it is cheap, and
    anything it misses can never be classified. Size bounds come from config so
    a 600dpi scan and a 150dpi one can both be served.

    ``page_edge`` is the shorter edge of the **page**, which is what the upper
    size bound should be measured against. It defaults to this image's own
    shorter edge, which is right for a page or a tile and badly wrong for a
    small crop: inside a 36px-tall frame compartment the default cap works out
    at four pixels, so every modifier in every frame was rejected before it
    could be classified. Callers passing a crop should pass the page's edge.
    """
    cfg = config or get_config()
    if cv2 is None:
        return []
    gray = to_gray(image)
    binary = binarize(gray)
    h, w = binary.shape[:2]
    edge = page_edge if page_edge is not None else min(h, w)
    # The fraction is the right rule for a real page, where a symbol is one or
    # two percent of the sheet. It collapses on a small image — a 456px-tall
    # test scene gives a 54px cap, just under the 56px symbols drawn on it —
    # so an absolute floor keeps small regions workable. On a real page the
    # fraction dominates and the floor never binds.
    max_side = max(cfg.min_glyph_px * 8, int(edge * cfg.max_glyph_frac))

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Size is judged on the long side. Judging it on the short side
        # rejected every stroke symbol: straightness drawn at a readable 46px
        # wide is only 7px tall, so a 9px floor on the short side threw it
        # away as noise. The resolution warning in classify_crop still fires
        # for genuinely tiny glyphs.
        if max(bw, bh) > max_side or max(bw, bh) < cfg.min_glyph_px:
            continue
        aspect = bw / max(bh, 1)
        # Most symbols are roughly square, but several are not: straightness
        # is one wide stroke, between is a wide double arrow, slope and
        # dimension-origin are both long and low. A 4:1 bound rejected all of
        # them before they could be classified — they scored zero recall while
        # matching their own templates perfectly. The bound is wide enough to
        # admit them and the classifier is left to reject what it cannot name.
        if aspect > GLYPH_MAX_ASPECT or aspect < 1.0 / GLYPH_MAX_ASPECT:
            continue
        boxes.append((x, y, bw, bh))
    return boxes


# ── Merging duplicates ───────────────────────────────────────────────────────


def iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def merge_detections(
    detections: Sequence[Detection], *, config: GdtConfig | None = None
) -> list[Detection]:
    """Collapse detections of the same symbol that are the same mark seen twice.

    Only *overlapping* detections of the *same* symbol merge. A drawing that
    legitimately carries four Ø callouts keeps four: they do not overlap. This
    is what the tiling pass needs, since a symbol in an overlap band is found
    by both tiles.
    """
    cfg = config or get_config()
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        duplicate = None
        for existing in kept:
            if existing.page != det.page:
                continue
            if existing.symbol_id != det.symbol_id:
                continue
            if iou(existing.bbox, det.bbox) >= cfg.merge_iou:
                duplicate = existing
                break
        if duplicate is None:
            kept.append(det)
            continue
        # Keep the stronger read but remember it was corroborated.
        duplicate.evidence.setdefault("merged_duplicates", 0)
        duplicate.evidence["merged_duplicates"] += 1
    return kept

