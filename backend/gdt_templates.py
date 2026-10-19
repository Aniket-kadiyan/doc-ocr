"""
Procedurally drawn templates for the GD&T glyphs.

Why draw them instead of shipping images
----------------------------------------
``symbol_vision._make_pm_template`` already proves the approach on ±: a GD&T
glyph is a handful of straight lines and circles specified by the standard, so
it can be *drawn* at whatever size and stroke weight the page needs. That beats
a folder of PNGs on every axis that matters here — a template matches at the
scale the drawing was rendered at rather than at whatever scale the asset was
captured; stroke weight tracks the glyph size the way a CAD pen does; there is
no binary to keep in the repository and no licence question about where the
artwork came from; and a reviewer can read the geometry in the code and check
it against the standard.

It also keeps the promise made in the brief not to add a large machine-learning
dependency without demonstrating need. Nothing here needs training data.

Conventions
-----------
Every template is ``uint8``, **0 background, 255 ink** — ink white on black.
That is the polarity ``image_preprocess.cad_ink_to_gray`` produces, so a
template feeds straight into ``cv2.matchTemplate`` against a converted crop
with no inversion in between. (Getting this backwards is not a subtle bug: a
correctly shaped template scores about 0.08 against its own glyph inverted,
and 1.0 the right way up.)

Geometry is written in a normalised 0..1 box and scaled on render, so one
definition serves every size. Each glyph is drawn with a margin, because a
template that touches its own border matches badly against a glyph that has
whitespace around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only on a cv2-less host
    cv2 = None  # type: ignore

#: Ink and background values, matching cad_ink_to_gray's polarity.
INK = 255
BG = 0

#: Fraction of the box left empty around a glyph.
MARGIN = 0.14

#: Stroke width as a fraction of the template size, before rounding. CAD pens
#: on a symbol this small are thin; too fat a stroke destroys the distinction
#: between, say, parallelism and a filled block.
STROKE_FRAC = 0.085

#: The sizes a multi-scale pass tries by default, as a multiple of the
#: estimated glyph height. A frame compartment is roughly square, so one size
#: family serves every characteristic symbol.
DEFAULT_SCALES: tuple[float, ...] = (0.72, 0.86, 1.0, 1.18, 1.4)


class TemplateUnavailable(RuntimeError):
    """Raised when a template is requested on a host with no cv2."""


def _stroke_for(size: int, stroke: int | None) -> int:
    if stroke is not None:
        return max(1, stroke)
    return max(1, int(round(size * STROKE_FRAC)))


@dataclass(frozen=True)
class _Canvas:
    """A normalised drawing surface: coordinates in 0..1, ink in pixels."""

    arr: np.ndarray
    size: int
    stroke: int

    def _px(self, x: float, y: float) -> tuple[int, int]:
        span = 1.0 - 2 * MARGIN
        px = MARGIN + x * span
        py = MARGIN + y * span
        return (
            int(round(px * (self.size - 1))),
            int(round(py * (self.size - 1))),
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, width: int | None = None) -> None:
        cv2.line(
            self.arr, self._px(x1, y1), self._px(x2, y2), INK,
            width or self.stroke, lineType=cv2.LINE_AA,
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: bool = False,
        width: int | None = None,
    ) -> None:
        # Radius is a fraction of the drawable span; the glyph box is square so
        # one scalar is enough. ``width`` overrides the canvas stroke, which a
        # small inner ring needs: at the canvas weight it closes up.
        span = (1.0 - 2 * MARGIN) * (self.size - 1)
        cv2.circle(
            self.arr, self._px(cx, cy), max(1, int(round(r * span))), INK,
            -1 if fill else (width or self.stroke), lineType=cv2.LINE_AA,
        )

    def poly(self, points: list[tuple[float, float]], *, fill: bool = False) -> None:
        pts = np.array([self._px(x, y) for x, y in points], dtype=np.int32)
        if fill:
            cv2.fillPoly(self.arr, [pts], INK, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(
                self.arr, [pts], True, INK, self.stroke, lineType=cv2.LINE_AA
            )

    def arc(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        start: float,
        end: float,
    ) -> None:
        span = (1.0 - 2 * MARGIN) * (self.size - 1)
        cv2.ellipse(
            self.arr,
            self._px(cx, cy),
            (max(1, int(round(rx * span))), max(1, int(round(ry * span)))),
            0,
            start,
            end,
            INK,
            self.stroke,
            lineType=cv2.LINE_AA,
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """A line ending in an arrowhead at (x2, y2)."""
        cv2.arrowedLine(
            self.arr, self._px(x1, y1), self._px(x2, y2), INK, self.stroke,
            line_type=cv2.LINE_AA, tipLength=0.34,
        )

    def letters(self, text: str, *, cx: float = 0.5, cy: float = 0.5, box: float = 0.56) -> None:
        """Draw ``text`` centred, scaled to fit ``box`` of the drawable span.

        Hershey is not an engineering font, but for a one- or two-letter
        modifier inside a ring the silhouette is what the match keys on, and a
        Hershey capital is close enough in proportion to carry it.
        """
        span = (1.0 - 2 * MARGIN) * (self.size - 1)
        target = box * span
        thickness = max(1, int(round(self.stroke * 0.9)))
        scale = 0.1
        for _ in range(60):
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )
            if max(tw, th) >= target:
                break
            scale += 0.05
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        ox, oy = self._px(cx, cy)
        cv2.putText(
            self.arr,
            text,
            (ox - tw // 2, oy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            INK,
            thickness,
            lineType=cv2.LINE_AA,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Geometric characteristic symbols
# ─────────────────────────────────────────────────────────────────────────────

def _straightness(c: _Canvas) -> None:
    """One horizontal stroke."""
    c.line(0.0, 0.5, 1.0, 0.5)


def _flatness(c: _Canvas) -> None:
    """A parallelogram leaning right."""
    c.poly([(0.22, 0.14), (1.0, 0.14), (0.78, 0.86), (0.0, 0.86)])


def _circularity(c: _Canvas) -> None:
    c.circle(0.5, 0.5, 0.46)


def _cylindricity(c: _Canvas) -> None:
    """A circle between two slanted strokes."""
    c.circle(0.5, 0.5, 0.33)
    c.line(0.1, 0.92, 0.26, 0.08)
    c.line(0.74, 0.92, 0.9, 0.08)


def _profile_of_a_line(c: _Canvas) -> None:
    """An open upward arc."""
    c.arc(0.5, 0.78, 0.5, 0.52, 180, 360)


def _profile_of_a_surface(c: _Canvas) -> None:
    """An upward arc closed across its ends."""
    c.arc(0.5, 0.74, 0.5, 0.5, 180, 360)
    c.line(0.0, 0.74, 1.0, 0.74)


def _angularity(c: _Canvas) -> None:
    """An acute angle opening to the right."""
    c.line(0.04, 0.9, 1.0, 0.9)
    c.line(0.04, 0.9, 0.96, 0.1)


def _perpendicularity(c: _Canvas) -> None:
    c.line(0.5, 0.06, 0.5, 0.94)
    c.line(0.06, 0.94, 0.94, 0.94)


def _parallelism(c: _Canvas) -> None:
    c.line(0.2, 0.94, 0.52, 0.06)
    c.line(0.62, 0.94, 0.94, 0.06)


def _position(c: _Canvas) -> None:
    """A circle with a cross whose arms reach past the ring."""
    c.circle(0.5, 0.5, 0.34)
    c.line(0.5, 0.0, 0.5, 1.0)
    c.line(0.0, 0.5, 1.0, 0.5)


def _concentricity(c: _Canvas) -> None:
    # Tuned against real ◎ glyphs, not drawn to look right. The original
    # 0.26-radius inner ring scored 0.446 against two concentricity symbols
    # cropped from an actual drawing — below the 0.52 bar to be a candidate at
    # all — so the glyph came back "unknown" on a real feature control frame
    # while passing every synthetic test. (It passed because those tests
    # render each glyph from this same template and match it against itself:
    # a tautology. benchmarks/fixtures/glyphs/ now holds real samples so that
    # cannot happen again.)
    #
    # These proportions merge the two rings into one thick band at the
    # normalised size, which does **not** look like a ◎ — and they are still
    # the right choice. Measured against both real glyphs:
    #
    #   inner 0.26, full stroke   (4 visible bands)   0.504
    #   inner 0.28, full stroke   (4 visible bands)   0.661
    #   inner 0.32, stroke x0.7   (bands merged)      0.800
    #
    # The same thing was found for the ± template in symbol_vision: at this
    # normalisation the correlation rewards matching ink mass, not glyph
    # fidelity. Re-measure against the fixtures before changing it.
    c.circle(0.5, 0.5, 0.46)
    c.circle(0.5, 0.5, 0.32, width=max(1, int(round(c.stroke * 0.7))))


def _symmetry(c: _Canvas) -> None:
    """Three horizontal strokes between two short uprights."""
    c.line(0.1, 0.5, 0.9, 0.5)
    c.line(0.24, 0.2, 0.76, 0.2)
    c.line(0.24, 0.8, 0.76, 0.8)
    c.line(0.1, 0.12, 0.1, 0.88)
    c.line(0.9, 0.12, 0.9, 0.88)


def _circular_runout(c: _Canvas) -> None:
    c.arrow(0.12, 0.9, 0.88, 0.1)


def _total_runout(c: _Canvas) -> None:
    # Kept clear of each other: two arrows that overlap merge into one blob at
    # template size, and the single/double arrow count is the only thing that
    # separates this from circular runout.
    c.arrow(0.04, 0.92, 0.42, 0.08)
    c.arrow(0.58, 0.92, 0.96, 0.08)


# ─────────────────────────────────────────────────────────────────────────────
# Datum symbols
# ─────────────────────────────────────────────────────────────────────────────

def _datum_triangle(c: _Canvas) -> None:
    """The ASME filled triangle sitting on a surface."""
    c.poly([(0.5, 0.08), (0.96, 0.86), (0.04, 0.86)], fill=True)
    c.line(0.0, 0.92, 1.0, 0.92)


def _datum_triangle_open(c: _Canvas) -> None:
    """The older ISO open triangle."""
    c.poly([(0.5, 0.08), (0.96, 0.86), (0.04, 0.86)])
    c.line(0.0, 0.92, 1.0, 0.92)


def _datum_target(c: _Canvas) -> None:
    """A circle split by a horizontal diameter."""
    c.circle(0.5, 0.5, 0.48)
    c.line(0.02, 0.5, 0.98, 0.5)


def _datum_target_point(c: _Canvas) -> None:
    c.line(0.08, 0.08, 0.92, 0.92)
    c.line(0.92, 0.08, 0.08, 0.92)


# ─────────────────────────────────────────────────────────────────────────────
# Dimensioning and manufacturing symbols
# ─────────────────────────────────────────────────────────────────────────────

def _diameter(c: _Canvas) -> None:
    """A circle with one diagonal slash through it."""
    c.circle(0.5, 0.5, 0.38)
    c.line(0.08, 0.92, 0.92, 0.08)


def _square_feature(c: _Canvas) -> None:
    c.poly([(0.06, 0.06), (0.94, 0.06), (0.94, 0.94), (0.06, 0.94)])


def _depth(c: _Canvas) -> None:
    """A down arrow beneath a short bar."""
    c.line(0.1, 0.08, 0.9, 0.08)
    c.arrow(0.5, 0.14, 0.5, 0.96)


def _counterbore(c: _Canvas) -> None:
    """A squared U, open at the top."""
    c.line(0.1, 0.1, 0.1, 0.9)
    c.line(0.1, 0.9, 0.9, 0.9)
    c.line(0.9, 0.9, 0.9, 0.1)


def _countersink(c: _Canvas) -> None:
    """A wide shallow V, open at the top."""
    c.line(0.02, 0.14, 0.5, 0.86)
    c.line(0.5, 0.86, 0.98, 0.14)


def _conical_taper(c: _Canvas) -> None:
    """A narrow cone pointing right."""
    c.line(0.02, 0.1, 0.98, 0.5)
    c.line(0.02, 0.9, 0.98, 0.5)


def _slope(c: _Canvas) -> None:
    """A right triangle lying on its long side, pointing right."""
    c.line(0.02, 0.9, 0.98, 0.9)
    c.line(0.02, 0.9, 0.98, 0.42)


def _all_around(c: _Canvas) -> None:
    """A small circle on a leader kink."""
    c.circle(0.5, 0.42, 0.3)
    c.line(0.5, 0.72, 0.5, 1.0)


def _all_over(c: _Canvas) -> None:
    """A double circle on a leader kink.

    **Untuned, and known to render its inner circle as a solid dot rather
    than a ring** — the same defect concentricity had. It is left alone
    deliberately: concentricity was only fixable because real ◎ glyphs were
    available to measure against, and the measurement overturned the
    change that *looked* right. There is no real all-over sample in
    ``benchmarks/fixtures/glyphs/`` yet, so any change here would be a guess.
    Add a sample first, then tune.
    """
    c.circle(0.5, 0.4, 0.32)
    c.circle(0.5, 0.4, 0.16)
    c.line(0.5, 0.72, 0.5, 1.0)


def _between(c: _Canvas) -> None:
    c.arrow(0.5, 0.5, 0.02, 0.5)
    c.arrow(0.5, 0.5, 0.98, 0.5)


def _parting_line(c: _Canvas) -> None:
    """Two arrowheads pointing away from each other across a line."""
    c.line(0.0, 0.5, 1.0, 0.5)
    c.arrow(0.5, 0.42, 0.5, 0.04)
    c.arrow(0.5, 0.58, 0.5, 0.96)


def _dimension_origin(c: _Canvas) -> None:
    """A small open circle terminating a dimension line."""
    c.circle(0.22, 0.5, 0.2)
    c.line(0.42, 0.5, 1.0, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Circled-letter modifiers
# ─────────────────────────────────────────────────────────────────────────────

def _circled(text: str) -> Callable[[_Canvas], None]:
    """A ring with ``text`` inside, sized to the letter count."""

    def draw(c: _Canvas) -> None:
        c.circle(0.5, 0.5, 0.48)
        c.letters(text, box=0.52 if len(text) == 1 else 0.74)

    return draw


_BUILDERS: dict[str, Callable[[_Canvas], None]] = {
    # geometric characteristics
    "straightness": _straightness,
    "flatness": _flatness,
    "circularity": _circularity,
    "cylindricity": _cylindricity,
    "profile_of_a_line": _profile_of_a_line,
    "profile_of_a_surface": _profile_of_a_surface,
    "angularity": _angularity,
    "perpendicularity": _perpendicularity,
    "parallelism": _parallelism,
    "position": _position,
    "concentricity": _concentricity,
    "symmetry": _symmetry,
    "circular_runout": _circular_runout,
    "total_runout": _total_runout,
    # datum
    "datum_triangle": _datum_triangle,
    "datum_triangle_open": _datum_triangle_open,
    "datum_target": _datum_target,
    "datum_target_point": _datum_target_point,
    # dimensioning / manufacturing
    "diameter": _diameter,
    "square_feature": _square_feature,
    "depth": _depth,
    "counterbore": _counterbore,
    "countersink": _countersink,
    "conical_taper": _conical_taper,
    "slope": _slope,
    "all_around": _all_around,
    "all_over": _all_over,
    "between": _between,
    "parting_line": _parting_line,
    "dimension_origin": _dimension_origin,
    # circled-letter modifiers
    "circled_M": _circled("M"),
    "circled_L": _circled("L"),
    "circled_I": _circled("I"),
    "circled_E": _circled("E"),
    "circled_P": _circled("P"),
    "circled_F": _circled("F"),
    "circled_T": _circled("T"),
    "circled_U": _circled("U"),
    "circled_S": _circled("S"),
    "circled_CF": _circled("CF"),
    "circled_ST": _circled("ST"),
}


def template_names() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def has_template(name: str) -> bool:
    return name in _BUILDERS


def render(name: str, size: int = 48, *, stroke: int | None = None) -> np.ndarray:
    """Draw ``name`` into a ``size``×``size`` uint8 array (0 bg, 255 ink)."""
    if cv2 is None:
        raise TemplateUnavailable("cv2 is required to render GD&T templates")
    builder = _BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"no template named {name!r}")
    size = max(8, int(size))
    arr = np.full((size, size), BG, dtype=np.uint8)
    builder(_Canvas(arr=arr, size=size, stroke=_stroke_for(size, stroke)))
    return arr


_CACHE: dict[tuple[str, int, int | None], np.ndarray] = {}


def render_cached(name: str, size: int = 48, *, stroke: int | None = None) -> np.ndarray:
    """:func:`render` memoised — a page scan re-requests the same few sizes."""
    key = (name, max(8, int(size)), stroke)
    hit = _CACHE.get(key)
    if hit is None:
        hit = render(name, size, stroke=stroke)
        _CACHE[key] = hit
    return hit.copy()


def scales_for(height: int, scales: tuple[float, ...] = DEFAULT_SCALES) -> tuple[int, ...]:
    """Template sizes to try for a glyph of roughly ``height`` pixels.

    Deduplicated and floored at 10px: below that a template carries no
    distinguishing detail and matching it is noise.
    """
    out: list[int] = []
    for s in scales:
        size = int(round(height * s))
        if size >= 10 and size not in out:
            out.append(size)
    return tuple(out) or (10,)


def contact_sheet(size: int = 48, columns: int = 8, pad: int = 6) -> np.ndarray:
    """Every template tiled into one image, for eyeballing the geometry.

    Used by ``scripts/gdt_contact_sheet.py`` and by the template test, which
    checks each glyph actually puts ink down rather than checking pixels.
    """
    if cv2 is None:
        raise TemplateUnavailable("cv2 is required to render GD&T templates")
    names = template_names()
    rows = (len(names) + columns - 1) // columns
    cell = size + 2 * pad
    sheet = np.full((rows * cell, columns * cell), BG, dtype=np.uint8)
    for index, name in enumerate(names):
        r, c = divmod(index, columns)
        tile = render(name, size)
        y, x = r * cell + pad, c * cell + pad
        sheet[y : y + size, x : x + size] = tile
    return sheet
