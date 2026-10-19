"""
Synthetic GD&T drawings with exact ground truth.

Why synthetic, and what it is and is not good for
-------------------------------------------------
There is no labelled GD&T drawing set in this repository — the three fixtures
in ``benchmarks/documents/`` are dimension-value ground truth for whole sheets,
with no symbol boxes — and inventing accuracy numbers for real scans would be
worse than having none. What *can* be built honestly is a generator: it draws
frames and callouts from the same geometry the registry describes, so every
symbol's identity, bounding box and field values are known exactly rather than
annotated by hand.

That makes this an honest measure of some things and no measure at all of
others, and the distinction has to stay visible in any report built on it:

* **It measures**: whether the classifier separates the symbols from each
  other, how it degrades under blur, noise, skew and low resolution, whether
  frame segmentation finds the right compartments, whether field parsing and
  datum precedence are right, and localisation quality.
* **It does not measure**: performance on real CAD fonts, on scanner artefacts
  this generator does not model, on drawings whose symbols touch dimension
  lines in ways not simulated here, or on any real-world class balance.

A number from this generator is a *regression baseline and a floor*, never a
claim about field accuracy. ``docs/GDT_DATASET.md`` records the annotation
workflow for real drawings, which is what would close that gap.

Determinism
-----------
Every scene takes a seed and is reproducible. A benchmark that moves when
nothing changed is worthless as a regression signal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from PIL import Image

import gdt_registry as reg
import gdt_templates as tpl

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


PAPER = 255
INK = 0


@dataclass
class TruthSymbol:
    """One symbol placed on a scene, with where it went."""

    symbol_id: str
    bbox: tuple[int, int, int, int]
    #: Set when the symbol sits in a frame compartment.
    frame_id: str | None = None
    role: str | None = None  # "characteristic" | "modifier" | "standalone"

    def to_dict(self) -> dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "symbol_id": self.symbol_id,
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "frame_id": self.frame_id,
            "role": self.role,
        }


@dataclass
class TruthFrame:
    """One feature control frame placed on a scene, with its field values."""

    id: str
    bbox: tuple[int, int, int, int]
    symbol_id: str
    tolerance: float
    diameter_zone: bool
    material_condition: str | None
    datums: tuple[str, ...]
    segment_index: int = 0
    composite_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "id": self.id,
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "symbol_id": self.symbol_id,
            "tolerance": self.tolerance,
            "diameter_zone": self.diameter_zone,
            "material_condition": self.material_condition,
            "datums": list(self.datums),
            "segment_index": self.segment_index,
            "composite_id": self.composite_id,
        }


@dataclass
class Scene:
    """A rendered drawing plus everything true about it."""

    name: str
    image: Image.Image
    symbols: list[TruthSymbol] = field(default_factory=list)
    frames: list[TruthFrame] = field(default_factory=list)
    #: What degradation was applied, for per-condition reporting.
    condition: str = "clean"

    def truth(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition": self.condition,
            "width": self.image.width,
            "height": self.image.height,
            "symbols": [s.to_dict() for s in self.symbols],
            "frames": [f.to_dict() for f in self.frames],
        }


# ── Drawing primitives ───────────────────────────────────────────────────────


def _blank(width: int, height: int) -> np.ndarray:
    return np.full((height, width), PAPER, dtype=np.uint8)


def _stamp(
    canvas: np.ndarray, glyph: np.ndarray, x: int, y: int
) -> tuple[int, int, int, int]:
    """Paste a template (ink-white) onto paper (ink-black) at x, y.

    Returns the box of the **ink actually laid down**, not the template cell.
    Those differ: a template is drawn with a margin inside its box, so a cell
    box overstates the glyph by about 28% on each axis. Recording the cell as
    ground truth made every correct detection score an IoU near 0.55 and made
    the localisation metric a measurement of the margin constant rather than
    of the detector.
    """
    h, w = glyph.shape[:2]
    ch, cw = canvas.shape[:2]
    if x < 0 or y < 0 or x + w > cw or y + h > ch:
        return (x, y, w, h)
    region = canvas[y : y + h, x : x + w]
    # Template is white-on-black; paper is black-on-white. Darken where ink is.
    canvas[y : y + h, x : x + w] = np.minimum(region, 255 - glyph)
    ys, xs = np.nonzero(glyph > 127)
    if len(xs) == 0:
        return (x, y, w, h)
    return (
        x + int(xs.min()),
        y + int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def _text(
    canvas: np.ndarray, text: str, x: int, y: int, height: int
) -> tuple[int, int]:
    """Draw text with its baseline box starting at (x, y). Returns its size."""
    if cv2 is None or not text:
        return (0, 0)
    thickness = max(1, height // 10)
    scale = 0.1
    for _ in range(80):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if th >= height * 0.72:
            break
        scale += 0.04
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        canvas,
        text,
        (x, y + (height + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        INK,
        thickness,
        lineType=cv2.LINE_AA,
    )
    return (tw, th)


def _rect(canvas: np.ndarray, x: int, y: int, w: int, h: int, thickness: int = 2) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), INK, thickness)


# ── Frame rendering ──────────────────────────────────────────────────────────

#: Width of each compartment as a multiple of the frame height.
_SYMBOL_CELL = 1.25
_TOLERANCE_CELL = 2.6
_DATUM_CELL = 0.95


def draw_frame(
    canvas: np.ndarray,
    x: int,
    y: int,
    height: int,
    *,
    symbol_id: str,
    tolerance: float,
    diameter_zone: bool = False,
    material_condition: str | None = None,
    datums: Sequence[str] = (),
    frame_id: str = "f",
    segment_index: int = 0,
    composite_id: str | None = None,
    min_width: int = 0,
) -> tuple[TruthFrame, list[TruthSymbol]]:
    """Draw one feature control frame and return its ground truth.

    ``min_width`` widens the tolerance compartment to reach a given total.
    The segments of a composite frame are drawn to one width in practice, so
    a generator that stacked different widths would be testing a shape that
    does not occur.
    """
    sym = reg.get(symbol_id)
    pad = max(2, height // 8)
    glyph_size = height - 2 * pad
    # Size the tolerance compartment to its contents. A fixed width let the
    # material-condition modifier overflow into the divider, where the crop
    # that reads it clipped the glyph in half.
    needed = pad * 2 + int(height * 0.42) * max(len(f"{tolerance:g}"), 1)
    if diameter_zone:
        needed += glyph_size + pad
    if material_condition:
        needed += glyph_size + pad
    widths = [
        int(height * _SYMBOL_CELL),
        max(int(height * _TOLERANCE_CELL), needed),
    ]
    widths += [int(height * _DATUM_CELL)] * len(datums)
    if min_width > sum(widths):
        widths[1] += min_width - sum(widths)
    total = sum(widths)

    _rect(canvas, x, y, total, height, thickness=max(1, height // 14))

    truths: list[TruthSymbol] = []
    cursor = x

    # compartment 0 — the characteristic symbol
    if sym.template:
        glyph = tpl.render(sym.template, glyph_size)
        gx = cursor + (widths[0] - glyph_size) // 2
        gy = y + pad
        ink = _stamp(canvas, glyph, gx, gy)
        truths.append(
            TruthSymbol(
                symbol_id=symbol_id,
                bbox=ink,
                frame_id=frame_id,
                role="characteristic",
            )
        )
    cursor += widths[0]
    cv2.line(canvas, (cursor, y), (cursor, y + height), INK, max(1, height // 14))

    # compartment 1 — the tolerance
    tol_text = f"{tolerance:g}"
    tx = cursor + pad
    if diameter_zone:
        dia = tpl.render("diameter", glyph_size)
        ink = _stamp(canvas, dia, tx, y + pad)
        truths.append(
            TruthSymbol(
                symbol_id="diameter",
                bbox=ink,
                frame_id=frame_id,
                role="modifier",
            )
        )
        tx += glyph_size + pad // 2
    tw, _ = _text(canvas, tol_text, tx, y, height)
    tx += tw + pad // 2
    if material_condition:
        letter = {"mmc": "M", "lmc": "L", "rfs": "S"}.get(material_condition)
        if letter:
            mod = tpl.render(f"circled_{letter}", glyph_size)
            ink = _stamp(canvas, mod, tx, y + pad)
            truths.append(
                TruthSymbol(
                    symbol_id=material_condition,
                    bbox=ink,
                    frame_id=frame_id,
                    role="modifier",
                )
            )
    cursor += widths[1]

    # datum compartments
    for index, letter in enumerate(datums):
        cv2.line(canvas, (cursor, y), (cursor, y + height), INK, max(1, height // 14))
        _text(canvas, letter, cursor + pad, y, height)
        cursor += widths[2 + index]

    return (
        TruthFrame(
            id=frame_id,
            bbox=(x, y, total, height),
            symbol_id=symbol_id,
            tolerance=tolerance,
            diameter_zone=diameter_zone,
            material_condition=material_condition,
            datums=tuple(datums),
            segment_index=segment_index,
            composite_id=composite_id,
        ),
        truths,
    )


def draw_datum_feature(
    canvas: np.ndarray, x: int, y: int, height: int, letter: str
) -> TruthSymbol:
    """A datum feature symbol: triangle, stem, boxed letter."""
    tri = tpl.render("datum_triangle", height)
    _stamp(canvas, tri, x, y + height + height // 2)
    cv2.line(
        canvas,
        (x + height // 2, y + height),
        (x + height // 2, y + height + height // 2),
        INK,
        max(1, height // 14),
    )
    _rect(canvas, x, y, height, height, thickness=max(1, height // 14))
    _text(canvas, letter, x + height // 4, y, height)
    return TruthSymbol(
        symbol_id="datum_feature_symbol",
        bbox=(x, y, height, height + height // 2 + height),
        role="standalone",
    )


# ── Degradations ─────────────────────────────────────────────────────────────


def degrade(
    image: Image.Image,
    *,
    blur: int = 0,
    noise: float = 0.0,
    skew_deg: float = 0.0,
    scale: float = 1.0,
    jpeg_quality: int | None = None,
    seed: int = 0,
) -> Image.Image:
    """Apply the degradations a real scan carries, reproducibly."""
    arr = np.array(image.convert("L"))
    if scale != 1.0 and cv2 is not None:
        h, w = arr.shape[:2]
        arr = cv2.resize(
            arr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
    if skew_deg and cv2 is not None:
        h, w = arr.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), skew_deg, 1.0)
        arr = cv2.warpAffine(arr, matrix, (w, h), borderValue=PAPER)
    if blur and cv2 is not None:
        k = blur if blur % 2 == 1 else blur + 1
        arr = cv2.GaussianBlur(arr, (k, k), 0)
    if noise:
        rng = np.random.RandomState(seed)
        arr = np.clip(
            arr.astype(np.int16) + rng.normal(0, noise, arr.shape), 0, 255
        ).astype(np.uint8)
    out = Image.fromarray(arr).convert("RGB")
    if jpeg_quality is not None:
        import io

        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


#: The conditions the evaluation sweeps. Named so a report can show per-class
#: results per condition rather than one aggregate that hides the hard cases.
CONDITIONS: dict[str, dict[str, Any]] = {
    "clean": {},
    "blurred": {"blur": 5},
    "noisy": {"noise": 28.0},
    "low_res": {"scale": 0.5},
    "skewed": {"skew_deg": 2.5},
    "scan": {"blur": 3, "noise": 18.0, "skew_deg": 1.2, "jpeg_quality": 60},
    "poor_scan": {"blur": 5, "noise": 30.0, "skew_deg": 3.0, "scale": 0.65,
                  "jpeg_quality": 40},
}


# ── Scene builders ───────────────────────────────────────────────────────────

_DATUM_LETTERS = ("A", "B", "C", "D", "E", "F")


def scene_symbol_grid(
    *, cell: int = 56, columns: int = 8, seed: int = 0
) -> Scene:
    """Every templated symbol on a grid — the per-class classification test."""
    ids = [
        s.id
        for s in reg.ALL_SYMBOLS
        if s.template and tpl.has_template(s.template) and s.detectable
    ]
    rows = (len(ids) + columns - 1) // columns
    pad = cell // 3
    width = columns * (cell + pad) + pad
    height = rows * (cell + pad) + pad
    canvas = _blank(width, height)
    truths: list[TruthSymbol] = []
    for index, symbol_id in enumerate(ids):
        r, c = divmod(index, columns)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + pad)
        sym = reg.get(symbol_id)
        glyph = tpl.render(sym.template, cell)
        ink = _stamp(canvas, glyph, x, y)
        truths.append(
            TruthSymbol(symbol_id=symbol_id, bbox=ink, role="standalone")
        )
    return Scene(
        name="symbol_grid",
        image=Image.fromarray(canvas).convert("RGB"),
        symbols=truths,
    )


def scene_frames(*, count: int = 6, height: int = 44, seed: int = 0) -> Scene:
    """A column of feature control frames with varied fields."""
    rng = random.Random(seed)
    characteristics = [s.id for s in reg.FRAME_CHARACTERISTICS]
    pad = height
    width = height * 10
    canvas = _blank(width, pad + count * (height + pad))
    frames: list[TruthFrame] = []
    truths: list[TruthSymbol] = []
    for index in range(count):
        symbol_id = characteristics[index % len(characteristics)]
        is_form = symbol_id in {
            "straightness", "flatness", "circularity", "cylindricity",
        }
        n_datums = 0 if is_form else rng.choice([1, 2, 3])
        frame, syms = draw_frame(
            canvas,
            pad // 2,
            pad + index * (height + pad),
            height,
            symbol_id=symbol_id,
            tolerance=round(rng.choice([0.05, 0.1, 0.2, 0.5, 1.0]), 3),
            diameter_zone=symbol_id == "position" and rng.random() < 0.8,
            material_condition=(
                rng.choice([None, "mmc", "lmc"]) if not is_form else None
            ),
            datums=_DATUM_LETTERS[:n_datums],
            frame_id=f"frame{index}",
        )
        frames.append(frame)
        truths.extend(syms)
    return Scene(
        name="frames",
        image=Image.fromarray(canvas).convert("RGB"),
        frames=frames,
        symbols=truths,
    )


def scene_composite_frame(*, height: int = 44, seed: int = 0) -> Scene:
    """A two-segment composite position control."""
    pad = height
    canvas = _blank(height * 10, pad * 2 + height * 2)
    frames: list[TruthFrame] = []
    truths: list[TruthSymbol] = []
    # Both segments drawn to the width of the wider one, as ASME draws them.
    full_width = (
        int(height * _SYMBOL_CELL)
        + int(height * _TOLERANCE_CELL)
        + 3 * int(height * _DATUM_CELL)
    )
    for segment in range(2):
        frame, syms = draw_frame(
            canvas,
            pad // 2,
            pad + segment * height,
            height,
            symbol_id="position",
            tolerance=0.5 if segment == 0 else 0.1,
            diameter_zone=True,
            material_condition="mmc" if segment == 0 else None,
            datums=("A", "B", "C") if segment == 0 else ("A",),
            frame_id=f"composite{segment}",
            segment_index=segment,
            composite_id="composite",
            min_width=full_width,
        )
        frames.append(frame)
        truths.extend(syms)
    return Scene(
        name="composite_frame",
        image=Image.fromarray(canvas).convert("RGB"),
        frames=frames,
        symbols=truths,
    )


def scene_dense(*, seed: int = 0, height: int = 38) -> Scene:
    """Frames packed close together with datum features between them.

    The crowded case: adjacent frames, symbols near dimension lines, and a
    datum feature symbol whose triangle sits on a drawn surface.
    """
    rng = random.Random(seed)
    canvas = _blank(height * 22, height * 14)
    frames: list[TruthFrame] = []
    truths: list[TruthSymbol] = []

    # A horizontal surface line the datum features stand on.
    cv2.line(
        canvas,
        (height, height * 12),
        (height * 20, height * 12),
        INK,
        max(1, height // 16),
    )

    positions = [
        (height, height),
        (height * 11, height),
        (height, height * 4),
        (height * 11, height * 4),
        (height, height * 7),
    ]
    characteristics = [
        "position", "perpendicularity", "profile_of_a_surface",
        "parallelism", "total_runout",
    ]
    for index, ((x, y), symbol_id) in enumerate(zip(positions, characteristics)):
        frame, syms = draw_frame(
            canvas, x, y, height,
            symbol_id=symbol_id,
            tolerance=rng.choice([0.02, 0.08, 0.25]),
            diameter_zone=symbol_id == "position",
            material_condition=rng.choice([None, "mmc"]),
            datums=_DATUM_LETTERS[: rng.choice([1, 2])],
            frame_id=f"dense{index}",
        )
        frames.append(frame)
        truths.extend(syms)

    for index, letter in enumerate(("A", "B")):
        truths.append(
            draw_datum_feature(
                canvas, height * (3 + index * 6), height * 9, height, letter
            )
        )

    return Scene(
        name="dense",
        image=Image.fromarray(canvas).convert("RGB"),
        frames=frames,
        symbols=truths,
    )


def scene_confusables(*, cell: int = 56, seed: int = 0) -> Scene:
    """The pairs that actually get confused, side by side.

    Deliberately stocked with the registry's own ``confusable_with`` entries so
    a regression in separating them shows up as a number rather than as a
    surprise on a customer drawing.
    """
    pairs = [
        ("circularity", "diameter"),
        ("circularity", "concentricity"),
        ("position", "circularity"),
        ("concentricity", "all_over"),
        ("all_around", "all_over"),
        ("circular_runout", "total_runout"),
        ("profile_of_a_line", "profile_of_a_surface"),
        ("perpendicularity", "angularity"),
        ("parallelism", "total_runout"),
        ("mmc", "lmc"),
        ("counterbore", "countersink"),
        ("depth", "countersink"),
        ("conical_taper", "slope"),
        ("straightness", "symmetry"),
    ]
    pad = cell // 3
    width = pad + 2 * (cell + pad)
    height = pad + len(pairs) * (cell + pad)
    canvas = _blank(width, height)
    truths: list[TruthSymbol] = []
    for row, pair in enumerate(pairs):
        for col, symbol_id in enumerate(pair):
            sym = reg.get(symbol_id)
            if not sym.template:
                continue
            x = pad + col * (cell + pad)
            y = pad + row * (cell + pad)
            ink = _stamp(canvas, tpl.render(sym.template, cell), x, y)
            truths.append(
                TruthSymbol(symbol_id=symbol_id, bbox=ink, role="standalone")
            )
    return Scene(
        name="confusables",
        image=Image.fromarray(canvas).convert("RGB"),
        symbols=truths,
    )


def scene_small_modifiers(*, cell: int = 18, seed: int = 0) -> Scene:
    """Circled modifiers at note size — the resolution floor case."""
    ids = ["mmc", "lmc", "projected_tolerance_zone", "free_state", "tangent_plane"]
    pad = cell
    canvas = _blank(pad + len(ids) * (cell + pad), cell + 2 * pad)
    truths: list[TruthSymbol] = []
    for index, symbol_id in enumerate(ids):
        sym = reg.get(symbol_id)
        x = pad + index * (cell + pad)
        ink = _stamp(canvas, tpl.render(sym.template, cell), x, pad)
        truths.append(
            TruthSymbol(symbol_id=symbol_id, bbox=ink, role="modifier")
        )
    return Scene(
        name="small_modifiers",
        image=Image.fromarray(canvas).convert("RGB"),
        symbols=truths,
    )


#: Every scene builder, by name.
BUILDERS = {
    "symbol_grid": scene_symbol_grid,
    "frames": scene_frames,
    "composite_frame": scene_composite_frame,
    "dense": scene_dense,
    "confusables": scene_confusables,
    "small_modifiers": scene_small_modifiers,
}


def build(name: str, *, seed: int = 0, **kwargs: Any) -> Scene:
    builder = BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"no scene named {name!r}; have {sorted(BUILDERS)}")
    return builder(seed=seed, **kwargs)


def build_all(*, seed: int = 0) -> list[Scene]:
    return [build(name, seed=seed) for name in BUILDERS]


def with_condition(scene: Scene, condition: str, *, seed: int = 0) -> Scene:
    """Re-render a scene under a named degradation.

    Ground-truth boxes are only valid for conditions that do not move pixels.
    A scaled or skewed scene keeps its symbol *identities* for classification
    scoring, and its boxes are marked invalid for localisation scoring by
    returning them scaled where that is exact and dropping them where it is
    not — see ``gdt_eval``, which checks ``condition`` before scoring IoU.
    """
    if condition not in CONDITIONS:
        raise KeyError(f"unknown condition {condition!r}")
    params = dict(CONDITIONS[condition])
    scale = params.get("scale", 1.0)
    degraded = degrade(scene.image, seed=seed, **params)
    symbols = scene.symbols
    frames = scene.frames
    if scale != 1.0:
        symbols = [
            TruthSymbol(
                symbol_id=s.symbol_id,
                bbox=(
                    int(s.bbox[0] * scale), int(s.bbox[1] * scale),
                    int(s.bbox[2] * scale), int(s.bbox[3] * scale),
                ),
                frame_id=s.frame_id,
                role=s.role,
            )
            for s in scene.symbols
        ]
        frames = [
            TruthFrame(
                id=f.id,
                bbox=(
                    int(f.bbox[0] * scale), int(f.bbox[1] * scale),
                    int(f.bbox[2] * scale), int(f.bbox[3] * scale),
                ),
                symbol_id=f.symbol_id,
                tolerance=f.tolerance,
                diameter_zone=f.diameter_zone,
                material_condition=f.material_condition,
                datums=f.datums,
                segment_index=f.segment_index,
                composite_id=f.composite_id,
            )
            for f in scene.frames
        ]
    return Scene(
        name=scene.name,
        image=degraded,
        symbols=symbols,
        frames=frames,
        condition=condition,
    )
