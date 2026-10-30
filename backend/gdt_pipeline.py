"""
The GD&T recognition pass, assembled.

This is the orchestration layer: it takes a rendered page (or a crop of one),
runs frame parsing and loose-symbol detection over it, applies the context
rules, and returns one structured result in the page's own coordinates.

Where it sits
-------------
It is **additive**. ``ocr_pipeline`` keeps doing exactly what it did; this runs
alongside and its output arrives in new response fields. Nothing here changes
an existing contract, and a failure here degrades the GD&T section of a
response to an error record rather than failing the document — which is what
the brief asks for and what a local operator mid-inspection needs.

Two paths, one annotation
-------------------------
There are two ways a GD&T control reaches this result, and they answer
different questions.

The **graphical** pass (``gdt_detect``, ``fcf_parse``) finds frames and glyphs
in pixels. The **textual** pass (``gdt_text``, spec section 16) reads controls
a drawing spells out in words, which have no glyph for a template to match and
are invisible to the graphical pass by construction.

Section 16 was switched off for most of this project's life because its rule
matched a GD&T *word* anywhere in a string — measured over 170 OCR strings
from four real drawings, that gave no true positives and one false positive.
It is on now, behind the evidence requirements in ``gdt_text``, which claim
nothing on that same corpus.

Where both paths describe the same callout, ``link_text_annotations`` attaches
the textual read to the frame rather than reporting it twice, and records
whether the two agree.

Tiling
------
A D-size drawing at 300dpi is around 10000×7000. Running contour detection and
41 shape comparisons over that in one pass is slow and, worse, makes the
adaptive threshold answer a question about the whole sheet when the question
is local. The page is tiled with an overlap wider than any symbol, and the
overlap duplicates are merged afterwards by ``gdt_detect.merge_detections``.
"""

from __future__ import annotations

import re

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

import fcf_parse
import gdt_context as ctx
import gdt_registry as reg
from fcf_parse import FeatureControlFrame, OcrFn
from gdt_config import GdtConfig, get_config
from gdt_detect import (
    STATUS_AMBIGUOUS,
    STATUS_CONFIRMED,
    STATUS_UNKNOWN,
    Detection,
    Transform,
    IDENTITY,
    classify_crop,
    find_glyph_boxes,
    merge_detections,
)


#: Shortest page edge, in pixels, below which the page is enlarged before it
#: is read. Frame field accuracy goes from 92% to 100% between 44px and 66px
#: of frame height, and a raster drawing uploaded as a photo or a screenshot
#: arrives at whatever size it was saved at — on a 1291px-wide sheet a frame
#: is 14px tall and its symbol 9px, which is below the floor at which
#: confusable glyphs can be told apart at all. A PDF page is rendered large
#: by the frontend before it gets here and is left alone.
MIN_WORK_SHORT_EDGE_PX = 2000

#: Enlarging further than this invents no detail and multiplies the cost.
MAX_WORK_UPSCALE = 3.0


@dataclass
class PageResult:
    """Everything the GD&T pass found on one page."""

    page: int
    width: int
    height: int
    frames: list[FeatureControlFrame] = field(default_factory=list)
    symbols: list[Detection] = field(default_factory=list)
    #: GD&T read from OCR *text* rather than from pixels — a control a drawing
    #: spells out in words has no glyph for the graphical pass to find. Each
    #: entry says whether it corroborates a frame found graphically or stands
    #: alone, so the two paths never read as two separate findings.
    text_annotations: list[dict[str, Any]] = field(default_factory=list)
    datum_associations: list[dict[str, Any]] = field(default_factory=list)
    standards_findings: list[dict[str, Any]] = field(default_factory=list)
    #: Non-fatal failures: a tile that threw, a frame that could not be read.
    errors: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    #: What the result can and cannot be held to.
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "frames": len(self.frames),
            "symbols": len(self.symbols),
            "confirmed": sum(1 for s in self.symbols if s.status == STATUS_CONFIRMED),
            "ambiguous": sum(1 for s in self.symbols if s.status == STATUS_AMBIGUOUS),
            "unknown": sum(1 for s in self.symbols if s.status == STATUS_UNKNOWN),
            "text_annotations": len(self.text_annotations),
            "errors": len(self.errors),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "image_size": {"width": self.width, "height": self.height},
            "frames": [f.to_dict() for f in self.frames],
            "symbols": [s.to_dict() for s in self.symbols],
            "text_annotations": self.text_annotations,
            "datum_associations": self.datum_associations,
            "standards_findings": self.standards_findings,
            "errors": self.errors,
            "counts": self.counts,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "provenance": self.provenance,
        }


# ── Deskewing ────────────────────────────────────────────────────────────────
#
# Frame detection finds a frame from its ruled edges using axis-aligned
# morphology, so it is only as good as the page's alignment. Measured on the
# synthetic suite: at 2.5° of skew, frame detection fell from 13 of 13 to 0 of
# 13 — not degraded, gone. A scan is never perfectly square, so without this
# the whole frame path is unavailable on exactly the documents that need it.
#
# The estimate deliberately uses the *long lines*, which is the opposite of
# what ``image_preprocess.estimate_skew_angle`` does — that one suppresses long
# lines to find a text baseline inside a small crop. On a whole drawing the
# long lines are the border, the frames and the dimension lines, and they are
# the most reliable statement of square the sheet has.

try:  # cv2 is optional throughout this package
    import cv2 as _cv2
    import numpy as _np
except ImportError:  # pragma: no cover
    _cv2 = None  # type: ignore
    _np = None  # type: ignore


#: Skews smaller than this are not worth resampling the page for.
MIN_SKEW_DEG = 0.3
#: Beyond this the page is not skewed, it is rotated or the estimate is wrong.
MAX_SKEW_DEG = 15.0


def estimate_page_skew(image: Image.Image) -> float | None:
    """Estimate a drawing's skew in degrees from its long straight lines.

    Returns the angle to rotate *by* to square the page (positive =
    counter-clockwise), or ``None`` when the page gives no clear answer.
    """
    if _cv2 is None or _np is None:
        return None
    from gdt_detect import binarize, to_gray

    gray = to_gray(image)
    binary = binarize(gray)
    h, w = binary.shape[:2]
    min_len = max(40, int(min(h, w) * 0.25))
    lines = _cv2.HoughLinesP(
        binary, 1, _np.pi / 360, threshold=min_len,
        minLineLength=min_len, maxLineGap=6,
    )
    if lines is None:
        return None

    offsets: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        angle = float(_np.degrees(_np.arctan2(y2 - y1, x2 - x1)))
        # Fold every line onto its deviation from the nearest axis. A drawing's
        # verticals carry the same skew information as its horizontals, and
        # folding them together roughly doubles the sample.
        offset = ((angle + 45.0) % 90.0) - 45.0
        if abs(offset) <= MAX_SKEW_DEG:
            offsets.append(offset)
            weights.append(length)
    if len(offsets) < 4:
        return None

    # The weighted median: robust to the handful of genuinely diagonal lines a
    # drawing contains (leaders, hatching, a chamfer) in a way a mean is not.
    order = sorted(range(len(offsets)), key=lambda i: offsets[i])
    total = sum(weights)
    running = 0.0
    angle = offsets[order[-1]]
    for index in order:
        running += weights[index]
        if running >= total / 2:
            angle = offsets[index]
            break
    if abs(angle) < MIN_SKEW_DEG:
        return None
    # The measured offset *is* the correction: ``deskew_page`` rotates by this
    # value and squares the page. Verified by residual — a page corrected by
    # this angle measures no remaining skew, and one corrected by its negative
    # measures twice the original.
    return angle


def _rotation_matrices(
    image: Image.Image, angle: float
) -> "tuple[Any, Any, tuple[int, int]]":
    """Forward and inverse affine matrices for rotating a page by ``angle``."""
    w, h = image.size
    center = (w / 2.0, h / 2.0)
    matrix = _cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    inverse = _cv2.invertAffineTransform(matrix)
    return matrix, inverse, (new_w, new_h)


def deskew_page(
    image: Image.Image, angle: float
) -> "tuple[Image.Image, Any]":
    """Rotate a page by ``angle`` and return it with the inverse transform."""
    matrix, inverse, (new_w, new_h) = _rotation_matrices(image, angle)
    arr = _np.array(image.convert("RGB"))
    rotated = _cv2.warpAffine(
        arr, matrix, (new_w, new_h),
        flags=_cv2.INTER_CUBIC, borderValue=(255, 255, 255),
    )
    return Image.fromarray(rotated), inverse


def _unrotate_box(
    box: tuple[float, float, float, float], inverse: Any
) -> tuple[float, float, float, float]:
    """Map a box from the deskewed page back to original page coordinates.

    The four corners are mapped and re-bounded rather than just the origin,
    because a rotated rectangle's axis-aligned bounds are not its corner plus
    the original width.
    """
    x, y, w, h = box
    corners = _np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=_np.float64
    )
    ones = _np.ones((4, 1))
    mapped = (_np.hstack([corners, ones]) @ inverse.T)
    xs, ys = mapped[:, 0], mapped[:, 1]
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min()),
        float(ys.max() - ys.min()),
    )


# ── Tiling ───────────────────────────────────────────────────────────────────


def tiles_for(
    width: int, height: int, *, config: GdtConfig | None = None
) -> list[tuple[int, int, int, int]]:
    """Cover a page with overlapping tiles.

    The overlap must exceed the largest symbol the detector will accept, or a
    glyph straddling a seam is cut in half in both tiles and recognized in
    neither.
    """
    cfg = config or get_config()
    size = max(256, cfg.tile_px)
    overlap = max(32, cfg.tile_overlap_px)
    if width <= size and height <= size:
        return [(0, 0, width, height)]

    step = max(64, size - overlap)
    out: list[tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        x = 0
        th = min(size, height - y)
        while x < width:
            tw = min(size, width - x)
            out.append((x, y, tw, th))
            if x + tw >= width:
                break
            x += step
        if y + th >= height:
            break
        y += step
    return out


# ── The pass ─────────────────────────────────────────────────────────────────


def detect_symbols_in_region(
    image: Image.Image,
    *,
    transform: Transform = IDENTITY,
    page: int = 1,
    texts: Sequence[ctx.TextBox] = (),
    config: GdtConfig | None = None,
) -> list[Detection]:
    """Classify every glyph-sized mark in one region.

    Marks that match nothing are dropped rather than published as "unknown
    symbol" — a page has thousands of them and they are drawing geometry, not
    unrecognized GD&T. An unknown is only worth reporting where something else
    says a symbol should be, which is why a frame compartment that cannot be
    read *does* produce one.
    """
    cfg = config or get_config()
    out: list[Detection] = []
    for x, y, w, h in find_glyph_boxes(image, config=cfg):
        crop = image.crop((x, y, x + w, y + h))
        local = Transform(
            offset_x=transform.offset_x + x / max(transform.scale_x, 1e-9),
            offset_y=transform.offset_y + y / max(transform.scale_y, 1e-9),
            scale_x=transform.scale_x,
            scale_y=transform.scale_y,
        )
        try:
            det = classify_crop(
                crop, transform=local, page=page, config=cfg
            )
        except Exception:  # noqa: BLE001 - one bad glyph must not stop the tile
            continue
        if det.symbol_id is None or det.status == STATUS_UNKNOWN:
            continue
        if det.confidence < cfg.ambiguous_confidence:
            continue
        out.append(det)
    return out


def analyze_page(
    image: Image.Image,
    *,
    page: int = 1,
    ocr_fn: OcrFn | None = None,
    texts: Sequence[ctx.TextBox] = (),
    declared_standard: str | None = None,
    source: str | None = None,
    config: GdtConfig | None = None,
    detect_loose_symbols: bool = True,
    deskew: bool = True,
) -> PageResult:
    """Run the whole GD&T pass over one rendered page.

    ``texts`` are OCR boxes already read from this page in *page* coordinates;
    passing the ones the existing pipeline produced avoids a second OCR pass
    and keeps symbol-to-value association consistent with what the rest of the
    app shows.

    Every stage is wrapped: a tile that throws is recorded in ``errors`` and
    the rest of the page still returns.
    """
    cfg = config or get_config()
    started = time.perf_counter()
    result = PageResult(page=page, width=image.width, height=image.height)
    result.provenance = {
        "source": source,
        "registry_version": reg.to_dict()["version"],
        "registry_symbols": len(reg.ALL_SYMBOLS),
        "declared_standard": declared_standard,
        "config": cfg.to_dict(),
        "note": (
            "Symbol recognition only. Nothing here asserts that the drawing is "
            "compliant or non-compliant with any standard; an interpretation "
            "that depends on the standard revision is reported as uncertain."
        ),
    }

    if not cfg.enabled:
        result.provenance["skipped"] = "GDT_ENABLED is off"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    # Square the page before anything measures it. Detections are found in the
    # deskewed page and mapped back, so every coordinate published is in the
    # caller's original space regardless of what happened in between.
    work = image
    inverse = None
    skew = None
    if deskew:
        try:
            skew = estimate_page_skew(image)
            if skew is not None:
                work, inverse = deskew_page(image, skew)
        except Exception as exc:  # noqa: BLE001 - never fail a page over this
            result.errors.append(
                {"stage": "deskew", "error": f"{type(exc).__name__}: {exc}"}
            )
            work, inverse, skew = image, None, None
    result.provenance["deskew_deg"] = round(skew, 3) if skew is not None else None

    # A small raster page is read enlarged — for its frames. The loose-symbol
    # pass stays at the page's own size: its glyph-size limits are fractions
    # of the page and were tuned there, and run over an enlarged page it kept
    # 3 of 28 glyphs on the confusables benchmark. Frame boxes are divided
    # back down as soon as they are read, so from the merge on every box is
    # in deskewed-page coordinates and the deskew is undone once at the end.
    base = work
    upscale = 1.0
    short_edge = min(work.width, work.height)
    if 0 < short_edge < MIN_WORK_SHORT_EDGE_PX:
        upscale = min(MAX_WORK_UPSCALE, MIN_WORK_SHORT_EDGE_PX / short_edge)
        if upscale > 1.05:
            work = work.resize(
                (round(work.width * upscale), round(work.height * upscale)),
                Image.LANCZOS,
            )
        else:
            upscale = 1.0
    result.provenance["work_scale"] = round(upscale, 3)

    def unscale(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if upscale == 1.0:
            return box
        return tuple(float(v) / upscale for v in box)  # type: ignore[return-value]

    all_symbols: list[Detection] = []

    for index, (tx, ty, tw, th) in enumerate(tiles_for(work.width, work.height, config=cfg)):
        tile = work.crop((tx, ty, tx + tw, ty + th))
        transform = Transform(offset_x=float(tx), offset_y=float(ty))
        try:
            frames = fcf_parse.parse_frames(
                tile,
                ocr_fn=ocr_fn,
                transform=transform,
                page=page,
                config=cfg,
                reference_edge=min(work.width, work.height),
            )
            for frame in frames:
                frame.bbox = unscale(frame.bbox)
                for cell in frame.compartments:
                    cell.bbox = unscale(cell.bbox)
                if frame.symbol is not None:
                    frame.symbol.bbox = unscale(frame.symbol.bbox)
                    frame.symbol.frame_id = frame.id
                    all_symbols.append(frame.symbol)
            result.frames.extend(frames)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(
                {
                    "stage": "feature_control_frames",
                    "tile": index,
                    "tile_box": [tx, ty, tw, th],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    if detect_loose_symbols:
        for index, (tx, ty, tw, th) in enumerate(tiles_for(base.width, base.height, config=cfg)):
            tile = base.crop((tx, ty, tx + tw, ty + th))
            transform = Transform(offset_x=float(tx), offset_y=float(ty))
            try:
                all_symbols.extend(
                    detect_symbols_in_region(
                        tile,
                        transform=transform,
                        page=page,
                        texts=texts,
                        config=cfg,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(
                    {
                        "stage": "loose_symbols",
                        "tile": index,
                        "tile_box": [tx, ty, tw, th],
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3),
                    }
                )

    # A symbol inside a frame is found by the frame pass *and* by the loose
    # pass; the merge collapses those. It keeps genuinely repeated symbols,
    # because those do not overlap.
    result.symbols = merge_detections(all_symbols, config=cfg)

    # Drop composite frames that duplicate across a tile seam, and boxes a
    # frame's own contents made that a seam separated from their frame.
    result.frames = fcf_parse.drop_nested_frames(
        _dedupe_frames(result.frames, config=cfg)
    )

    # Everything so far is in deskewed-page coordinates. Put it back.
    if inverse is not None:
        for det in result.symbols:
            det.bbox = _unrotate_box(det.bbox, inverse)
        for frame in result.frames:
            frame.bbox = _unrotate_box(frame.bbox, inverse)
            for cell in frame.compartments:
                cell.bbox = _unrotate_box(cell.bbox, inverse)
            if frame.symbol is not None:
                # The frame's symbol is also in result.symbols; mapping it
                # twice would move it off the page.
                if frame.symbol not in result.symbols:
                    frame.symbol.bbox = _unrotate_box(frame.symbol.bbox, inverse)

    try:
        ctx.attach_text(result.symbols, list(texts))
    except Exception as exc:  # noqa: BLE001
        result.errors.append(
            {"stage": "text_association", "error": f"{type(exc).__name__}: {exc}"}
        )

    try:
        _apply_context(result, texts=texts, config=cfg)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(
            {"stage": "context", "error": f"{type(exc).__name__}: {exc}"}
        )

    try:
        result.text_annotations = link_text_annotations(result, texts)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(
            {"stage": "text_annotations", "error": f"{type(exc).__name__}: {exc}"}
        )

    try:
        result.datum_associations = ctx.associate_datums(result.frames, result.symbols)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(
            {"stage": "datum_association", "error": f"{type(exc).__name__}: {exc}"}
        )

    try:
        result.standards_findings = ctx.check_standards_consistency(
            result.symbols, declared_standard=declared_standard
        )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(
            {"stage": "standards_check", "error": f"{type(exc).__name__}: {exc}"}
        )

    result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result


def link_text_annotations(
    result: PageResult, texts: Sequence[ctx.TextBox]
) -> list[dict[str, Any]]:
    """Read GD&T out of the page's OCR text and tie it to the frames found.

    Two paths can describe one annotation. The graphical pass finds a frame
    from its ruled edges; the text of that same frame, read by OCR, can also
    name the control. Reporting both as independent findings would double-count
    a single callout on the sheet.

    So a textual match overlapping a frame is attached to it as corroboration
    — ``frame_id`` set, ``duplicate_of_frame`` true — and only a match with no
    frame under it is reported as a finding in its own right. That is the case
    worth having: a control a drawing spells out in words has no glyph, so the
    graphical pass is blind to it by construction.
    """
    import gdt_text

    out: list[dict[str, Any]] = []
    for box in texts:
        match, _ = gdt_text.recognize(box.text)
        if match is None:
            continue
        entry = match.to_dict()
        entry["page"] = result.page
        entry["bounding_box"] = {
            "x": round(float(box.bbox[0]), 2),
            "y": round(float(box.bbox[1]), 2),
            "width": round(float(box.bbox[2]), 2),
            "height": round(float(box.bbox[3]), 2),
        }
        covering = next(
            (f for f in result.frames if _overlaps(f.bbox, box.bbox)), None
        )
        entry["frame_id"] = covering.id if covering else None
        entry["duplicate_of_frame"] = covering is not None
        if covering is not None and covering.symbol is not None:
            # Agreement between an independent path is worth recording; so is
            # disagreement, which says one of them is wrong.
            entry["agrees_with_frame"] = (
                covering.symbol.symbol_id == match.symbol_id
            )
        out.append(entry)
    return out


def _overlaps(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    min_iou: float = 0.1,
) -> bool:
    from gdt_detect import iou

    return iou(a, b) >= min_iou


def _apply_context(
    result: PageResult,
    *,
    texts: Sequence[ctx.TextBox],
    config: GdtConfig,
) -> None:
    """Run the Stage 4 rules over a page's detections."""
    frame_boxes = [(f.id, f.bbox) for f in result.frames]

    for det in result.symbols:
        in_frame = det.frame_id is not None
        compartment: int | None = None
        if not in_frame:
            for frame_id, box in frame_boxes:
                if _contains(box, det.bbox):
                    in_frame = True
                    det.frame_id = frame_id
                    break
        if in_frame and det.frame_id:
            frame = next((f for f in result.frames if f.id == det.frame_id), None)
            if frame is not None:
                for cell in frame.compartments:
                    if _contains(cell.bbox, det.bbox):
                        compartment = cell.index
                        break
        ctx.resolve_by_position(
            det,
            in_frame=in_frame,
            frame_compartment=compartment,
            on_leader=False,
            nearby_text=det.associated_text or "",
        )

    ctx.demote_loose_characteristics(result.symbols)


def _contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    *,
    slack: float = 0.25,
) -> bool:
    """True when ``inner``'s centre falls inside ``outer``, with slack."""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    cx, cy = ix + iw / 2, iy + ih / 2
    pad_x, pad_y = ow * slack * 0.1, oh * slack
    return (ox - pad_x) <= cx <= (ox + ow + pad_x) and (
        oy - pad_y
    ) <= cy <= (oy + oh + pad_y)


def _dedupe_frames(
    frames: Sequence[FeatureControlFrame], *, config: GdtConfig
) -> list[FeatureControlFrame]:
    """Collapse frames found twice because they straddle a tile seam."""
    from gdt_detect import iou

    kept: list[FeatureControlFrame] = []
    for frame in sorted(frames, key=lambda f: -f.confidence):
        if any(
            other.page == frame.page and iou(other.bbox, frame.bbox) >= config.merge_iou
            for other in kept
        ):
            continue
        kept.append(frame)
    return sorted(kept, key=lambda f: (f.bbox[1], f.bbox[0]))


# ── Multi-page ───────────────────────────────────────────────────────────────


# ── The page scan's view of a frame ──────────────────────────────────────────

#: The scan region type a frame is published under. It is the same string the
#: rule engine's ``CAT_GDT`` carries, so a frame balloon sorts and filters
#: with the textual GD&T callouts the engine already finds.
SCAN_REGION_TYPE = "GD&T"


def frame_text(frame: FeatureControlFrame) -> str:
    """One frame as the string a balloon shows: ``⌖ Ø0.3 Ⓜ A B``.

    The symbol is its registry glyph, so what the reader sees is the mark on
    the drawing and not its name. A field that could not be read is shown as
    ``?`` rather than left out, because a tolerance that is missing and a
    tolerance that was not read are different things to an inspector.
    """
    parts: list[str] = []
    if frame.symbol is not None and frame.symbol.symbol_id:
        sym = reg.find(frame.symbol.symbol_id)
        parts.append((sym.unicode if sym and sym.unicode else None) or frame.symbol.name)
    elif frame.symbol is not None:
        # Kept unresolved: shown as unread, never guessed.
        parts.append("?")
    tolerance = ""
    if frame.diameter_zone.value is True:
        tolerance += "Ø"
    if frame.tolerance.state == fcf_parse.PRESENT and frame.tolerance.value is not None:
        tolerance += f"{frame.tolerance.value:g}"
    elif frame.tolerance.state == fcf_parse.UNREADABLE:
        tolerance += "?"
    if tolerance:
        parts.append(tolerance)
    mc = frame.material_condition
    if mc.state == fcf_parse.PRESENT and mc.value:
        glyph = reg.find(str(mc.value))
        if glyph and glyph.unicode:
            parts.append(glyph.unicode)
    elif mc.state == fcf_parse.UNREADABLE:
        # A modifier is there and was not read: shown, not dropped.
        parts.append("?")
    # The other modifiers (Ⓤ, Ⓟ, Ⓕ, Ⓣ …) were detected and kept on the frame
    # but never written into the text, so a profile frame "⌓ 0.2 Ⓤ" reached
    # the drawing and every export as "⌓ 0.2".
    for modifier in frame.modifiers:
        glyph = reg.find(str(modifier))
        parts.append(glyph.unicode if glyph and glyph.unicode else str(modifier))
    for datum in frame.datums:
        parts.append(datum.label)
    return " ".join(parts)


def _frame_needs_review(frame: FeatureControlFrame) -> str:
    """Why a frame should be confirmed by a person, or ``""`` when it need not.

    Every reason, not the first: a reviewer told only that the symbol is
    uncertain would confirm the symbol and miss that the tolerance beside it
    was never read.
    """
    reasons: list[str] = []
    if frame.symbol is None or frame.symbol.status != STATUS_CONFIRMED:
        reasons.append("the geometric characteristic symbol is uncertain")
    if frame.tolerance.state != fcf_parse.PRESENT:
        reasons.append("the tolerance could not be read")
    if frame.material_condition.state == fcf_parse.UNREADABLE:
        reasons.append("a modifier is present and could not be read")
    return "; ".join(reasons)


#: A geometric tolerance larger than this is not one. GD&T tolerances are
#: fractions of a millimetre or of an inch; a value in the tens is a boxed
#: basic dimension or a dimension line read through a frame-shaped gap.
MAX_PLAUSIBLE_TOLERANCE = 5.0


def _plausible(frame: FeatureControlFrame) -> bool:
    """Whether a parsed frame is worth a balloon at all.

    The finder over-produces and the parser turns away what has no
    characteristic; what reaches here opens with something a template
    explains. A sheet's part outline still yields a few boxes whose first
    cell holds a stroke that passes for straightness or perpendicularity at
    an ambiguous score. Those have no tolerance beside them, or a "tolerance"
    of 18. A frame that is uncertain on *both* counts is not published even
    for review — a confirmed symbol or a readable tolerance earns the box.
    """
    symbol_sure = frame.symbol is not None and frame.symbol.status == STATUS_CONFIRMED
    tolerance = frame.tolerance
    tolerance_read = (
        tolerance.state == fcf_parse.PRESENT
        and tolerance.value is not None
        and 0 < float(tolerance.value) <= MAX_PLAUSIBLE_TOLERANCE
    )
    if tolerance.state == fcf_parse.PRESENT and tolerance.value is not None:
        if float(tolerance.value) > MAX_PLAUSIBLE_TOLERANCE:
            return False
    return symbol_sure or tolerance_read


def scan_regions(
    image: Image.Image,
    *,
    ocr_fn: OcrFn | None,
    page: int = 1,
    config: GdtConfig | None = None,
    deskew: bool = True,
) -> list[dict[str, Any]]:
    """Every feature control frame on a page, as whole-page scan regions.

    This is what puts a frame into the ballooning flow. The text detector the
    scan is built on finds *text*: handed ``⌖ | Ø0.3`` it returns a box around
    ``0.3`` and publishes a linear dimension, and the frame itself — the
    symbol, the diametral zone, the datums — never reaches the sheet. Frames
    are found here from their rules, read structurally, and handed back in the
    shape ``segment_page`` publishes, so a frame arrives as one balloon that
    carries the whole callout.

    The segments of a composite frame are one region: they are one
    requirement, and an inspector balloons them once.

    ``deskew=False`` is used for vector-PDF renders, whose geometry is already
    square to the page. It avoids an expensive skew estimate without changing
    the normal scanned-image path.
    """
    result = analyze_page(
        image,
        page=page,
        ocr_fn=ocr_fn,
        source="page_scan",
        config=config,
        detect_loose_symbols=False,
        deskew=deskew,
    )
    groups: dict[str, list[FeatureControlFrame]] = {}
    order: list[str] = []
    for frame in result.frames:
        if not _plausible(frame):
            continue
        key = frame.composite_id or frame.id
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(frame)

    regions: list[dict[str, Any]] = []
    for key in order:
        frames = sorted(groups[key], key=lambda f: f.segment_index)
        x0 = min(f.bbox[0] for f in frames)
        y0 = min(f.bbox[1] for f in frames)
        x1 = max(f.bbox[0] + f.bbox[2] for f in frames)
        y1 = max(f.bbox[1] + f.bbox[3] for f in frames)
        lead = frames[0]
        name = (
            lead.symbol.name
            if lead.symbol is not None and lead.symbol.symbol_id
            else "Feature Control Frame"
        )
        reasons = [r for r in (_frame_needs_review(f) for f in frames) if r]
        regions.append(
            {
                "candidate_id": f"fcf-{key}",
                "bbox": {
                    "x": round(x0, 2),
                    "y": round(y0, 2),
                    "width": round(x1 - x0, 2),
                    "height": round(y1 - y0, 2),
                },
                "text": " / ".join(frame_text(f) for f in frames),
                "confidence": round(min(f.confidence for f in frames), 4),
                "type": SCAN_REGION_TYPE,
                "category": SCAN_REGION_TYPE,
                "subtype": name,
                "label": f"GD&T {name}" if len(frames) == 1 else "GD&T Composite",
                "orientation": "horizontal",
                "rotation": 0,
                "needs_review": bool(reasons),
                "recognized": True,
                "boundary_review": False,
                "agreement": 1.0,
                "engine": "gdt",
                "symbols_detected": None,
                "ocr_profile": "feature_control_frame",
                "page_filter_rule": "feature_control_frame",
                "page_filter_reason": (
                    "ruled feature control frame opening with a geometric "
                    "characteristic symbol"
                ),
                "review_reason": "; ".join(dict.fromkeys(reasons)),
                "recovery_attempted": False,
                "authoritative_reread": False,
                "authoritative_target_owned": False,
                "numeric_conflict": False,
                "gdt": {"frames": [f.to_dict() for f in frames]},
            }
        )
    return regions


def _centre_inside(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    cx = float(inner["x"]) + float(inner["width"]) / 2
    cy = float(inner["y"]) + float(inner["height"]) / 2
    return (
        float(outer["x"]) <= cx <= float(outer["x"]) + float(outer["width"])
        and float(outer["y"]) <= cy <= float(outer["y"]) + float(outer["height"])
    )


def absorb_frames(
    regions: list[dict[str, Any]],
    review_candidates: list[dict[str, Any]],
    frames: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Replace whatever the scan read inside a frame with the frame.

    Returns the new regions, the new review candidates, and the net change in
    the number of detected objects, so the caller's accounting still balances.
    A frame's tolerance compartment is a perfectly good value on its own —
    ``0.3`` — and the detector finds it every time; left in place it would be
    ballooned beside the frame as a linear dimension with no tolerance. The
    rest of the frame comes back as a review box with no readable text.
    Nothing inside a frame is a dimension, so all of it gives way.
    """
    delta = 0
    for frame in frames:
        box = frame["bbox"]
        before = len(regions)
        regions = [r for r in regions if not _centre_inside(r["bbox"], box)]
        delta -= before - len(regions)
        before = len(review_candidates)
        review_candidates = [
            c for c in review_candidates if not _centre_inside(c["bbox"], box)
        ]
        delta -= before - len(review_candidates)
        regions.append(dict(frame))
        delta += 1
    return regions, review_candidates, delta


def analyze_document(
    pages: Iterable[Image.Image],
    *,
    ocr_fn: OcrFn | None = None,
    texts_by_page: dict[int, Sequence[ctx.TextBox]] | None = None,
    declared_standard: str | None = None,
    source: str | None = None,
    config: GdtConfig | None = None,
) -> list[PageResult]:
    """Run the pass over every page, isolating per-page failures."""
    cfg = config or get_config()
    out: list[PageResult] = []
    for index, image in enumerate(pages, start=1):
        try:
            out.append(
                analyze_page(
                    image,
                    page=index,
                    ocr_fn=ocr_fn,
                    texts=(texts_by_page or {}).get(index, ()),
                    declared_standard=declared_standard,
                    source=source,
                    config=cfg,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a bad page must not lose the rest
            failed = PageResult(
                page=index, width=image.width if image else 0,
                height=image.height if image else 0,
            )
            failed.errors.append(
                {
                    "stage": "page",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            out.append(failed)
    return out


#: Prose this close to a frame (in frame heights, above or below) and sharing
#: its column is taken to annotate it: "OUTWARDS" under a profile frame.
ADJACENT_TEXT_GAP_FRAMES = 1.0


def attach_adjacent_text(
    frame_regions: list[dict[str, Any]],
    texts: Sequence[tuple[str, dict[str, Any]]],
) -> None:
    """
    Record the words printed against each frame, as ``associated_text``.

    The text is kept, not interpreted: whether "OUTWARDS" states the side of
    an unequally disposed zone or something else is for the reader. It is
    also carried into the frame's label, so the drawing view and every export
    show it beside the frame instead of losing it as an unattached label.
    ``texts`` are (text, bbox) in the frames' coordinates; only words count —
    a value beside a frame is its own callout.
    """
    for region in frame_regions:
        box = region["bbox"]
        fx0, fy0 = float(box["x"]), float(box["y"])
        fx1, fy1 = fx0 + float(box["width"]), fy0 + float(box["height"])
        reach = ADJACENT_TEXT_GAP_FRAMES * (fy1 - fy0)
        found: list[tuple[float, str]] = []
        for text, tb in texts:
            words = re.findall(r"[A-Za-z]{3,}", text or "")
            if not words or any(ch.isdigit() for ch in text):
                continue
            tx0, ty0 = float(tb["x"]), float(tb["y"])
            tx1, ty1 = tx0 + float(tb["width"]), ty0 + float(tb["height"])
            if _centre_inside(tb, box):
                continue
            gap = max(ty0 - fy1, fy0 - ty1)
            overlap = min(fx1, tx1) - max(fx0, tx0)
            if gap <= reach and overlap > 0.3 * min(tx1 - tx0, fx1 - fx0):
                found.append((ty0, " ".join(text.split())))
        if found:
            associated = " ".join(t for _, t in sorted(found))
            region["associated_text"] = associated
            region["label"] = f"{region.get('label') or 'GD&T'} · {associated}"
