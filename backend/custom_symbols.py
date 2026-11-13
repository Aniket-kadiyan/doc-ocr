"""
User-defined symbols: a durable library and the detector that finds them.

A user uploads a picture of a symbol their drawings use (a company mark, a
GD&T glyph the built-in reader does not know, an inspection stamp) with a
name, a category and an optional balloon label. Every later scan looks for
each enabled symbol on the page, and every match becomes an accepted balloon
linked back to the symbol by id.

Storage
-------
One shared library on the backend: ``index.json`` plus one cleaned template
PNG per symbol, under ``backend/data/custom_symbols`` (or
``CUSTOM_SYMBOL_DATA_DIR``). Writes go through a temp file and ``os.replace``
so a crash never leaves a half-written index. A symbol stays until it is
deleted.

Detection
---------
Normalised template matching on ink masks, in two stages:

1. **Coarse**: for each candidate glyph size, page and template are shrunk
   together so the template is about ``COARSE_SIZE`` px. Peaks above a relaxed
   threshold are kept as candidates. This keeps a 4000 px page fast.
2. **Verify**: each candidate is re-matched at full resolution over nearby
   sizes, and must pass three independent checks: the correlation threshold,
   *recall* (the template's ink is present on the page) and *precision* (the
   page has no extra ink inside the symbol's box that the template lacks).
   The last one is what stops a plain circle matching inside a circle-with-a-
   cross.

Overlapping matches are reduced to the best one, across sizes, rotations and
symbols alike.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Sequence
from uuid import uuid4

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only on a cv2-less host
    cv2 = None  # type: ignore

import feature_dictionary as fd

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

#: Categories a symbol may be filed under: the rule engine's CAT_* values,
#: which the frontend's DIMENSION_TYPES mirrors.
ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {value for key, value in vars(fd).items() if key.startswith("CAT_") and isinstance(value, str)}
    | {"Note"}
)

DEFAULT_CATEGORY = fd.CAT_GDT
DEFAULT_THRESHOLD = 0.82
MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 0.99
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_NAME_LENGTH = 80
MAX_LABEL_LENGTH = 120
#: A cleaned template smaller than this has too little shape to match safely.
MIN_TEMPLATE_SIDE = 6


class CustomSymbolError(RuntimeError):
    """Base error translated to an HTTP response by the route layer."""


class CustomSymbolNotFoundError(CustomSymbolError):
    pass


class CustomSymbolConflictError(CustomSymbolError):
    pass


class CustomSymbolValidationError(CustomSymbolError):
    pass


@dataclass
class CustomSymbol:
    id: str
    name: str
    label: str
    category: str
    threshold: float
    match_rotations: bool
    enabled: bool
    width: int
    height: int
    created_at: str
    updated_at: str
    created_by: str | None = None
    #: Sides where the uploaded crop cut through ink, so the symbol may carry
    #: on past them on the drawing (a centre line, a leader).
    open_sides: list[str] = field(default_factory=list)
    #: Search every glyph size instead of sizes near the upload's. For clean,
    #: large pictures whose size has nothing to do with the drawing's.
    any_size: bool = False

    def to_public(self, image_png: bytes | None = None) -> dict[str, Any]:
        data = asdict(self)
        if image_png is not None:
            data["image_data_url"] = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        return data


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _configured_root() -> Path:
    configured = os.environ.get("CUSTOM_SYMBOL_DATA_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        return path.resolve()
    return (Path(__file__).resolve().parent / "data" / "custom_symbols").resolve()


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    if not cleaned:
        raise CustomSymbolValidationError("A symbol needs a name")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise CustomSymbolValidationError(f"The name is longer than {MAX_NAME_LENGTH} characters")
    return cleaned


def _clean_label(label: str | None) -> str:
    cleaned = re.sub(r"\s+", " ", str(label or "")).strip()
    if len(cleaned) > MAX_LABEL_LENGTH:
        raise CustomSymbolValidationError(f"The label is longer than {MAX_LABEL_LENGTH} characters")
    return cleaned


def _clean_category(category: str | None) -> str:
    cleaned = str(category or "").strip() or DEFAULT_CATEGORY
    if cleaned not in ALLOWED_CATEGORIES:
        raise CustomSymbolValidationError(f"Unknown category: {cleaned}")
    return cleaned


def _clean_threshold(threshold: float | None) -> float:
    value = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    if not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
        raise CustomSymbolValidationError(
            f"Match strictness must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}"
        )
    return round(value, 3)


class CustomSymbolStore:
    """The shared symbol library on disk."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or _configured_root()).resolve()
        self.images = self.root / "images"
        self.index_path = self.root / "index.json"
        self._lock = Lock()
        self._template_cache: dict[tuple[str, str], np.ndarray] = {}

    # -- index -------------------------------------------------------------

    def _read(self) -> list[CustomSymbol]:
        if not self.index_path.exists():
            return []
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        return [CustomSymbol(**item) for item in raw.get("symbols", [])]

    def _write(self, symbols: Sequence[CustomSymbol]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"version": 1, "symbols": [asdict(s) for s in symbols]}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.index_path)

    def _image_path(self, symbol_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", symbol_id):
            raise CustomSymbolNotFoundError("Custom symbol not found")
        return self.images / f"{symbol_id}.png"

    # -- queries -----------------------------------------------------------

    def list(self) -> list[CustomSymbol]:
        with self._lock:
            return sorted(self._read(), key=lambda s: s.name.casefold())

    def get(self, symbol_id: str) -> CustomSymbol:
        for symbol in self.list():
            if symbol.id == symbol_id:
                return symbol
        raise CustomSymbolNotFoundError("Custom symbol not found")

    def image_png(self, symbol_id: str) -> bytes:
        path = self._image_path(symbol_id)
        if not path.exists():
            raise CustomSymbolNotFoundError("Custom symbol image not found")
        return path.read_bytes()

    def template(self, symbol: CustomSymbol) -> np.ndarray:
        """The symbol's ink mask (uint8, ink 255), cached per revision."""
        key = (symbol.id, symbol.updated_at)
        cached = self._template_cache.get(key)
        if cached is None:
            png = Image.open(io.BytesIO(self.image_png(symbol.id))).convert("L")
            cached = np.where(np.asarray(png) < 128, 255, 0).astype(np.uint8)
            self._template_cache[key] = cached
        return cached

    # -- mutations ---------------------------------------------------------

    def create(
        self,
        *,
        name: str,
        image_bytes: bytes,
        label: str | None = None,
        category: str | None = None,
        threshold: float | None = None,
        match_rotations: bool = False,
        any_size: bool = False,
        created_by: str | None = None,
    ) -> CustomSymbol:
        clean_name = _clean_name(name)
        clean_label = _clean_label(label)
        clean_category = _clean_category(category)
        clean_threshold = _clean_threshold(threshold)
        if not image_bytes:
            raise CustomSymbolValidationError("The symbol image is empty")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise CustomSymbolValidationError("The symbol image is larger than 5 MB")
        mask, open_sides = prepare_template_with_sides(decode_image(image_bytes))

        now = _utc_now()
        symbol = CustomSymbol(
            id=uuid4().hex,
            name=clean_name,
            label=clean_label,
            category=clean_category,
            threshold=clean_threshold,
            match_rotations=bool(match_rotations),
            enabled=True,
            width=int(mask.shape[1]),
            height=int(mask.shape[0]),
            created_at=now,
            updated_at=now,
            created_by=created_by,
            open_sides=open_sides,
            any_size=bool(any_size),
        )
        with self._lock:
            symbols = self._read()
            if any(s.name.casefold() == clean_name.casefold() for s in symbols):
                raise CustomSymbolConflictError(f"A symbol named “{clean_name}” already exists")
            self.images.mkdir(parents=True, exist_ok=True)
            # Stored dark-on-white so the library page shows exactly what is matched.
            Image.fromarray(255 - mask).save(self._image_path(symbol.id), format="PNG")
            self._write([*symbols, symbol])
        return symbol

    def update(self, symbol_id: str, changes: dict[str, Any]) -> CustomSymbol:
        with self._lock:
            symbols = self._read()
            for index, symbol in enumerate(symbols):
                if symbol.id != symbol_id:
                    continue
                if "name" in changes and changes["name"] is not None:
                    new_name = _clean_name(changes["name"])
                    if any(
                        s.id != symbol_id and s.name.casefold() == new_name.casefold()
                        for s in symbols
                    ):
                        raise CustomSymbolConflictError(f"A symbol named “{new_name}” already exists")
                    symbol.name = new_name
                if "label" in changes and changes["label"] is not None:
                    symbol.label = _clean_label(changes["label"])
                if "category" in changes and changes["category"] is not None:
                    symbol.category = _clean_category(changes["category"])
                if "threshold" in changes and changes["threshold"] is not None:
                    symbol.threshold = _clean_threshold(changes["threshold"])
                if "match_rotations" in changes and changes["match_rotations"] is not None:
                    symbol.match_rotations = bool(changes["match_rotations"])
                if "enabled" in changes and changes["enabled"] is not None:
                    symbol.enabled = bool(changes["enabled"])
                if "any_size" in changes and changes["any_size"] is not None:
                    symbol.any_size = bool(changes["any_size"])
                symbol.updated_at = _utc_now()
                symbols[index] = symbol
                self._write(symbols)
                return symbol
        raise CustomSymbolNotFoundError("Custom symbol not found")

    def delete(self, symbol_id: str) -> None:
        with self._lock:
            symbols = self._read()
            remaining = [s for s in symbols if s.id != symbol_id]
            if len(remaining) == len(symbols):
                raise CustomSymbolNotFoundError("Custom symbol not found")
            self._write(remaining)
            self._image_path(symbol_id).unlink(missing_ok=True)
            self._template_cache = {
                key: value for key, value in self._template_cache.items() if key[0] != symbol_id
            }

    def enabled_templates(self) -> list[tuple[CustomSymbol, np.ndarray]]:
        out: list[tuple[CustomSymbol, np.ndarray]] = []
        for symbol in self.list():
            if not symbol.enabled:
                continue
            try:
                out.append((symbol, self.template(symbol)))
            except CustomSymbolError:
                continue
        return out


_store: CustomSymbolStore | None = None
_store_lock = Lock()


def get_custom_symbol_store() -> CustomSymbolStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = CustomSymbolStore()
        return _store


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as error:
        raise CustomSymbolValidationError("The file is not a readable image") from error
    if image.mode in {"RGBA", "LA", "P"}:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def ink_mask(image: Image.Image | np.ndarray) -> np.ndarray:
    """Binary ink mask (uint8, ink 255) of a drawing or symbol picture.

    Darkness is taken from the brightest channel, the same way
    ``image_preprocess.cad_ink_to_gray`` treats CAD line colours, then split
    with Otsu.
    """
    if cv2 is None:  # pragma: no cover
        raise CustomSymbolError("OpenCV is required for custom symbol matching")
    arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if arr.ndim == 3:
        darkness = 255 - arr.max(axis=2)
    else:
        darkness = 255 - arr
    darkness = darkness.astype(np.uint8)
    if int(darkness.max()) - int(darkness.min()) < 40:
        return np.zeros(darkness.shape, dtype=np.uint8)
    _, mask = cv2.threshold(darkness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def prepare_template(image: Image.Image) -> np.ndarray:
    """Clean an uploaded picture into a tight ink mask."""
    return prepare_template_with_sides(image)[0]


def prepare_template_with_sides(image: Image.Image) -> tuple[np.ndarray, list[str]]:
    """The tight ink mask, and the sides of the picture its ink touches."""
    mask = ink_mask(image)
    # A light-on-dark picture: the "ink" Otsu found is the background.
    if mask.mean() > 127:
        mask = 255 - mask
    if cv2 is not None:
        # Drop isolated specks (JPEG noise, scanner dust) that are not the symbol.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            keep = np.zeros(count, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(3, largest * 0.02)
            mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise CustomSymbolValidationError("No symbol could be found in the image")
    height, width = mask.shape
    open_sides = [
        side
        for side, touches in (
            ("left", xs.min() <= 1),
            ("right", xs.max() >= width - 2),
            ("top", ys.min() <= 1),
            ("bottom", ys.max() >= height - 2),
        )
        if touches
    ]
    mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if min(mask.shape) < MIN_TEMPLATE_SIDE and max(mask.shape) < MIN_TEMPLATE_SIDE * 3:
        raise CustomSymbolValidationError("The symbol in the image is too small; upload a larger crop")
    fill = float((mask > 0).mean())
    if fill > 0.85:
        raise CustomSymbolValidationError(
            "The image is almost solid ink; upload a tight crop of the symbol on a light background"
        )
    return mask, open_sides


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

#: Longest side, in page pixels, of the smallest and largest glyph searched.
MIN_GLYPH_PX = 14
MAX_GLYPH_PX = 260
#: Sizes searched relative to the uploaded crop. A crop taken from a drawing
#: is within this of its size on the scanned page (screenshot zoom vs scan
#: resolution); searching every size instead lets a thin symbol shrink until it
#: matches a digit.
MIN_RELATIVE_SCALE = 0.4
MAX_RELATIVE_SCALE = 2.5
SIZE_STEP = 1.12
#: Template size at the coarse stage.
COARSE_SIZE = 18
COARSE_SLACK = 0.14
MAX_COARSE_PEAKS = 400
#: Recall/precision floors at verification, measured with a 1 px tolerance
#: per ~25 px of glyph so anti-aliasing and pen-width differences do not count.
MIN_RECALL = 0.8
MIN_PRECISION = 0.78
NMS_IOU = 0.3


@dataclass
class SymbolMatch:
    symbol: CustomSymbol
    x: int
    y: int
    width: int
    height: int
    score: float
    recall: float
    precision: float
    rotation: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


def _pad(mask: np.ndarray) -> np.ndarray:
    """Template with a small empty border so surrounding whitespace counts."""
    margin = max(2, int(round(max(mask.shape) * 0.08)))
    return cv2.copyMakeBorder(mask, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)


def _soft(mask: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    width, height = max(1, width), max(1, height)
    interp = cv2.INTER_AREA if width < mask.shape[1] else cv2.INTER_LINEAR
    resized = cv2.resize(mask, (width, height), interpolation=interp)
    return np.where(resized >= 96, 255, 0).astype(np.uint8)


def _topology(mask: np.ndarray) -> tuple[int, int]:
    """(ink pieces, holes) of a mask, ignoring specks."""
    min_area = max(2, int(mask.size * 0.002))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0
    pieces = holes = 0
    for contour, link in zip(contours, hierarchy[0]):
        if cv2.contourArea(contour) < min_area:
            continue
        if link[3] < 0:
            pieces += 1
        else:
            holes += 1
    return pieces, holes


def _smallest_faithful_size(template: np.ndarray) -> float:
    """Smallest glyph size (longest side, px) at which the template still has
    its own shape: the same pieces and holes. Below that, ◎'s two rings run
    into one and it is just a 0.

    A picture whose fine detail never survives shrinking (rings a hairline
    apart) falls back to the size where the shrunk glyph still covers the
    original closely.
    """
    target = _topology(template)
    longest = max(template.shape)
    fallback: float | None = None
    size = float(MIN_GLYPH_PX)
    while size < longest:
        scale = size / longest
        small = _resize_mask(
            template,
            int(round(template.shape[1] * scale)),
            int(round(template.shape[0] * scale)),
        )
        if _topology(small) == target:
            return size
        if fallback is None:
            back = _resize_mask(small, template.shape[1], template.shape[0]) > 0
            ink = template > 0
            if (back & ink).sum() / max(1, (back | ink).sum()) >= 0.85:
                fallback = size
        size *= SIZE_STEP
    return fallback if fallback is not None else float(MIN_GLYPH_PX)


def _sizes(page_shape: tuple[int, int], template: np.ndarray, any_size: bool = False) -> list[float]:
    """Glyph scales (template-px → page-px) to search, smallest first."""
    longest = max(template.shape)
    limit = min(MAX_GLYPH_PX, max(page_shape))
    size = _smallest_faithful_size(template)
    if not any_size:
        limit = min(limit, longest * MAX_RELATIVE_SCALE)
        size = max(size, longest * MIN_RELATIVE_SCALE)
    out: list[float] = []
    while size <= limit:
        out.append(size / longest)
        size *= SIZE_STEP
    if not out and size <= max(page_shape):
        out.append(size / longest)  # a crop already near the size limits
    return out


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _contained(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix = max(0.0, min(inner[0] + inner[2], outer[0] + outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[1] + inner[3], outer[1] + outer[3]) - max(inner[1], outer[1]))
    area = inner[2] * inner[3]
    return (ix * iy) / area if area > 0 else 0.0


def _shape_agreement(window: np.ndarray, template: np.ndarray) -> tuple[float, float]:
    """(recall, precision) of page ink vs template ink, with a small tolerance."""
    tol = max(1, int(round(max(template.shape) / 25)))
    kernel = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    t_ink = template > 0
    w_ink = window > 0
    if not t_ink.any() or not w_ink.any():
        return 0.0, 0.0
    w_grown = cv2.dilate(window, kernel) > 0
    t_grown = cv2.dilate(template, kernel) > 0
    precision = float((w_ink & t_grown).sum() / w_ink.sum())
    # Recall is the weakest piece's: an "A" above an arrow must be there too,
    # not outvoted by the arrowhead landing on a decimal point.
    count, labels = cv2.connectedComponents(template, connectivity=8)
    recall = 1.0
    for label in range(1, count):
        piece = labels == label
        recall = min(recall, float((piece & w_grown).sum() / piece.sum()))
    return recall, precision


def _leaks(page: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Whether the matched ink is part of a bigger shape that continues outside.

    A plain circle "matches" the top of every 0 and O; what gives those away is
    that their stroke carries on past the box. Ink connected to the match that
    runs outside it is allowed only when it is line-like: a leader or a frame
    edge running into the symbol.
    """
    size = max(w, h)
    grow = max(2, int(round(size * 0.6)))
    # Frame edges and extension lines are long and axis-aligned; take them out
    # first so a leader that runs into a frame is not judged as one bent shape.
    # "Long" is well past any character, or the straight sides of a 0 would go
    # too; the lines are found in a wider view than the one judged.
    long_px = max(12, int(round(size * 2.2)))
    reach = long_px
    vx0, vy0 = max(0, x - reach), max(0, y - reach)
    vx1, vy1 = min(page.shape[1], x + w + reach), min(page.shape[0], y + h + reach)
    view = page[vy0:vy1, vx0:vx1].copy()
    for kernel in (np.ones((1, long_px), np.uint8), np.ones((long_px, 1), np.uint8)):
        view[cv2.morphologyEx(view, cv2.MORPH_OPEN, kernel) > 0] = 0
    x0, y0 = max(0, x - grow), max(0, y - grow)
    x1, y1 = min(page.shape[1], x + w + grow), min(page.shape[0], y + h + grow)
    window = view[y0 - vy0 : y1 - vy0, x0 - vx0 : x1 - vx0]
    count, labels = cv2.connectedComponents(window, connectivity=8)
    if count <= 1:
        return False
    tol = max(1, int(round(max(w, h) / 25)))
    inside = np.zeros(window.shape, dtype=bool)
    inside[
        max(0, y - y0 - tol) : y - y0 + h + tol,
        max(0, x - x0 - tol) : x - x0 + w + tol,
    ] = True
    stroke_limit = max(3.0, max(w, h) * 0.12)
    in_box_ink = int((window[inside] > 0).sum())
    leaked = 0
    for label in np.unique(labels[inside & (window > 0)]):
        if label == 0:
            continue
        outside = ((labels == label) & ~inside).astype(np.uint8)
        if not outside.any():
            continue
        # Judge each piece that runs out on its own, so two leaders crossing
        # the symbol are two straight pieces rather than one tangle.
        n_pieces, pieces = cv2.connectedComponents(outside, connectivity=8)
        for piece in range(1, n_pieces):
            pts = np.column_stack(np.nonzero(pieces == piece)).astype(np.float32)
            if len(pts) <= stroke_limit * stroke_limit:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            if min(rw, rh) <= stroke_limit:
                continue  # a straight line running in: leader, frame edge
            leaked += len(pts)
    return leaked > 0.15 * max(in_box_ink, 1)


SIDES = ("left", "right", "top", "bottom")
#: Where each side of the upright template ends up after a clockwise turn.
_TURN_CW = {"left": "top", "top": "right", "right": "bottom", "bottom": "left"}


def _turned_sides(sides: Iterable[str], degrees: int) -> list[str]:
    out = list(sides)
    for _ in range((degrees // 90) % 4):
        out = [_TURN_CW[side] for side in out]
    return out


def _continues(
    page: np.ndarray,
    glyph: np.ndarray,
    x: int,
    y: int,
    open_sides: Iterable[str] = (),
) -> bool:
    """Whether the page carries one of the symbol's stroke ends on past the box.

    A ⊥ drawn alone ends where its strokes end; the same two strokes inside a
    T-junction of construction lines keep going. A stroke "ends" at a side of
    the box when the template ink near that side is a narrow band running
    towards it. A circle touching its box is tangent there and spreads along
    the side instead, so a leader meeting a circle is not counted.
    """
    h, w = glyph.shape
    depth = max(2, int(round(min(w, h) * 0.15)))
    gap = max(1, int(round(max(w, h) / 25)))
    reach = max(3, int(round(max(w, h) * 0.25)))
    band_limit = max(3, int(round(max(w, h) * 0.22)))
    ink = glyph > 0

    def stroke_end(edge: np.ndarray) -> np.ndarray | None:
        """Indices across the side where a stroke runs into it, if it is narrow."""
        across = np.nonzero(edge.any(axis=0))[0]
        if across.size == 0 or across[-1] - across[0] + 1 > band_limit:
            return None
        # A line end keeps its width; an apex (a datum triangle's tip, where
        # the leader legitimately attaches) widens away from the side.
        widths = [int(row.sum()) for row in edge if row.any()]
        if max(widths) - min(widths) > 2 * gap + 1:
            return None
        return across

    ph, pw = page.shape
    # (template edge strip as rows×across, page strip outside that side)
    sides = (
        (ink[:, :depth].T, lambda a: page[y + a[0] : y + a[-1] + 1, max(0, x - gap - reach) : max(0, x - gap)]),
        (ink[:, w - depth :].T, lambda a: page[y + a[0] : y + a[-1] + 1, min(pw, x + w + gap) : min(pw, x + w + gap + reach)]),
        (ink[:depth, :], lambda a: page[max(0, y - gap - reach) : max(0, y - gap), x + a[0] : x + a[-1] + 1]),
        (ink[h - depth :, :], lambda a: page[min(ph, y + h + gap) : min(ph, y + h + gap + reach), x + a[0] : x + a[-1] + 1]),
    )
    skip = set(open_sides)
    for index, (edge, outside) in enumerate(sides):
        if SIDES[index] in skip:
            continue
        across = stroke_end(edge)
        if across is None:
            continue
        strip = outside(across) > 0
        if not strip.size:
            continue
        # How far ink keeps going away from the box: every step outward having
        # some is a line carrying on, however thin it is.
        steps = strip.any(axis=0) if index < 2 else strip.any(axis=1)
        if steps.mean() >= 0.8:
            return True
    return False


def _coarse_candidates(
    page: np.ndarray,
    template: np.ndarray,
    threshold: float,
    any_size: bool = False,
) -> list[tuple[float, int, int, float]]:
    """(scale, x, y, score) candidates in page pixels, top-left of the glyph."""
    found: list[tuple[float, int, int, float]] = []
    page_cache: dict[float, np.ndarray] = {}
    floor = max(0.3, threshold - COARSE_SLACK)
    for scale in _sizes(page.shape, template, any_size):
        glyph_w = template.shape[1] * scale
        glyph_h = template.shape[0] * scale
        shrink = min(1.0, COARSE_SIZE / max(glyph_w, glyph_h))
        shrink = round(shrink, 3)
        small_page = page_cache.get(shrink)
        if small_page is None:
            if shrink < 1.0:
                small_page = cv2.resize(page, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
            else:
                small_page = page
            small_page = _soft(small_page)
            page_cache[shrink] = small_page
        tw = int(round(glyph_w * shrink))
        th = int(round(glyph_h * shrink))
        if tw < 3 or th < 3:
            continue
        small_t = _pad(_resize_mask(template, tw, th))
        if small_t.shape[0] >= small_page.shape[0] or small_t.shape[1] >= small_page.shape[1]:
            continue
        result = cv2.matchTemplate(small_page, _soft(small_t), cv2.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        # Local maxima only, so a glyph contributes one candidate per size.
        dilated = cv2.dilate(result, np.ones((3, 3), np.uint8))
        peaks = np.argwhere((result >= floor) & (result >= dilated))
        margin = (small_t.shape[0] - th) / 2.0
        for py, px in peaks:
            x = int(round((px + margin) / shrink))
            y = int(round((py + margin) / shrink))
            found.append((scale, x, y, float(result[py, px])))
    found.sort(key=lambda item: -item[3])
    return found[:MAX_COARSE_PEAKS]


def _verify(
    page: np.ndarray,
    template: np.ndarray,
    scale: float,
    x: int,
    y: int,
) -> tuple[float, float, float, int, int, int, int] | None:
    """Best full-resolution (score, recall, precision, x, y, w, h) near a candidate."""
    best: tuple[float, float, float, int, int, int, int] | None = None
    page_h, page_w = page.shape
    for factor in (0.92, 0.96, 1.0, 1.04, 1.08):
        tw = int(round(template.shape[1] * scale * factor))
        th = int(round(template.shape[0] * scale * factor))
        if tw < 4 or th < 4:
            continue
        glyph = _resize_mask(template, tw, th)
        padded = _pad(glyph)
        margin = (padded.shape[0] - th) // 2
        reach = max(3, int(round(max(tw, th) * 0.2)))
        x0 = max(0, x - margin - reach)
        y0 = max(0, y - margin - reach)
        x1 = min(page_w, x + tw + margin + reach)
        y1 = min(page_h, y + th + margin + reach)
        window = page[y0:y1, x0:x1]
        if window.shape[0] < padded.shape[0] or window.shape[1] < padded.shape[1]:
            continue
        result = cv2.matchTemplate(_soft(window), _soft(padded), cv2.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        _, score, _, loc = cv2.minMaxLoc(result)
        gx = x0 + loc[0] + margin
        gy = y0 + loc[1] + margin
        if best is None or score > best[0]:
            recall, precision = _shape_agreement(page[gy : gy + th, gx : gx + tw], glyph)
            best = (float(score), recall, precision, gx, gy, tw, th)
    return best


def _rotations(template: np.ndarray, allow: bool) -> list[tuple[int, np.ndarray]]:
    if not allow:
        return [(0, template)]
    out = [(0, template)]
    seen = [template]
    for k, degrees in ((1, 90), (2, 180), (3, 270)):
        rotated = np.ascontiguousarray(np.rot90(template, -k))
        # A symmetric glyph rotated onto itself adds work and nothing else.
        if any(r.shape == rotated.shape and np.array_equal(r, rotated) for r in seen):
            continue
        seen.append(rotated)
        out.append((degrees, rotated))
    return out


def detect_custom_symbols(
    image: Image.Image,
    symbols: Iterable[tuple[CustomSymbol, np.ndarray]],
    *,
    scope_bbox: dict[str, float] | None = None,
    skip_boxes: Sequence[dict[str, float]] = (),
) -> list[SymbolMatch]:
    """Every confident match of every given symbol on ``image``.

    ``scope_bbox`` limits the search (page pixels); ``skip_boxes`` are existing
    balloons a match must not duplicate. Results are in page pixels.
    """
    symbols = list(symbols)
    if not symbols or cv2 is None:
        return []
    full = ink_mask(image)
    ox = oy = 0
    if scope_bbox:
        ox = max(0, int(scope_bbox.get("x", 0)))
        oy = max(0, int(scope_bbox.get("y", 0)))
        x1 = min(full.shape[1], int(scope_bbox.get("x", 0) + scope_bbox.get("width", full.shape[1])))
        y1 = min(full.shape[0], int(scope_bbox.get("y", 0) + scope_bbox.get("height", full.shape[0])))
        full = full[oy:y1, ox:x1]
    if full.size == 0 or not full.any():
        return []

    matches: list[SymbolMatch] = []
    for symbol, template in symbols:
        for degrees, oriented in _rotations(template, symbol.match_rotations):
            accepted: list[SymbolMatch] = []
            for scale, cx, cy, _ in _coarse_candidates(
                full, oriented, symbol.threshold, symbol.any_size
            ):
                # A candidate inside one already verified for this orientation
                # is the same glyph at a neighbouring size.
                if any(
                    m.x - 2 <= cx <= m.x + m.width and m.y - 2 <= cy <= m.y + m.height
                    for m in accepted
                ):
                    continue
                verified = _verify(full, oriented, scale, cx, cy)
                if verified is None:
                    continue
                score, recall, precision, gx, gy, w, h = verified
                if score < symbol.threshold or recall < MIN_RECALL or precision < MIN_PRECISION:
                    continue
                if _leaks(full, gx, gy, w, h) or _continues(
                    full,
                    _resize_mask(oriented, w, h),
                    gx,
                    gy,
                    _turned_sides(symbol.open_sides, degrees),
                ):
                    continue
                accepted.append(
                    SymbolMatch(
                        symbol=symbol,
                        x=gx,
                        y=gy,
                        width=w,
                        height=h,
                        score=score,
                        recall=recall,
                        precision=precision,
                        rotation=degrees,
                    )
                )
            matches.extend(accepted)

    # One match per place: the strongest across sizes, rotations and symbols.
    ranked = sorted(matches, key=lambda m: -(m.score + m.precision * 0.5))
    kept: list[SymbolMatch] = []
    for match in ranked:
        if any(
            _iou(match.box, other.box) > NMS_IOU
            or _contained(match.box, other.box) > 0.7
            or _contained(other.box, match.box) > 0.7
            for other in kept
        ):
            continue
        kept.append(match)

    for match in kept:
        match.x += ox
        match.y += oy
    return [
        m
        for m in kept
        if not any(
            _contained(m.box, (b["x"], b["y"], b["width"], b["height"])) >= 0.5
            or _contained((b["x"], b["y"], b["width"], b["height"]), m.box) >= 0.5
            for b in skip_boxes
        )
    ]


# ---------------------------------------------------------------------------
# Scan result integration
# ---------------------------------------------------------------------------

#: An OCR region at least this much inside a symbol match is a reading of the
#: symbol itself (Paddle reading a glyph as "O" or "+"), not a separate value.
SWALLOW_FRACTION = 0.7


def match_region(match: SymbolMatch, sequence: int) -> dict[str, Any]:
    symbol = match.symbol
    reason = f"Matched custom symbol “{symbol.name}” (score {match.score:.2f})"
    return {
        "candidate_id": f"S{sequence:04d}",
        "bbox": {"x": match.x, "y": match.y, "width": match.width, "height": match.height},
        "text": symbol.name,
        "raw_text": symbol.name,
        "confidence": round(match.score, 4),
        "type": symbol.category,
        "category": symbol.category,
        "subtype": None,
        "label": symbol.label or symbol.name,
        "orientation": "horizontal",
        "rotation": 0,
        "needs_review": False,
        "recognized": True,
        "page_filter_rule": "custom_symbol",
        "page_filter_reason": reason,
        "custom_symbol_id": symbol.id,
        "custom_symbol_rotation": match.rotation,
    }


def merge_custom_symbol_matches(seg: dict[str, Any], matches: Sequence[SymbolMatch]) -> int:
    """Add symbol matches to a scan result as accepted regions, in place.

    OCR regions that are readings of a matched glyph are dropped and their
    audit outcome marked excluded, so the symbol gets one balloon, not two.
    Returns the number of symbol regions added.
    """
    if not matches:
        return 0
    boxes = [(m.x, m.y, m.width, m.height) for m in matches]

    def swallowed(item: dict[str, Any]) -> bool:
        b = item.get("bbox") or {}
        try:
            box = (float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"]))
        except (KeyError, TypeError, ValueError):
            return False
        return any(_contained(box, m) >= SWALLOW_FRACTION for m in boxes)

    dropped_ids: set[str] = set()
    kept_regions: list[dict[str, Any]] = []
    dropped_regions = 0
    for region in seg.get("regions", []):
        if swallowed(region):
            dropped_regions += 1
            if region.get("candidate_id"):
                dropped_ids.add(str(region["candidate_id"]))
        else:
            kept_regions.append(region)
    kept_reviews: list[dict[str, Any]] = []
    dropped_reviews = 0
    for review in seg.get("review_candidates", []):
        if swallowed(review):
            dropped_reviews += 1
            if review.get("candidate_id"):
                dropped_ids.add(str(review["candidate_id"]))
        else:
            kept_reviews.append(review)

    outcomes = [dict(o) for o in seg.get("candidate_outcomes", [])]
    for outcome in outcomes:
        if str(outcome.get("candidate_id") or "") in dropped_ids:
            outcome["state"] = "excluded"
            outcome["rule"] = "custom_symbol_overlap"
            outcome["reason"] = "Part of a matched custom symbol"

    added = [match_region(m, i + 1) for i, m in enumerate(matches)]
    for region in added:
        outcomes.append(
            {
                "candidate_id": region["candidate_id"],
                "bbox": dict(region["bbox"]),
                "state": "eligible",
                "text": region["text"],
                "raw_text": region["raw_text"],
                "confidence": region["confidence"],
                "recognized": True,
                "reason": region["page_filter_reason"],
                "rule": "custom_symbol",
                "type": region["type"],
                "category": region["category"],
                "subtype": None,
                "label": region["label"],
                "orientation": "horizontal",
                "rotation": 0,
            }
        )

    regions_before = len(seg.get("regions", []))
    reviews_before = len(seg.get("review_candidates", []))
    seg["regions"] = [*kept_regions, *added]
    seg["review_candidates"] = kept_reviews
    seg["candidate_outcomes"] = outcomes
    n = len(added)
    dropped = dropped_regions + dropped_reviews
    seg["detected_count"] = int(seg.get("detected_count", regions_before)) + n
    seg["recognized_count"] = int(seg.get("recognized_count", regions_before)) + n
    seg["eligible_count"] = int(seg.get("eligible_count", regions_before)) + n - dropped_regions
    seg["review_count"] = max(0, int(seg.get("review_count", reviews_before)) - dropped_reviews)
    seg["excluded_count"] = int(seg.get("excluded_count", 0)) + dropped
    seg["count"] = len(seg["regions"])
    rules = dict(seg.get("filter_rule_counts") or {})
    rules["custom_symbol"] = rules.get("custom_symbol", 0) + n
    if dropped:
        rules["custom_symbol_overlap"] = rules.get("custom_symbol_overlap", 0) + dropped
    seg["filter_rule_counts"] = rules
    return n
