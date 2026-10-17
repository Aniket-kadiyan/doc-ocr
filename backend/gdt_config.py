"""
Tunables for GD&T symbol recognition, read through the existing config system.

Everything here comes from the project's own ``.env.local`` / ``.env`` via
``config_env``, so the backend and the Next.js frontend keep drawing their
settings from one file. Names are prefixed ``GDT_`` so they cannot collide with
the ``OCR_*`` settings already in use.

Thresholds, not switches
------------------------
The two confidence bars are the important ones and they are deliberately
*two*, not one. ``accept`` is the bar above which a detection is published as a
named symbol. ``ambiguous`` is the floor below which nothing is reported at
all. Between them a detection is published with ``status="ambiguous"`` and its
alternatives attached, which is what the brief asks for: an uncertain glyph is
never silently promoted to a definitive engineering symbol, but neither is it
thrown away, because an operator reviewing the sheet wants to see that
*something* is there.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from config_env import get_bool, get_int, get_str


def _get_float(name: str, default: float) -> float:
    raw = get_str(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class GdtConfig:
    """Resolved GD&T recognition settings."""

    #: Run the GD&T pass at all.
    enabled: bool = True

    #: Publish a named symbol at or above this confidence.
    accept_confidence: float = 0.62
    #: Report nothing at all below this.
    ambiguous_confidence: float = 0.42
    #: A second candidate within this margin of the best makes the result
    #: ambiguous even when the best cleared ``accept_confidence`` — two
    #: near-equal explanations are not a confident answer.
    rival_margin: float = 0.06

    #: Normalised template-match score below which a template is not a match.
    template_min_score: float = 0.52
    #: Circularity floor for a contour to count as a ring.
    ring_min_circularity: float = 0.62

    #: Minimum glyph side in pixels. Below this a symbol carries too little
    #: detail to tell apart from its confusables, whatever the match score.
    min_glyph_px: int = 9
    #: Largest glyph side, as a fraction of the shorter page edge. A "symbol"
    #: bigger than this is drawing geometry.
    max_glyph_frac: float = 0.12

    #: Feature control frame geometry.
    frame_min_aspect: float = 1.6      # a frame is wider than it is tall
    frame_max_aspect: float = 40.0
    frame_min_height_px: int = 12
    frame_min_width_px: int = 24

    #: Page tiling for large drawings.
    tile_px: int = 1400
    tile_overlap_px: int = 180

    #: Two detections of the same symbol overlapping by more than this IoU are
    #: one detection seen twice.
    merge_iou: float = 0.55

    #: Upscale factor applied to a crop before template matching when the glyph
    #: is small. Small symbols are the whole problem on a scanned sheet.
    small_glyph_upscale: float = 3.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load() -> GdtConfig:
    """Build the config from the environment, falling back to the defaults."""
    defaults = GdtConfig()
    return GdtConfig(
        enabled=get_bool("GDT_ENABLED", defaults.enabled),
        accept_confidence=_get_float(
            "GDT_ACCEPT_CONFIDENCE", defaults.accept_confidence
        ),
        ambiguous_confidence=_get_float(
            "GDT_AMBIGUOUS_CONFIDENCE", defaults.ambiguous_confidence
        ),
        rival_margin=_get_float("GDT_RIVAL_MARGIN", defaults.rival_margin),
        template_min_score=_get_float(
            "GDT_TEMPLATE_MIN_SCORE", defaults.template_min_score
        ),
        ring_min_circularity=_get_float(
            "GDT_RING_MIN_CIRCULARITY", defaults.ring_min_circularity
        ),
        min_glyph_px=get_int("GDT_MIN_GLYPH_PX", defaults.min_glyph_px),
        max_glyph_frac=_get_float("GDT_MAX_GLYPH_FRAC", defaults.max_glyph_frac),
        frame_min_aspect=_get_float(
            "GDT_FRAME_MIN_ASPECT", defaults.frame_min_aspect
        ),
        frame_max_aspect=_get_float(
            "GDT_FRAME_MAX_ASPECT", defaults.frame_max_aspect
        ),
        frame_min_height_px=get_int(
            "GDT_FRAME_MIN_HEIGHT_PX", defaults.frame_min_height_px
        ),
        frame_min_width_px=get_int(
            "GDT_FRAME_MIN_WIDTH_PX", defaults.frame_min_width_px
        ),
        tile_px=get_int("GDT_TILE_PX", defaults.tile_px),
        tile_overlap_px=get_int("GDT_TILE_OVERLAP_PX", defaults.tile_overlap_px),
        merge_iou=_get_float("GDT_MERGE_IOU", defaults.merge_iou),
        small_glyph_upscale=_get_float(
            "GDT_SMALL_GLYPH_UPSCALE", defaults.small_glyph_upscale
        ),
    )


_config: GdtConfig | None = None


def get_config() -> GdtConfig:
    """The process-wide config, loaded once."""
    global _config
    if _config is None:
        _config = load()
    return _config


def reset_config() -> None:
    """Drop the cached config — used by tests that patch the environment."""
    global _config
    _config = None
