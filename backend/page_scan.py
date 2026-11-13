"""Bounded whole-page detector planning and strict candidate deduplication.

Whole-page auto-ballooning has a dedicated, detector-only route.  Page layout
analysis now supplies four to eight adaptive overlapping panels; each panel is
inspected at 0 and 90 degrees before candidates are deduplicated globally.

This module owns only page orchestration geometry.  It does not perform model
inference, recognition, or value filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from math import inf
from typing import Mapping, Sequence

from PIL import Image

from detection_passes import DetectionTile


PAGE_SCAN_ROTATIONS_CW = (0, 90)
PAGE_SCAN_MAX_TILES = 8
# Two detector calls per panel at the render's own resolution, and two more
# over an enlarged panel to recover the text too small for the first pair.
PAGE_SCAN_MAX_DETECTOR_CALLS = 32
# A gap object has to stand clear of everything the primary passes found. The
# threshold is deliberately low: an enlarged panel resolves a callout into its
# separate glyphs, and this route does no grouping, so a box that merely
# touches a value already read is a fragment of it and not a new object.
GAP_CANDIDATE_CLEARANCE = 0.2
# A gap object also has to look like the thing the primary passes miss, which
# is a small compact callout. Both bounds are in the pixels of the 250dpi
# render this route scans, where a CAD character stands about 30 to 40 high.
#
# The shape bound is what the sheet margin runs into: a drawing's boilerplate
# is set in a vertical strip, and the enlarged pass offered nine of those as
# values -- 39x622, 38x566, 35x537 -- none of which a primary pass could have
# missed. The size bound is for the ink that is not writing at all: an
# arrowhead or a centre mark at 11x7 reads as a confident "1".
GAP_CANDIDATE_MAX_ASPECT = 6.0
GAP_CANDIDATE_MIN_SHORT_EDGE = 12.0
GAP_CANDIDATE_MIN_LONG_EDGE = 20.0
# Preserve approximately the text scale that worked in selection scans without
# restoring the selection cascade. The shared detector still caps at 2400px.
PAGE_SCAN_DETECTOR_MIN_LONG_EDGE = 2000
PAGE_SCAN_RECOGNITION_BATCH_SIZE = 16
EXISTING_VALUE_OVERLAP_THRESHOLD = 0.55
# Diagonal neighbourhoods the slanted-callout pass may work through on a whole
# sheet, and how tightly it groups them. A selection holds its angled callouts
# in one or two places; a sheet spreads them over every view, at a different
# angle in each, and each neighbourhood is levelled by one angle only. See the
# call site for what the selection-sized defaults do to a full sheet.
PAGE_SCAN_MAX_SLANT_NEIGHBOURHOODS = 24
# Measured on 129E01-13300-ID, whose middle view packs four angles together.
# The merge margin is what decides how much lands in one neighbourhood, and a
# neighbourhood is levelled by one angle only, so it is the whole story:
#   1.00 (the selection default): 8 groups, one of them 21% of the sheet,
#         and "3.8±0.3", "(28°)", "17" and "R2" all inside it, none read.
#   0.35: 16 groups, "R4.5" and "R25" come back, the other three still fuse
#         into one read, "3.8±0.37(28°)R2".
#   0.15: 23 groups, "(28°)" separates, "17" still fuses onto "3.8±0.3".
#   0.08: 23 groups, and "3.8±0.3" and "(28°)" both read on their own.
# Context is added back AFTER grouping, which is what keeps a value with its
# deviation stack at a margin this tight.
PAGE_SCAN_SLANT_MERGE_MARGIN = 0.08
PAGE_SCAN_SLANT_CONTEXT_MARGIN = 0.5
# Large sheets contain several unrelated slant directions. Levelling the
# entire image by one angle adds multiple full-page OCR calls but finds no
# extra values; the neighbourhood pass below remains enabled.
PAGE_SCAN_SLANT_WHOLE_IMAGE_MAX_EDGE = 3000
# A section scan's slanted, notes and GD&T passes read a copy of the page that
# is blank beyond the section and this margin. They used to read the whole
# page and drop what fell outside afterwards: on 113-M02-292A a 1.7 MP section
# spent 20.5 of its 23 minutes there and found nothing outside. The margin is
# context for a note, frame or leader that crosses the section's edge.
SECTION_CONTEXT_MARGIN_FRAC = 0.25
SECTION_CONTEXT_MARGIN_MIN_PX = 300


def section_context_box(
    scope: Mapping[str, float],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """The section plus its context margin, clamped to the page."""

    width, height = size
    margin = max(
        SECTION_CONTEXT_MARGIN_MIN_PX,
        SECTION_CONTEXT_MARGIN_FRAC * max(float(scope["width"]), float(scope["height"])),
    )
    return (
        max(0, int(float(scope["x"]) - margin)),
        max(0, int(float(scope["y"]) - margin)),
        min(width, int(float(scope["x"]) + float(scope["width"]) + margin + 0.5)),
        min(height, int(float(scope["y"]) + float(scope["height"]) + margin + 0.5)),
    )


def blank_outside_section(image: Image.Image, scope: Mapping[str, float]) -> Image.Image:
    """
    The page with everything beyond the section's context blanked.

    Same size, so every pass keeps working, and publishing, in page
    coordinates; only the text it has to read shrinks to the section.
    """

    box = section_context_box(scope, image.size)
    blank = Image.new(image.mode, image.size, "white")
    blank.paste(image.crop(box), box[:2])
    return blank


def bbox_overlap_fraction(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    """Intersection as a fraction of the smaller box."""

    x0 = max(float(left["x"]), float(right["x"]))
    y0 = max(float(left["y"]), float(right["y"]))
    x1 = min(
        float(left["x"]) + float(left["width"]),
        float(right["x"]) + float(right["width"]),
    )
    y1 = min(
        float(left["y"]) + float(left["height"]),
        float(right["y"]) + float(right["height"]),
    )
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = max(float(left["width"]) * float(left["height"]), 1.0)
    right_area = max(float(right["width"]) * float(right["height"]), 1.0)
    return intersection / min(left_area, right_area)


def overlaps_existing_value(
    bbox: Mapping[str, float],
    existing_value_boxes: Sequence[Mapping[str, float]],
    *,
    threshold: float = EXISTING_VALUE_OVERLAP_THRESHOLD,
) -> bool:
    """Return whether a detected object is already represented by a balloon."""

    return any(
        bbox_overlap_fraction(bbox, existing) >= threshold
        for existing in existing_value_boxes
    )


def bbox_intersects_scope(
    bbox: Mapping[str, float],
    scope: Mapping[str, float],
) -> bool:
    """Whether two page-coordinate rectangles have positive overlap."""

    return (
        float(bbox["x"]) < float(scope["x"]) + float(scope["width"])
        and float(scope["x"]) < float(bbox["x"]) + float(bbox["width"])
        and float(bbox["y"]) < float(scope["y"]) + float(scope["height"])
        and float(scope["y"]) < float(bbox["y"]) + float(bbox["height"])
    )


def bbox_center_in_scope(
    bbox: Mapping[str, float],
    scope: Mapping[str, float],
) -> bool:
    """Whether an object's centre belongs to the requested page scope."""

    centre_x = float(bbox["x"]) + float(bbox["width"]) / 2.0
    centre_y = float(bbox["y"]) + float(bbox["height"]) / 2.0
    return (
        float(scope["x"])
        <= centre_x
        <= float(scope["x"]) + float(scope["width"])
        and float(scope["y"])
        <= centre_y
        <= float(scope["y"]) + float(scope["height"])
    )


@dataclass(frozen=True)
class PageCandidate:
    """One detector-pass object restored to whole-page coordinates."""

    bbox: dict[str, float]
    tile_index: int
    tile: DetectionTile
    boundary_clearance: float
    boundary_review: bool
    detection_confidence: float = 0.0
    pass_index: int = 0
    rotation_cw: int = 0
    #: Every rotation whose detector pass produced this object. A box seen
    #: only by the quarter-turn pass is text written up the side of a view,
    #: and that is the one reliable signal for a one or two glyph callout:
    #: "10" turned on its side is no taller than it is wide, so its shape
    #: cannot say which way it reads and both readings are real numbers.
    rotations_cw: tuple[int, ...] = ()
    polygon: tuple[tuple[float, float], ...] = ()
    candidate_id: str = ""

    @property
    def quarter_turn_only(self) -> bool:
        """Whether no upright pass ever saw this object."""

        return bool(self.rotations_cw) and set(self.rotations_cw) == {90}


def _candidate_boundary_clearance(
    bbox: dict[str, float],
    tile: DetectionTile,
    *,
    page_size: tuple[int, int],
) -> float:
    """Distance to the nearest internal tile edge.

    Page exterior edges are ignored because no neighbouring tile can provide a
    more complete view there.
    """

    page_width, page_height = page_size
    local_x = float(bbox["x"])
    local_y = float(bbox["y"])
    width = float(bbox["width"])
    height = float(bbox["height"])
    clearances: list[float] = []

    if tile.x > 0:
        clearances.append(local_x)
    if tile.y > 0:
        clearances.append(local_y)
    if tile.x + tile.width < page_width:
        clearances.append(float(tile.width) - (local_x + width))
    if tile.y + tile.height < page_height:
        clearances.append(float(tile.height) - (local_y + height))

    return max(0.0, min(clearances)) if clearances else inf


def map_tile_candidate(
    bbox: dict[str, float],
    tile: DetectionTile,
    *,
    tile_index: int,
    page_size: tuple[int, int],
    detection_confidence: float = 0.0,
    pass_index: int = 0,
    rotation_cw: int = 0,
    polygon: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
) -> PageCandidate | None:
    """Clip and restore one tile-local object to whole-page coordinates."""

    page_width, page_height = page_size
    local_x = float(bbox["x"])
    local_y = float(bbox["y"])
    local_width = float(bbox["width"])
    local_height = float(bbox["height"])
    if local_width <= 0 or local_height <= 0:
        return None

    page_x0 = max(0.0, min(float(page_width), tile.x + local_x))
    page_y0 = max(0.0, min(float(page_height), tile.y + local_y))
    page_x1 = max(
        0.0,
        min(float(page_width), tile.x + local_x + local_width),
    )
    page_y1 = max(
        0.0,
        min(float(page_height), tile.y + local_y + local_height),
    )
    if page_x1 - page_x0 < 1.0 or page_y1 - page_y0 < 1.0:
        return None

    clearance = _candidate_boundary_clearance(
        bbox,
        tile,
        page_size=page_size,
    )
    boundary_margin = max(8.0, min(local_width, local_height) * 0.5)
    local_polygon = polygon or (
        (local_x, local_y),
        (local_x + local_width, local_y),
        (local_x + local_width, local_y + local_height),
        (local_x, local_y + local_height),
    )
    page_polygon: list[tuple[float, float]] = []
    for point in local_polygon:
        if len(point) < 2:
            continue
        point_x = max(
            0.0,
            min(float(page_width), tile.x + float(point[0])),
        )
        point_y = max(
            0.0,
            min(float(page_height), tile.y + float(point[1])),
        )
        page_polygon.append((round(point_x, 1), round(point_y, 1)))
    return PageCandidate(
        bbox={
            "x": round(page_x0, 1),
            "y": round(page_y0, 1),
            "width": round(page_x1 - page_x0, 1),
            "height": round(page_y1 - page_y0, 1),
        },
        tile_index=tile_index,
        tile=tile,
        boundary_clearance=clearance,
        boundary_review=clearance < boundary_margin,
        detection_confidence=float(detection_confidence),
        pass_index=pass_index,
        rotation_cw=rotation_cw,
        rotations_cw=(rotation_cw,),
        polygon=tuple(page_polygon),
    )


def _intersection_over_union(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    x0 = max(float(left["x"]), float(right["x"]))
    y0 = max(float(left["y"]), float(right["y"]))
    x1 = min(
        float(left["x"] + left["width"]),
        float(right["x"] + right["width"]),
    )
    y1 = min(
        float(left["y"] + left["height"]),
        float(right["y"] + right["height"]),
    )
    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    left_area = max(float(left["width"] * left["height"]), 1.0)
    right_area = max(float(right["width"] * right["height"]), 1.0)
    return intersection / max(left_area + right_area - intersection, 1.0)


def strict_same_object(
    left: dict[str, float],
    right: dict[str, float],
    *,
    min_size_ratio: float = 0.65,
) -> bool:
    """Match only similar-size boxes at the same position.

    Containment alone is intentionally insufficient: a large box containing
    several dimensions must never suppress its individual children.
    """

    left_width = max(float(left["width"]), 1.0)
    left_height = max(float(left["height"]), 1.0)
    right_width = max(float(right["width"]), 1.0)
    right_height = max(float(right["height"]), 1.0)
    width_ratio = min(left_width, right_width) / max(left_width, right_width)
    height_ratio = min(left_height, right_height) / max(left_height, right_height)
    if width_ratio < min_size_ratio or height_ratio < min_size_ratio:
        return False

    left_center = (
        float(left["x"]) + left_width / 2,
        float(left["y"]) + left_height / 2,
    )
    right_center = (
        float(right["x"]) + right_width / 2,
        float(right["y"]) + right_height / 2,
    )
    centre_is_close = (
        abs(left_center[0] - right_center[0])
        <= 0.35 * max(left_width, right_width)
        and abs(left_center[1] - right_center[1])
        <= 0.35 * max(left_height, right_height)
    )
    return (
        _intersection_over_union(left, right) >= 0.45
        or centre_is_close
    )


def _candidate_preference(candidate: PageCandidate) -> tuple[float, ...]:
    area = float(candidate.bbox["width"] * candidate.bbox["height"])
    return (
        0.0 if candidate.boundary_review else 1.0,
        candidate.boundary_clearance,
        candidate.detection_confidence,
        area,
        -float(candidate.tile_index),
    )


def deduplicate_page_candidates(
    candidates: list[PageCandidate],
) -> list[PageCandidate]:
    """Remove only repeated same-size views of the same atomic object."""

    preferred_first = sorted(
        candidates,
        key=_candidate_preference,
        reverse=True,
    )
    accepted: list[PageCandidate] = []
    for candidate in preferred_first:
        duplicate = next(
            (
                index
                for index, existing in enumerate(accepted)
                if strict_same_object(existing.bbox, candidate.bbox)
            ),
            None,
        )
        if duplicate is None:
            accepted.append(candidate)
            continue
        # The box is dropped, but which passes saw this object is evidence
        # about the object and not about the box that won, so it is carried
        # over. Without this every candidate looks like it was found by one
        # rotation and "only the quarter turn saw it" means nothing.
        kept = accepted[duplicate]
        merged = tuple(
            sorted(set(kept.rotations_cw) | set(candidate.rotations_cw))
        )
        if merged != kept.rotations_cw:
            accepted[duplicate] = replace(kept, rotations_cw=merged)

    return sorted(
        accepted,
        key=lambda candidate: (
            candidate.bbox["y"],
            candidate.bbox["x"],
            candidate.tile_index,
        ),
    )


def looks_like_a_small_callout(bbox: Mapping[str, float]) -> bool:
    """Whether a box has the shape and size of the text a gap pass is for."""

    width = float(bbox["width"])
    height = float(bbox["height"])
    long_edge = max(width, height)
    short_edge = max(min(width, height), 1.0)
    if short_edge < GAP_CANDIDATE_MIN_SHORT_EDGE:
        return False
    if long_edge < GAP_CANDIDATE_MIN_LONG_EDGE:
        return False
    return long_edge / short_edge <= GAP_CANDIDATE_MAX_ASPECT


def uncovered_gap_candidates(
    gap_candidates: list[PageCandidate],
    covered: list[PageCandidate],
    *,
    clearance: float = GAP_CANDIDATE_CLEARANCE,
) -> list[PageCandidate]:
    """Keep only the enlarged-pass objects that sit where nothing was found.

    The primary passes see a panel at the render's own resolution, where they
    read whole callouts but walk past the smallest text on the sheet: a "10"
    turned up the side of a view is two ~35px glyphs and is detected at
    neither rotation. Enlarging the panel finds it -- and also splits "0.5"
    into "0" and "5", and "13.8±0.3" into pieces, because a detector given
    more pixels resolves more boxes. So the enlarged pass is not allowed to
    replace anything: it may only speak where the primary passes were silent,
    which is the one place a fragment is better than nothing -- and only
    about something shaped like the small callout it is there to recover.
    """

    return [
        candidate
        for candidate in gap_candidates
        if looks_like_a_small_callout(candidate.bbox)
        and not any(
            bbox_overlap_fraction(candidate.bbox, existing.bbox) >= clearance
            for existing in covered
        )
    ]


def assign_candidate_ids(
    candidates: list[PageCandidate],
) -> list[PageCandidate]:
    """Give deduplicated page objects stable, page-order identifiers."""

    return [
        replace(candidate, candidate_id=f"C{index:04d}")
        for index, candidate in enumerate(candidates, start=1)
    ]


# One limit of a stacked pair: a plain decimal, the upper one optionally with
# its Ø or R prefix. Anything richer ("3×2.03", "(0.902 REF)") is its own
# callout that only happens to sit above or below another.
_LIMIT_UPPER = re.compile(r"^([ØR]?)(\d+\.(\d+))$")
_LIMIT_LOWER = re.compile(r"^(\d+\.(\d+))$")
_QUALIFIER = re.compile(r"^(MAX|MIN|REF)\.?$", re.IGNORECASE)
# Two lines of one callout: the gap between them is under this share of the
# line height, they are about the same height, and they overlap across most
# of the narrower one.
STACK_MAX_GAP_LINES = 0.8
STACK_HEIGHT_RATIO = 1.5
STACK_MIN_OVERLAP = 0.6


def _stacked_below(upper: Mapping[str, float], lower: Mapping[str, float], vertical: bool) -> bool:
    """Whether ``lower`` is the next line of ``upper`` in one stacked callout.

    Text written up the page turns the stack on its side: its first line is
    the one on the left.
    """
    if vertical:
        u0, u1, l0, l1 = upper["x"], upper["x"] + upper["width"], lower["x"], lower["x"] + lower["width"]
        a0, a1, b0, b1 = upper["y"], upper["y"] + upper["height"], lower["y"], lower["y"] + lower["height"]
    else:
        u0, u1, l0, l1 = upper["y"], upper["y"] + upper["height"], lower["y"], lower["y"] + lower["height"]
        a0, a1, b0, b1 = upper["x"], upper["x"] + upper["width"], lower["x"], lower["x"] + lower["width"]
    upper_line, lower_line = u1 - u0, l1 - l0
    if min(upper_line, lower_line) <= 0:
        return False
    if max(upper_line, lower_line) > STACK_HEIGHT_RATIO * min(upper_line, lower_line):
        return False
    gap = l0 - u1
    if not -0.25 * min(upper_line, lower_line) <= gap <= STACK_MAX_GAP_LINES * max(upper_line, lower_line):
        return False
    overlap = min(a1, b1) - max(a0, b0)
    return overlap >= STACK_MIN_OVERLAP * min(a1 - a0, b1 - b0)


def _union(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    x0, y0 = min(a["x"], b["x"]), min(a["y"], b["y"])
    x1 = max(a["x"] + a["width"], b["x"] + b["width"])
    y1 = max(a["y"] + a["height"], b["y"] + b["height"])
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _is_vertical(region: Mapping[str, object]) -> bool:
    return region.get("orientation") == "vertical"


def group_stacked_values(
    regions: list[dict],
    word_boxes: Sequence[tuple[str, Mapping[str, float]]] = (),
) -> tuple[list[dict], int]:
    """Join the lines of one stacked callout into a single value.

    The detector boxes text lines, so a limit dimension drawn as "4.15" over
    "3.97" arrived as two balloons, and "5.20" over "MAX" as a bare 5.20. Read
    together they are one requirement: ``4.15/3.97`` and ``5.20 MAX``. Only
    two plain decimals with the same number of places pair as limits, upper
    above lower and the larger one first; a qualifier word joins the value it
    sits directly under. Returns the regions and how many were absorbed.
    """
    absorbed: set[int] = set()
    out = [dict(region) for region in regions]
    order = sorted(range(len(out)), key=lambda i: (out[i]["bbox"]["y"], out[i]["bbox"]["x"]))
    for i in order:
        if i in absorbed:
            continue
        upper = out[i]
        m_up = _LIMIT_UPPER.match(str(upper.get("text") or "").replace(" ", ""))
        if not m_up:
            continue
        vertical = _is_vertical(upper)
        for j in order:
            if j == i or j in absorbed or _is_vertical(out[j]) != vertical:
                continue
            m_low = _LIMIT_LOWER.match(str(out[j].get("text") or "").replace(" ", ""))
            if not m_low or len(m_low.group(2)) != len(m_up.group(3)):
                continue
            if float(m_low.group(1)) >= float(m_up.group(2)):
                continue
            if not _stacked_below(upper["bbox"], out[j]["bbox"], vertical):
                continue
            upper["text"] = f"{m_up.group(1)}{m_up.group(2)}/{m_low.group(1)}"
            upper["bbox"] = _union(upper["bbox"], out[j]["bbox"])
            upper["stacked_with"] = out[j].get("candidate_id")
            upper["grouped"] = True
            absorbed.add(j)
            break
    regions_out = [region for index, region in enumerate(out) if index not in absorbed]

    for region in regions_out:
        text = str(region.get("text") or "").strip()
        if not re.fullmatch(r"[ØR]?\d+(?:\.\d+)?", text):
            continue
        for word, box in word_boxes:
            qualifier = _QUALIFIER.match(str(word or "").strip())
            if qualifier and _stacked_below(region["bbox"], box, _is_vertical(region)):
                region["text"] = f"{text} {qualifier.group(1).upper()}"
                region["bbox"] = _union(region["bbox"], box)
                region["grouped"] = True
                break
    return regions_out, len(absorbed)


# A reading this much inside a published value that runs the other way is a
# piece of that value read across its lines, not a callout of its own.
CROSS_DIRECTION_INSIDE_SHARE = 0.6


def drop_cross_direction_fragments(
    regions: list[dict],
    review_candidates: list[dict],
) -> tuple[list[dict], list[dict], int]:
    """Keep one reading direction per box.

    A callout written up the page is read bottom to top; nothing inside its
    box reads left to right, and the reverse. A reading that crosses the lines
    of a published value -- on 2H-G772 a vertical "43" taken through the
    horizontal "4.15/3.97" stack -- is dropped, whether it was published or
    waiting for review. Returns both lists and how many were dropped.
    """

    def direction(region: Mapping[str, object]) -> str:
        return "vertical" if _is_vertical(region) else "horizontal"

    def crosses(item: Mapping[str, object]) -> bool:
        box = item["bbox"]
        area = max(float(box["width"]) * float(box["height"]), 1.0)
        for owner in regions:
            if owner is item or direction(owner) == direction(item):
                continue
            # A note block or a frame is not one value: a real vertical
            # dimension can sit inside a paragraph's bounding box.
            if owner.get("type") == "GD&T" or "Note" in str(owner.get("type") or ""):
                continue
            owner_area = float(owner["bbox"]["width"]) * float(owner["bbox"]["height"])
            if (
                area < owner_area
                and bbox_overlap_fraction(box, owner["bbox"])
                >= CROSS_DIRECTION_INSIDE_SHARE
            ):
                return True
        return False

    kept_regions = [region for region in regions if not crosses(region)]
    kept_review = [item for item in review_candidates if not crosses(item)]
    dropped = (len(regions) - len(kept_regions)) + (len(review_candidates) - len(kept_review))
    return kept_regions, kept_review, dropped


_LIMIT_PAIR = re.compile(r"^\(?([ØR]?)(\d+\.(\d+))/(\d+\.(\d+))\)?$")


def is_limit_pair(text: str) -> bool:
    """Whether ``text`` already reads as one limit dimension, upper/lower.

    Such a box is tall because it holds two lines, but it is one requirement
    read correctly: splitting it costs a detector and OCR pass, and the
    grouping step would join the halves straight back.
    """
    match = _LIMIT_PAIR.match(str(text or "").replace(" ", ""))
    if not match or len(match.group(3)) != len(match.group(5)):
        return False
    return float(match.group(2)) > float(match.group(4))
