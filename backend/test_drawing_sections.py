"""Sub-drawing sectioning on synthetic sheets; no OCR models are loaded."""

from __future__ import annotations

import math

import numpy as np

from PIL import Image, ImageDraw

from drawing_sections import (
    DrawingSection,
    _ring_groups,
    _seam_merges,
    _sheet_frame,
    detect_drawing_sections,
    section_crop,
    tag_regions_with_sections,
)

SHEET = (2400, 1600)


def _sheet() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SHEET, "white")
    return image, ImageDraw.Draw(image)


def _part_view(draw: ImageDraw.ImageDraw, x: int, y: int, value: str) -> tuple[int, int]:
    """A machined part: outline, bore, hatching, and one dimension with its
    value written above the dimension line. Returns the value's position."""

    draw.rectangle((x, y, x + 360, y + 260), outline="black", width=3)
    draw.rectangle((x + 60, y + 60, x + 200, y + 200), outline="black", width=2)
    draw.ellipse((x + 240, y + 90, x + 320, y + 170), outline="black", width=2)
    for offset in range(0, 140, 14):
        draw.line((x + 60 + offset, y + 200, x + 200, y + 60 + offset), fill="black", width=1)
    # Extension lines start a small gap off the outline, as drafted.
    draw.line((x, y - 8, x, y - 70), fill="black", width=1)
    draw.line((x + 360, y - 8, x + 360, y - 70), fill="black", width=1)
    draw.line((x, y - 50, x + 360, y - 50), fill="black", width=1)
    draw.polygon([(x, y - 50), (x + 16, y - 55), (x + 16, y - 45)], fill="black")
    draw.polygon([(x + 360, y - 50), (x + 344, y - 55), (x + 344, y - 45)], fill="black")
    draw.text((x + 160, y - 82), value, fill="black", font_size=24)
    return x + 175, y - 70


def _views(sections: tuple[DrawingSection, ...]) -> list[DrawingSection]:
    return [section for section in sections if section.kind == "view"]


def _holding(sections, x: int, y: int) -> list[DrawingSection]:
    return [
        section
        for section in sections
        if section.bbox.x <= x <= section.bbox.x1 and section.bbox.y <= y <= section.bbox.y1
    ]


def test_separate_views_become_separate_sections_with_their_values() -> None:
    image, draw = _sheet()
    left_value = _part_view(draw, 200, 400, "48.5")
    right_value = _part_view(draw, 1500, 900, "27")

    views = _views(detect_drawing_sections(image, table_masks=()))

    assert len(views) == 2
    # The values touch no line, yet each lands in its own view's section.
    assert len(_holding(views, *left_value)) == 1
    assert len(_holding(views, *right_value)) == 1
    assert _holding(views, *left_value) != _holding(views, *right_value)


def test_a_leader_line_between_two_views_does_not_fuse_them() -> None:
    image, draw = _sheet()
    _part_view(draw, 200, 400, "48.5")
    _part_view(draw, 1100, 400, "27")
    # One thin leader runs from the first view right into the second.
    draw.line((560, 530, 1180, 560), fill="black", width=1)

    views = _views(detect_drawing_sections(image, table_masks=()))

    assert len(views) == 2
    left, right = sorted(views, key=lambda section: section.bbox.x)
    assert left.line_bbox.x1 < 1100
    assert right.line_bbox.x > 560


def test_a_notes_paragraph_is_a_text_section_not_a_view() -> None:
    image, draw = _sheet()
    _part_view(draw, 1500, 300, "12")
    for row in range(8):
        draw.text(
            (150, 900 + row * 40),
            f"{row + 1}) UNLESS OTHERWISE SPECIFIED DRAFTS SHALL BE 3 MAX",
            fill="black",
            font_size=26,
        )

    sections = detect_drawing_sections(image, table_masks=())

    assert len(_views(sections)) == 1
    texts = [section for section in sections if section.kind == "text"]
    assert len(texts) == 1
    assert texts[0].bbox.x < 200 and texts[0].bbox.y1 > 1150


def test_a_view_in_sparse_thin_lines_stays_one_section() -> None:
    """A gear face: a thin outer circle and bore, spokes, and the dense
    detail only at the rim. Its corners would each core on their own."""

    image, draw = _sheet()
    cx, cy, radius = 1200, 800, 500
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="black", width=2)
    draw.ellipse((cx - 120, cy - 120, cx + 120, cy + 120), outline="black", width=2)
    for spoke in range(6):
        angle = spoke * math.pi / 3
        draw.line(
            (
                cx + 120 * math.cos(angle),
                cy + 120 * math.sin(angle),
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle),
            ),
            fill="black",
            width=2,
        )
        # A dense block of detail where each spoke meets the rim.
        bx = cx + (radius - 60) * math.cos(angle)
        by = cy + (radius - 60) * math.sin(angle)
        for step in range(0, 80, 8):
            draw.line((bx - 40, by - 40 + step, bx + 40, by - 40 + step), fill="black", width=2)

    views = _views(detect_drawing_sections(image, table_masks=()))

    assert len(views) == 1
    assert views[0].line_bbox.width > 900


def test_a_blank_sheet_has_no_sections() -> None:
    image, _ = _sheet()

    assert detect_drawing_sections(image, table_masks=()) == ()


def test_nothing_outside_the_drawing_frame_is_a_section() -> None:
    """Margin notes and stamps sit between the frame and the paper edge.
    The frame is drawn slightly skewed, as a scanned sheet's is."""

    image, draw = _sheet()
    draw.line((60, 60, 2340, 66), fill="black", width=3)
    draw.line((60, 1540, 2340, 1534), fill="black", width=3)
    draw.line((60, 60, 64, 1540), fill="black", width=3)
    draw.line((2340, 66, 2336, 1534), fill="black", width=3)
    _part_view(draw, 400, 500, "48.5")
    draw.text((70, 8), "NOTE: DO NOT SCALE DRAWING", fill="black", font_size=30)
    draw.text((1700, 1550), "CATIA V5 DRAWING", fill="black", font_size=30)

    sections = detect_drawing_sections(image, table_masks=())

    assert [section.kind for section in sections] == ["view"]
    assert all(60 <= section.bbox.y and section.bbox.y1 <= 1540 for section in sections)


def _region_mask(section: DrawingSection) -> Image.Image:
    mask = Image.new("1", SHEET, 0)
    draw = ImageDraw.Draw(mask)
    for outline in section.outlines:
        draw.polygon(list(outline), fill=1)
    return mask


def test_regions_of_close_views_never_overlap() -> None:
    """Two views close enough that their bounding boxes overlap; their
    regions claim each pixel once (outlines are simplified polygons, so a
    shared border may cross by a pixel or two)."""

    image, draw = _sheet()
    _part_view(draw, 200, 400, "48.5")
    _part_view(draw, 600, 720, "27")

    views = sorted(
        _views(detect_drawing_sections(image, table_masks=())),
        key=lambda view: view.bbox.x,
    )

    assert len(views) == 2
    assert views[0].bbox.x1 > views[1].bbox.x and views[0].bbox.y1 > views[1].bbox.y
    first, second = (np.asarray(_region_mask(view)) for view in views)
    assert (first & second).sum() < 0.02 * min(first.sum(), second.sum())


def test_a_section_crop_shows_none_of_its_neighbour() -> None:
    image, draw = _sheet()
    _part_view(draw, 200, 400, "48.5")
    _part_view(draw, 600, 720, "27")
    views = sorted(
        _views(detect_drawing_sections(image, table_masks=())),
        key=lambda view: view.bbox.x,
    )

    crop, (x0, y0) = section_crop(image, views[0])

    # The right view's extension line runs inside the left view's box.
    assert views[0].bbox.x1 > 603
    line = np.asarray(crop.convert("L"))[650 - y0 : 705 - y0, 598 - x0 : 603 - x0]
    assert line.min() == 255
    # And the crop keeps the left view's own value.
    value = np.asarray(crop.convert("L"))[326 - y0 : 344 - y0, 360 - x0 : 410 - x0]
    assert value.min() < 128


def _owned_pieces(*, gap: int, seam_lines: int) -> tuple[np.ndarray, np.ndarray]:
    """Two pieces in one region, each a stack of lines every 20 px, ``gap``
    px apart. ``seam_lines`` of the left piece's lines run on to the right
    piece; the rest stop short."""

    section_ink = np.zeros((400, 900), np.int32)
    right = 290 + gap
    for y in range(40, 360, 20):
        section_ink[y : y + 2, 60:290] = 1
        section_ink[y : y + 2, right : right + 230] = 2
    for y in range(40, 40 + 20 * seam_lines, 20):
        middle = (290 + right) // 2
        section_ink[y : y + 2, 290:middle] = 1
        section_ink[y : y + 2, middle:right] = 2
    return section_ink, (section_ink > 0).astype(np.uint8)


def _merges(section_ink, lines, origin=None) -> dict[int, int]:
    origin = np.arange(3) if origin is None else origin
    return _seam_merges(section_ink, lines, {1: [1, 2]}, origin, pad=10, min_seam=40)


def test_a_view_cut_through_its_own_lines_is_rejoined() -> None:
    """A wire-frame view cores only at its dense spots; the cut between two
    of them runs through drawing as inked as the pieces."""

    assert _merges(*_owned_pieces(gap=20, seam_lines=16)) == {2: 1}


def test_two_views_meeting_on_a_leader_stay_apart() -> None:
    assert _merges(*_owned_pieces(gap=300, seam_lines=1)) == {}


def test_the_halves_of_a_necked_core_stay_apart() -> None:
    pieces = _owned_pieces(gap=20, seam_lines=16)

    assert _merges(*pieces, origin=np.array([0, 1, 1])) == {}


def test_regions_are_tagged_with_the_section_they_lie_in() -> None:
    image, draw = _sheet()
    left_value = _part_view(draw, 200, 400, "48.5")
    right_value = _part_view(draw, 1500, 900, "27")
    sections = detect_drawing_sections(image, table_masks=())

    def region(point: tuple[int, int]) -> dict:
        return {"bbox": {"x": point[0] - 10, "y": point[1] - 5, "width": 20, "height": 10}}

    regions = [region(right_value), region(left_value), region((5, 5))]
    tag_regions_with_sections(regions, sections)

    ids = [section.section_id for section in sections]
    left, right, corner = regions
    assert right["section_id"] != left["section_id"]
    for tagged in regions:
        assert ids[tagged["section_rank"]] == tagged["section_id"]
    # Off every outline, the nearest section still takes it.
    assert corner["section_id"] in ids


def test_two_short_rules_in_one_row_are_not_the_frame() -> None:
    # 3015-100-2232F: a note box's bottom edge (45% of the sheet) and a
    # dimension line (17%) share a row near the top. Summed they passed as
    # the frame's top rule, and everything above it, a value included, was
    # cut off the drawing.
    ink = np.zeros((1400, 2000), np.uint8)
    ink[40:43, 40:1960] = 1
    ink[1357:1360, 40:1960] = 1
    ink[40:1360, 40:43] = 1
    ink[40:1360, 1957:1960] = 1
    ink[110:113, 1000:1900] = 1
    ink[111:114, 150:490] = 1

    frame = _sheet_frame(ink)

    assert frame is not None
    assert frame[1] < 60


def test_arcs_of_one_ring_stay_one_view_but_a_neck_splits_two() -> None:
    import cv2

    # A flange's concentric outline opens into two arcs, linked at both ends.
    ring = np.zeros((300, 400), np.uint8)
    cv2.ellipse(ring, (200, 150), (160, 110), 0, 0, 360, 1, 30)
    parts = np.zeros(ring.shape, np.int32)
    parts[(ring > 0) & (np.arange(400)[None, :] < 170)] = 1
    parts[(ring > 0) & (np.arange(400)[None, :] > 230)] = 2
    assert _ring_groups(ring, parts, [1, 2]) == [[1, 2]]

    # Two views joined by one neck of dimensions split as before.
    pair = np.zeros((300, 400), np.uint8)
    pair[50:250, 20:160] = 1
    pair[50:250, 240:380] = 1
    pair[140:160, 160:240] = 1
    parts = np.zeros(pair.shape, np.int32)
    parts[50:250, 20:160] = 1
    parts[50:250, 240:380] = 2
    assert _ring_groups(pair, parts, [1, 2]) == [[1], [2]]
