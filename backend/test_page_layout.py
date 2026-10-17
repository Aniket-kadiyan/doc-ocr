"""Fast page-layout tests; no OCR or OpenCV models are loaded."""

from __future__ import annotations

from PIL import Image, ImageDraw

from page_layout import (
    LAYOUT_MAX_PANELS,
    LayoutBox,
    _retain_repeated_cell_groups,
    analyze_page_layout,
    crop_section,
    detect_table_masks,
)
from page_layout_cache import PageLayoutCache


def synthetic_drawing() -> Image.Image:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)

    # One ordinary drawing rectangle must not be classified as a table.
    draw.rectangle((40, 40, 260, 250), outline="black", width=2)
    draw.line((270, 300, 350, 300), fill="black", width=2)

    # A repeated 3x4 grid with text in its cells is unambiguously tabular.
    x_values = (458, 518, 578, 638)
    y_values = (30, 54, 78, 102, 126)
    _draw_populated_grid(draw, x_values, y_values)
    return image


def _draw_populated_grid(draw: ImageDraw.ImageDraw, x_values, y_values) -> None:
    """Rules plus a short label in every cell, which is what a table has.

    An empty grid is not a table: a machined part in orthographic projection
    is exactly a set of adjacent empty rectangles, and the mask it produced
    erased every dimension inside the view.
    """
    for x in x_values:
        draw.line((x, y_values[0], x, y_values[-1]), fill="black", width=2)
    for y in y_values:
        draw.line((x_values[0], y, x_values[-1], y), fill="black", width=2)
    for left, right in zip(x_values, x_values[1:]):
        for top, bottom in zip(y_values, y_values[1:]):
            draw.text((left + 8, top + 5), "12", fill="black")


def _contains(box, x: int, y: int) -> bool:
    return box.x <= x <= box.x1 and box.y <= y <= box.y1


def test_table_detection_requires_repeated_cells_not_one_rectangle() -> None:
    masks = detect_table_masks(synthetic_drawing())

    assert masks
    assert any(_contains(mask, 560, 120) for mask in masks)
    assert not any(_contains(mask, 120, 120) for mask in masks)


def test_interior_repeated_grid_is_a_page_table_mask() -> None:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    x_values = (280, 340, 400, 460)
    y_values = (80, 104, 128, 152, 176)
    _draw_populated_grid(draw, x_values, y_values)

    masks = detect_table_masks(image)

    assert masks
    assert any(_contains(mask, 370, 130) for mask in masks)


def test_adaptive_layout_never_uses_the_complete_page_as_one_panel() -> None:
    layout = analyze_page_layout(synthetic_drawing())

    assert 4 <= len(layout.panels) <= LAYOUT_MAX_PANELS
    assert all(
        panel.bbox.width < layout.width or panel.bbox.height < layout.height
        for panel in layout.panels
    )
    for x, y in ((0, 0), (layout.width - 1, 0), (0, layout.height - 1)):
        assert any(_contains(panel.bbox, x, y) for panel in layout.panels)
    assert layout.overlaps


def test_section_crop_keeps_selected_table_and_drawing_pixels() -> None:
    image = synthetic_drawing()
    layout = analyze_page_layout(image)
    section, origin, clipped = crop_section(
        image,
        {"x": 250, "y": 0, "width": 360, "height": 340},
        padding=20,
    )

    assert origin == (230, -20)
    assert clipped.box == (250, 0, 610, 340)
    # Boundary-table ink remains because section mode is user-controlled.
    assert section.getpixel((578 - origin[0], 120 - origin[1])) == (0, 0, 0)
    # Page (300, 300) is drawing ink outside the table and remains visible.
    assert section.getpixel((300 - origin[0], 300 - origin[1])) == (0, 0, 0)
    assert layout.overlay(scope_kind="section")["table_masks"] == []


def test_page_layout_cache_is_bounded_and_reuses_geometry() -> None:
    layout = analyze_page_layout(synthetic_drawing())
    cache = PageLayoutCache(max_entries=2)
    cache.put("one", layout)
    cached, hit = cache.get_or_create("one", lambda: layout)

    assert hit and cached is layout
    cache.put("two", layout)
    cache.put("three", layout)
    assert cache.get("one") is None
    assert cache.get("two") is layout
    assert cache.get("three") is layout


def test_a_table_the_page_cuts_through_keeps_its_last_column() -> None:
    """A media box that clips the sheet leaves a column with three sides.

    Without the page edge standing in for the missing rule no cell forms
    there, the column is never masked, and whatever it holds is scanned as
    free drawing text — on a real sheet that published a tensile-strength
    figure from a materials table as if it were a dimension.
    """

    width, height = 640, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y_values = (80, 104, 128, 152, 176)
    # Rules at 300, 400 and 500; the fourth column runs off the paper.
    for x in (300, 400, 500):
        draw.line((x, y_values[0], x, y_values[-1]), fill="black", width=2)
    for y in y_values:
        draw.line((300, y, width, y), fill="black", width=2)
    for left in (300, 400, 500):
        for top in y_values[:-1]:
            draw.text((left + 8, top + 5), "12", fill="black")

    masks = detect_table_masks(image)

    assert masks
    # The closed columns were always found.
    assert any(_contains(mask, 450, 130) for mask in masks)
    # The clipped one is what this test is about.
    assert any(_contains(mask, 580, 130) for mask in masks)


def test_a_framed_sheet_does_not_grow_cells_out_to_the_paper() -> None:
    """The border frame must not become the far side of every row."""

    width, height = 640, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, width - 20, height - 20), outline="black", width=2)
    x_values = (280, 340, 400, 460)
    y_values = (80, 104, 128, 152, 176)
    _draw_populated_grid(draw, x_values, y_values)

    masks = detect_table_masks(image)

    assert masks
    assert any(_contains(mask, 370, 130) for mask in masks)
    # Nothing between the frame and the paper is a table cell.
    assert not any(_contains(mask, 10, 180) for mask in masks)
    assert not any(_contains(mask, width - 10, 180) for mask in masks)


def _sheet_with_a_margin_legend() -> Image.Image:
    """A sheet-proportioned page: a wide title block and a narrow margin table."""

    image = Image.new("RGB", (2400, 1700), "white")
    draw = ImageDraw.Draw(image)

    # The boilerplate legend down the left margin: rules only 78px long on a
    # 2400px page, well under the page-wide "a rule is long" threshold.
    left, right = 40, 118
    for index in range(12):
        y = 500 + index * 60
        draw.line((left, y, right, y), fill="black", width=2)
    draw.line((left, 500, left, 500 + 11 * 60), fill="black", width=2)
    draw.line((right, 500, right, 500 + 11 * 60), fill="black", width=2)
    draw.line((79, 500, 79, 500 + 11 * 60), fill="black", width=2)
    for index in range(11):
        for x in (left, 79):
            draw.text((x + 10, 500 + index * 60 + 24), "12", fill="black")

    # A title block across the bottom, which the page-wide pass already finds.
    for index in range(4):
        y = 1500 + index * 50
        draw.line((900, y, 2300, y), fill="black", width=2)
    for x in (900, 1600, 2300):
        draw.line((x, 1500, x, 1650), fill="black", width=2)
    for x in (900, 1600):
        for index in range(3):
            draw.text((x + 20, 1500 + index * 50 + 18), "ABC 12", fill="black")

    # An ordinary drawing rectangle in the middle must stay unmasked.
    draw.rectangle((700, 300, 1500, 900), outline="black", width=2)
    return image


def test_a_narrow_margin_legend_is_masked_like_any_other_table() -> None:
    # A rule counts as a rule by being long relative to the page, which is
    # right for a title block spanning the sheet and wrong for the boilerplate
    # column down a margin. This drawing's tolerance legend ruled its rows
    # over 190px of a 5875px page, so not one row was found and every garbled
    # label in it was ballooned as a value.
    masks = detect_table_masks(_sheet_with_a_margin_legend())

    legend = [
        mask
        for mask in masks
        if mask.x < 200 and mask.y < 1200 and mask.y1 > 700
    ]
    assert legend, [ (m.x, m.y, m.width, m.height) for m in masks ]
    covered = legend[0]
    assert covered.y <= 520 and covered.y1 >= 1040

    # The drawing rectangle in the middle is not a table.
    assert not [
        mask
        for mask in masks
        if mask.x > 600 and mask.x1 < 1600 and mask.y > 250 and mask.y1 < 1000
    ]


def test_the_margin_pass_does_not_run_on_a_small_section() -> None:
    # The band would be a few pixels wide, so there is nothing to measure
    # against and the page-wide pass stands alone.
    masks = detect_table_masks(synthetic_drawing())

    assert all(mask.width >= 2 and mask.height >= 2 for mask in masks)


def test_two_framed_values_on_a_view_are_not_a_table():
    """A pair of adjacent rectangles is drawing geometry, not a ruled grid.

    Masking happens by whitening the page before detection, so a mask landing
    on the views deletes the values under it outright — no balloon and no
    review box. A drawing that frames its dimension text produces such pairs
    all over, and two cells used to be enough to call them a table.
    """
    pair = [
        LayoutBox(100, 100, 80, 40),
        LayoutBox(100, 140, 80, 40),
    ]
    assert _retain_repeated_cell_groups(pair, tolerance=4) == []


def test_a_ruled_grid_is_still_a_table():
    grid = [
        LayoutBox(100, 100 + row * 40, 80, 40)
        for row in range(4)
    ]
    assert len(_retain_repeated_cell_groups(grid, tolerance=4)) == 4


def test_an_empty_grid_is_a_part_view_not_a_table() -> None:
    """Adjacent empty rectangles are what a machined block looks like."""
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    x_values = (280, 340, 400, 460)
    y_values = (80, 130, 180, 230, 280)
    for x in x_values:
        draw.line((x, y_values[0], x, y_values[-1]), fill="black", width=2)
    for y in y_values:
        draw.line((x_values[0], y, x_values[-1], y), fill="black", width=2)
    # One dimension printed inside the view, which the mask used to erase.
    draw.text((300, 100), "21.42", fill="black")

    assert not any(_contains(mask, 370, 180) for mask in detect_table_masks(image))


def test_a_grid_of_large_cells_is_not_a_table_even_with_text_in_it() -> None:
    """A section view with a dimension in every pocket is still a view."""
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    x_values = (100, 300, 500)
    y_values = (60, 200, 340)
    _draw_populated_grid(draw, x_values, y_values)

    assert not any(_contains(mask, 300, 200) for mask in detect_table_masks(image))
