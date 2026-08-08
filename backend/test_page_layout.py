"""Fast page-layout tests; no OCR or OpenCV models are loaded."""

from __future__ import annotations

from PIL import Image, ImageDraw

from page_layout import (
    LAYOUT_MAX_PANELS,
    analyze_page_layout,
    crop_masked_section,
    detect_table_masks,
)
from page_layout_cache import PageLayoutCache


def synthetic_drawing() -> Image.Image:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)

    # One ordinary drawing rectangle must not be classified as a table.
    draw.rectangle((40, 40, 260, 250), outline="black", width=2)
    draw.line((270, 300, 350, 300), fill="black", width=2)

    # A repeated 3x4 grid is unambiguously tabular.
    x_values = (400, 460, 520, 580)
    y_values = (30, 90, 150, 210, 270)
    for x in x_values:
        draw.line((x, y_values[0], x, y_values[-1]), fill="black", width=2)
    for y in y_values:
        draw.line((x_values[0], y, x_values[-1], y), fill="black", width=2)
    return image


def _contains(box, x: int, y: int) -> bool:
    return box.x <= x <= box.x1 and box.y <= y <= box.y1


def test_table_detection_requires_repeated_cells_not_one_rectangle() -> None:
    masks = detect_table_masks(synthetic_drawing())

    assert masks
    assert any(_contains(mask, 500, 120) for mask in masks)
    assert not any(_contains(mask, 120, 120) for mask in masks)


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


def test_mixed_section_masks_only_the_table_part() -> None:
    image = synthetic_drawing()
    layout = analyze_page_layout(image)
    section, origin, clipped = crop_masked_section(
        image,
        {"x": 250, "y": 0, "width": 360, "height": 340},
        layout.table_masks,
        padding=20,
    )

    assert origin == (230, -20)
    assert clipped.box == (250, 0, 610, 340)
    # Page (500, 120) is inside the grid and is whitened.
    assert section.getpixel((500 - origin[0], 120 - origin[1])) == (
        255,
        255,
        255,
    )
    # Page (300, 300) is drawing ink outside the table and remains visible.
    assert section.getpixel((300 - origin[0], 300 - origin[1])) == (0, 0, 0)


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
