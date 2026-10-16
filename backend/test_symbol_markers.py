"""Symbol-marker matching tests. OpenCV only; no OCR model is loaded.

The sheets here are drawn with PIL's own font so the test owns both sides of
the comparison: the "table" cells and the "views" print the same characters,
and the matcher has to tell the point names apart from the digits and from
the words they are printed beside.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from page_layout import LayoutBox
from symbol_markers import build_templates, locate_markers


TABLE_SIZE = 44
VIEW_SIZE = 22


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    size: int,
    fill: str | tuple[int, int, int] = "black",
) -> None:
    draw.text(position, text, fill=fill, font=_font(size))


class Sheet:
    """A synthetic drawing with a name/coordinate table and some views."""

    def __init__(self, width: int = 1400, height: int = 900) -> None:
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.name_cells: dict[str, LayoutBox] = {}
        self.value_cells: list[LayoutBox] = []

    def table(self, rows: dict[str, str], *, left: int, top: int) -> None:
        """One name column and one coordinate column, in ruled cells."""

        name_width, value_width, height = 90, 220, 70
        for index, (symbol, value) in enumerate(rows.items()):
            y = top + index * height
            name = LayoutBox(left, y, name_width, height)
            value_box = LayoutBox(left + name_width, y, value_width, height)
            for box in (name, value_box):
                self.draw.rectangle(
                    (box.x, box.y, box.x1, box.y1),
                    outline="black",
                    width=2,
                )
            _write(self.draw, symbol, (left + 28, y + 6), TABLE_SIZE)
            _write(
                self.draw,
                value,
                (left + name_width + 20, y + 6),
                TABLE_SIZE,
            )
            self.name_cells[symbol] = name
            self.value_cells.append(value_box)

    def callout(
        self,
        symbol: str,
        position: tuple[int, int],
        fill: str | tuple[int, int, int] = "black",
    ) -> None:
        """A point name printed on its own with a leader line beside it."""

        x, y = position
        _write(self.draw, symbol, (x, y), VIEW_SIZE, fill)
        self.draw.line((x + 34, y + 16, x + 110, y + 60), fill="black", width=2)

    def text(self, words: str, position: tuple[int, int]) -> None:
        _write(self.draw, words, position, VIEW_SIZE)

    def templates(self):
        return build_templates(self.image, self.name_cells, self.value_cells)


def _found(markers, symbol: str) -> int:
    return sum(1 for marker in markers if marker.symbol == symbol)


def test_a_point_name_is_found_where_the_view_prints_it() -> None:
    sheet = Sheet()
    sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
    sheet.callout("a", (200, 200))
    sheet.callout("b", (500, 300))

    markers = locate_markers(sheet.image, sheet.templates())

    assert _found(markers, "a") == 1
    assert _found(markers, "b") == 1
    assert _found(markers, "c") == 0


def test_the_same_name_is_found_in_every_view_that_calls_it_out() -> None:
    sheet = Sheet()
    sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
    sheet.callout("b", (200, 200))
    sheet.callout("b", (600, 200))
    sheet.callout("b", (200, 420))

    markers = locate_markers(sheet.image, sheet.templates())

    assert _found(markers, "b") == 3


def test_the_table_itself_is_never_reported_as_a_callout() -> None:
    sheet = Sheet()
    sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
    sheet.callout("a", (200, 200))
    table_region = LayoutBox(880, 540, 340, 240)

    markers = locate_markers(
        sheet.image,
        sheet.templates(),
        excluded=[table_region],
    )

    assert len(markers) == 1
    assert markers[0].bbox.x < 400


def test_a_letter_inside_a_word_is_not_a_callout() -> None:
    sheet = Sheet()
    sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
    sheet.text("detail cab shown above", (150, 200))

    assert locate_markers(sheet.image, sheet.templates()) == []


def test_a_letter_spaced_heading_is_not_a_row_of_callouts() -> None:
    sheet = Sheet()
    sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
    sheet.text("G u a r a n t e e d   c a b l e", (150, 200))

    assert locate_markers(sheet.image, sheet.templates()) == []


def test_a_lone_digit_is_not_mistaken_for_a_point_name() -> None:
    sheet = Sheet()
    sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
    for index, digit in enumerate("035672"):
        _write(sheet.draw, digit, (150 + index * 160, 200), VIEW_SIZE)

    assert locate_markers(sheet.image, sheet.templates()) == []


def test_nothing_is_reported_without_any_point_names() -> None:
    sheet = Sheet()
    sheet.callout("a", (200, 200))

    assert locate_markers(sheet.image, []) == []


def test_every_marker_carries_its_score_and_its_lead() -> None:
    sheet = Sheet()
    sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
    sheet.callout("a", (200, 200))

    markers = locate_markers(sheet.image, sheet.templates())

    assert len(markers) == 1
    assert 0.0 < markers[0].score <= 1.0
    assert markers[0].margin > 0.0


def test_a_callout_printed_in_grey_is_still_found() -> None:
    """Point names are routinely set lighter than the artwork around them.

    A global Otsu threshold splits a mostly-white sheet near mid-grey, so
    markers printed above that level used to be dropped as paper before they
    were ever compared against a template — the sheet reported them as named
    in the table but absent from the views.
    """

    for grey in (110, 140, 170, 195):
        sheet = Sheet()
        sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
        sheet.callout("b", (200, 200), (grey, grey, grey))
        sheet.callout("c", (500, 300), (grey, grey, grey))

        markers = locate_markers(sheet.image, sheet.templates())

        assert _found(markers, "b") == 1, f"lost 'b' at grey {grey}"
        assert _found(markers, "c") == 1, f"lost 'c' at grey {grey}"


def test_grey_callouts_survive_a_noisy_scan() -> None:
    """The looser threshold must not simply read paper noise as ink."""

    import numpy as np

    sheet = Sheet()
    sheet.table({"a": "0.0", "b": "35.6", "c": "72.2"}, left=900, top=560)
    sheet.callout("b", (200, 200), (165, 165, 165))
    sheet.callout("c", (500, 300), (165, 165, 165))

    speckled = np.asarray(sheet.image).astype(np.int16)
    speckled -= 10
    speckled += np.random.default_rng(0).normal(0, 9, speckled.shape).astype(np.int16)
    sheet.image = Image.fromarray(np.clip(speckled, 0, 255).astype(np.uint8))

    markers = locate_markers(sheet.image, sheet.templates())

    assert _found(markers, "b") == 1
    assert _found(markers, "c") == 1
