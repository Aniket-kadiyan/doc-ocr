"""Reference-table grid and header tests; no OCR model is loaded.

Recognized text is supplied directly, which is the module's contract: it
decides *which* ruled table is a coordinate table and what its rows say, and
leaves recognition to the caller.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from page_layout import LayoutBox
from reference_table import (
    GridCell,
    GridTable,
    assign_text_to_cells,
    extract_grid_tables,
    identify_reference_table,
)


def ruled_table(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    column_width: int,
    row_height: int,
    columns: int,
    rows: int,
) -> None:
    right = left + column_width * columns
    bottom = top + row_height * rows
    for index in range(columns + 1):
        x = left + column_width * index
        draw.line((x, top, x, bottom), fill="black", width=2)
    for index in range(rows + 1):
        y = top + row_height * index
        draw.line((left, y, right, y), fill="black", width=2)


def sheet_with_two_tables() -> Image.Image:
    image = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(image)
    # Artwork that is not a table: one rectangle and a leader line.
    draw.rectangle((40, 40, 300, 260), outline="black", width=2)
    draw.line((320, 120, 420, 120), fill="black", width=2)
    ruled_table(
        draw,
        left=500,
        top=40,
        column_width=90,
        row_height=46,
        columns=4,
        rows=6,
    )
    ruled_table(
        draw,
        left=60,
        top=420,
        column_width=120,
        row_height=50,
        columns=3,
        rows=4,
    )
    return image


def synthetic_lattice(
    *,
    left: int = 0,
    top: int = 0,
    column_width: int = 80,
    row_height: int = 40,
    columns: int = 4,
    rows: int = 5,
) -> GridTable:
    cells = tuple(
        GridCell(
            LayoutBox(
                left + column * column_width,
                top + row * row_height,
                column_width,
                row_height,
            ),
            row,
            column,
        )
        for row in range(rows)
        for column in range(columns)
    )
    return GridTable(
        bbox=LayoutBox(left, top, column_width * columns, row_height * rows),
        cells=cells,
        row_count=rows,
        column_count=columns,
    )


def coordinate_text(
    header: tuple[str, str, str, str] = ("POINT", "X", "Y", "Z"),
) -> dict[tuple[int, int], str]:
    text = {(0, column): value for column, value in enumerate(header)}
    rows = (
        ("a", "0.0", "0.0", "0.0"),
        ("b", "0.0", "35.6", "6.9"),
        ("c", "10.0", "72.2", "9.3"),
        ("d", "-0.5", "108.8", "25.3"),
    )
    for index, row in enumerate(rows, start=1):
        for column, value in enumerate(row):
            text[(index, column)] = value
    return text


def test_grids_are_recovered_as_separate_lattices() -> None:
    tables = extract_grid_tables(sheet_with_two_tables())

    assert len(tables) == 2
    shapes = sorted((table.row_count, table.column_count) for table in tables)
    assert shapes == [(4, 3), (6, 4)]


def test_a_lone_rectangle_is_not_a_table() -> None:
    image = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 300, 300), outline="black", width=2)

    assert extract_grid_tables(image) == []


def test_text_lands_in_the_cell_holding_its_centre() -> None:
    table = synthetic_lattice()
    boxes = [
        {"x": 10.0, "y": 8.0, "w": 40.0, "h": 20.0, "text": "POINT"},
        {"x": 92.0, "y": 50.0, "w": 30.0, "h": 20.0, "text": "0.0"},
        # A box wider than its cell still belongs to the cell it is centred in.
        {"x": 150.0, "y": 50.0, "w": 90.0, "h": 20.0, "text": "35.6"},
    ]

    assert assign_text_to_cells(table, boxes) == {
        (0, 0): "POINT",
        (1, 1): "0.0",
        (1, 2): "35.6",
    }


def test_header_identifies_the_table_without_any_caption() -> None:
    table = synthetic_lattice()

    reference = identify_reference_table([table], [coordinate_text()])

    assert reference is not None
    assert reference.axes == ("X", "Y", "Z")
    assert [point.symbol for point in reference.points] == ["a", "b", "c", "d"]
    assert reference.points[1].coordinates == (
        ("X", "0.0"),
        ("Y", "35.6"),
        ("Z", "6.9"),
    )


def test_a_blank_name_header_is_still_a_reference_table() -> None:
    reference = identify_reference_table(
        [synthetic_lattice()],
        [coordinate_text(header=("", "X", "Y", "Z"))],
    )

    assert reference is not None
    assert reference.name_header == ""
    assert len(reference.points) == 4


def test_the_multiplication_sign_is_accepted_as_the_x_header() -> None:
    reference = identify_reference_table(
        [synthetic_lattice()],
        [coordinate_text(header=("POINT", "×", "Y", "Z"))],
    )

    assert reference is not None
    assert reference.axes == ("X", "Y", "Z")


def test_a_table_without_axis_headers_is_rejected() -> None:
    text = coordinate_text(header=("NO", "P.NAME", "QTY", "MATL"))

    assert identify_reference_table([synthetic_lattice()], [text]) is None


def test_one_axis_column_is_not_enough() -> None:
    text = coordinate_text(header=("POINT", "X", "QTY", "MATL"))

    assert identify_reference_table([synthetic_lattice()], [text]) is None


def test_axis_columns_with_no_name_column_to_their_left_are_rejected() -> None:
    text = coordinate_text(header=("X", "Y", "Z", "NOTE"))

    assert identify_reference_table([synthetic_lattice()], [text]) is None


def test_rows_whose_coordinates_are_not_numbers_are_dropped() -> None:
    text = coordinate_text()
    text[(2, 1)] = "see note"
    text[(2, 2)] = "see note"
    text[(2, 3)] = "see note"

    reference = identify_reference_table([synthetic_lattice()], [text])

    assert reference is not None
    assert [point.symbol for point in reference.points] == ["a", "c", "d"]


def test_the_table_bbox_covers_only_its_own_columns() -> None:
    # A coordinate table printed against the title block comes back as one
    # lattice with it, so the reported bbox must not include the neighbour.
    table = synthetic_lattice(columns=6)
    text = coordinate_text()
    text[(0, 4)] = "DWG NO"
    text[(1, 4)] = "747587"

    reference = identify_reference_table([table], [text])

    assert reference is not None
    assert reference.bbox.x1 <= 320


def test_a_capital_misread_follows_the_case_the_table_is_printed_in() -> None:
    text = coordinate_text()
    text[(3, 0)] = "C"

    reference = identify_reference_table([synthetic_lattice()], [text])

    assert reference is not None
    assert [point.symbol for point in reference.points] == ["a", "b", "c", "d"]


def test_the_table_defining_the_most_points_wins() -> None:
    small = synthetic_lattice(rows=4)
    large = synthetic_lattice(top=400, rows=5)
    small_text = {key: value for key, value in coordinate_text().items() if key[0] < 3}

    reference = identify_reference_table(
        [small, large],
        [small_text, coordinate_text()],
    )

    assert reference is not None
    assert len(reference.points) == 4


def test_coordinate_cells_are_reported_for_counter_templates() -> None:
    reference = identify_reference_table([synthetic_lattice()], [coordinate_text()])

    assert reference is not None
    # Four point rows across three axis columns, and never the name column.
    assert len(reference.value_boxes) == 12
    assert all(box.x >= 80 for box in reference.value_boxes)
