"""Fast geometry tests for selection-sized whole-page orchestration."""

from __future__ import annotations

from detection_passes import DetectionTile
from page_scan import (
    build_page_tiles,
    deduplicate_page_candidates,
    map_tile_candidate,
    strict_same_object,
)


def test_page_tiles_cover_both_axes_with_overlap_and_no_tiny_tail() -> None:
    tiles = build_page_tiles((3800, 2600), max_edge=2000, overlap=160)
    x_starts = sorted({tile.x for tile in tiles})
    y_starts = sorted({tile.y for tile in tiles})

    assert x_starts[0] == 0
    assert y_starts[0] == 0
    assert x_starts[-1] + 2000 == 3800
    assert y_starts[-1] + 2000 == 2600
    assert all(
        right - left <= 2000 - 160
        for left, right in zip(x_starts, x_starts[1:])
    )
    assert all(
        bottom - top <= 2000 - 160
        for top, bottom in zip(y_starts, y_starts[1:])
    )
    assert all(tile.width == 2000 and tile.height == 2000 for tile in tiles)


def test_small_page_remains_one_selection_sized_tile() -> None:
    tiles = build_page_tiles((600, 400))

    assert tiles == [DetectionTile(x=0, y=0, width=600, height=400)]


def test_representative_drawing_is_split_into_four_automatic_selections() -> None:
    tiles = build_page_tiles((1263, 893))

    assert len(tiles) == 4
    assert {tile.x for tile in tiles} == {0, 551}
    assert {tile.y for tile in tiles} == {0, 386}
    assert all(tile.width == 712 and tile.height == 507 for tile in tiles)


def test_coordinate_restoration_prefers_the_non_boundary_duplicate() -> None:
    page_size = (3600, 1800)
    left_tile = DetectionTile(x=0, y=0, width=2000, height=1800)
    right_tile = DetectionTile(x=1600, y=0, width=2000, height=1800)
    clipped_view = map_tile_candidate(
        {"x": 1985, "y": 500, "width": 40, "height": 16},
        left_tile,
        tile_index=1,
        page_size=page_size,
        detection_confidence=0.99,
    )
    complete_view = map_tile_candidate(
        {"x": 385, "y": 500, "width": 40, "height": 16},
        right_tile,
        tile_index=2,
        page_size=page_size,
        detection_confidence=0.80,
    )

    assert clipped_view is not None and clipped_view.boundary_review
    assert complete_view is not None and not complete_view.boundary_review
    deduped = deduplicate_page_candidates([clipped_view, complete_view])

    assert deduped == [complete_view]
    assert deduped[0].bbox == {
        "x": 1985.0,
        "y": 500.0,
        "width": 40.0,
        "height": 16.0,
    }


def test_containment_does_not_make_a_large_parent_and_child_duplicates() -> None:
    parent = {"x": 10, "y": 10, "width": 300, "height": 180}
    child = {"x": 40, "y": 50, "width": 40, "height": 14}

    assert not strict_same_object(parent, child)


def test_parallel_neighbouring_dimensions_are_not_overlap_duplicates() -> None:
    left = {"x": 100, "y": 100, "width": 16, "height": 100}
    right = {"x": 125, "y": 100, "width": 16, "height": 100}

    assert not strict_same_object(left, right)


def test_unmatched_boundary_candidate_is_retained_for_review() -> None:
    page_size = (3600, 1800)
    tile = DetectionTile(x=0, y=0, width=2000, height=1800)
    candidate = map_tile_candidate(
        {"x": 1990, "y": 300, "width": 60, "height": 16},
        tile,
        tile_index=1,
        page_size=page_size,
    )

    assert candidate is not None and candidate.boundary_review
    assert deduplicate_page_candidates([candidate]) == [candidate]
