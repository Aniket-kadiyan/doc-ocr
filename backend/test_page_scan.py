"""Fast geometry tests for bounded whole-page detector orchestration."""

from __future__ import annotations

from detection_passes import DetectionTile
from page_scan import (
    PAGE_SCAN_ROTATIONS_CW,
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


def test_small_page_remains_one_detector_tile() -> None:
    tiles = build_page_tiles((600, 400))

    assert tiles == [DetectionTile(x=0, y=0, width=600, height=400)]


def test_representative_drawing_uses_one_page_detector_tile() -> None:
    tiles = build_page_tiles((1263, 893))

    assert tiles == [DetectionTile(x=0, y=0, width=1263, height=893)]


def test_high_resolution_page_uses_at_most_four_balanced_tiles() -> None:
    tiles = build_page_tiles((4200, 3000))

    assert len(tiles) == 4
    assert {tile.x for tile in tiles} == {0, 2020}
    assert {tile.y for tile in tiles} == {0, 1440}
    assert all(tile.width == 2180 and tile.height == 1560 for tile in tiles)
    assert len(tiles) * len(PAGE_SCAN_ROTATIONS_CW) == 8


def test_tiny_page_does_not_require_an_overlap_smaller_than_the_page() -> None:
    assert build_page_tiles((80, 40)) == [
        DetectionTile(x=0, y=0, width=80, height=40)
    ]


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


def test_same_tile_orientation_duplicates_are_removed() -> None:
    tile = DetectionTile(x=0, y=0, width=1263, height=893)
    zero_degree = map_tile_candidate(
        {"x": 100, "y": 200, "width": 40, "height": 16},
        tile,
        tile_index=1,
        page_size=(1263, 893),
        pass_index=1,
        rotation_cw=0,
    )
    ninety_degree = map_tile_candidate(
        {"x": 101, "y": 200, "width": 39, "height": 16},
        tile,
        tile_index=1,
        page_size=(1263, 893),
        pass_index=2,
        rotation_cw=90,
    )

    assert zero_degree is not None and ninety_degree is not None
    assert len(deduplicate_page_candidates([zero_degree, ninety_degree])) == 1
