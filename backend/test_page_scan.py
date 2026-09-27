"""Fast geometry tests for bounded whole-page detector orchestration."""

from __future__ import annotations

from detection_passes import DetectionTile
from page_scan import (
    assign_candidate_ids,
    bbox_overlap_fraction,
    deduplicate_page_candidates,
    map_tile_candidate,
    overlaps_existing_value,
    strict_same_object,
)


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


def test_page_candidate_keeps_polygon_and_receives_stable_id() -> None:
    tile = DetectionTile(x=100, y=50, width=300, height=200)
    candidate = map_tile_candidate(
        {"x": 20, "y": 30, "width": 42, "height": 18},
        tile,
        tile_index=1,
        page_size=(600, 400),
        polygon=[[20, 32], [60, 30], [62, 46], [22, 48]],
    )

    assert candidate is not None
    assert candidate.polygon == (
        (120.0, 82.0),
        (160.0, 80.0),
        (162.0, 96.0),
        (122.0, 98.0),
    )
    [identified] = assign_candidate_ids([candidate])
    assert identified.candidate_id == "C0001"


def test_existing_balloon_overlap_uses_the_smaller_box() -> None:
    detector_box = {"x": 105, "y": 102, "width": 35, "height": 14}
    saved_box = {"x": 100, "y": 98, "width": 46, "height": 22}
    neighbour = {"x": 160, "y": 98, "width": 40, "height": 22}

    assert bbox_overlap_fraction(detector_box, saved_box) > 0.55
    assert overlaps_existing_value(detector_box, (saved_box,))
    assert not overlaps_existing_value(detector_box, (neighbour,))
