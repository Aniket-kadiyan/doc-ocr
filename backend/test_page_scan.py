"""Fast geometry tests for bounded whole-page detector orchestration."""

from __future__ import annotations

from detection_passes import DetectionTile
from page_scan import (
    assign_candidate_ids,
    bbox_center_in_scope,
    bbox_intersects_scope,
    bbox_overlap_fraction,
    deduplicate_page_candidates,
    map_tile_candidate,
    overlaps_existing_value,
    strict_same_object,
    looks_like_a_small_callout,
    uncovered_gap_candidates,
)


def test_scope_geometry_uses_intersection_for_work_and_centre_for_output() -> None:
    scope = {"x": 100, "y": 50, "width": 80, "height": 60}

    # A processing panel only has to touch the selection: using the complete
    # panel is what gives section mode the same detector context as page mode.
    assert bbox_intersects_scope(
        {"x": 0, "y": 0, "width": 120, "height": 100},
        scope,
    )
    assert not bbox_intersects_scope(
        {"x": 0, "y": 0, "width": 90, "height": 40},
        scope,
    )

    # Publication follows the existing native-PDF rule: a partially clipped
    # object belongs to the selection only when its centre is inside it.
    assert bbox_center_in_scope(
        {"x": 95, "y": 60, "width": 20, "height": 10},
        scope,
    )
    assert not bbox_center_in_scope(
        {"x": 80, "y": 60, "width": 30, "height": 10},
        scope,
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


def _candidate(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    rotation_cw: int = 0,
):
    tile = DetectionTile(x=0, y=0, width=2600, height=1500)
    mapped = map_tile_candidate(
        {"x": x, "y": y, "width": width, "height": height},
        tile,
        tile_index=1,
        page_size=(2600, 1500),
        rotation_cw=rotation_cw,
    )
    assert mapped is not None
    return mapped


def test_an_enlarged_pass_only_speaks_where_the_primary_passes_were_silent() -> None:
    # An enlarged panel resolves "0.5" into "0" and "5" as well as finding the
    # quarter-turn "10" the primary passes missed entirely. This route does no
    # grouping, so the glyphs have to be dropped and only the new object kept.
    primary = [_candidate(500, 400, 120, 44)]
    glyph_of_a_value = _candidate(500, 400, 40, 44)
    second_glyph = _candidate(580, 400, 40, 44)
    new_object = _candidate(1200, 900, 33, 48, rotation_cw=90)

    kept = uncovered_gap_candidates(
        [glyph_of_a_value, second_glyph, new_object],
        primary,
    )

    assert [candidate.bbox for candidate in kept] == [new_object.bbox]


def test_a_gap_object_records_the_only_rotation_that_saw_it() -> None:
    # Which pass found an object is the one reliable signal for a short
    # callout: "10" turned on its side is no taller than it is wide, and both
    # readings of it are real numbers.
    quarter_turn = _candidate(1200, 900, 33, 48, rotation_cw=90)

    assert quarter_turn.rotations_cw == (90,)
    assert quarter_turn.quarter_turn_only


def test_duplicate_views_pool_the_rotations_that_saw_the_object() -> None:
    # The surviving box loses its duplicates, but which passes saw the object
    # is evidence about the object, so it has to outlive the box that carried
    # it -- otherwise every candidate looks quarter-turn-only or none does.
    upright = _candidate(500, 400, 120, 44, rotation_cw=0)
    turned = _candidate(502, 401, 118, 45, rotation_cw=90)

    accepted = deduplicate_page_candidates([upright, turned])

    assert len(accepted) == 1
    assert accepted[0].rotations_cw == (0, 90)
    assert not accepted[0].quarter_turn_only


def test_a_gap_object_has_to_look_like_the_text_a_gap_pass_is_for() -> None:
    # Every box below is one the enlarged passes really offered on
    # 129E01-13300-ID, with what the recogniser made of it.
    recovered_quarter_turn_ten = {"x": 2201, "y": 3157, "width": 34, "height": 50}
    assert looks_like_a_small_callout(recovered_quarter_turn_ten)
    for small_callout in (
        {"x": 97, "y": 3480, "width": 18, "height": 50},
        {"x": 226, "y": 3391, "width": 38, "height": 38},
        {"x": 2034, "y": 686, "width": 25, "height": 24},
    ):
        assert looks_like_a_small_callout(small_callout), small_callout

    # The sheet margin's boilerplate, set in a vertical strip. No primary pass
    # could have missed text this size, so it is not a gap.
    for margin_strip in (
        {"x": 100, "y": 1988, "width": 39, "height": 622},
        {"x": 102, "y": 1582, "width": 38, "height": 566},
        {"x": 191, "y": 1602, "width": 35, "height": 537},
        {"x": 212, "y": 2729, "width": 39, "height": 370},
    ):
        assert not looks_like_a_small_callout(margin_strip), margin_strip

    # Ink that is not writing: an arrowhead or a centre mark, each of which
    # read back as a confident "1".
    for speck in (
        {"x": 2501, "y": 1197, "width": 11, "height": 7},
        {"x": 3355, "y": 1614, "width": 17, "height": 11},
        {"x": 3448, "y": 1641, "width": 19, "height": 16},
    ):
        assert not looks_like_a_small_callout(speck), speck


def test_shape_and_size_are_checked_before_clearance() -> None:
    # A margin strip standing in empty space is still not a value.
    strip = _candidate(100, 500, 39, 622)

    assert uncovered_gap_candidates([strip], []) == []
