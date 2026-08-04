"""
Unit tests for the auto-segment box clustering (backend/region_cluster.py).

Run: PYTHONPATH=. .venv/bin/python -m pytest test_region_cluster.py
"""

from __future__ import annotations

from region_cluster import (
    cluster_boxes,
    drop_bridge_boxes,
    merge_fragment_clusters,
    merge_overlapping_clusters,
    order_clusters,
    split_mixed_clusters,
    union_bbox,
)


def _box(x, y, w, h, text="t"):
    return {"x": x, "y": y, "w": w, "h": h, "text": text, "conf": 0.9}


def test_empty():
    assert cluster_boxes([]) == []
    assert order_clusters([]) == []


def test_single_box():
    clusters = cluster_boxes([_box(0, 0, 20, 40)])
    assert len(clusters) == 1


def test_vertical_columns_stay_separate():
    """5 vertical dimensions in columns, each split into 2 stacked fragments."""
    boxes = []
    for col, x in enumerate([10, 70, 130, 190, 250]):
        boxes.append(_box(x, 20, 14, 40, f"d{col}a"))
        boxes.append(_box(x, 64, 14, 30, f"d{col}b"))
    clusters = cluster_boxes(boxes)
    assert len(clusters) == 5
    # Every cluster holds exactly the two fragments of one column.
    for c in clusters:
        xs = {b["x"] for b in c}
        assert len(xs) == 1


def test_mixed_orientation():
    """A vertical column next to a 2-fragment horizontal angle -> 2 clusters."""
    boxes = [
        _box(10, 20, 14, 60, "phiCol"),
        _box(80, 40, 30, 12, "ang1"),
        _box(112, 40, 28, 12, "ang2"),
    ]
    clusters = cluster_boxes(boxes)
    assert len(clusters) == 2


def test_perpendicular_nearby_stay_separate():
    """Inflated boxes must not merge a vertical column with a nearby angle."""
    boxes = [
        _box(10, 20, 14, 70, "phiCol"),
        _box(30, 95, 55, 12, "angle"),
    ]
    clusters = cluster_boxes(boxes)
    assert len(clusters) == 2


def test_stacked_vertical_dimensions_stay_separate():
    """Two vertical dimensions in one column with a gap -> 2 clusters."""
    boxes = [
        _box(10, 20, 14, 45, "top"),
        _box(10, 90, 14, 45, "bottom"),
    ]
    clusters = cluster_boxes(boxes)
    assert len(clusters) == 2


def test_split_mixed_orientation_cluster():
    cluster = [
        _box(10, 20, 14, 60, "phi"),
        _box(12, 95, 50, 12, "angle"),
    ]
    split = split_mixed_clusters([cluster])
    assert len(split) == 2


def test_reading_order_top_to_bottom_then_left():
    boxes = [
        _box(200, 10, 14, 40, "tr"),
        _box(10, 10, 14, 40, "tl"),
        _box(10, 200, 14, 40, "bl"),
    ]
    ordered = order_clusters(cluster_boxes(boxes))
    first_texts = [c[0]["text"] for c in ordered]
    assert first_texts == ["tl", "tr", "bl"]


def test_union_bbox():
    ub = union_bbox([_box(10, 20, 14, 40), _box(10, 64, 14, 30)])
    assert ub == {"x": 10, "y": 20, "width": 14, "height": 74}


def test_merge_fragment_clusters():
    """Tiny slivers beside a full vertical dimension merge into one cluster."""
    anchor = [_box(10, 20, 16, 72, "full")]
    sliver = [_box(11, 24, 6, 8, "phi")]
    merged = merge_fragment_clusters([anchor, sliver])
    assert len(merged) == 1
    assert len(merged[0]) == 2


def test_merge_overlapping_clusters_joins_two_liner():
    """A value split from its REF. tag onto a perpendicular axis re-joins."""
    tall = [_box(300, 210, 48, 72, "val")]   # vertical Ø175,32
    wide = [_box(285, 245, 111, 32, "ref")]  # wider REF. tag, overlaps tall
    merged = merge_overlapping_clusters([tall, wide])
    assert len(merged) == 1
    assert len(merged[0]) == 2


def test_merge_overlapping_keeps_distinct_columns():
    """Separate columns never overlap, so they stay distinct."""
    cols = [[_box(90, 250, 35, 190)], [_box(205, 245, 34, 195)], [_box(690, 330, 35, 120)]]
    assert len(merge_overlapping_clusters(cols)) == 3


def test_drop_bridge_boxes_removes_cross_column_connector():
    """A wide horizontal box overlapping two separate vertical columns is dropped."""
    col_a = _box(83, 104, 22, 90)    # Ø174 column (vertical)
    col_b = _box(159, 77, 36, 60)    # Ø175,32 column (vertical)
    bridge = _box(91, 88, 97, 18)    # wide horizontal spanning both
    kept = drop_bridge_boxes([col_a, col_b, bridge])
    assert bridge not in kept
    assert col_a in kept and col_b in kept


def test_drop_bridge_keeps_real_perpendicular_text():
    """A horizontal value (angle) overlapping no separate columns is kept."""
    col_a = _box(90, 250, 35, 190)
    col_b = _box(690, 330, 35, 120)
    angle = _box(300, 400, 110, 30)  # horizontal, overlaps neither column
    kept = drop_bridge_boxes([col_a, col_b, angle])
    assert angle in kept and col_a in kept and col_b in kept


def test_drop_bridge_keeps_fragment_of_one_dimension():
    """A small tag overlapping only ONE vertical column is not a bridge."""
    value = _box(159, 77, 36, 60)    # Ø175,32
    ref = _box(150, 120, 50, 20)     # REF. tag, overlaps only this column
    kept = drop_bridge_boxes([value, ref])
    assert value in kept and ref in kept
