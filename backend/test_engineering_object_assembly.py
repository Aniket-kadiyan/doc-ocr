"""Fast geometry/grammar tests for M4 engineering-object assembly."""

from __future__ import annotations

from engineering_object_assembly import plan_engineering_object_assemblies


def _record(
    candidate_id: str,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float = 12,
    *,
    orientation: str = "horizontal",
    rotation: float = 0,
    source_conflict: bool = False,
    table_excluded: bool = False,
) -> dict:
    bbox = {"x": x, "y": y, "width": width, "height": height}
    return {
        "candidate_id": candidate_id,
        "bbox": bbox,
        "polygon": [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        "text": text,
        "recognized": True,
        "table_excluded": table_excluded,
        "result": {
            "text": text,
            "raw_ocr": text,
            "confidence": 0.95,
            "orientation": orientation,
            "rotation": rotation,
            "recognition_source": "ocr",
            "source_conflict": source_conflict,
        },
    }


def test_inline_multiplier_value_and_qualifier_become_one_object() -> None:
    records = [
        _record("C1", "4X", 10, 20, 18),
        _record("C2", "Ø10", 32, 20, 28),
        _record("C3", "THRU", 65, 20, 35),
    ]

    plans = plan_engineering_object_assemblies(records)

    assert len(plans) == 1
    assert plans[0].text == "4X Ø10 THRU"
    assert plans[0].member_indexes == (0, 1, 2)
    assert [child["candidate_id"] for child in plans[0].children] == [
        "C1",
        "C2",
        "C3",
    ]


def test_stacked_tolerance_uses_nominal_as_the_object_anchor() -> None:
    records = [
        _record("C1", "+0.02", 42, 10, 32),
        _record("C2", "25", 40, 24, 20),
        _record("C3", "-0.01", 42, 38, 32),
    ]

    plans = plan_engineering_object_assemblies(records)

    assert len(plans) == 1
    assert plans[0].object_id == "C2"
    assert plans[0].text == "25 +0.02/-0.01"
    assert plans[0].needs_review is False


def test_unsigned_zero_joins_only_after_a_signed_deviation() -> None:
    assembled = plan_engineering_object_assemblies(
        [
            _record("C1", "Ø8", 10, 20, 24, 20),
            _record("C2", "+0.2", 38, 14, 28, 10),
            _record("C3", "0", 38, 27, 10, 10),
        ]
    )
    unqualified = plan_engineering_object_assemblies(
        [
            _record("C1", "Ø8", 10, 20, 24, 20),
            _record("C2", "0", 38, 24, 10, 10),
        ]
    )

    assert assembled[0].text == "Ø8 +0.2/0"
    assert assembled[0].needs_review is False
    assert unqualified == []


def test_two_neighbouring_complete_values_stay_separate() -> None:
    records = [
        _record("C1", "10.0", 10, 20, 30),
        _record("C2", "20.0", 44, 20, 30),
    ]

    assert plan_engineering_object_assemblies(records) == []


def test_explicit_multiplication_operator_assembles_chamfer() -> None:
    records = [
        _record("C1", "2", 10, 20, 10),
        _record("C2", "X", 23, 20, 10),
        _record("C3", "45°", 36, 20, 25),
    ]

    plans = plan_engineering_object_assemblies(records)

    assert len(plans) == 1
    assert plans[0].text == "2 X 45°"
    assert "explicit_multiplication" in plans[0].rule


def test_vertical_fragments_use_their_reading_axis() -> None:
    records = [
        _record("C1", "4X", 30, 10, 12, 18, orientation="vertical", rotation=90),
        _record("C2", "Ø10", 30, 32, 12, 28, orientation="vertical", rotation=90),
        _record("C3", "THRU", 30, 65, 12, 35, orientation="vertical", rotation=90),
    ]

    plans = plan_engineering_object_assemblies(records)

    assert len(plans) == 1
    assert plans[0].text == "4X Ø10 THRU"
    assert plans[0].orientation == "vertical"


def test_table_and_structured_gdt_fragments_are_not_assembled() -> None:
    assert plan_engineering_object_assemblies(
        [
            _record("C1", "4X", 10, 20, 18, table_excluded=True),
            _record("C2", "Ø10", 32, 20, 28),
        ]
    ) == []
    assert plan_engineering_object_assemblies(
        [
            _record("C1", "⌖|Ø0.1|A", 10, 20, 55),
            _record("C2", "+0.1", 68, 20, 28),
        ]
    ) == []


def test_source_conflict_is_preserved_and_forces_review() -> None:
    records = [
        _record("C1", "R", 10, 20, 10, source_conflict=True),
        _record("C2", "0.3", 22, 20, 26),
        _record("C3", "+0.05", 52, 20, 34),
    ]

    plan = plan_engineering_object_assemblies(records)[0]

    assert plan.text == "R0.3 +0.05"
    assert plan.conflict is True
    assert plan.needs_review is True
    assert plan.recognition_evidence["conflict"] is True
