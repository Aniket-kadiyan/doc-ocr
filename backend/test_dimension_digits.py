"""
Tests for the context-aware letter-for-digit repair in ``dimension_digits``.

The earlier rule replaced any table letter that touched a digit. That turned
thread classes, fit grades and grid references into different numbers: "UNC-2B"
became "UNC-28", "6g" became "69", "B1" became "81". A repair now needs the
shape of a number around it, and these tests pin both halves: the misreads it
must still fix, and the notation it must leave alone.
"""

from __future__ import annotations

import pytest

from dimension_digits import correct_numeric_confusables


@pytest.mark.parametrize(
    ("raw", "fixed"),
    [
        ("2l5.37", "215.37"),
        ("1O.0", "10.0"),
        ("O.59", "0.59"),
        ("0.O04", "0.004"),
        ("Ø2O", "Ø20"),
        ("R0.2S0", "R0.250"),
        ("1S.5", "15.5"),
        ("2OO", "200"),
        ("1l4.5", "114.5"),
        ("Ø1O.5±O.1", "Ø10.5±0.1"),
        ("215,37O.05", "215,370.05"),
    ],
)
def test_misread_digits_inside_numbers_are_repaired(raw: str, fixed: str) -> None:
    got, changed = correct_numeric_confusables(raw)
    assert got == fixed
    assert changed


@pytest.mark.parametrize(
    "text",
    [
        "UNC-2B",            # UN thread class, not 28
        "1/2-13 UNC-2B",
        "M54×1.5-6H",        # metric thread, tolerance class
        "4g6g",              # metric thread tolerance classes
        "Ø20H10",            # ISO fit
        "Ø4 H7",
        "SC-3",              # significant-characteristic tag
        "3D",                # a view label, not 30
        "B1",                # a grid or revision reference, not 81
        "REV B2",
        "0.5S",              # trailing S is not evidence of a 5
        "12OKPSI",           # an O inside a word stays a letter
        "SIO",               # a word made of look-alikes is still a word
        "SAE52100",
        "60.8 PCD",
        "10.0 TO INT",
        "CHF0.25×45°",
        "Ø0.148[3.76]REF.",
        "R0.75",
        "A B C",
        "45o",               # a trailing o after an angle is not a zero
    ],
)
def test_engineering_notation_is_left_alone(text: str) -> None:
    got, changed = correct_numeric_confusables(text)
    assert got == text
    assert not changed


def test_empty_input() -> None:
    assert correct_numeric_confusables("") == ("", False)


@pytest.mark.parametrize(
    ("glued", "limits"),
    [
        ("0.62000.612", "0.620/0.612"),
        ("Ø0.62000.612", "Ø0.620/0.612"),
        ("12.5012.48", "12.50/12.48"),
    ],
)
def test_stacked_limit_pair_read_on_one_line_is_separated(glued: str, limits: str) -> None:
    from dimension_digits import normalize_cad_number_string

    assert normalize_cad_number_string(glued) == limits


@pytest.mark.parametrize(
    ("glued", "expected"),
    [
        ("215.370.05", "215.37±0.05"),   # nominal + tolerance, not two limits
        ("0.5000.400", "0.5000.400"),    # 20% apart: not limits of one feature
    ],
)
def test_glued_values_that_are_not_limits_keep_their_old_reading(glued: str, expected: str) -> None:
    from dimension_digits import normalize_cad_number_string

    assert normalize_cad_number_string(glued) == expected


@pytest.mark.parametrize(
    "angle",
    ["90°±0°30'", "6×60°±0°30'", "45°±5°0'", "(8×33°45')"],
)
def test_a_toleranced_angle_keeps_its_minute_mark(angle: str) -> None:
    # Only a bare DMS triple used to keep its ticks; "90°±0°30'" came out as
    # "90°±0°30", which the check sheet cannot turn into a range.
    from dimension_digits import normalize_cad_number_string

    assert normalize_cad_number_string(angle) == angle


def test_a_stray_tick_is_still_dropped() -> None:
    from dimension_digits import normalize_cad_number_string

    assert normalize_cad_number_string("Ø37.5'") == "Ø37.5"
    assert normalize_cad_number_string("'45°") == "45°"
