"""Tests for angular-dimension parsing, incl. degree-symbol OCR recovery."""

from __future__ import annotations

from angle_utils import parse_dms
from dimension_digits import normalize_cad_number_string


def test_recovers_degree_read_as_quote():
    # OCR commonly reads the degree ° as a double-quote: 32"20'40" -> 32°20'40"
    assert parse_dms("32\"20'40\"") == "32°20'40\""
    assert normalize_cad_number_string("32\"20'40\"") == "32°20'40\""


def test_clean_and_partial_angles():
    assert parse_dms("32°20'40\"") == "32°20'40\""
    assert parse_dms("32 20'40\"") == "32°20'40\""  # degree dropped entirely
    assert parse_dms("45.5°") == "45.5°"


def test_does_not_invent_angles():
    assert parse_dms("215,37") is None
    assert parse_dms("203.20") is None
    assert parse_dms("40\"") is None  # lone seconds/inch mark, not a DMS triple
    # linear values pass through normalization unchanged (no bogus degree)
    assert normalize_cad_number_string("215.37±0.05") == "215.37±0.05"
    assert "°" not in normalize_cad_number_string("40\"")


if __name__ == "__main__":
    test_recovers_degree_read_as_quote()
    test_clean_and_partial_angles()
    test_does_not_invent_angles()
    print("OK")
