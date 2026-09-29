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


def test_tickless_dms_recovered():
    # PaddleOCR drops the ' and " ticks; trailing "|" / "1" is the extension line.
    from angle_utils import parse_dms

    assert parse_dms("32°20 40") == "32°20'40\""
    assert parse_dms("32°20 40|") == "32°20'40\""
    assert parse_dms("32°20 401") == "32°20'40\""
    assert parse_dms("32°20401") == "32°20'40\""
    # Not an angle: no degree mark, no ticks.
    assert parse_dms("32 20 40") is None


def test_strip_angle_tail():
    from angle_utils import strip_angle_tail

    assert strip_angle_tail("12°±3°1") == "12°±3°"
    assert strip_angle_tail("12°±3°|") == "12°±3°"
    assert strip_angle_tail("45°") == "45°"
    assert strip_angle_tail("32°20'40\"") == "32°20'40\""
    assert strip_angle_tail("215.37±0.05") == "215.37±0.05"
