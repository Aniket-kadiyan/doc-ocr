"""Tests for segment_quality.is_segment_worthy."""

from __future__ import annotations

from segment_quality import dedupe_regions, is_segment_worthy


def test_accepts_full_dimensions():
    assert is_segment_worthy("Ø215.37±0.05")
    assert is_segment_worthy("32°20'40\"")
    assert is_segment_worthy("Ø174.07±0.05")


def test_rejects_fragments():
    assert not is_segment_worthy(".05")
    assert not is_segment_worthy("05")
    assert not is_segment_worthy("215.")
    assert not is_segment_worthy("174.")


def _r(text, x, y, w, h, conf=0.9):
    return {"text": text, "confidence": conf, "bbox": {"x": x, "y": y, "width": w, "height": h}}


def test_dedupe_suppresses_overlapping_duplicate():
    # Ø175,32 column detected as a proper box and an overlapping sliver.
    regions = [
        _r("Ø175.32REF.", 355, 215, 60, 130, 0.92),
        _r("175REF", 350, 250, 111, 32, 0.71),
    ]
    out = dedupe_regions(regions)
    assert [r["text"] for r in out] == ["Ø175.32REF."]


def test_dedupe_keeps_distinct_columns():
    regions = [
        _r("Ø215,37±0.05", 90, 250, 35, 190),
        _r("Ø174.07±0.05", 205, 245, 34, 195),
        _r("203.20", 690, 330, 35, 120),
    ]
    assert len(dedupe_regions(regions)) == 3


def test_dedupe_preserves_reading_order():
    regions = [
        _r("Ø215,37±0.05", 90, 250, 35, 190),
        _r("Ø175.32REF.", 355, 215, 60, 130, 0.92),
        _r("175REF", 350, 250, 111, 32, 0.71),  # dup of the previous
        _r("203.20", 690, 330, 35, 120),
    ]
    out = [r["text"] for r in dedupe_regions(regions)]
    assert out == ["Ø215,37±0.05", "Ø175.32REF.", "203.20"]


if __name__ == "__main__":
    test_accepts_full_dimensions()
    test_rejects_fragments()
    test_dedupe_suppresses_overlapping_duplicate()
    test_dedupe_keeps_distinct_columns()
    test_dedupe_preserves_reading_order()
    print("OK")


def test_is_annotation_note():
    from segment_quality import is_annotation_note
    assert is_annotation_note("(BOTH SIDES)")
    assert is_annotation_note("BOTH SIDES")
    assert is_annotation_note("TYP")
    assert is_annotation_note("SIDES")
    # A fused box that still carries a real value must be kept.
    assert not is_annotation_note("0.5×45 BOTH SIDES 123")
    assert not is_annotation_note("Ø20.5")
    assert not is_annotation_note("45°")
    assert not is_annotation_note("")


def test_short_radius_is_worthy():
    assert is_segment_worthy("R1")
    assert is_segment_worthy("R2.5")
    assert not is_segment_worthy("R")


def test_finish_symbol_letters_are_notes():
    from segment_quality import is_annotation_note

    for t in ("W", "VV", "NV", "vvv", "▽▽"):
        assert is_annotation_note(t), t
    # Real values and longer words are not.
    for t in ("WV0.5", "NOM.", "R1", "45°±3°", "VVVV"):
        assert not is_annotation_note(t), t


def test_strip_foreign_glyphs():
    from segment_quality import strip_foreign_glyphs

    assert strip_foreign_glyphs("四1") == "1"
    assert strip_foreign_glyphs("Ø215.37±0.05") == "Ø215.37±0.05"
    assert strip_foreign_glyphs("//0.03B ⟂ ∠ ◎ ⌖ ▽") == "//0.03B ⟂ ∠ ◎ ⌖ ▽"


def test_has_dimension_value_keeps_real_dimensions():
    from segment_quality import has_dimension_value

    for text in (
        "Ø215.37±0.05", "32°20'40\"", "R5.00", "15-0.2", "R10±2", "114.5±0.05",
        "Ø33-0.05-0.1", "5°±0°30", "0.5×45°", "Ø20H10+0.084/0", "20×19×1",
        "INV.S.20×19×1", "12.55", "R1", "Ø23.5-0.05", "1.28[32.51]", "3,245",
    ):
        assert has_dimension_value(text), text


def test_has_dimension_value_drops_text_that_is_not_a_dimension():
    """Digits alone do not make a balloon worth placing."""
    from segment_quality import has_dimension_value

    cases = {
        "no digits at all": "SHARP",
        "a date": "07.08.2020",
        "a date with dashes": "Dat.:23-11-2018",
        "a part code": "NEIC00052001",
        "a material code": "SAE52100/SUJ2",
        "a drawing number": "B51801006.020",
        "a labelled table row": "Mat.:56-5-2",
        "a scale": "SCALE 2:1",
        "an opening word": "ROUNDNESS(2PT./180)",
        "a revision note": "65432HRC MODIFIED AS PART NO.&HRC MC1",
        "a hardness range": "Rc58~63(AFTER H/T)2",
        "a tolerance with no value": "±0.2",
        "an unresolved glyph": "1.2?20—1",
    }
    for why, text in cases.items():
        assert not has_dimension_value(text), f"{why}: {text!r}"


def test_deviation_pair_is_not_read_as_a_date():
    """Ø33-0.05-0.1 has the shape of a date but is a diameter."""
    from segment_quality import has_dimension_value

    assert has_dimension_value("Ø33-0.05-0.1")
    assert has_dimension_value("33-0.1-0.05")
