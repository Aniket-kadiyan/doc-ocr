"""Whole-page filter regressions; no OCR models are loaded."""

from __future__ import annotations

from dataclasses import replace

from page_value_filters import (
    NEVER_BALLOON_RULES,
    PageValueCandidate,
    candidate_is_in_table,
    evaluate_page_value,
    evaluate_scan_value,
    needs_expanded_filter_context,
    normalize_page_value_text,
)


def candidate(
    text: str,
    *,
    x: float = 10,
    y: float = 10,
    width: float = 80,
    height: float = 16,
    context: str = "",
) -> PageValueCandidate:
    return PageValueCandidate(
        text=text,
        bbox={
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        context_text=context,
    )


def test_complete_engineering_values_need_neither_units_nor_geometry() -> None:
    values = (
        "50",
        ".25",
        "25.00",
        "R5.00",
        "Ø24.80",
        "15°±3°",
        "30 +/- 3",
        "30+-3",
        "32°20'40\"",
        "25 +0.1 -0.2",
        "M8",
        "M8 x 1.25",
        "2X Ø10",
        "SS304",
        "TS232224",
        "R0.2 MAX",
        "2:1",
    )
    for text in values:
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, text
        assert decision.rule_name == "engineering_value"


def test_harmless_symbol_and_boundary_noise_is_normalized() -> None:
    cases = {
        "?30º +/- 3º": "30° ± 3°",
        "|R5.00;": "R5.00",
        "12˚±3˚": "12°±3°",
        "30 + - 3": "30 ± 3",
    }

    for source, expected in cases.items():
        normalized = normalize_page_value_text(source)
        decision = evaluate_page_value(candidate(source))

        assert normalized == expected
        assert decision.accepted


def test_noise_is_not_used_to_extract_a_number_from_mixed_text() -> None:
    text = "?ZONE A 25;"

    assert normalize_page_value_text(text) == text
    decision = evaluate_page_value(candidate(text))
    assert not decision.accepted
    assert decision.rule_name == "invalid_engineering_value"


def test_pure_text_is_rejected() -> None:
    decision = evaluate_page_value(candidate("MATERIAL"))

    assert not decision.accepted
    assert decision.rule_name == "no_numeric_component"


def test_direct_never_balloon_values_are_rejected_before_numeric_acceptance() -> None:
    cases = {
        "DETAIL B SCALE 2:1": "detail_view_section",
        "D3TAIL B": "detail_view_section",
        "SC4LE 2:1": "scale_information",
        "15.07.2024": "date",
        "REV1SION 3": "revision_history",
        "MOD1FICATIONS 3": "revision_history",
        "RELEA5ED 3": "revision_history",
        "N0TE 4": "note_information",
        "PART NO TS232224": "document_metadata",
        "SHEET 1 OF 1": "document_metadata",
    }

    for text, rule_name in cases.items():
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted, text
        assert decision.rule_name == rule_name


def test_split_metadata_value_is_rejected_only_when_its_label_is_nearby() -> None:
    label = candidate("PART NUMBER", x=10, y=10, width=90)
    value = candidate("TS232224", x=110, y=10, width=85)

    associated = evaluate_page_value(
        value,
        page_candidates=(label, value),
    )
    standalone = evaluate_page_value(value, page_candidates=(value,))

    assert not associated.accepted
    assert associated.rule_name == "document_metadata"
    assert standalone.accepted


def test_a_drawing_number_beside_its_label_is_not_a_dimension() -> None:
    # A plain ten-digit run reads as a valid linear value, so without this it
    # never got as far as the "DWG NO" printed next to it and the sheet's own
    # drawing number came back as the first balloon on the page.
    label = candidate("DWG NO", x=10, y=10, width=60)
    number = candidate("7475870300", x=80, y=10, width=200)

    decision = evaluate_page_value(number, page_candidates=(label, number))

    assert not decision.accepted
    assert decision.rule_name == "document_metadata"


def test_a_long_bare_number_with_no_label_is_still_a_value() -> None:
    # Length alone proves nothing: dropping it would need a label beside it.
    number = candidate("123456", x=900, y=900, width=140)

    assert evaluate_page_value(number, page_candidates=(number,)).accepted


def test_a_short_integer_dimension_keeps_its_balloon() -> None:
    label = candidate("DWG NO", x=10, y=10, width=60)
    for text in ("21", "140", "12345"):
        value = candidate(text, x=80, y=10, width=60)
        decision = evaluate_page_value(value, page_candidates=(label, value))
        assert decision.accepted, text


def test_distant_metadata_label_does_not_exclude_a_numeric_value() -> None:
    label = candidate("PART NUMBER", x=10, y=10, width=90)
    value = candidate("50", x=700, y=500, width=30)

    decision = evaluate_page_value(
        value,
        page_candidates=(label, value),
    )

    assert decision.accepted


def test_reconstructed_context_filters_a_split_scale_value_only() -> None:
    value = candidate("2:1", context="DETAIL B SCALE 2:1")

    decision = evaluate_page_value(value, page_candidates=(value,))

    assert not decision.accepted
    assert decision.rule_name == "detail_view_section"
    assert needs_expanded_filter_context("2:1", value.bbox)
    assert not needs_expanded_filter_context("50", value.bbox)


def test_nearby_labels_do_not_exclude_strong_dimensions() -> None:
    angle = candidate("30°±3°", x=100, y=100, width=70)
    nearby_labels = (
        candidate("DETAIL B", x=175, y=100, width=70),
        candidate("SCALE", x=100, y=120, width=55),
        candidate("REVISION", x=160, y=120, width=70),
        angle,
    )

    decision = evaluate_page_value(angle, page_candidates=nearby_labels)

    assert decision.accepted
    assert decision.rule_name == "engineering_value"


def test_nearby_labels_exclude_only_context_prone_values() -> None:
    scale_label = candidate("SC4LE", x=10, y=10, width=55)
    ratio = candidate("2:1", x=70, y=10, width=35)
    revision_label = candidate("MODIFICATIONS", x=10, y=50, width=110)
    revision_number = candidate("3", x=125, y=50, width=12)

    scale_decision = evaluate_page_value(
        ratio,
        page_candidates=(scale_label, ratio),
    )
    revision_decision = evaluate_page_value(
        revision_number,
        page_candidates=(revision_label, revision_number),
    )

    assert not scale_decision.accepted
    assert scale_decision.rule_name == "scale_information"
    assert not revision_decision.accepted
    assert revision_decision.rule_name == "revision_history"


def test_isolated_confusable_digits_and_mixed_numeric_text_require_review() -> None:
    for text in ("0", "1", "8"):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted
        assert decision.rule_name == "ambiguous_single_character"

    for text in ("ZONE A 25", "MATERIAL 50", "A B 12.5"):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted
        assert decision.rule_name == "invalid_engineering_value"


def test_each_never_balloon_rule_can_be_disabled_in_code() -> None:
    scale_rules = tuple(
        replace(rule, enabled=False)
        if rule.name == "scale_information"
        else rule
        for rule in NEVER_BALLOON_RULES
    )

    label = candidate("SCALE", x=10, y=10, width=55)
    value = candidate("2:1", x=70, y=10, width=35)
    decision = evaluate_page_value(
        value,
        page_candidates=(label, value),
        rules=scale_rules,
    )

    assert decision.accepted


def test_table_region_is_available_to_both_scope_policies() -> None:
    mask = {"x": 100, "y": 100, "width": 200, "height": 120}
    inside = candidate("50", x=120, y=130, width=30, height=14)

    for scope_kind in ("page", "section"):
        decision = evaluate_scan_value(
            inside,
            scope_kind=scope_kind,
            table_masks=(mask,),
        )
        assert not decision.accepted
        assert decision.rule_name == "table_region"


def test_section_light_filter_keeps_short_values_and_rejects_garbage() -> None:
    accepted = ("0", "1", "6", "8", "50", "105", "Ø6", "R3", "30+-3")
    rejected = ("A", "B", "D", "---", "DETAIL B (1:1)", "ZONE 25 VALUE")

    for text in accepted:
        decision = evaluate_scan_value(candidate(text), scope_kind="section")
        assert decision.accepted, text
        assert decision.rule_name == "section_engineering_value"

    for text in rejected:
        decision = evaluate_scan_value(candidate(text), scope_kind="section")
        assert not decision.accepted, text


def test_mixed_section_rejects_only_candidate_inside_the_table() -> None:
    mask = {"x": 100, "y": 100, "width": 200, "height": 120}
    table_value = candidate("50", x=120, y=130, width=30, height=14)
    drawing_value = candidate("50", x=20, y=30, width=30, height=14)

    assert candidate_is_in_table(table_value.bbox, (mask,))
    assert not candidate_is_in_table(drawing_value.bbox, (mask,))
    decision = evaluate_scan_value(
        drawing_value,
        scope_kind="section",
        table_masks=(mask,),
    )
    assert decision.accepted
    assert decision.rule_name == "section_engineering_value"


def test_a_chamfer_is_a_complete_value() -> None:
    # A chamfer matched neither grammar: the linear one takes a leading count
    # ("4× Ø5") but then wants a plain number, and the angle one takes no
    # prefix at all. So a correctly read chamfer was judged incomplete and
    # demoted to review, on every drawing that carries one.
    for text in (
        "1×45°",
        "1.0×45°",
        "0.25×45°",
        "2X45°",
        "1 × 45°",
        "0.5x30°",
        "CHF1.0×45°",
        "CHF0.25×45°",
        "C1×45°",
        # Written the other way round, which is equally standard.
        "45°×1",
    ):
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, text
        assert decision.rule_name == "engineering_value", text


def test_the_short_chamfer_form_is_a_complete_value() -> None:
    # JIS and ISO drawings print a 45 degree chamfer as its leg alone, so the
    # pattern above -- which needs both the multiplication sign and the degree
    # mark -- matched nothing and a perfectly read "C3" was dropped from the
    # page scan as "not one complete engineering value".
    for text in ("C3", "C0.5", "C 3", "CHF2", "C3MAX", "C2.5 REF"):
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, text
        assert decision.rule_name == "engineering_value", text


def test_a_bare_letter_is_not_a_chamfer() -> None:
    # The letter has to carry a number, which is what keeps a datum letter and
    # a zone label out of the short form above.
    for text in ("C", "CC", "CHF", "A3", "B2"):
        assert not evaluate_page_value(candidate(text)).accepted, text


def test_a_qualifier_needs_no_space_in_front_of_it() -> None:
    # OCR returns "Ø52.5REF" for "Ø52.5 REF". The space-less form used to pass
    # only by falling through to the compact-identifier rule, whose character
    # class has no Ø and no °, so whether a real value was accepted depended
    # on whether it happened to carry a symbol.
    for text in (
        "Ø52.5REF",
        "Ø52.5 REF",
        "Ø52.5REF.",
        "Ø12.5MAX",
        "30.0°TYP",
        "R2.5MIN",
        "8.00THRU",
    ):
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, text


def test_widening_the_grammar_still_rejects_prose() -> None:
    for text in (
        "REFERENCE ONLY",
        "SECTION A-A",
        "SHEET 1 OF 2",
        "DO NOT SCALE",
        "CHAMFER ALL EDGES",
    ):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted, text


def _at(text: str, x: float, y: float, width: float, height: float):
    return PageValueCandidate(
        text=text,
        bbox={"x": x, "y": y, "width": width, "height": height},
    )


# The upright detector pass returns this line as one box; the quarter-turn
# pass cannot join a vertical run of glyphs into a line and returns its
# letters one at a time. Neither box's TEXT is used by the rule: the
# preliminary pass reads a long line of hairline type as an empty string.
TEXT_LINE = _at("", 100, 200, 460, 16)


def _letters_along_the_line(count: int = 8):
    return [
        _at("", 110 + index * 50, 201, 20, 14)
        for index in range(count)
    ]


def test_a_letter_of_a_printed_line_is_not_a_value() -> None:
    # "properties" read back as "Roper0" at this size, and a single letter of
    # it as "031" — three characters with a letter and a digit.
    others = _letters_along_the_line()
    for fragment in (
        _at("Roper0", 250, 201, 70, 15),
        _at("031", 330, 202, 22, 14),
        _at("35", 410, 203, 12, 12),
    ):
        decision = evaluate_page_value(
            fragment,
            page_candidates=[TEXT_LINE, fragment, *others],
        )
        assert not decision.accepted, fragment.text
        assert decision.rule_name == "text_line_fragment", fragment.text


def test_an_ascender_still_counts_as_sitting_on_the_line() -> None:
    # The line box is tight to the x-height; a 'd' or a 'p' stands outside it.
    tall = _at("031", 330, 193, 20, 30)
    decision = evaluate_page_value(
        tall,
        page_candidates=[TEXT_LINE, tall, *_letters_along_the_line()],
    )
    assert decision.rule_name == "text_line_fragment"


def test_two_pieces_are_a_split_callout_not_a_line_of_type() -> None:
    # A detector box spanning two stacked dimensions must not suppress them.
    fused = _at("", 100, 200, 460, 16)
    left = _at("25.4", 110, 201, 180, 14)
    right = _at("31.8", 310, 201, 180, 14)
    decision = evaluate_page_value(
        left,
        page_candidates=[fused, left, right],
    )
    assert decision.accepted, decision.rule_name


def test_a_dimension_beside_a_line_is_still_a_value() -> None:
    value = _at("Ø18.6", 700, 201, 60, 15)
    decision = evaluate_page_value(
        value,
        page_candidates=[TEXT_LINE, value, *_letters_along_the_line()],
    )
    assert decision.accepted, decision.rule_name


def test_a_compact_box_cannot_be_a_line_of_type() -> None:
    # Wide enough to hold several callouts, but not line-shaped.
    block = _at("", 100, 200, 460, 120)
    value = _at("25.4", 150, 240, 60, 16)
    decision = evaluate_page_value(
        value,
        page_candidates=[block, value, *_letters_along_the_line()],
    )
    assert decision.rule_name != "text_line_fragment"


def test_a_value_the_size_of_the_line_is_not_its_fragment() -> None:
    twin = _at("119.0", 100, 200, 450, 16)
    decision = evaluate_page_value(
        twin,
        page_candidates=[TEXT_LINE, twin, *_letters_along_the_line()],
    )
    assert decision.accepted, decision.rule_name


def test_a_misread_word_is_no_longer_a_part_number() -> None:
    # The compact-identifier escape hatch is for ISD-600-500Y, not for what
    # the recogniser makes of three letters at note size.
    for text in ("e11", "listed2", "Roper0", "mus1", "1el", "proper1"):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted, text


def test_a_real_part_number_still_passes() -> None:
    for text in ("ISD-600-500Y", "ISC-A00-008Y", "SCM415", "E545", "R24"):
        decision = evaluate_page_value(candidate(text))
        assert decision.accepted, text


# --- hole callouts -----------------------------------------------------------
#
# A hole is written as a feature term plus the machining instructions that
# apply to it, in glyph notation: ↧ depth, ⌴ counterbore, ⌵ countersink. None
# of it matched any grammar, so on a sheet that dimensions its holes this way
# every callout was rejected as an incomplete value and sent to review.


def test_a_hole_callout_is_one_complete_value() -> None:
    for text in (
        "Ø2.50 ↧13",
        "Ø 2.50 ↧ 13",
        "M3x0.5 - 6H ↧3",
        "M3x0.5 - 6H THRU ALL",
        "2 x Ø2.50 ↧8.50",
        "2 x Ø2.50 THRU ALL",
        "Ø4 H7 ↧10",
        "Ø6.50 ⌴Ø11 ↧4.40",
        "Ø3.50 ⌵Ø7 X 90°",
        "Ø3.50 X 90° NEAR SIDE",
        "13 DEEP",
        # Drill on one printed line, thread on the next, read as one string.
        "Ø2.50 ↧13 M3x0.5 - 6H ↧13",
    ):
        decision = evaluate_page_value(candidate(normalize_page_value_text(text)))
        assert decision.accepted, (text, decision.rule_name)


def test_a_depth_on_its_own_keeps_its_glyph() -> None:
    # The arrow is the whole meaning of the callout: "0.90" alone does not say
    # that the 0.90 is a depth. It used to be stripped as crop-boundary noise.
    assert normalize_page_value_text("↧ 0.90") == "↧0.90"
    assert evaluate_page_value(candidate("↧0.90")).accepted


def test_the_down_arrow_paddle_returns_is_folded_onto_the_depth_glyph() -> None:
    # PaddleOCR has no U+21A7 in its charset and returns U+2193 for the glyph.
    assert normalize_page_value_text("2.50 ↓13") == "2.50 ↧13"
    assert normalize_page_value_text("M3x0.5 - 6H ↓13") == "M3x0.5 - 6H ↧13"


def test_an_arrow_with_no_number_behind_it_is_still_boundary_noise() -> None:
    # A leader arrowhead intruding on a crop. Only an arrow that runs straight
    # into a number is read as a depth.
    assert normalize_page_value_text("→ 25") == "25"
    assert normalize_page_value_text("↑25") == "25"


def test_the_diameter_glyph_paddle_returns_is_folded_anywhere_in_the_text() -> None:
    # U+2205 EMPTY SET is what the recogniser emits for Ø. Only a LEADING one
    # was folded, so a counted callout kept a glyph no grammar recognised.
    assert normalize_page_value_text("2 x ∅2.50 THRU") == "2 x Ø2.50 THRU"
    assert evaluate_page_value(candidate("2 x Ø2.50 THRU")).accepted


def test_a_fit_class_and_a_stacked_deviation_are_complete_values() -> None:
    for text in ("Ø4 H7", "Ø20H10 +0.084/0", "Ø33 +0.012 0", "+0.012 0"):
        decision = evaluate_page_value(candidate(normalize_page_value_text(text)))
        assert decision.accepted, (text, decision.rule_name)


def test_two_fused_numbers_are_still_not_a_value() -> None:
    # The unsigned lower deviation is allowed only behind a signed upper one.
    for text in ("33 0.5", "25 40", "12.5 0.25"):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted, text


def test_prose_is_not_a_hole_callout() -> None:
    for text in (
        "3 PLACES",
        "DO NOT SCALE",
        "REMOVE ALL BURRS 0.2",
        "ALL DIMENSIONS ARE IN mm",
    ):
        decision = evaluate_page_value(candidate(text))
        assert not decision.accepted, text


def test_a_depth_glyph_read_as_a_letter_is_repaired() -> None:
    # The recogniser substitutes "T" for ↧ at small text sizes.
    assert normalize_page_value_text("M3x0.5-6H T4.50") == "M3x0.5-6H↧4.50"
    assert normalize_page_value_text("Ø2.50 T13") == "Ø2.50↧13"


def test_the_letter_repair_leaves_everything_else_alone() -> None:
    # A fit class set tight against its size, a part number, a range, and a
    # phrase that is not a callout at all.
    for text in ("Ø50T7", "ISD-600T500", "20 TO 25", "PART T5", "T10"):
        assert normalize_page_value_text(text) == text, text


def test_a_tolerance_zone_with_its_datums_is_a_value() -> None:
    # Read correctly by the authoritative pass out of a feature control frame,
    # then thrown out by the grammar and sent to review.
    for text in ("Ø0.02A", "Ø0.02 A", "Ø0.1 A B", "Ø0.3 A B C"):
        assert evaluate_page_value(candidate(text)).accepted, text


def test_a_length_beside_a_datum_letter_is_not_one() -> None:
    # The Ø is what separates a tolerance zone from a fused read of a length
    # and the section label next to it.
    for text in ("25 A", "0.5 m", "12.5 B"):
        assert not evaluate_page_value(candidate(text)).accepted, text


def test_an_orphaned_closing_bracket_is_boundary_noise() -> None:
    # A reference dimension's "(" is routinely read as something else.
    assert normalize_page_value_text("Ø20.86)") == "Ø20.86"
    # ...but a balanced pair is left alone, because stripping it leaves
    # something incomplete and the strip is then discarded.
    assert normalize_page_value_text("(20.86)") == "(20.86)"
    assert normalize_page_value_text("1.00[25.40]") == "1.00[25.40]"


# --- view captions label a region, not a value -------------------------------


def _detail_caption_page(value_text: str) -> list[PageValueCandidate]:
    caption = candidate("Detail C", x=1500, y=300, width=200, height=34)
    scale = candidate("SCALE 5:1", x=1500, y=340, width=200, height=34)
    value = candidate(value_text, x=1520, y=450, width=160, height=34)
    return [caption, scale, value]


def test_a_dimension_inside_a_detail_view_survives_its_caption() -> None:
    # The nearby-label radius is sized for a title-block field beside its
    # value — about 240px at 250dpi. A view caption is a heading over a whole
    # region, so every dimension under one was being hard-excluded: no
    # balloon, and no review box either.
    for text in ("Ø20.6 -0.1", "0.30 X 45° TYP.", "Ø2.50 ↧13"):
        page = _detail_caption_page(normalize_page_value_text(text))
        decision = evaluate_page_value(page[-1], page_candidates=page)
        assert decision.rule_name != "detail_view_section", text
        assert decision.accepted, (text, decision.rule_name)


def test_a_damaged_dimension_under_a_caption_goes_to_review_not_oblivion() -> None:
    # "Ø20.6 -0.1" read as "20.6~0.1". It is not a value, but it is not part
    # of the caption either, so it must reach an inspector.
    for text in ("20.6~0.1", "20.86=0.01"):
        page = _detail_caption_page(text)
        decision = evaluate_page_value(page[-1], page_candidates=page)
        assert decision.rule_name == "invalid_engineering_value", text


def test_the_pieces_of_a_split_caption_are_still_excluded() -> None:
    for text in ("C", "A-A", "(3x)", "1:1", "5:1"):
        page = _detail_caption_page(text)
        decision = evaluate_page_value(page[-1], page_candidates=page)
        assert not decision.accepted, text
        assert decision.rule_name in {"detail_view_section", "scale_information"}, (
            text,
            decision.rule_name,
        )


def test_a_caption_reached_through_line_context_is_also_a_heading() -> None:
    # reconstruct_line_context sweeps up the caption on the line above, and
    # the keyword branch reads text and context together — so "R212.5" was
    # excluded by a "SECTION VIEW B-B" it merely sat under.
    value = PageValueCandidate(
        text="R212.5",
        bbox={"x": 2646, "y": 1480, "width": 39, "height": 159},
        context_text="SECTION VIEW B-B",
    )
    decision = evaluate_page_value(value)
    assert decision.accepted, decision.rule_name


def test_a_caption_fragment_reached_through_context_is_still_excluded() -> None:
    piece = PageValueCandidate(
        text="C",
        bbox={"x": 2646, "y": 1480, "width": 39, "height": 40},
        context_text="SECTION VIEW B-B",
    )
    assert not evaluate_page_value(piece).accepted


def test_the_caption_itself_is_excluded_by_its_own_words() -> None:
    for text in ("SECTION VIEW B-B", "Detail D", "Section A-A"):
        assert not evaluate_page_value(candidate(text)).accepted, text
