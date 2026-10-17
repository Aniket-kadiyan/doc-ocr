"""
Tests for GD&T glyph detection (backend/gdt_detect.py, gdt_templates.py).

These need cv2 and numpy but no OCR model and no network, so they belong in
the fast suite. Symbols are rendered from the same templates the detector
matches against, which makes these a test of the *machinery* — normalisation,
thresholds, the ambiguity rules, coordinate transforms — rather than a claim
about real drawings. The claim about drawings is what the synthetic benchmark
in ``benchmarks/`` measures, and it is explicit about its own limits.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_gdt_detect.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import gdt_detect as detect
import gdt_registry as reg
import gdt_templates as tpl

cv2 = pytest.importorskip("cv2")


# --- helpers -----------------------------------------------------------------


def paper(
    template: str,
    *,
    size: int = 44,
    pad: int = 14,
    blur: int = 0,
    noise: float = 0.0,
    seed: int = 7,
) -> Image.Image:
    """Render one glyph as black ink on white paper, optionally degraded."""
    glyph = 255 - tpl.render(template, size)
    # render() floors the size, so trust the array rather than the request.
    drawn = glyph.shape[0]
    canvas = np.full((drawn + 2 * pad, drawn + 2 * pad), 255, np.uint8)
    canvas[pad : pad + drawn, pad : pad + drawn] = glyph
    if blur:
        canvas = cv2.GaussianBlur(canvas, (blur, blur), 0)
    if noise:
        rng = np.random.RandomState(seed)
        canvas = np.clip(
            canvas.astype(np.int16) + rng.normal(0, noise, canvas.shape), 0, 255
        ).astype(np.uint8)
    return Image.fromarray(canvas).convert("RGB")


TEMPLATED = sorted(
    {s.template for s in reg.ALL_SYMBOLS if s.template and tpl.has_template(s.template)}
)


# --- templates ---------------------------------------------------------------


def test_every_registry_template_has_a_builder():
    missing = {
        s.template
        for s in reg.ALL_SYMBOLS
        if s.template and not tpl.has_template(s.template)
    }
    assert not missing, f"registry names templates with no builder: {missing}"


@pytest.mark.parametrize("name", TEMPLATED)
def test_each_template_draws_a_plausible_amount_of_ink(name):
    """A blank template silently matches nothing; a solid one matches all."""
    arr = tpl.render(name, 48)
    ink = float((arr > 127).mean())
    assert 0.02 < ink < 0.6, f"{name} ink fraction {ink:.3f}"


def test_templates_are_ink_white_on_black():
    """The polarity cad_ink_to_gray produces. Inverting it breaks everything."""
    arr = tpl.render("circularity", 48)
    assert arr[0, 0] == tpl.BG == 0
    assert arr.max() == tpl.INK == 255


def test_template_matches_its_own_rendering_almost_perfectly():
    from image_preprocess import cad_ink_to_gray

    gray = np.array(cad_ink_to_gray(paper("circularity")).convert("L"))
    norm, _ = detect.normalize_glyph(gray)
    assert detect.compare_normalized(norm, "circularity") > 0.9


# --- normalisation -----------------------------------------------------------


def test_normalize_glyph_returns_the_ink_box_not_the_crop():
    """The published bbox must be the symbol, not the region it was found in."""
    image = paper("circularity", size=40, pad=20)
    gray = detect.to_gray(image)
    norm, ink_box = detect.normalize_glyph(gray)
    assert norm.shape == (detect.NORM_PX, detect.NORM_PX)
    x, y, w, h = ink_box
    assert 15 <= x <= 25 and 15 <= y <= 25
    assert 25 <= w <= 45 and 25 <= h <= 45


def test_normalize_glyph_is_none_on_blank_paper():
    blank = Image.new("RGB", (60, 60), "white")
    norm, box = detect.normalize_glyph(detect.to_gray(blank))
    assert norm is None and box is None


def test_despeckle_keeps_a_thin_stroke_and_drops_isolated_noise():
    """The brief forbids preprocessing that removes thin lines."""
    binary = np.zeros((60, 60), np.uint8)
    binary[30, 5:55] = 255          # a one-pixel-tall stroke
    binary[5, 5] = 255              # a speck
    binary[50, 44] = 255            # another speck
    cleaned = detect.despeckle(binary)
    assert cleaned[30, 5:55].all(), "the thin stroke was destroyed"
    assert cleaned[5, 5] == 0 and cleaned[50, 44] == 0


# --- classification ----------------------------------------------------------


@pytest.mark.parametrize("name", TEMPLATED)
def test_each_glyph_is_either_named_correctly_or_offered_as_an_alternative(name):
    """The thing that must never happen is a confident *wrong* answer."""
    owners = [s.id for s in reg.ALL_SYMBOLS if s.template == name]
    det = detect.classify_crop(paper(name))
    assert det.symbol_id is not None, f"{name} was not recognized at all"
    offered = {det.symbol_id, *(a.symbol_id for a in det.alternatives)}
    assert offered & set(owners), (
        f"{name} produced {det.symbol_id!r} with alternatives "
        f"{[a.symbol_id for a in det.alternatives]}, none of which is {owners}"
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("circularity", "circularity"),
        ("position", "position"),
        ("perpendicularity", "perpendicularity"),
        ("concentricity", "concentricity"),
        ("flatness", "flatness"),
        ("cylindricity", "cylindricity"),
        ("angularity", "angularity"),
        ("countersink", "countersink"),
        ("symmetry", "symmetry"),
    ],
)
def test_distinctive_glyphs_are_named_confidently(name, expected):
    det = detect.classify_crop(paper(name))
    assert det.symbol_id == expected
    assert det.status == detect.STATUS_CONFIRMED
    assert det.confidence > 0.5


def test_symbols_drawn_with_one_glyph_are_reported_ambiguous():
    """Counterbore and spotface are the same mark; shape cannot decide."""
    det = detect.classify_crop(paper("counterbore"))
    offered = {det.symbol_id, *(a.symbol_id for a in det.alternatives)}
    assert {"counterbore", "spotface"} <= offered
    assert det.status == detect.STATUS_AMBIGUOUS
    assert any("same glyph" in w for w in det.warnings)


def test_blank_input_is_unknown_not_a_guess():
    det = detect.classify_crop(Image.new("RGB", (40, 40), "white"))
    assert det.symbol_id is None
    assert det.status == detect.STATUS_UNKNOWN
    assert det.confidence == 0.0


def test_allowed_restricts_the_field():
    """A frame's first compartment may only hold a characteristic."""
    det = detect.classify_crop(
        paper("circularity"), allowed=["flatness", "straightness"]
    )
    assert det.symbol_id in {"flatness", "straightness", None}


def test_a_legacy_control_is_named_with_its_retirement_noted():
    det = detect.classify_crop(paper("concentricity"))
    assert det.symbol_id == "concentricity"
    assert any("retired in ASME Y14.5-2018" in w for w in det.warnings)


def test_an_iso_only_modifier_says_so():
    det = detect.classify_crop(paper("circled_E"), allowed=["envelope_requirement"])
    assert det.symbol_id == "envelope_requirement"
    assert any("ISO" in w for w in det.warnings)


@pytest.mark.parametrize("blur,noise", [(5, 0.0), (0, 28.0), (5, 22.0)])
def test_degraded_glyphs_never_produce_a_confident_wrong_answer(blur, noise):
    for name in ("circularity", "position", "perpendicularity", "flatness"):
        owners = [s.id for s in reg.ALL_SYMBOLS if s.template == name]
        det = detect.classify_crop(paper(name, blur=blur, noise=noise))
        if det.status == detect.STATUS_CONFIRMED:
            assert det.symbol_id in owners, (
                f"{name} under blur={blur} noise={noise} was confidently "
                f"called {det.symbol_id}"
            )


def test_a_tiny_glyph_is_flagged_rather_than_silently_read():
    det = detect.classify_crop(paper("circularity", size=7, pad=4))
    assert det.status != detect.STATUS_CONFIRMED or any(
        "too small" in w or "below the" in w for w in det.warnings
    )


# --- OCR normalization -------------------------------------------------------


def test_an_exact_glyph_scores_above_a_bare_letter():
    _, strong = detect.normalize_ocr("⌖")      # ⌖ position
    _, weak = detect.normalize_ocr("O")
    assert strong and weak
    assert max(c.score for c in strong) > max(c.score for c in weak)


def test_a_shared_ocr_string_spreads_across_its_claimants():
    _, candidates = detect.normalize_ocr("O")
    ids = {c.symbol_id for c in candidates}
    assert {"circularity", "diameter"} <= ids
    # None of them is strong enough on its own to confirm anything.
    assert all(c.score < 0.5 for c in candidates)


def test_empty_ocr_yields_nothing():
    canonical, candidates = detect.normalize_ocr("")
    assert canonical is None and candidates == []


# --- decisions ---------------------------------------------------------------


def test_a_close_rival_forces_ambiguity_even_above_the_accept_bar():
    from gdt_config import GdtConfig

    cfg = GdtConfig()
    candidates = [
        detect.Candidate("position", cfg.accept_confidence + 0.2, detect.M_TEMPLATE),
        detect.Candidate("circularity", cfg.accept_confidence + 0.18, detect.M_TEMPLATE),
    ]
    status, confidence, alternatives = detect.decide(candidates, config=cfg)
    assert status == detect.STATUS_AMBIGUOUS
    assert alternatives and alternatives[0].symbol_id == "circularity"


def test_a_clear_winner_is_confirmed():
    from gdt_config import GdtConfig

    cfg = GdtConfig()
    candidates = [
        detect.Candidate("position", 0.9, detect.M_TEMPLATE),
        detect.Candidate("circularity", 0.3, detect.M_TEMPLATE),
    ]
    status, _, _ = detect.decide(candidates, config=cfg)
    assert status == detect.STATUS_CONFIRMED


def test_everything_below_the_floor_is_unknown():
    from gdt_config import GdtConfig

    cfg = GdtConfig()
    candidates = [detect.Candidate("position", cfg.ambiguous_confidence - 0.1, "t")]
    status, _, _ = detect.decide(candidates, config=cfg)
    assert status == detect.STATUS_UNKNOWN


def test_agreement_between_methods_raises_confidence_without_reaching_certainty():
    merged = detect.combine(
        [
            [detect.Candidate("position", 0.7, detect.M_TEMPLATE)],
            [detect.Candidate("position", 0.6, detect.M_OCR)],
        ]
    )
    assert len(merged) == 1
    assert 0.7 < merged[0].score < 1.0
    assert "+" in merged[0].method


# --- geometry ----------------------------------------------------------------


def test_transform_maps_a_box_back_to_page_coordinates():
    t = detect.Transform(offset_x=100, offset_y=50, scale_x=2.0, scale_y=2.0)
    assert t.apply((10, 20, 30, 40)) == (105.0, 60.0, 15.0, 20.0)


def test_identity_transform_changes_nothing():
    assert detect.IDENTITY.apply((3, 4, 5, 6)) == (3.0, 4.0, 5.0, 6.0)


def test_iou_of_identical_boxes_is_one_and_disjoint_is_zero():
    assert detect.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert detect.iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0


def test_find_glyph_boxes_admits_a_wide_stroke_symbol():
    """Straightness is one wide, low stroke — an aspect filter must not eat it."""
    boxes = detect.find_glyph_boxes(paper("straightness", size=48, pad=12))
    assert boxes, "no proposal for a straightness stroke"


def test_merge_keeps_repeated_symbols_and_drops_overlapping_duplicates():
    def make(x):
        return detect.Detection(
            symbol_id="diameter", name="Diameter",
            status=detect.STATUS_CONFIRMED, confidence=0.8,
            bbox=(x, 0, 20, 20),
        )

    far_apart = detect.merge_detections([make(0), make(500), make(1000)])
    assert len(far_apart) == 3, "legitimate repeats were merged away"

    duplicates = detect.merge_detections([make(0), make(1)])
    assert len(duplicates) == 1
    assert duplicates[0].evidence.get("merged_duplicates") == 1


def test_merge_does_not_combine_different_symbols_that_overlap():
    a = detect.Detection("diameter", "Diameter", detect.STATUS_CONFIRMED, 0.8, (0, 0, 20, 20))
    b = detect.Detection("circularity", "Circularity", detect.STATUS_CONFIRMED, 0.8, (0, 0, 20, 20))
    assert len(detect.merge_detections([a, b])) == 2


# --- serialization -----------------------------------------------------------


def test_detection_to_dict_carries_everything_the_brief_requires():
    import json

    det = detect.classify_crop(paper("position"), ocr_text="+")
    payload = det.to_dict()
    for key in (
        "symbol_id", "name", "status", "confidence", "bounding_box", "page",
        "method", "raw_ocr", "normalized", "alternatives", "associated_text",
        "value", "warnings",
    ):
        assert key in payload, f"missing {key}"
    assert payload["raw_ocr"] == "+"
    assert set(payload["bounding_box"]) == {"x", "y", "width", "height"}
    json.dumps(payload)


# --- real drawn glyphs, not rendered from the templates ----------------------
#
# Everything above renders a glyph with `tpl.render(name)` and matches it
# against `canonical_template(name)` — the same artwork on both sides. That is
# a test of the matching machinery and *not* of whether a template resembles
# the symbol it claims to be, and the difference is not academic: the
# concentricity template shipped with an inner ring so small it closed up,
# scored 0.446 against a real ◎ (below the 0.52 candidate bar), and classified
# a real feature control frame's symbol as unknown — while passing every test
# in this file.

GLYPHS = Path(__file__).parent / "benchmarks" / "fixtures" / "glyphs"


def _real_glyphs() -> list[dict]:
    manifest = GLYPHS / "manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))["glyphs"]


REAL_GLYPHS = _real_glyphs()


@pytest.mark.skipif(not REAL_GLYPHS, reason="no real glyph fixtures present")
@pytest.mark.parametrize(
    "spec", REAL_GLYPHS, ids=[g["file"] for g in REAL_GLYPHS]
)
def test_a_real_drawn_glyph_is_identified(spec):
    """A glyph cropped from an actual drawing must classify correctly."""
    image = Image.open(GLYPHS / spec["file"]).convert("RGB")
    allowed = [s.id for s in reg.FRAME_CHARACTERISTICS]
    det = detect.classify_crop(image, allowed=allowed)
    assert det.symbol_id == spec["symbol_id"], (
        f"{spec['file']} ({spec['note']}) classified as {det.symbol_id!r}"
    )
    assert det.status == detect.STATUS_CONFIRMED


@pytest.mark.skipif(not REAL_GLYPHS, reason="no real glyph fixtures present")
@pytest.mark.parametrize(
    "spec", REAL_GLYPHS, ids=[g["file"] for g in REAL_GLYPHS]
)
def test_a_real_glyph_clears_the_candidate_bar_against_its_template(spec):
    """Pins the shape score, which is what silently fell below the bar."""
    from gdt_config import get_config

    image = Image.open(GLYPHS / spec["file"]).convert("RGB")
    norm, _ = detect.normalize_glyph(detect.to_gray(image))
    assert norm is not None
    template = reg.get(spec["symbol_id"]).template
    score = detect.compare_normalized(norm, template)
    assert score >= get_config().template_min_score, (
        f"{spec['file']} scores {score:.3f} against its own template"
    )
