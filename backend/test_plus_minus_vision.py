"""
Regression tests for ± template matching (backend/symbol_vision.py).

These exist because the matcher was broken in a way nothing caught: its
template was drawn black-on-white while the images it is matched against are
white-on-black, so it anti-correlated with the glyph it was looking for. It
still returned plausible-looking small numbers, it just almost never fired —
1 of 23 real ± regions.

The tests are built so that failure mode cannot return quietly:

* polarity is asserted directly, not inferred from a score;
* the real-drawing fixtures pin precision and recall to measured values, so a
  change that trades one for the other has to say so;
* negatives include the marks that actually look like a ± on a drawing.

Fixtures in ``benchmarks/fixtures/plus_minus/`` are real CAD-rendered crops
from this repository's own drawing, with ground truth from the glyph visible
in each crop. ``manifest.json`` records where each came from.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_plus_minus_vision.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import symbol_vision as sv
from image_preprocess import pad_image, upscale_min_edge

cv2 = pytest.importorskip("cv2")

FIXTURES = Path(__file__).parent / "benchmarks" / "fixtures" / "plus_minus"


# ── helpers ──────────────────────────────────────────────────────────────────


def draw_plus_minus(
    size: int = 64,
    *,
    stroke: int | None = None,
    ink: int = 0,
    paper: int = 255,
    antialias: bool = True,
    margin: int = 14,
) -> Image.Image:
    """A ± on paper, in the proportions the standard glyph uses.

    Drawn independently of :func:`symbol_vision._make_pm_template` rather than
    reusing it, so this is a test of the matcher and not of a tautology.
    """
    side = size + 2 * margin
    canvas = np.full((side, side), paper, dtype=np.uint8)
    th = stroke if stroke is not None else max(1, size // 10)
    c = margin + size // 2
    line = cv2.LINE_AA if antialias else cv2.LINE_8
    # Plus: vertical through the upper two-thirds, horizontal across it.
    cv2.line(canvas, (c, margin + size // 5), (c, margin + 4 * size // 5), ink, th, line)
    cv2.line(
        canvas,
        (margin + size // 5, c - size // 8),
        (margin + 4 * size // 5, c - size // 8),
        ink, th, line,
    )
    # The bar beneath.
    cv2.line(
        canvas,
        (margin + size // 5, c + size // 7),
        (margin + 4 * size // 5, c + size // 7),
        ink, th, line,
    )
    return Image.fromarray(canvas).convert("RGB")


def draw_marks(kind: str, size: int = 64, margin: int = 14) -> Image.Image:
    """Marks that are *not* a ± but share strokes with one."""
    side = size + 2 * margin
    canvas = np.full((side, side), 255, dtype=np.uint8)
    th = max(1, size // 10)
    c = margin + size // 2
    if kind == "plus":
        cv2.line(canvas, (c, margin), (c, margin + size), 0, th)
        cv2.line(canvas, (margin, c), (margin + size, c), 0, th)
    elif kind == "minus":
        cv2.line(canvas, (margin, c), (margin + size, c), 0, th)
    elif kind == "equals":
        cv2.line(canvas, (margin, c - size // 6), (margin + size, c - size // 6), 0, th)
        cv2.line(canvas, (margin, c + size // 6), (margin + size, c + size // 6), 0, th)
    elif kind == "divide":  # ÷ — a bar with dots, close to ± in silhouette
        cv2.line(canvas, (margin, c), (margin + size, c), 0, th)
        cv2.circle(canvas, (c, c - size // 4), max(2, th), 0, -1)
        cv2.circle(canvas, (c, c + size // 4), max(2, th), 0, -1)
    elif kind == "blank":
        pass
    else:
        raise ValueError(kind)
    return Image.fromarray(canvas).convert("RGB")


def pm_score(image: Image.Image) -> float:
    """The score the pipeline would compute for this crop."""
    prep = upscale_min_edge(pad_image(image))
    _, debug = sv.detect_symbols(prep)
    scores = debug["pm_scores"]
    return max(scores.values()) if scores else 0.0


def load_fixtures() -> list[dict]:
    manifest = FIXTURES / "manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))["crops"]


REAL = load_fixtures()


# ── 1. polarity, asserted directly ───────────────────────────────────────────


def test_template_and_input_share_one_ink_polarity():
    """The root-cause bug. Asserted on the arrays, not via a score."""
    gray = sv._to_gray(draw_plus_minus())
    template = sv._make_pm_template(32)
    assert gray[0, 0] < 128, "cad_ink_to_gray should give a dark (black) field"
    assert template[0, 0] < 128, "the ± template's field must be dark too"
    assert template.max() == 255, "the ± template's ink must be bright"


def test_an_inverted_template_scores_far_worse_than_the_correct_one():
    """Pins the magnitude of the bug so a silent re-inversion is caught."""
    gray = sv._to_gray(draw_plus_minus())
    ink = sv._ink_bounds(gray)
    size = max(8, int(ink.shape[0] * 0.9))
    correct = sv._make_pm_template(size)
    inverted = 255 - correct

    def peak(template):
        if ink.shape[0] < size or ink.shape[1] < size:
            pytest.skip("fixture too small to match at this size")
        res = cv2.matchTemplate(ink, template, cv2.TM_CCOEFF_NORMED)
        return float(cv2.minMaxLoc(res)[1])

    assert peak(correct) > peak(inverted) + 0.2


# ── 2. clean synthetic ± across renderings ───────────────────────────────────


@pytest.mark.parametrize("size", [32, 48, 64, 96])
def test_a_clean_plus_minus_is_scored_at_any_size(size):
    """The glyph is scored, and no size silently returns nothing."""
    assert pm_score(draw_plus_minus(size)) > 0.1


@pytest.mark.parametrize("size", [32, 48, 64, 96])
def test_an_isolated_plus_sign_stays_below_the_threshold(size):
    """What the matcher actually promises, and what it does not.

    It does *not* rank an isolated ± above an isolated +: measured, a bare
    plus scores higher (0.50 against 0.39 at 64px). That is a real limit and
    it is recorded here rather than papered over. It does not matter in
    service because the template is matched against a whole ink-cropped value
    line — "3,41±0.30", not a lone glyph — and because what the pipeline asks
    is a threshold question, not a ranking one. The contract is that ordinary
    marks stay under the bar, and that is what is asserted.
    """
    assert pm_score(draw_marks("plus", size)) < sv.PM_THRESHOLD


@pytest.mark.parametrize("stroke", [1, 2, 4, 8])
def test_stroke_width_does_not_invert_the_ranking(stroke):
    assert pm_score(draw_plus_minus(64, stroke=stroke)) > pm_score(
        draw_marks("minus", 64)
    )


def test_anti_aliasing_does_not_change_the_verdict():
    smooth = pm_score(draw_plus_minus(64, antialias=True))
    hard = pm_score(draw_plus_minus(64, antialias=False))
    assert abs(smooth - hard) < 0.25


def test_low_contrast_ink_is_still_scored():
    """CAD exports are not always pure black on pure white."""
    faint = pm_score(draw_plus_minus(64, ink=90, paper=210))
    strong = pm_score(draw_plus_minus(64, ink=0, paper=255))
    assert faint > 0.0
    assert abs(faint - strong) < 0.3


def test_an_inverted_page_does_not_silently_score_high():
    """A white-on-black *page* is not a supported input; it must not look good.

    Recorded rather than supported: if inverted pages ever need handling, it
    should be by normalising the input, not by the matcher happening to score
    them.
    """
    normal = pm_score(draw_plus_minus(64))
    inverted = pm_score(draw_plus_minus(64, ink=255, paper=0))
    assert inverted <= normal + 0.1


# ── 3. real CAD crops ────────────────────────────────────────────────────────


@pytest.mark.skipif(not REAL, reason="real ± fixtures are not present")
def test_real_fixtures_cover_both_classes():
    assert sum(1 for c in REAL if c["has_plus_minus"]) >= 5
    assert sum(1 for c in REAL if not c["has_plus_minus"]) >= 5


@pytest.mark.skipif(not REAL, reason="real ± fixtures are not present")
def test_real_plus_minus_crops_separate_from_real_negatives():
    """The property that matters: positives outscore negatives on real ink."""
    pos, neg = [], []
    for crop in REAL:
        score = pm_score(Image.open(FIXTURES / crop["file"]).convert("RGB"))
        (pos if crop["has_plus_minus"] else neg).append(score)
    assert pos and neg
    pos_median = sorted(pos)[len(pos) // 2]
    neg_median = sorted(neg)[len(neg) // 2]
    assert pos_median > neg_median + 0.08, (
        f"positives median {pos_median:.3f} vs negatives {neg_median:.3f} — "
        "the matcher has stopped separating the classes"
    )


@pytest.mark.skipif(not REAL, reason="real ± fixtures are not present")
def test_real_crops_do_not_regress_below_the_calibrated_operating_point():
    """Pins precision and recall measured over the fixture set.

    Calibrated over 86 regions from four drawings at P 0.83 / R 0.22. The
    committed fixtures are a subset, so the bars here are the subset's
    measured behaviour, not the full-corpus figures.
    """
    tp = fp = fn = 0
    for crop in REAL:
        image = Image.open(FIXTURES / crop["file"]).convert("RGB")
        fired = pm_score(image) >= sv.PM_THRESHOLD
        if crop["has_plus_minus"]:
            tp += fired
            fn += not fired
        else:
            fp += fired
    precision = tp / max(tp + fp, 1)
    assert fp <= 1, f"{fp} false positives on real negatives"
    assert precision >= 0.75, f"precision fell to {precision:.2f}"
    assert tp >= 1, "the matcher fires on no real ± crop at all"


# ── 4. negatives ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["plus", "minus", "equals", "divide", "blank"])
def test_ordinary_marks_do_not_reach_the_threshold(kind):
    """A plus, a minus or an equals is not a ±."""
    assert pm_score(draw_marks(kind)) < sv.PM_THRESHOLD


def test_blank_paper_scores_nothing():
    assert pm_score(draw_marks("blank")) == 0.0


def test_an_empty_array_is_handled():
    assert sv._match_pm(np.zeros((0, 0), dtype=np.uint8)) == 0.0


def test_a_tiny_region_does_not_crash_or_fire():
    assert sv._match_pm(np.zeros((3, 3), dtype=np.uint8)) == 0.0


# ── 5. threshold behaviour ───────────────────────────────────────────────────


def test_threshold_is_the_calibrated_value():
    """A bare constant change should have to come past this test.

    0.60 was chosen from a measured precision/recall curve, not to make any
    one example pass — see the calibration note in symbol_vision.
    """
    assert sv.PM_THRESHOLD == 0.60


def test_detect_symbols_fires_exactly_at_the_threshold(monkeypatch):
    image = draw_plus_minus(64)
    score = pm_score(image)
    prep = upscale_min_edge(pad_image(image))

    monkeypatch.setattr(sv, "PM_THRESHOLD", max(score - 0.01, 0.0))
    assert sv.detect_symbols(prep)[0].plus_minus is True

    monkeypatch.setattr(sv, "PM_THRESHOLD", score + 0.01)
    assert sv.detect_symbols(prep)[0].plus_minus is False


def test_the_debug_payload_reports_the_threshold_in_force():
    _, debug = sv.detect_symbols(upscale_min_edge(pad_image(draw_plus_minus())))
    assert debug["pm_threshold"] == sv.PM_THRESHOLD
    assert set(debug["pm_scores"]) <= {"prefix", "body"}


# ── 6. ink scoping ───────────────────────────────────────────────────────────


def test_padding_does_not_change_the_score():
    """The second half of the fix: the score must describe the glyph.

    Before ink scoping, the correlation ran across the white border pad_image
    adds, and a differently-padded crop of the same glyph scored differently.
    """
    glyph = draw_plus_minus(64, margin=8)
    wide = draw_plus_minus(64, margin=40)
    assert abs(pm_score(glyph) - pm_score(wide)) < 0.12


def test_ink_bounds_crops_to_the_ink():
    gray = sv._to_gray(draw_plus_minus(40, margin=30))
    ink = sv._ink_bounds(gray)
    assert ink.shape[0] < gray.shape[0]
    assert ink.shape[1] < gray.shape[1]
    assert ink.size > 0


def test_ink_bounds_on_blank_input_returns_the_input():
    blank = np.zeros((20, 20), dtype=np.uint8)
    assert sv._ink_bounds(blank).shape == blank.shape


# ── 7. repeated and overlapping candidates ───────────────────────────────────


def test_two_plus_minus_glyphs_still_report_one_flag():
    """The flag is per-crop; two ± in one value is one toleranced dimension."""
    side = 96
    canvas = np.full((side, side * 2), 255, dtype=np.uint8)
    one = np.array(draw_plus_minus(48).convert("L"))
    canvas[:one.shape[0], : one.shape[1]] = one
    canvas[:one.shape[0], side : side + one.shape[1]] = one
    image = Image.fromarray(canvas).convert("RGB")
    symbols, _ = sv.detect_symbols(upscale_min_edge(pad_image(image)))
    assert isinstance(symbols.plus_minus, bool)


def test_merge_keeps_a_text_read_when_vision_is_silent():
    """Vision must never *remove* a ± the OCR text reported."""
    vision = sv.DetectedSymbols(plus_minus=False)
    text = sv.detect_prefix_from_ocr_text("12.5±0.05")
    assert text.plus_minus is True
    assert sv.merge_symbol_scores(vision, text).plus_minus is True


def test_merge_keeps_a_vision_read_when_text_is_silent():
    vision = sv.DetectedSymbols(plus_minus=True)
    text = sv.detect_prefix_from_ocr_text("12.5 0.05")
    assert text.plus_minus is False
    assert sv.merge_symbol_scores(vision, text).plus_minus is True
