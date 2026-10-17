"""
Tests for the GD&T pass orchestration (backend/gdt_pipeline.py) and its
scoring (backend/benchmarks/gdt_eval.py).

Covers the properties that only appear once the stages are assembled: tiling
and its overlap, deskewing and mapping coordinates back, error isolation, and
the promise that this pass is additive and never fails a document.

Run: PYTHONPATH=. .venv/bin/python -m pytest test_gdt_pipeline.py
"""

from __future__ import annotations

import pytest
from PIL import Image

import gdt_context as ctx
import gdt_pipeline as pipeline
import gdt_registry as reg
from gdt_config import GdtConfig

pytest.importorskip("cv2")

from benchmarks import gdt_eval, gdt_synthetic as syn  # noqa: E402


# --- tiling ------------------------------------------------------------------


def test_a_small_page_is_one_tile():
    assert pipeline.tiles_for(800, 600) == [(0, 0, 800, 600)]


def test_a_large_page_is_tiled_with_overlap():
    cfg = GdtConfig(tile_px=500, tile_overlap_px=100)
    tiles = pipeline.tiles_for(1800, 1200, config=cfg)
    assert len(tiles) > 1
    xs = sorted({t[0] for t in tiles})
    # Consecutive tile origins must advance by less than the tile width, or
    # a symbol on the seam is cut in half in both tiles and found in neither.
    assert all(b - a < 500 for a, b in zip(xs, xs[1:]))


def test_tiles_cover_the_whole_page():
    cfg = GdtConfig(tile_px=400, tile_overlap_px=80)
    width, height = 1500, 900
    tiles = pipeline.tiles_for(width, height, config=cfg)
    assert max(x + w for x, _, w, _ in tiles) == width
    assert max(y + h for _, y, _, h in tiles) == height


# --- deskew ------------------------------------------------------------------


def test_a_square_page_reports_no_skew():
    scene = syn.build("frames")
    assert pipeline.estimate_page_skew(scene.image) is None


def test_a_skewed_page_is_measured_and_corrected():
    scene = syn.with_condition(syn.build("frames"), "skewed")
    angle = pipeline.estimate_page_skew(scene.image)
    assert angle is not None and abs(angle) > 1.0
    corrected, _ = pipeline.deskew_page(scene.image, angle)
    # Correcting by the reported angle must leave nothing behind.
    assert pipeline.estimate_page_skew(corrected) is None


def test_skew_correction_keeps_every_frame_findable():
    """Deskewing must never cost a frame the finder could already see.

    The contour-based finder this replaced found nothing at all on a page
    skewed by 2.5 degrees; the rule-pair finder tolerates that much on its
    own. Deskewing stays, because the compartment split and glyph crops are
    axis-aligned, and this pins the one thing that must hold either way: the
    corrected page yields every frame.
    """
    import fcf_parse

    scene = syn.with_condition(syn.build("frames"), "skewed")
    angle = pipeline.estimate_page_skew(scene.image)
    assert angle is not None and abs(angle) > 1.0
    corrected, _ = pipeline.deskew_page(scene.image, angle)
    assert len(fcf_parse.find_frames(corrected)) == len(scene.frames)


def test_boxes_from_a_deskewed_page_come_back_in_original_coordinates():
    from gdt_detect import iou

    scene = syn.with_condition(syn.build("frames"), "skewed")
    result = pipeline.analyze_page(scene.image)
    assert result.frames, "no frames found on the skewed page"
    assert result.provenance["deskew_deg"] is not None
    # Each found frame must land on a frame of the *original* layout.
    for frame in result.frames:
        best = max(iou(frame.bbox, t.bbox) for t in scene.frames)
        assert best > 0.4, "a frame came back in deskewed coordinates"


def test_deskew_can_be_switched_off():
    scene = syn.build("frames")
    result = pipeline.analyze_page(scene.image, deskew=False)
    assert result.provenance["deskew_deg"] is None


# --- the pass ----------------------------------------------------------------


def test_a_blank_page_returns_an_empty_result_not_an_error():
    result = pipeline.analyze_page(Image.new("RGB", (600, 400), "white"))
    assert result.frames == []
    assert result.errors == []
    assert result.counts["symbols"] == 0


def test_frames_are_found_and_their_symbols_classified():
    scene = syn.build("frames")
    result = pipeline.analyze_page(scene.image)
    assert len(result.frames) == len(scene.frames)
    found = {f.symbol.symbol_id for f in result.frames if f.symbol}
    assert found == {f.symbol_id for f in scene.frames}


def test_a_composite_frame_keeps_its_segments_grouped():
    scene = syn.build("composite_frame")
    result = pipeline.analyze_page(scene.image)
    composite_ids = {f.composite_id for f in result.frames if f.composite_id}
    assert len(composite_ids) == 1
    segments = [f for f in result.frames if f.composite_id]
    assert len(segments) == 2
    assert {f.segment_index for f in segments} == {0, 1}


def test_a_frames_symbol_is_linked_to_its_frame():
    scene = syn.build("frames")
    result = pipeline.analyze_page(scene.image)
    frame_ids = {f.id for f in result.frames}
    linked = [s for s in result.symbols if s.frame_id in frame_ids]
    assert linked, "no symbol was linked to the frame it sits in"


def test_the_pass_can_be_switched_off_entirely():
    cfg = GdtConfig(enabled=False)
    scene = syn.build("frames")
    result = pipeline.analyze_page(scene.image, config=cfg)
    assert result.frames == []
    assert "skipped" in result.provenance


def test_provenance_records_what_the_result_may_be_held_to():
    scene = syn.build("frames")
    result = pipeline.analyze_page(scene.image, source="unit-test")
    provenance = result.provenance
    assert provenance["source"] == "unit-test"
    assert provenance["registry_symbols"] > 0
    assert "config" in provenance
    # It must not claim compliance.
    assert "compliant" in provenance["note"]


# --- error isolation ---------------------------------------------------------


def test_a_failing_ocr_hook_does_not_fail_the_page():
    def exploding_ocr(image):
        raise RuntimeError("model unavailable")

    scene = syn.build("frames")
    result = pipeline.analyze_page(scene.image, ocr_fn=exploding_ocr)
    # Frames are still found and their symbols still read from pixels.
    assert result.frames
    assert any(f.symbol and f.symbol.symbol_id for f in result.frames)


def test_a_failing_page_is_isolated_from_the_rest_of_the_document():
    class Exploding:
        width = 10
        height = 10

        def crop(self, *a, **k):
            raise RuntimeError("corrupt page")

        def convert(self, *a, **k):
            raise RuntimeError("corrupt page")

    scene = syn.build("frames")
    results = pipeline.analyze_document([scene.image, Exploding(), scene.image])
    assert len(results) == 3
    assert results[0].frames and results[2].frames
    assert results[1].errors and results[1].errors[0]["stage"] == "page"


def test_pages_are_numbered_from_one():
    scene = syn.build("frames")
    results = pipeline.analyze_document([scene.image, scene.image])
    assert [r.page for r in results] == [1, 2]
    assert all(f.page == r.page for r in results for f in r.frames)


# --- serialization -----------------------------------------------------------


def test_page_result_serializes_to_json():
    import json

    scene = syn.build("frames")
    payload = pipeline.analyze_page(scene.image).to_dict()
    json.dumps(payload)
    for key in (
        "page", "image_size", "frames", "symbols", "datum_associations",
        "standards_findings", "errors", "counts", "elapsed_ms", "provenance",
    ):
        assert key in payload


def test_every_published_frame_passes_its_own_schema():
    import fcf_parse

    scene = syn.build("frames")
    result = pipeline.analyze_page(scene.image)
    for frame in result.frames:
        assert fcf_parse.validate(frame.to_dict()) == []


# --- evaluation framework ----------------------------------------------------


def test_iou_matches_the_detector_implementation():
    from gdt_detect import iou as detect_iou

    box_a, box_b = (0, 0, 10, 10), (5, 5, 10, 10)
    assert gdt_eval.iou(box_a, box_b) == pytest.approx(detect_iou(box_a, box_b))


def test_a_perfect_run_scores_one():
    class FakeDetection:
        def __init__(self, symbol_id, bbox):
            self.symbol_id = symbol_id
            self.bbox = bbox
            self.status = "confirmed"
            self.alternatives = []

    class FakeTruth:
        def __init__(self, symbol_id, bbox):
            self.symbol_id = symbol_id
            self.bbox = bbox

    report = gdt_eval.Report(name="t")
    truth = [FakeTruth("position", (0, 0, 10, 10))]
    found = [FakeDetection("position", (0, 0, 10, 10))]
    report.total_detections = 1
    gdt_eval.score_symbols(report, truth, found)
    precision, recall, f1 = report.macro
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)


def test_one_detection_cannot_satisfy_two_truths():
    class FakeDetection:
        def __init__(self, symbol_id, bbox):
            self.symbol_id = symbol_id
            self.bbox = bbox
            self.status = "confirmed"
            self.alternatives = []

    class FakeTruth:
        def __init__(self, symbol_id, bbox):
            self.symbol_id = symbol_id
            self.bbox = bbox

    report = gdt_eval.Report(name="t")
    truth = [
        FakeTruth("diameter", (0, 0, 10, 10)),
        FakeTruth("diameter", (2, 2, 10, 10)),
    ]
    found = [FakeDetection("diameter", (1, 1, 10, 10))]
    gdt_eval.score_symbols(report, truth, found)
    score = report.classes["diameter"]
    assert score.true_positives == 1
    assert score.false_negatives == 1


def test_macro_and_micro_differ_when_classes_are_imbalanced():
    """The reason macro is the headline: rare classes must not be hidden."""
    report = gdt_eval.Report(name="t")
    common = report.cls("diameter")
    common.true_positives, common.false_negatives = 40, 0
    rare = report.cls("cylindricity")
    rare.true_positives, rare.false_negatives = 0, 1
    _, macro_recall, _ = report.macro
    _, micro_recall, _ = report.micro
    assert macro_recall == pytest.approx(0.5)
    assert micro_recall > 0.9
    assert macro_recall < micro_recall


def test_localization_is_not_scored_on_a_rotated_page():
    """Axis-aligned truth boxes stop describing a rotated symbol."""
    scene = syn.with_condition(syn.build("symbol_grid"), "skewed")
    report = gdt_eval.evaluate_scene(scene, pipeline.analyze_page)
    assert report.localization_scored is False
    assert any("not scored" in n for n in report.notes)
    assert report.to_dict()["aggregate"]["mean_iou"] is None


def test_the_report_serializes_with_per_class_detail():
    import json

    scene = syn.build("symbol_grid")
    report = gdt_eval.evaluate_scene(scene, pipeline.analyze_page)
    payload = report.to_dict()
    json.dumps(payload)
    assert payload["per_class"], "no per-class breakdown"
    assert "macro_f1" in payload["aggregate"]
    assert "declined_rate" in payload["aggregate"]


def test_top_k_recall_is_never_below_strict_recall():
    """They share a denominator; mixing macro and micro broke this once."""
    scene = syn.build("symbol_grid")
    report = gdt_eval.evaluate_scene(scene, pipeline.analyze_page)
    _, macro_recall, _ = report.macro
    assert report.macro_top_k_recall >= macro_recall - 1e-9


# --- textual GD&T, and its relationship to the graphical pass ----------------


def test_a_textual_control_over_a_frame_is_linked_not_double_counted():
    """Both paths can describe one callout; it must be reported once."""
    scene = syn.build("frames")
    frame = scene.frames[0]
    texts = [
        ctx.TextBox(
            "POSITION Ø0.1 A B",
            (float(frame.bbox[0]), float(frame.bbox[1]),
             float(frame.bbox[2]), float(frame.bbox[3])),
            0.9,
        )
    ]
    result = pipeline.analyze_page(scene.image, texts=texts)
    assert len(result.text_annotations) == 1
    entry = result.text_annotations[0]
    assert entry["duplicate_of_frame"] is True
    assert entry["frame_id"] is not None


def test_a_textual_control_with_no_frame_under_it_stands_alone():
    """The case the graphical pass cannot see: a control written in words."""
    scene = syn.build("frames")
    texts = [ctx.TextBox("FLATNESS 0.05", (5000.0, 5000.0, 120.0, 20.0), 0.9)]
    result = pipeline.analyze_page(scene.image, texts=texts)
    assert len(result.text_annotations) == 1
    entry = result.text_annotations[0]
    assert entry["duplicate_of_frame"] is False
    assert entry["frame_id"] is None
    assert entry["name"] == "Flatness"


def test_disagreement_between_the_two_paths_is_recorded():
    """Agreement is worth knowing; so is its absence."""
    scene = syn.build("frames")
    straightness = next(f for f in scene.frames if f.symbol_id == "straightness")
    texts = [
        ctx.TextBox(
            "POSITION Ø0.1 A B",
            (float(straightness.bbox[0]), float(straightness.bbox[1]),
             float(straightness.bbox[2]), float(straightness.bbox[3])),
            0.9,
        )
    ]
    result = pipeline.analyze_page(scene.image, texts=texts)
    entry = result.text_annotations[0]
    assert entry["agrees_with_frame"] is False


def test_ordinary_text_produces_no_annotations():
    scene = syn.build("frames")
    texts = [
        ctx.TextBox("Ø25.4", (5000.0, 5000.0, 60.0, 20.0), 0.9),
        ctx.TextBox("ROUNDNESS(2PT./180)50µm MAX.", (5000.0, 5100.0, 300.0, 20.0), 0.9),
    ]
    result = pipeline.analyze_page(scene.image, texts=texts)
    assert result.text_annotations == []


def test_text_annotations_carry_their_source_and_confidence():
    scene = syn.build("frames")
    texts = [ctx.TextBox("FLATNESS 0.05", (5000.0, 5000.0, 120.0, 20.0), 0.9)]
    entry = pipeline.analyze_page(scene.image, texts=texts).text_annotations[0]
    assert entry["source_text"] == "FLATNESS 0.05"
    assert entry["method"] == "text_rule"
    assert 0.0 < entry["confidence"] <= 1.0
    assert entry["evidence"]
    assert set(entry["bounding_box"]) == {"x", "y", "width", "height"}


def test_text_annotations_appear_in_the_serialized_payload():
    import json

    scene = syn.build("frames")
    texts = [ctx.TextBox("FLATNESS 0.05", (5000.0, 5000.0, 120.0, 20.0), 0.9)]
    payload = pipeline.analyze_page(scene.image, texts=texts).to_dict()
    json.dumps(payload)
    assert "text_annotations" in payload
    assert payload["counts"]["text_annotations"] == 1


# --- the page scan's view ----------------------------------------------------


def test_scan_regions_publish_a_frame_as_one_gdt_balloon():
    import fcf_parse

    scene = syn.build("frames")
    ocr = lambda image: ("0.5", 0.9)  # noqa: E731
    regions = pipeline.scan_regions(scene.image, ocr_fn=ocr)
    assert len(regions) == len(scene.frames)
    by_y = sorted(regions, key=lambda r: r["bbox"]["y"])
    for region, truth in zip(by_y, scene.frames):
        assert region["type"] == "GD&T" and region["category"] == "GD&T"
        assert region["label"].startswith("GD&T ")
        glyph = reg.get(truth.symbol_id).unicode
        assert region["text"].startswith(glyph), (region["text"], glyph)
        assert "0.5" in region["text"]
        assert region["gdt"]["frames"][0]["type"] == "feature_control_frame"
        assert region["page_filter_rule"] == "feature_control_frame"


def test_a_composite_frame_is_one_scan_region():
    scene = syn.build("composite_frame")
    regions = pipeline.scan_regions(scene.image, ocr_fn=lambda image: ("0.5", 0.9))
    assert len(regions) == 1
    assert len(regions[0]["gdt"]["frames"]) == 2
    assert " / " in regions[0]["text"]
    assert regions[0]["label"] == "GD&T Composite"


def test_a_frame_that_could_not_be_fully_read_asks_for_review():
    scene = syn.build("frames")
    regions = pipeline.scan_regions(scene.image, ocr_fn=lambda image: ("", 0.0))
    assert regions
    assert all(r["needs_review"] for r in regions)
    assert all("tolerance" in r["review_reason"] for r in regions)
    assert all("?" in r["text"] for r in regions)


def test_absorbing_a_frame_replaces_what_the_scan_read_inside_it():
    """The tolerance compartment reads as a perfectly good "0.3" on its own,
    and the rest of the frame came back as an unreadable review box."""
    frame = {
        "candidate_id": "fcf-1",
        "bbox": {"x": 100, "y": 100, "width": 120, "height": 30},
        "text": "⌖ Ø0.3",
        "type": "GD&T",
    }
    inner_value = {"bbox": {"x": 150, "y": 105, "width": 40, "height": 20}, "text": "0.3"}
    elsewhere = {"bbox": {"x": 400, "y": 400, "width": 40, "height": 20}, "text": "12"}
    review = {"bbox": {"x": 101, "y": 101, "width": 118, "height": 28}, "text": ""}
    regions, candidates, delta = pipeline.absorb_frames(
        [inner_value, elsewhere], [review], [frame]
    )
    assert regions == [elsewhere, frame]
    assert candidates == []
    # Two objects gave way to one.
    assert delta == -1
