"""Progress instrumentation tests that do not load PaddleOCR models."""

from __future__ import annotations

from PIL import Image

from ocr_pipeline import OcrPipeline
from page_layout import LayoutBox, LayoutPanel, PageLayout


def test_segment_reports_real_stages_and_recognition_counters() -> None:
    pipeline = object.__new__(OcrPipeline)

    def detect_regions(_image, *, progress_callback=None, **_kwargs):
        if progress_callback is not None:
            progress_callback(
                phase="proposing",
                completed=0,
                total=1,
                state="running",
                label="source-resolution morphology proposals",
                proposals=0,
                deskew_angle=0.0,
                pass_current=1,
                pass_total=1,
                tile_current=1,
                tile_total=1,
            )
            progress_callback(
                phase="proposing",
                completed=1,
                total=1,
                state="completed",
                label="source-resolution morphology proposals",
                proposals=1,
                deskew_angle=0.0,
                pass_current=1,
                pass_total=1,
                tile_current=1,
                tile_total=1,
            )
            progress_callback(
                phase="detecting",
                completed=0,
                total=2,
                state="running",
                label="source contrast, 0 deg, 1100px",
                proposals=1,
                deskew_angle=0.0,
                pass_current=1,
                pass_total=2,
                tile_current=1,
                tile_total=1,
            )
            progress_callback(
                phase="detecting",
                completed=1,
                total=2,
                state="completed",
                label="source contrast, 0 deg, 1100px",
                proposals=1,
                deskew_angle=0.0,
                pass_current=1,
                pass_total=2,
                tile_current=1,
                tile_total=1,
            )
            progress_callback(
                phase="detecting",
                completed=1,
                total=2,
                state="running",
                label="source contrast, 90 deg, 1100px",
                proposals=1,
                deskew_angle=0.0,
                pass_current=2,
                pass_total=2,
                tile_current=1,
                tile_total=1,
            )
            progress_callback(
                phase="detecting",
                completed=2,
                total=2,
                state="completed",
                label="source contrast, 90 deg, 1100px",
                proposals=1,
                deskew_angle=0.0,
                pass_current=2,
                pass_total=2,
                tile_current=1,
                tile_total=1,
            )
        return [
            {
                "x": 10.0,
                "y": 10.0,
                "w": 40.0,
                "h": 12.0,
                "text": "25",
                "conf": 0.9,
            }
        ]

    pipeline.detect_regions = detect_regions
    pipeline._expand_clusters = lambda _image, clusters, **_kwargs: clusters
    pipeline.recognize = lambda _image, **_kwargs: {
        # Section mode must pass through short/plain values that the former
        # is_segment_worthy policy rejected before publication.
        "text": "50",
        "confidence": 0.98,
        "type": "Linear",
        "orientation": "horizontal",
        "rotation": 0,
        "needs_review": False,
        "agreement": 1.0,
        "engine": "paddleocr",
        "symbols_detected": {},
    }
    pipeline._complete_angle_regions = lambda _image, regions: regions
    events: list[dict] = []

    result = pipeline.segment(
        Image.new("RGB", (100, 60), "white"),
        progress_callback=lambda **event: events.append(event),
    )

    assert result["count"] == 1
    assert result["regions"][0]["text"] == "50"
    assert result["detected_count"] == result["recognized_count"] == 1
    assert result["eligible_count"] == 1
    assert result["excluded_count"] == result["review_count"] == 0
    assert events[0]["stage"] == "preparing"
    assert events[-1]["stage"] == "finalizing"
    detection_events = [
        event
        for event in events
        if event["stage"] == "detecting" and event["total"] > 0
    ]
    assert detection_events[0]["message"].startswith(
        "Running detection pass 1 of 2"
    )
    assert detection_events[0]["pass_current"] == 1
    assert detection_events[0]["tile_current"] == 1
    assert detection_events[0]["tile_total"] == 1
    assert detection_events[-1]["completed"] == 2
    assert detection_events[-1]["total"] == 2
    grouping_events = [event for event in events if event["stage"] == "grouping"]
    assert [event["operation_label"] for event in grouping_events] == [
        "Bridge filtering",
        "Bridge filtering",
        "Spatial grouping",
        "Spatial grouping",
        "Fragment merging",
        "Fragment merging",
        "Orientation splitting",
        "Orientation splitting",
        "Overlap merging",
        "Grouping complete",
    ]
    assert grouping_events[0]["candidate_count"] == 1
    assert grouping_events[-1]["completed"] == grouping_events[-1]["total"] == 5
    recognition_events = [
        event for event in events if event["stage"] == "recognizing"
    ]
    assert recognition_events[0]["completed"] == 0
    assert recognition_events[0]["total"] == 1
    assert recognition_events[-1]["completed"] == 1
    assert recognition_events[-1]["total"] == 1
    assert [event["percent"] for event in events] == sorted(
        event["percent"] for event in events
    )


def test_section_skips_existing_balloon_before_recognition() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline.detect_regions = lambda *_args, **_kwargs: [
        {
            "x": 10.0,
            "y": 10.0,
            "w": 40.0,
            "h": 12.0,
            "text": "50",
            "conf": 0.9,
        }
    ]
    pipeline._expand_clusters = lambda _image, clusters, **_kwargs: clusters
    pipeline.recognize = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("an existing section balloon must not be reread")
    )

    result = pipeline.segment(
        Image.new("RGB", (100, 60), "white"),
        existing_value_boxes=(
            {"x": 8, "y": 8, "width": 45, "height": 18},
        ),
    )

    assert result["detected_count"] == 0
    assert result["count"] == 0
    assert result["skipped_existing_count"] == 1


def test_oversized_cluster_refinement_is_detector_only_and_capped() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    calls: list[tuple[int, int]] = []

    def fake_detector(image, **_kwargs):
        calls.append(image.size)
        return []

    pipeline._detector_only_boxes = fake_detector
    clusters = [
        [
            {
                "x": 20.0,
                "y": float(20 + index * 70),
                "w": 450.0,
                "h": 20.0,
                "text": "",
                "conf": 0.0,
            }
        ]
        for index in range(12)
    ]
    events: list[dict] = []

    refined = pipeline._expand_clusters(
        Image.new("RGB", (1000, 1000), "white"),
        clusters,
        cluster_margin=0.72,
        max_refinements=3,
        progress_callback=lambda **event: events.append(event),
    )

    assert len(calls) == 3
    assert len(refined) == len(clusters)
    assert events[-1]["completed"] == events[-1]["total"] == 3


def test_cluster_refinement_never_falls_back_to_full_ocr() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = False

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("full OCR must not run during grouping")

    pipeline._run_paddle = fail_if_called
    cluster = [
        {
            "x": 10.0,
            "y": 10.0,
            "w": 180.0,
            "h": 20.0,
            "text": "",
            "conf": 0.0,
        }
    ]
    events: list[dict] = []

    refined = pipeline._expand_clusters(
        Image.new("RGB", (200, 200), "white"),
        [cluster],
        cluster_margin=0.72,
        progress_callback=lambda **event: events.append(event),
    )

    assert refined == [cluster]
    assert events[0]["state"] == "skipped"
    assert events[0]["candidate_count"] == 1


def test_page_scan_uses_adaptive_panels_then_bounded_recovery_batches() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    pipeline._page_batch_recognition_available = True
    detector_calls: list[tuple[tuple[int, int], int]] = []
    recognition_batches: list[tuple[str, int]] = []
    authoritative_crops: list[tuple[int, int]] = []
    authoritative_texts = iter(("30°±3°", "R5.00"))
    source_box = {"x": 40.0, "y": 30.0, "w": 40.0, "h": 16.0}
    layout = PageLayout(
        width=400,
        height=240,
        table_masks=(),
        panels=tuple(
            LayoutPanel(panel_id=f"P{index}", bbox=box)
            for index, box in enumerate(
                (
                    LayoutBox(0, 0, 200, 120),
                    LayoutBox(200, 0, 200, 120),
                    LayoutBox(0, 120, 200, 120),
                    LayoutBox(200, 120, 200, 120),
                ),
                start=1,
            )
        ),
        overlaps=(),
    )

    def fake_detector(image, *, target_long_edge):
        detector_calls.append((image.size, target_long_edge))
        if len(detector_calls) % 2 == 1:
            return [{**source_box, "text": "", "conf": 0.9}]
        return [
            {
                "x": 120.0 - (source_box["y"] + source_box["h"]),
                "y": source_box["x"],
                "w": source_box["h"],
                "h": source_box["w"],
                "text": "",
                "conf": 0.8,
            }
        ]

    def fake_batch(crops, *, batch_size, profile="batch_recognition"):
        assert batch_size == 16
        recognition_batches.append((profile, len(crops)))
        if profile == "batch_recognition":
            texts = ("?30º +/- 3º", "SC4LE 2:1", "N0TES", "")
        elif profile == "recovery_rectified":
            texts = ("R5.00",)
        else:
            raise AssertionError(
                "a usable first recovery read must skip the second variant"
            )
        return [
            {
                "text": text,
                "raw_ocr": text,
                "confidence": (
                    0.35 if text == "25.00" else 0.96 if text else 0.0
                ),
                "confusable_corrected": False,
                "orientation_confidence": 0.15 if text == "25.00" else 0.99,
                "type": "Linear" if text else "Unknown",
                "orientation": "horizontal",
                "rotation": 0,
                "needs_review": not bool(text),
                "ocr_profile": "batch_recognition",
            }
            for text in texts[: len(crops)]
        ]

    def fail_selection(*_args, **_kwargs):
        raise AssertionError("page scan must not call the section pipeline")

    def fake_authoritative(crop, **kwargs):
        assert kwargs["compute_text_bbox"] is True
        authoritative_crops.append(crop.size)
        text = next(authoritative_texts)
        return {
            "text": text,
            "raw_ocr": text,
            "confidence": 0.99,
            "agreement": 1.0,
            "needs_review": False,
            "type": "Angle" if "°" in text else "Radius",
            "orientation": "horizontal",
            "rotation": 0,
            "engine": "paddleocr+compose",
            "symbols_detected": {},
            "text_bbox": {
                "x": 2,
                "y": 2,
                "width": max(1, crop.width - 4),
                "height": max(1, crop.height - 4),
            },
        }

    pipeline._detector_only_boxes = fake_detector
    pipeline.segment = fail_selection
    pipeline.recognize = fake_authoritative
    pipeline._recognize_page_batch = fake_batch
    events: list[dict] = []

    result = pipeline.segment_page(
        Image.new("RGB", (400, 240), "white"),
        layout=layout,
        progress_callback=lambda **event: events.append(event),
    )

    assert len(detector_calls) == 8
    assert [target for _size, target in detector_calls] == [1000] * 8
    assert result["detected_count"] == 4
    assert result["recognized_count"] == 4
    assert result["eligible_count"] == 2, [
        (item["candidate_id"], item["text"], item["state"], item["reason"])
        for item in result["candidate_outcomes"]
    ]
    assert result["excluded_count"] == 2
    assert result["review_count"] == 0
    assert result["unread_count"] == 0
    assert result["detected_count"] == (
        result["eligible_count"]
        + result["excluded_count"]
        + result["review_count"]
    )
    assert recognition_batches == [
        ("batch_recognition", 4),
        ("recovery_rectified", 1),
    ]
    # Only the two preliminarily publishable values use accurate OCR; scale and
    # note exclusions do not pay for a reread. Padding changes the crop only,
    # never the published detector bbox.
    assert len(authoritative_crops) == 2
    assert all(width > 40 and height > 16 for width, height in authoritative_crops)
    assert any(
        event["pass_total"] == 8 and event["tile_total"] == 4
        for event in events
    )
    assert any(
        event["batch_current"] == 1 and event["batch_total"] == 1
        for event in events
    )
    assert any(event["stage"] == "recovering" for event in events)
    assert any(event["stage"] == "rereading" for event in events)
    assert any(event["stage"] == "filtering" for event in events)
    assert any(event["overlay"] is not None for event in events)
    assert events[-1]["candidate_count"] == 2
    assert [event["percent"] for event in events] == sorted(
        event["percent"] for event in events
    )
    assert [region["text"] for region in result["regions"]] == [
        "30°±3°",
        "R5.00",
    ]
    assert result["regions"][0]["page_filter_rule"] == "engineering_value"
    assert all(region["authoritative_reread"] for region in result["regions"])
    assert result["filter_rule_counts"] == {
        "engineering_value": 2,
        "note_information": 1,
        "scale_information": 1,
    }
    assert len(result["candidate_outcomes"]) == result["detected_count"]


def test_page_scan_never_falls_back_to_full_ocr_for_detection() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = False

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("whole-page detection must not use full OCR")

    pipeline._run_paddle = fail_if_called

    try:
        pipeline.segment_page(Image.new("RGB", (200, 100), "white"))
    except RuntimeError as exc:
        assert "standalone PaddleOCR text detector" in str(exc)
    else:
        raise AssertionError("missing standalone detector should fail clearly")


def test_page_scan_skips_existing_balloon_before_primary_ocr() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    pipeline._page_batch_recognition_available = True
    detector_calls = 0
    layout = PageLayout(
        width=200,
        height=100,
        table_masks=(),
        panels=(LayoutPanel("P1", LayoutBox(0, 0, 200, 100)),),
        overlaps=(),
    )

    def fake_detector(_image, **_kwargs):
        nonlocal detector_calls
        detector_calls += 1
        return (
            [{"x": 40, "y": 30, "w": 45, "h": 16, "conf": 0.95}]
            if detector_calls == 1
            else []
        )

    pipeline._detector_only_boxes = fake_detector
    pipeline._recognize_page_batch = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("an existing page balloon must not enter primary OCR"))

    result = pipeline.segment_page(
        Image.new("RGB", (200, 100), "white"),
        layout=layout,
        existing_value_boxes=(
            {"x": 38, "y": 28, "width": 50, "height": 22},
        ),
    )

    assert result["detected_count"] == 0
    assert result["skipped_existing_count"] == 1
    assert result["candidate_outcomes"] == []


def test_page_boundary_is_diagnostic_and_does_not_block_auto_acceptance() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    pipeline._page_batch_recognition_available = True
    detector_calls = 0
    layout = PageLayout(
        width=200,
        height=100,
        table_masks=(),
        panels=(
            LayoutPanel("P1", LayoutBox(0, 0, 120, 100)),
            LayoutPanel("P2", LayoutBox(80, 0, 120, 100)),
        ),
        overlaps=(),
    )

    def fake_detector(_image, **_kwargs):
        nonlocal detector_calls
        detector_calls += 1
        return (
            [{"x": 108, "y": 30, "w": 12, "h": 16, "conf": 0.9}]
            if detector_calls == 1
            else []
        )

    def fake_batch(crops, *, batch_size, profile="batch_recognition"):
        assert batch_size == 16
        assert profile == "batch_recognition"
        return [
            {
                "text": "25.00",
                "raw_ocr": "25.00",
                "confidence": 0.41,
                "confusable_corrected": False,
                "orientation_confidence": 0.22,
                "orientation": "horizontal",
                "rotation": 0,
                "ocr_profile": profile,
            }
            for _crop in crops
        ]

    pipeline._detector_only_boxes = fake_detector
    pipeline._recognize_page_batch = fake_batch
    def fake_authoritative(crop, **_kwargs):
        return {
            "text": "25.00",
            "raw_ocr": "25.00",
            "confidence": 0.41,
            "agreement": 0.25,
            # Confidence alone must not demote a structurally valid accurate read.
            "needs_review": True,
            "type": "Linear",
            "orientation": "horizontal",
            "rotation": 0,
            "engine": "paddleocr",
            "symbols_detected": {},
            "text_bbox": {
                "x": 2,
                "y": 2,
                "width": max(1, crop.width - 4),
                "height": max(1, crop.height - 4),
            },
        }

    pipeline.recognize = fake_authoritative

    result = pipeline.segment_page(
        Image.new("RGB", (200, 100), "white"),
        layout=layout,
    )

    assert result["eligible_count"] == 1
    assert result["review_count"] == 0
    assert result["regions"][0]["boundary_review"]


def test_page_scan_corrects_preliminary_review_text_with_full_ocr() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    pipeline._page_batch_recognition_available = True
    detector_calls = 0
    recognition_profiles: list[str] = []
    authoritative_calls = 0
    layout = PageLayout(
        width=200,
        height=100,
        table_masks=(),
        panels=(LayoutPanel("P1", LayoutBox(0, 0, 200, 100)),),
        overlaps=(),
    )

    def fake_detector(_image, **_kwargs):
        nonlocal detector_calls
        detector_calls += 1
        return (
            [{"x": 30, "y": 30, "w": 120, "h": 16, "conf": 0.9}]
            if detector_calls == 1
            else []
        )

    def fake_batch(crops, *, batch_size, profile="batch_recognition"):
        assert batch_size == 16
        recognition_profiles.append(profile)
        return [
            {
                "text": "ZONE A 25",
                "raw_ocr": "ZONE A 25",
                "confidence": 0.98,
                "confusable_corrected": False,
                "orientation_confidence": 0.99,
                "orientation": "horizontal",
                "rotation": 0,
                "needs_review": False,
                "ocr_profile": profile,
            }
            for _crop in crops
        ]

    pipeline._detector_only_boxes = fake_detector
    pipeline._recognize_page_batch = fake_batch

    def fake_authoritative(crop, **_kwargs):
        nonlocal authoritative_calls
        authoritative_calls += 1
        return {
            "text": "25.00",
            "raw_ocr": "25.00",
            "confidence": 0.99,
            "agreement": 1.0,
            "needs_review": False,
            "type": "Linear",
            "orientation": "horizontal",
            "rotation": 0,
            "engine": "paddleocr",
            "symbols_detected": {},
            "text_bbox": {
                "x": 2,
                "y": 2,
                "width": max(1, crop.width - 4),
                "height": max(1, crop.height - 4),
            },
        }

    pipeline.recognize = fake_authoritative

    result = pipeline.segment_page(
        Image.new("RGB", (200, 100), "white"),
        layout=layout,
    )

    assert result["eligible_count"] == 1
    assert result["excluded_count"] == 0
    assert result["review_count"] == 0
    assert result["regions"][0]["text"] == "25.00"
    assert result["candidate_outcomes"][0]["rule"] == (
        "engineering_value"
    )
    assert recognition_profiles == ["batch_recognition"]
    assert authoritative_calls == 1


def test_page_scan_never_publishes_preliminary_garbage_as_review_text() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    pipeline._page_batch_recognition_available = True
    detector_calls = 0
    recognition_profiles: list[str] = []
    layout = PageLayout(
        width=160,
        height=100,
        table_masks=(),
        panels=(
            LayoutPanel("P1", LayoutBox(0, 0, 160, 100)),
        ),
        overlaps=(),
    )

    def fake_detector(_image, **_kwargs):
        nonlocal detector_calls
        detector_calls += 1
        return (
            [{"x": 40, "y": 30, "w": 18, "h": 16, "conf": 0.9}]
            if detector_calls == 1
            else []
        )

    def fake_batch(crops, *, batch_size, profile="batch_recognition"):
        assert batch_size == 16
        recognition_profiles.append(profile)
        text = "8" if profile == "batch_recognition" else "B"
        return [
            {
                "text": text,
                "raw_ocr": text,
                "confidence": 0.95,
                "confusable_corrected": False,
                "orientation_confidence": 0.99,
                "orientation": "horizontal",
                "rotation": 0,
                "ocr_profile": profile,
            }
            for _crop in crops
        ]

    pipeline._detector_only_boxes = fake_detector
    pipeline._recognize_page_batch = fake_batch
    pipeline.recognize = lambda _crop, **_kwargs: {
        "text": "",
        "raw_ocr": "",
        "confidence": 0.0,
        "agreement": 0.0,
        "needs_review": False,
        "type": "Unknown",
        "orientation": "horizontal",
        "rotation": 0,
        "engine": "paddleocr",
        "symbols_detected": {},
    }

    result = pipeline.segment_page(
        Image.new("RGB", (160, 100), "white"),
        layout=layout,
    )

    assert result["detected_count"] == 1
    assert result["eligible_count"] == 0
    assert result["excluded_count"] == 0
    assert result["review_count"] == 1
    assert result["review_candidates"][0]["candidate_id"] == "C0001"
    assert result["review_candidates"][0]["text"] == ""
    assert "detector target" in result["review_candidates"][0]["review_reason"].lower()
    assert result["candidate_outcomes"][0]["state"] == "review"
    assert result["candidate_outcomes"][0]["text"] == ""
    assert recognition_profiles == [
        "batch_recognition",
        "recovery_rectified",
        "recovery_expanded_sharp",
    ]


def test_page_scan_never_falls_back_when_batch_recognition_is_unavailable() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    pipeline._page_batch_recognition_available = False

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("whole-page recognition must not use full OCR")

    pipeline.recognize = fail_if_called

    try:
        pipeline.segment_page(Image.new("RGB", (200, 100), "white"))
    except RuntimeError as exc:
        assert "slow per-object fallback is intentionally disabled" in str(exc)
    else:
        raise AssertionError("missing standalone recognition should fail clearly")


def test_page_batch_recognition_calls_each_standalone_module_once() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._page_batch_recognition_available = True
    pipeline._text_orientation_available = True
    pipeline._text_recognizer_available = True
    calls: list[tuple[str, int, int]] = []

    class OrientationModule:
        def predict(self, *, input, batch_size):
            calls.append(("orientation", len(input), batch_size))
            return [
                {
                    "res": {
                        "label_names": ["0_degree"],
                        "scores": [0.99],
                    }
                }
                for _item in input
            ]

    class RecognitionModule:
        def predict(self, *, input, batch_size):
            calls.append(("recognition", len(input), batch_size))
            texts = ("M8", "25")
            return [
                {"res": {"rec_text": text, "rec_score": 0.97}}
                for text in texts[: len(input)]
            ]

    pipeline._text_orientation = OrientationModule()
    pipeline._text_recognizer = RecognitionModule()
    pipeline.recognize = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("batch mode must not call the complete OCR pipeline")
    )

    results = pipeline._recognize_page_batch(
        [
            Image.new("RGB", (60, 20), "white"),
            Image.new("RGB", (60, 20), "white"),
        ],
        batch_size=16,
    )

    assert calls == [("orientation", 2, 16), ("recognition", 2, 16)]
    assert [result["text"] for result in results] == ["M8", "25"]
    assert all(result["ocr_profile"] == "batch_recognition" for result in results)


def test_paddle_prediction_limit_stops_after_one_call(monkeypatch) -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._paddle_api = 3
    calls: list[tuple[tuple[int, int], bool]] = []

    def fail_full_variant_plan(_image):
        raise AssertionError("single-pass OCR must not build retry variants")

    monkeypatch.setattr(
        "ocr_pipeline.prepare_ocr_variants",
        fail_full_variant_plan,
    )

    def fake_run(image, det=True, **_kwargs):
        calls.append((image.size, det))
        return "25.00", 0.70, []

    pipeline._run_paddle = fake_run
    result = pipeline.recognize_paddle(
        Image.new("RGB", (80, 30), "white"),
        max_predictions=1,
    )

    assert len(calls) == 1
    assert result[0] == "25.00"
