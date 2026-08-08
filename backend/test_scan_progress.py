"""Progress instrumentation tests that do not load PaddleOCR models."""

from __future__ import annotations

from PIL import Image

from ocr_pipeline import OcrPipeline


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
        "text": "25.00",
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


def test_page_scan_uses_two_detectors_then_filters_single_pass_ocr() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._text_detector_available = True
    detector_calls: list[tuple[tuple[int, int], int]] = []
    recognition_calls: list[dict] = []

    source_boxes = [
        {"x": 100.0, "y": 50.0, "w": 40.0, "h": 16.0},
        {"x": 200.0, "y": 100.0, "w": 70.0, "h": 16.0},
        {"x": 300.0, "y": 150.0, "w": 55.0, "h": 16.0},
        {"x": 400.0, "y": 200.0, "w": 45.0, "h": 16.0},
    ]

    def fake_detector(image, *, target_long_edge):
        detector_calls.append((image.size, target_long_edge))
        if len(detector_calls) == 1:
            return [
                {**box, "text": "", "conf": 0.9}
                for box in source_boxes
            ]

        # Same source boxes expressed in the 90-degree detector coordinates.
        return [
            {
                "x": 400.0 - (box["y"] + box["h"]),
                "y": box["x"],
                "w": box["h"],
                "h": box["w"],
                "text": "",
                "conf": 0.8,
            }
            for box in source_boxes
        ]

    def fake_recognize(_image, **kwargs):
        recognition_calls.append(kwargs)
        texts = ("25.00", "SCALE 2:1", "NOTES", "")
        text = texts[len(recognition_calls) - 1]
        return {
            "text": text,
            "confidence": 0.96 if text else 0.0,
            "type": "Linear" if text else "Unknown",
            "orientation": "horizontal",
            "rotation": 0,
            "needs_review": not bool(text),
        }

    def fail_selection(*_args, **_kwargs):
        raise AssertionError("page scan must not call the section pipeline")

    pipeline._detector_only_boxes = fake_detector
    pipeline.segment = fail_selection
    pipeline.recognize = fake_recognize
    events: list[dict] = []

    result = pipeline.segment_page(
        Image.new("RGB", (1263, 400), "white"),
        progress_callback=lambda **event: events.append(event),
    )

    assert len(detector_calls) == 2
    assert [target for _size, target in detector_calls] == [2000, 2000]
    assert result["detected_count"] == 4
    assert result["recognized_count"] == 3
    assert result["eligible_count"] == 1
    assert result["excluded_count"] == 2
    assert result["unread_count"] == 1
    assert result["detected_count"] == (
        result["eligible_count"]
        + result["excluded_count"]
        + result["unread_count"]
    )
    assert len(recognition_calls) == result["detected_count"]
    assert all(
        call["max_paddle_predictions"] == 1
        and call["allow_prefix_ocr"] is False
        and call["compute_text_bbox"] is False
        for call in recognition_calls
    )
    assert any(
        event["pass_total"] == 2 and event["tile_total"] == 1
        for event in events
    )
    assert any(event["stage"] == "filtering" for event in events)
    assert events[-1]["candidate_count"] == 1
    assert [event["percent"] for event in events] == sorted(
        event["percent"] for event in events
    )
    assert [region["text"] for region in result["regions"]] == ["25.00"]
    assert result["regions"][0]["page_filter_rule"] == "numeric_component"
    assert result["filter_rule_counts"] == {
        "no_numeric_component": 1,
        "numeric_component": 1,
        "scale_information": 1,
        "unread": 1,
    }


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
