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
