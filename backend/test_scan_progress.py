"""Progress instrumentation tests that do not load PaddleOCR models."""

from __future__ import annotations

from PIL import Image

from ocr_pipeline import OcrPipeline


def test_segment_reports_real_stages_and_recognition_counters() -> None:
    pipeline = object.__new__(OcrPipeline)

    def detect_regions(_image, *, progress_callback=None, **_kwargs):
        if progress_callback is not None:
            progress_callback(
                completed=1,
                total=2,
                label="source contrast, 0 deg, 1100px",
                proposals=1,
                deskew_angle=0.0,
            )
            progress_callback(
                completed=2,
                total=2,
                label="morphology",
                proposals=1,
                deskew_angle=0.0,
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
    assert [event["stage"] for event in events] == [
        "preparing",
        "detecting",
        "detecting",
        "detecting",
        "detecting",
        "grouping",
        "grouping",
        "recognizing",
        "recognizing",
        "finalizing",
        "finalizing",
    ]
    detection_events = [
        event
        for event in events
        if event["stage"] == "detecting" and event["total"] > 0
    ]
    assert detection_events[0]["message"].startswith("Detection pass 1 of 2")
    assert detection_events[-1]["completed"] == 2
    assert detection_events[-1]["total"] == 2
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
