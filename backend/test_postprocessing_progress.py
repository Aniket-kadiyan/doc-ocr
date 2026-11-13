"""Progress checkpoints for the OCR work that follows final-value filtering."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from ocr_pipeline import OcrPipeline


def test_slanted_recovery_reports_each_expensive_read(monkeypatch) -> None:
    import region_detect

    pipeline = object.__new__(OcrPipeline)
    pipeline._slant_neighbourhoods = lambda *_args, **_kwargs: [
        (0, 0, 20, 20)
    ]
    pipeline._angled_in_roi = lambda *_args, **_kwargs: []
    pipeline._neighbourhood_sign = lambda *_args, **_kwargs: None
    pipeline._leader_angles_for_roi = lambda *_args, **_kwargs: []
    pipeline._supersede_fused_base = lambda *_args, **_kwargs: set()
    pipeline._dedupe_angled = lambda found, _base: found
    monkeypatch.setattr(region_detect, "detect_leader_lines", lambda _image: [])
    events: list[dict] = []

    result = pipeline._detect_angled_regions(
        Image.new("RGB", (20, 20), "white"),
        [],
        whole_image=True,
        progress_callback=lambda **event: events.append(event),
    )

    assert result == []
    assert [event["completed"] for event in events] == [0, 1, 1, 2, 2, 3]
    assert {event["total"] for event in events} == {3}
    assert events[0]["label"] == "whole-page slanted-text read"
    assert events[-1]["label"].endswith("at 1.5x")


def test_note_scan_reports_its_internal_phases() -> None:
    pipeline = object.__new__(OcrPipeline)
    pipeline._detect_regions_ocr = lambda _image: []
    pipeline._repair_note_boxes = lambda _image, boxes, _ink: boxes
    pipeline._detect_notes_blocks = lambda _boxes: []
    pipeline._rotated_text_boxes = lambda _image, _boxes: ([], set())
    pipeline._reconcile_notes = lambda found, _ink: found
    events: list[dict] = []

    result = pipeline._notes_regions(
        Image.new("RGB", (40, 30), "white"),
        progress_callback=lambda **event: events.append(event),
    )

    assert result == []
    assert [event["completed"] for event in events] == [0, 1, 2, 3, 4, 5]
    assert {event["total"] for event in events} == {5}
    assert events[0]["label"] == "Detecting note text lines"
    assert events[-1]["label"] == "Prepared 0 note regions"


def test_gdt_scan_reports_each_frame_cell_ocr(monkeypatch) -> None:
    import gdt_config
    import gdt_pipeline

    pipeline = object.__new__(OcrPipeline)
    pipeline._predict_text_recognition = lambda crops, *, batch_size: [
        ("0.1", 0.9) for _ in crops
    ]
    monkeypatch.setattr(
        gdt_config,
        "get_config",
        lambda: SimpleNamespace(enabled=True),
    )

    def fake_scan_regions(image, *, ocr_fn, progress_callback=None):
        assert ocr_fn is not None
        assert progress_callback is not None
        ocr_fn(image)
        ocr_fn(image)
        return []

    monkeypatch.setattr(gdt_pipeline, "scan_regions", fake_scan_regions)
    events: list[dict] = []

    result = pipeline._frame_regions(
        Image.new("RGB", (20, 10), "white"),
        progress_callback=lambda **event: events.append(event),
    )

    assert result == []
    assert [event["completed"] for event in events] == [0, 1, 1, 2]
    assert events[0]["label"] == "Reading GD&T frame cell 1"
    assert events[-1]["label"] == "Read GD&T frame cell 2"


def test_slanted_recovery_reports_what_each_zoom_kept(monkeypatch) -> None:
    import region_detect
    import run_report

    pipeline = object.__new__(OcrPipeline)
    pipeline._slant_neighbourhoods = lambda *_args, **_kwargs: [
        (0, 0, 20, 20)
    ]

    def angled(view, *, tally=None, **_kwargs):
        # The 1x read rejects one cluster; the 1.5x read keeps one value.
        if view.width == 20:
            tally.count(read=1, rejected_low_confidence=1)
            return []
        tally.count(read=1, values=1)
        return [{"text": "R4.5", "bbox": {"x": 3, "y": 3, "width": 6, "height": 3}}]

    pipeline._angled_in_roi = angled
    pipeline._neighbourhood_sign = lambda *_args, **_kwargs: None
    pipeline._leader_angles_for_roi = lambda *_args, **_kwargs: []
    pipeline._supersede_fused_base = lambda *_args, **_kwargs: set()
    pipeline._dedupe_angled = lambda found, _base: found
    monkeypatch.setattr(region_detect, "detect_leader_lines", lambda _image: [])
    report = run_report.RunReport("scan-job", "zooms", "test scan")
    monkeypatch.setattr(report, "_write_running", lambda: None)
    report.start()

    with run_report.activate(report):
        result = pipeline._detect_angled_regions(
            Image.new("RGB", (20, 20), "white"), [], whole_image=False
        )

    assert [r["text"] for r in result] == ["R4.5"]
    zoom_1x = report.steps[("Neighbourhood 01", "Zoom 1x")]
    zoom_15x = report.steps[("Neighbourhood 01", "Zoom 1.5x")]
    assert zoom_1x.counts == {"read": 1, "rejected_low_confidence": 1}
    assert zoom_15x.counts == {"read": 1, "values": 1}
    assert report.steps[("Dedupe slanted reads",)].counts == {
        "before": 1,
        "after": 1,
        "kept_from_whole_area": 0,
        "kept_from_1x": 0,
        "kept_from_1_5x": 1,
    }
