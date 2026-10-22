from __future__ import annotations

from ocr_capacity import build_ocr_capacity_status


def test_ready_singleton_reports_one_effective_scan_slot() -> None:
    status = build_ocr_capacity_status(
        {
            "paddleocr": True,
            "ocr_device_effective": "gpu:0",
            "text_detector": True,
            "text_recognizer": True,
            "text_line_orientation": True,
        },
        {
            "executor": "thread_pool",
            "worker_threads": 1,
            "maximum_concurrent_jobs": 1,
            "running_jobs": 0,
            "queued_jobs": 0,
            "active_jobs": 0,
            "available_worker_slots": 1,
        },
    )

    assert status["architecture"] == "single_process_singleton"
    assert status["model_bundles"] == {
        "configured": 1,
        "created": 1,
        "inference_ready": 1,
        "concurrent_scans_per_bundle": 1,
        "device": "gpu:0",
        "components": {
            "paddleocr_pipeline": True,
            "text_detector": True,
            "text_recognizer": True,
            "text_line_orientation": True,
        },
    }
    assert status["auto_balloon_scans"]["effective_concurrent_slots"] == 1
    assert status["direct_ocr_requests"] == {
        "share_model_bundle": True,
        "controlled_by_scan_executor": False,
    }


def test_unavailable_pipeline_reports_zero_effective_slots() -> None:
    status = build_ocr_capacity_status(
        {
            "paddleocr": False,
            "ocr_device_effective": "cpu",
        },
        {
            "worker_threads": 2,
            "running_jobs": 0,
            "queued_jobs": 0,
        },
    )

    assert status["model_bundles"]["inference_ready"] == 0
    assert status["auto_balloon_scans"]["effective_concurrent_slots"] == 0


def test_more_executor_threads_do_not_invent_more_model_capacity() -> None:
    status = build_ocr_capacity_status(
        {
            "paddleocr": True,
            "ocr_device_effective": "cpu",
        },
        {
            "worker_threads": 4,
            "running_jobs": 0,
            "queued_jobs": 0,
        },
    )

    assert status["model_bundles"]["inference_ready"] == 1
    assert status["auto_balloon_scans"]["effective_concurrent_slots"] == 1
