from __future__ import annotations

from ocr_capacity import build_ocr_capacity_status


def test_ready_model_pool_reports_loaded_effective_scan_slots() -> None:
    status = build_ocr_capacity_status(
        {
            "paddleocr": True,
            "ocr_device_effective": "gpu:0",
            "text_detector": True,
            "text_recognizer": True,
            "text_line_orientation": True,
            "model_pool": {
                "requested": 3,
                "loaded": 2,
                "failed": 1,
                "load_errors": ["Bundle 3 was not added"],
            },
        },
        {
            "executor": "thread_pool",
            "worker_threads": 2,
            "maximum_concurrent_jobs": 2,
            "running_jobs": 0,
            "queued_jobs": 0,
            "active_jobs": 0,
            "available_worker_slots": 2,
        },
    )

    assert status["architecture"] == "single_process_model_pool"
    assert status["model_bundles"] == {
        "configured": 3,
        "created": 2,
        "inference_ready": 2,
        "failed": 1,
        "load_errors": ["Bundle 3 was not added"],
        "concurrent_scans_per_bundle": 1,
        "device": "gpu:0",
        "components": {
            "paddleocr_pipeline": True,
            "text_detector": True,
            "text_recognizer": True,
            "text_line_orientation": True,
        },
    }
    assert status["auto_balloon_scans"]["effective_concurrent_slots"] == 2
    assert status["direct_ocr_requests"] == {
        "share_model_pool": True,
        "controlled_by_scan_executor": False,
        "controlled_by_inference_gate": True,
    }
    assert status["inference_gate"]["configured_slots"] == 2
    assert status["inference_gate"]["available_slots"] == 2


def test_health_uses_live_inference_gate_counts() -> None:
    gate = {
        "configured_slots": 1,
        "active_calls": 1,
        "available_slots": 0,
        "waiting_interactive": 2,
        "waiting_background": 1,
        "peak_active_calls": 1,
        "total_acquisitions": 9,
        "active_operations": {"text_recognizer": 1},
    }
    status = build_ocr_capacity_status(
        {"paddleocr": True, "ocr_device_effective": "gpu:0"},
        {"worker_threads": 1},
        gate,
    )

    assert status["inference_gate"] == gate


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
