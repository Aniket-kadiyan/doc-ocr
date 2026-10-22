"""Read-only OCR capacity diagnostics.

The values in this module describe the architecture that exists today.  They
are deliberately not environment settings: advertising a configurable model
pool before the backend can actually create and coordinate that pool would
make deployment diagnostics misleading.
"""

from __future__ import annotations

from typing import Any, Mapping


# These constants describe model objects the process really creates today.
# Increase them only together with a corresponding OcrPipeline instance pool;
# executor threads alone are not additional inference capacity.
OCR_MODEL_BUNDLE_COUNT = 1
OCR_CONCURRENT_SCANS_PER_BUNDLE = 1
AUTO_BALLOON_SCAN_CAPACITY = (
    OCR_MODEL_BUNDLE_COUNT * OCR_CONCURRENT_SCANS_PER_BUNDLE
)


def build_ocr_capacity_status(
    pipeline_status: Mapping[str, Any],
    scan_executor_status: Mapping[str, Any],
    inference_gate_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine model and executor facts into one stable health payload."""

    inference_ready = bool(pipeline_status.get("paddleocr"))
    components = {
        "paddleocr_pipeline": inference_ready,
        "text_detector": bool(pipeline_status.get("text_detector")),
        "text_recognizer": bool(pipeline_status.get("text_recognizer")),
        "text_line_orientation": bool(
            pipeline_status.get("text_line_orientation")
        ),
    }
    worker_threads = max(
        0,
        int(scan_executor_status.get("worker_threads", 0) or 0),
    )

    # There is currently one OcrPipeline singleton per backend process.  The
    # detector/recognizer entries above are components inside that bundle, not
    # additional independently schedulable Paddle instances.
    model_bundle_slots = OCR_MODEL_BUNDLE_COUNT if inference_ready else 0
    inference_slots = (
        model_bundle_slots * OCR_CONCURRENT_SCANS_PER_BUNDLE
    )
    effective_slots = min(inference_slots, worker_threads)
    gate_status = dict(
        inference_gate_status
        or {
            "configured_slots": AUTO_BALLOON_SCAN_CAPACITY,
            "active_calls": 0,
            "available_slots": AUTO_BALLOON_SCAN_CAPACITY,
            "waiting_interactive": 0,
            "waiting_background": 0,
            "peak_active_calls": 0,
            "total_acquisitions": 0,
            "active_operations": {},
        }
    )

    return {
        "architecture": "single_process_singleton",
        "model_bundles": {
            "configured": OCR_MODEL_BUNDLE_COUNT,
            "created": OCR_MODEL_BUNDLE_COUNT,
            "inference_ready": model_bundle_slots,
            "concurrent_scans_per_bundle": (
                OCR_CONCURRENT_SCANS_PER_BUNDLE
            ),
            "device": str(
                pipeline_status.get("ocr_device_effective") or "unknown"
            ),
            "components": components,
        },
        "auto_balloon_scans": {
            **dict(scan_executor_status),
            "effective_concurrent_slots": effective_slots,
        },
        "inference_gate": gate_status,
        "direct_ocr_requests": {
            "share_model_bundle": True,
            "controlled_by_scan_executor": False,
            "controlled_by_inference_gate": True,
        },
        "capacity_basis": (
            "one concurrent Paddle inference call per ready singleton model "
            "bundle; interactive calls are preferred between background "
            "scan calls"
        ),
    }
