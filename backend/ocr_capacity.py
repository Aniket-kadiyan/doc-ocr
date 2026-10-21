"""Read-only OCR capacity diagnostics.

The values in this module describe the architecture that exists today.  They
are deliberately not environment settings: advertising a configurable model
pool before the backend can actually create and coordinate that pool would
make deployment diagnostics misleading.
"""

from __future__ import annotations

from typing import Any, Mapping


def build_ocr_capacity_status(
    pipeline_status: Mapping[str, Any],
    scan_executor_status: Mapping[str, Any],
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
    model_bundle_slots = 1 if inference_ready else 0
    effective_slots = min(model_bundle_slots, worker_threads)

    return {
        "architecture": "single_process_singleton",
        "model_bundles": {
            "configured": 1,
            "created": 1,
            "inference_ready": model_bundle_slots,
            "device": str(
                pipeline_status.get("ocr_device_effective") or "unknown"
            ),
            "components": components,
        },
        "auto_balloon_scans": {
            **dict(scan_executor_status),
            "effective_concurrent_slots": effective_slots,
        },
        "direct_ocr_requests": {
            "share_model_bundle": True,
            "controlled_by_scan_executor": False,
        },
        "capacity_basis": (
            "one concurrent auto-balloon scan per scan worker and ready "
            "singleton model bundle"
        ),
    }
