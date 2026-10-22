"""Read-only OCR model-pool and scheduler capacity diagnostics."""

from __future__ import annotations

from typing import Any, Mapping


# Backward-compatible defaults used by isolated tests and callers that do not
# load runtime configuration. Production capacity comes from the model pool's
# successfully loaded bundle count, never from these constants alone.
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
    raw_pool = pipeline_status.get("model_pool")
    pool_status = raw_pool if isinstance(raw_pool, Mapping) else {}
    configured_bundles = max(
        0,
        int(pool_status.get("requested", OCR_MODEL_BUNDLE_COUNT) or 0),
    )
    created_bundles = max(
        0,
        int(
            pool_status.get(
                "loaded",
                OCR_MODEL_BUNDLE_COUNT if inference_ready else 0,
            )
            or 0
        ),
    )
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

    # Detector/recognizer entries are components inside every validated
    # bundle, not additional independently schedulable Paddle instances.
    model_bundle_slots = created_bundles if inference_ready else 0
    inference_slots = (
        model_bundle_slots * OCR_CONCURRENT_SCANS_PER_BUNDLE
    )
    effective_slots = min(inference_slots, worker_threads)
    gate_status = dict(
        inference_gate_status
        or {
            "configured_slots": model_bundle_slots,
            "active_calls": 0,
            "available_slots": model_bundle_slots,
            "waiting_interactive": 0,
            "waiting_background": 0,
            "peak_active_calls": 0,
            "total_acquisitions": 0,
            "active_operations": {},
        }
    )

    return {
        "architecture": "single_process_model_pool",
        "model_bundles": {
            "configured": configured_bundles,
            "created": created_bundles,
            "inference_ready": model_bundle_slots,
            "failed": max(0, configured_bundles - created_bundles),
            "load_errors": list(pool_status.get("load_errors", [])),
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
            "share_model_pool": True,
            "controlled_by_scan_executor": False,
            "controlled_by_inference_gate": True,
        },
        "capacity_basis": (
            "one concurrent Paddle inference call per successfully loaded "
            "model bundle; interactive calls are preferred between "
            "background scan calls"
        ),
    }
