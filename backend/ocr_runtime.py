"""Runtime device selection and diagnostics for the local PaddleOCR service.

This module deliberately has no Paddle import at module load time.  Unit tests
and non-OCR tooling can therefore inspect configuration without importing the
large inference runtime or initializing a GPU context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config_env import get_str


_AUTO_DEVICE_VALUES = frozenset(("", "auto", "default"))


@dataclass(frozen=True)
class PaddleRuntimeInfo:
    """Read-only facts reported by the installed Paddle runtime."""

    global_device: str = "unknown"
    cuda_compiled: bool | None = None
    available_devices: tuple[str, ...] = ()
    probe_error: str = ""


def configured_ocr_device() -> str | None:
    """Return the explicit Paddle device, or ``None`` for automatic choice."""

    raw = get_str("OCR_DEVICE", "auto").strip()
    return None if raw.lower() in _AUTO_DEVICE_VALUES else raw


def configured_device_label(device: str | None) -> str:
    """Stable, JSON-friendly label for the requested device."""

    return device or "auto"


def is_gpu_device(device: str | None) -> bool:
    return bool(device and device.lower().startswith("gpu"))


def probe_paddle_runtime(paddle_module: Any) -> PaddleRuntimeInfo:
    """Inspect Paddle without changing its global device."""

    errors: list[str] = []
    global_device = "unknown"
    cuda_compiled: bool | None = None
    available_devices: tuple[str, ...] = ()

    try:
        get_device = getattr(getattr(paddle_module, "device", None), "get_device")
        global_device = str(get_device())
    except Exception as exc:  # noqa: BLE001 - diagnostics must never hide OCR
        errors.append(f"get_device: {exc}")

    try:
        compiled = getattr(paddle_module, "is_compiled_with_cuda")
        cuda_compiled = bool(compiled())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"is_compiled_with_cuda: {exc}")

    try:
        get_available = getattr(
            getattr(paddle_module, "device", None),
            "get_available_device",
        )
        available_devices = tuple(str(item) for item in get_available())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"get_available_device: {exc}")

    return PaddleRuntimeInfo(
        global_device=global_device,
        cuda_compiled=cuda_compiled,
        available_devices=available_devices,
        probe_error="; ".join(errors),
    )


def validate_configured_device(
    device: str | None,
    runtime: PaddleRuntimeInfo,
) -> str | None:
    """Return a clear configuration error for an impossible GPU request."""

    if not is_gpu_device(device):
        return None
    if runtime.cuda_compiled is False:
        return (
            f"OCR_DEVICE={device} was requested, but the installed "
            "PaddlePaddle build has no CUDA support"
        )
    if runtime.available_devices and not any(
        item.lower().startswith("gpu") for item in runtime.available_devices
    ):
        return (
            f"OCR_DEVICE={device} was requested, but Paddle reports no "
            "available GPU device"
        )
    return None


def effective_device(
    configured: str | None,
    runtime: PaddleRuntimeInfo,
) -> str:
    """Describe the intended device without claiming more than Paddle reports."""

    return configured or runtime.global_device
