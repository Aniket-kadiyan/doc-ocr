from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ocr_runtime import (
    PaddleRuntimeInfo,
    configured_device_label,
    configured_ocr_device,
    effective_device,
    probe_paddle_runtime,
    validate_configured_device,
)


class OcrRuntimeTests(unittest.TestCase):
    def test_device_defaults_to_automatic(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(configured_ocr_device())
            self.assertEqual(
                configured_device_label(configured_ocr_device()),
                "auto",
            )

    def test_auto_aliases_keep_paddle_automatic(self) -> None:
        for value in ("auto", "AUTO", "default", ""):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"OCR_DEVICE": value},
                clear=True,
            ):
                self.assertIsNone(configured_ocr_device())

    def test_explicit_gpu_device_is_preserved(self) -> None:
        with patch.dict(
            os.environ,
            {"OCR_DEVICE": "gpu:0"},
            clear=True,
        ):
            self.assertEqual(configured_ocr_device(), "gpu:0")

    def test_runtime_probe_reports_device_and_cuda(self) -> None:
        fake_paddle = SimpleNamespace(
            is_compiled_with_cuda=lambda: True,
            device=SimpleNamespace(
                get_device=lambda: "gpu:0",
                get_available_device=lambda: ["cpu", "gpu:0"],
            ),
        )
        info = probe_paddle_runtime(fake_paddle)
        self.assertEqual(
            info,
            PaddleRuntimeInfo(
                global_device="gpu:0",
                cuda_compiled=True,
                available_devices=("cpu", "gpu:0"),
                probe_error="",
            ),
        )

    def test_gpu_request_rejects_cpu_only_paddle(self) -> None:
        info = PaddleRuntimeInfo(
            global_device="cpu",
            cuda_compiled=False,
            available_devices=("cpu",),
        )
        error = validate_configured_device("gpu:0", info)
        self.assertIsNotNone(error)
        self.assertIn("no CUDA support", error or "")

    def test_cpu_and_auto_requests_do_not_require_cuda(self) -> None:
        info = PaddleRuntimeInfo(global_device="cpu", cuda_compiled=False)
        self.assertIsNone(validate_configured_device("cpu", info))
        self.assertIsNone(validate_configured_device(None, info))

    def test_effective_device_prefers_explicit_configuration(self) -> None:
        info = PaddleRuntimeInfo(global_device="gpu:1", cuda_compiled=True)
        self.assertEqual(effective_device("gpu:0", info), "gpu:0")
        self.assertEqual(effective_device(None, info), "gpu:1")


if __name__ == "__main__":
    unittest.main()
