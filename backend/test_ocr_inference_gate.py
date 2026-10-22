from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from PIL import Image

from ocr_pipeline import OcrPipeline


class _RecordingGate:
    def __init__(self) -> None:
        self.operations: list[str] = []
        self.active: str | None = None

    @contextmanager
    def slot(self, operation: str):
        assert self.active is None
        self.active = operation
        self.operations.append(operation)
        try:
            yield
        finally:
            self.active = None


class _MainPaddle:
    def __init__(self, gate: _RecordingGate) -> None:
        self.gate = gate

    def predict(self, _arr):
        assert self.gate.active == "paddleocr_pipeline"
        def output():
            assert self.gate.active == "paddleocr_pipeline"
            yield from ()

        return output()

    def ocr(self, _arr, **_kwargs):
        assert self.gate.active == "paddleocr_pipeline"
        return []


class _Detector:
    def __init__(self, gate: _RecordingGate) -> None:
        self.gate = gate

    def predict(self, _arr, **_kwargs):
        assert self.gate.active == "text_detector"
        def output():
            assert self.gate.active == "text_detector"
            yield from ()

        return output()


class _Orientation:
    def __init__(self, gate: _RecordingGate) -> None:
        self.gate = gate

    def predict(self, **_kwargs):
        assert self.gate.active == "text_line_orientation"
        return [{"label_names": ["0_degree"], "scores": [0.99]}]


class _Recognizer:
    def __init__(self, gate: _RecordingGate) -> None:
        self.gate = gate

    def predict(self, **_kwargs):
        assert self.gate.active == "text_recognizer"
        return [{"rec_text": ["12.5"], "rec_score": [0.98]}]


def test_every_paddle_entry_point_uses_the_shared_gate() -> None:
    gate = _RecordingGate()
    pipeline = OcrPipeline(inference_gate=gate)  # type: ignore[arg-type]
    pipeline._paddle = _MainPaddle(gate)
    pipeline._text_detector = _Detector(gate)
    pipeline._text_orientation = _Orientation(gate)
    pipeline._text_recognizer = _Recognizer(gate)
    pipeline._text_orientation_available = True
    pipeline._text_recognizer_available = True
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = Image.new("RGB", (30, 20), "white")

    pipeline._predict_3x(arr)
    pipeline._predict_2x(arr, det=True)
    pipeline._predict_text_detector(arr)
    assert pipeline._predict_text_orientations([crop], batch_size=1) == [
        (0, 0.99)
    ]
    assert pipeline._predict_text_recognition([crop], batch_size=1) == [
        ("12.5", 0.98)
    ]

    assert gate.operations == [
        "paddleocr_pipeline",
        "paddleocr_pipeline",
        "text_detector",
        "text_line_orientation",
        "text_recognizer",
    ]
