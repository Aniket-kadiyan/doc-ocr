"""Regression checks for PaddleOCR 2.x and 3.x result normalization.

Run from backend/:
    python test_paddle_parse.py
"""

from paddle_parse import (
    extract_paddle_detection_boxes,
    extract_paddle_lines,
    extract_text_orientation_result,
    extract_text_recognition_result,
)


BOX = [[1, 2], [21, 2], [21, 10], [1, 10]]


class Paddle3Result:
    """Minimal stand-in for the Result object returned by predict()."""

    @property
    def json(self):
        return {
            "res": {
                "rec_texts": ["42.50"],
                "rec_scores": [0.98],
                "rec_polys": [BOX],
            }
        }


class Paddle3DetectionResult:
    """Minimal standalone TextDetection result with no recognition fields."""

    @property
    def json(self):
        return {
            "res": {
                "dt_polys": [BOX],
                "dt_scores": [0.03],
            }
        }


class Paddle3TextRecognitionResult:
    @property
    def json(self):
        return {"res": {"rec_text": "M8", "rec_score": 0.97}}


class Paddle3OrientationResult:
    @property
    def json(self):
        return {
            "res": {
                "class_ids": [1],
                "scores": [0.998],
                "label_names": ["180_degree"],
            }
        }


def assert_one_line(result, expected_text: str, expected_confidence: float) -> None:
    parsed = extract_paddle_lines(result)
    assert len(parsed) == 1, parsed
    text, x, y, width, height, confidence = parsed[0]
    assert text == expected_text, parsed
    assert (x, y, width, height) == (1.0, 2.0, 20.0, 8.0), parsed
    assert abs(confidence - expected_confidence) < 1e-9, parsed


def test_paddle_3_result_object() -> None:
    assert_one_line([Paddle3Result()], "42.50", 0.98)


def test_paddle_2_nested_result() -> None:
    # PaddleOCR 2.x det=True: one image containing one recognized line.
    result = [[[BOX, ("17.25", 0.96)]]]
    assert_one_line(result, "17.25", 0.96)


def test_paddle_2_recognition_only_result() -> None:
    # det=False has text/confidence but no detection box.
    parsed = extract_paddle_lines([[("8.00", 0.94)]])
    assert parsed == [("8.00", 0.0, 0.0, 0.0, 0.0, 0.94)], parsed


def test_paddle_3_detector_only_result_keeps_low_confidence_box() -> None:
    parsed = extract_paddle_detection_boxes(iter([Paddle3DetectionResult()]))
    assert parsed == [(1.0, 2.0, 20.0, 8.0, 0.03)]


def test_standalone_text_modules_are_parsed_per_input_crop() -> None:
    assert extract_text_recognition_result(Paddle3TextRecognitionResult()) == (
        "M8",
        0.97,
    )
    assert extract_text_orientation_result(Paddle3OrientationResult()) == (
        180,
        0.998,
    )


if __name__ == "__main__":
    test_paddle_3_result_object()
    test_paddle_2_nested_result()
    test_paddle_2_recognition_only_result()
    test_paddle_3_detector_only_result_keeps_low_confidence_box()
    print("OK")
