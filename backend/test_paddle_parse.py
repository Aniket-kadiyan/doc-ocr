"""Regression checks for PaddleOCR 2.x and 3.x result normalization.

Run from backend/:
    python test_paddle_parse.py
"""

from paddle_parse import extract_paddle_lines


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


if __name__ == "__main__":
    test_paddle_3_result_object()
    test_paddle_2_nested_result()
    test_paddle_2_recognition_only_result()
    print("OK")
