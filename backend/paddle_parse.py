"""Normalize PaddleOCR output across v2/v3 and det=True/det=False."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real
from typing import Any, Iterator


def _is_point_pair(obj: Any) -> bool:
    return (
        isinstance(obj, (list, tuple))
        and len(obj) >= 2
        and isinstance(obj[0], Real)
        and isinstance(obj[1], Real)
    )


def _is_box(obj: Any) -> bool:
    if obj is None or isinstance(obj, str):
        return False
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        return False
    if _is_point_pair(obj[0]):
        return True
    if len(obj) == 4 and all(isinstance(v, Real) for v in obj):
        return True
    return False


def _box_to_rect(box: Any) -> tuple[float, float, float, float]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if isinstance(box, (list, tuple)) and len(box) == 4:
        if all(isinstance(v, Real) for v in box):
            x0, y0, x1, y1 = map(float, box)
            return x0, y0, x1 - x0, y1 - y0

    points: list[tuple[float, float]] = []
    for p in box:
        if _is_point_pair(p):
            points.append((float(p[0]), float(p[1])))

    if not points:
        return 0.0, 0.0, 0.0, 0.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _parse_recognition(rec: Any) -> tuple[str, float]:
    if isinstance(rec, (list, tuple)):
        if len(rec) == 0:
            return "", 0.0
        if len(rec) == 1:
            return str(rec[0]), 0.0
        return str(rec[0]), float(rec[1])
    if isinstance(rec, dict):
        text = rec.get("text") or rec.get("rec_text") or ""
        score = rec.get("score") or rec.get("confidence") or 0.0
        return str(text), float(score)
    return str(rec), 0.0


def _as_list(value: Any) -> list[Any]:
    """Convert Paddle lists/tuples/NumPy arrays without truth-value checks."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first_present(page: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-None field without evaluating arrays."""
    for key in keys:
        if key in page and page[key] is not None:
            return page[key]
    return None


def _unwrap_result(obj: Any) -> Any:
    """Unwrap the serializable payload from a PaddleOCR 3.x Result object."""
    for _ in range(3):
        if not isinstance(obj, dict) and hasattr(obj, "json"):
            payload = getattr(obj, "json")
            obj = payload() if callable(payload) else payload
            continue
        if isinstance(obj, dict) and isinstance(obj.get("res"), dict):
            obj = obj["res"]
            continue
        break
    return obj


def unwrap_paddle_result(obj: Any) -> Any:
    """Public wrapper used by standalone PaddleOCR module integrations."""

    return _unwrap_result(obj)


def extract_text_recognition_result(obj: Any) -> tuple[str, float]:
    """Return one standalone ``TextRecognition`` result as text/confidence."""

    payload = _unwrap_result(obj)
    if not isinstance(payload, dict):
        return "", 0.0
    texts = _as_list(_first_present(payload, "rec_text", "rec_texts"))
    scores = _as_list(_first_present(payload, "rec_score", "rec_scores"))
    text = str(texts[0]).strip() if texts else ""
    score = float(scores[0]) if scores else 0.0
    return text, score


def extract_text_orientation_result(obj: Any) -> tuple[int, float]:
    """Return one standalone text-line orientation as degrees/confidence."""

    payload = _unwrap_result(obj)
    if not isinstance(payload, dict):
        return 0, 0.0
    labels = _as_list(_first_present(payload, "label_names", "label_name"))
    scores = _as_list(_first_present(payload, "scores", "score"))
    label = str(labels[0]).lower() if labels else ""
    degrees = 180 if "180" in label else 0
    score = float(scores[0]) if scores else 0.0
    return degrees, score


def _yield_from_dict_page(
    page: dict[str, Any],
) -> Iterator[tuple[str, float, float, float, float, float]]:
    texts = _as_list(_first_present(page, "rec_texts", "rec_text"))
    scores = _as_list(_first_present(page, "rec_scores", "rec_score"))
    # rec_polys aligns with the confidence-filtered recognition arrays.
    polys = _as_list(_first_present(page, "rec_polys", "dt_polys"))

    for i, text in enumerate(texts):
        conf = float(scores[i]) if i < len(scores) else 0.0
        if i < len(polys) and _is_box(polys[i]):
            x, y, w, h = _box_to_rect(polys[i])
        else:
            x, y, w, h = 0.0, 0.0, 0.0, 0.0
        if str(text).strip():
            yield str(text), x, y, w, h, conf


def _yield_from_line(line: Any) -> Iterator[tuple[str, float, float, float, float, float]]:
    if line is None:
        return

    if isinstance(line, dict):
        yield from _yield_from_dict_page(line)
        return

    if not isinstance(line, (list, tuple)):
        return

    if len(line) == 0:
        return

    # det=False: ("text", score) or ["text", score]
    if len(line) == 2 and isinstance(line[0], str) and not _is_box(line[0]):
        text, conf = _parse_recognition(line)
        if text.strip():
            yield text, 0.0, 0.0, 0.0, 0.0, conf
        return

    # Sometimes: [box, text, score]
    if len(line) >= 3 and _is_box(line[0]) and isinstance(line[1], str):
        text = line[1]
        conf = float(line[2])
        if text.strip():
            x, y, w, h = _box_to_rect(line[0])
            yield text, x, y, w, h, conf
        return

    # Classic: [box, (text, score)]
    if len(line) >= 2 and _is_box(line[0]):
        box = line[0]
        rec = line[1]
        text, conf = _parse_recognition(rec)
        if text.strip():
            x, y, w, h = _box_to_rect(box)
            yield text, x, y, w, h, conf
        return


def _looks_like_recognition_line(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)) or not obj:
        return False
    # det=False: ("text", confidence)
    if len(obj) == 2 and isinstance(obj[0], str):
        return True
    # det=True: [box, ("text", confidence)] or [box, "text", confidence]
    return len(obj) >= 2 and _is_box(obj[0])


def _walk_result(
    obj: Any,
) -> Iterator[tuple[str, float, float, float, float, float]]:
    """Recursively walk v2 nested lists and v3 Result objects."""
    obj = _unwrap_result(obj)
    if obj is None:
        return

    if isinstance(obj, dict):
        yield from _yield_from_dict_page(obj)
        return

    if _looks_like_recognition_line(obj):
        yield from _yield_from_line(obj)
        return

    if isinstance(obj, (list, tuple)):
        for child in obj:
            yield from _walk_result(child)


def extract_paddle_lines(
    result: Any,
) -> list[tuple[str, float, float, float, float, float]]:
    """Returns list of (text, x, y, width, height, confidence)."""
    return list(_walk_result(result))


def _walk_detection_result(
    obj: Any,
) -> Iterator[tuple[float, float, float, float, float]]:
    """Walk standalone PaddleOCR ``TextDetection`` result objects."""

    obj = _unwrap_result(obj)
    if obj is None:
        return

    if isinstance(obj, dict):
        polys = _as_list(_first_present(obj, "dt_polys", "rec_polys"))
        scores = _as_list(_first_present(obj, "dt_scores", "rec_scores"))
        for index, polygon in enumerate(polys):
            if not _is_box(polygon):
                continue
            x, y, width, height = _box_to_rect(polygon)
            score = float(scores[index]) if index < len(scores) else 0.0
            if width > 0 and height > 0:
                yield x, y, width, height, score
        return

    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        for child in obj:
            yield from _walk_detection_result(child)


def extract_paddle_detection_boxes(
    result: Any,
) -> list[tuple[float, float, float, float, float]]:
    """Return detector-only ``(x, y, width, height, confidence)`` boxes."""

    return list(_walk_detection_result(result))
