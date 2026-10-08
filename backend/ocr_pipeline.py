"""
PaddleOCR pipeline: best digit read + balanced symbol compose.
"""

from __future__ import annotations
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
import re

import numpy as np
from PIL import Image

from debug_dump import StepDumper, dump_force, dump_status, should_dump
from dimension_compose import compose_engineering_dimension
from feature_classifier import classify_feature
from dimension_digits import normalize_cad_number_string
from image_preprocess import (
    bbox_from_oriented_to_original,
    is_vertical_dimension,
    pad_image,
    prepare_primary_ocr_variant,
    prepare_ocr_variants,
    primary_oriented,
    sharpen_rgb,
    upscale_min_edge,
    clahe_rgb,
)
from ocr_runtime import (
    PaddleRuntimeInfo,
    configured_device_label,
    configured_ocr_device,
    effective_device,
    is_gpu_device,
    probe_paddle_runtime,
    validate_configured_device,
)
from paddle_parse import (
    PaddleLine,
    extract_paddle_detection_boxes,
    extract_paddle_detection_regions,
    extract_paddle_lines,
    extract_text_orientation_result,
    extract_text_recognition_result,
)
from symbol_normalize import fix_engineering_symbols_light
from symbol_regions import enlarge_zone, split_symbol_zones
from symbol_vision import (
    DetectedSymbols,
    detect_prefix_from_ocr_text,
    detect_symbols,
    merge_symbol_scores,
    symbols_to_dict,
)

LINE_THRESHOLD = 15
# A single callout is wider than it is tall; a stacked pair is not. Used to
# spot a fused read whose text collapsed to one value and hid the second.
_PAGE_SPLIT_LINE_RATIO = 0.35
# Long edge a slanted-callout neighbourhood is enlarged to before it is read.
_ANGLED_MIN_EDGE = 700
# A whole-image angled pass (one slant vote over everything) is what a zoomed-in
# section crop needs; on a page-sized image the vote is diluted by the rest of
# the drawing and the levelled full sheet is expensive to re-detect, so above
# this edge only the per-callout neighbourhoods are read.
_ANGLED_WHOLE_IMAGE_MAX_EDGE = 1400
# Each per-callout neighbourhood is grown to at least this edge so the levelled
# view keeps its surroundings and the free enlargement stays moderate: a 2.3x
# upscale of a tight crop broke "Ø20H10" into pieces, 1.4-1.6x read it cleanly.
_ANGLED_ROI_MIN_EDGE = 450
# Voting-recogniser passes per crop in the page route's authoritative reread:
# the three upright variants. See OcrPipeline._authoritative_attempts.
_PAGE_REREAD_MAX_PREDICTIONS = 3
# Paddle confidence is expressed from 0.0 to 1.0.
# This comparison is intentionally strict: exactly 0.95 continues.
EARLY_ACCEPT_CONFIDENCE = 0.95

# If visual Ø detection is close to its 0.32 acceptance threshold, use the
# dedicated prefix OCR as a second opinion.
PREFIX_RECHECK_PHI_SCORE = 0.20

# Oversized-cluster refinement is a best-effort accuracy improvement. It must
# never expand into unbounded OCR work during a whole-page scan.
GROUPING_MAX_REFINEMENTS = 8


def sort_reading_order(
    items: list[PaddleLine],
    vertical: bool,
) -> list[PaddleLine]:
    if vertical:
        return sorted(items, key=lambda x: x[2])
    return sorted(
        items,
        key=lambda x: (round(x[2] / LINE_THRESHOLD), x[1]),
    )


def assemble_paddle_lines(
    result: Any,
    force_vertical: bool = False,
) -> tuple[str, float, list[dict[str, Any]]]:
    parsed = extract_paddle_lines(result)
    if not parsed:
        return "", 0.0, []

    confidences = [p[5] for p in parsed]
    avg_h = sum(p[4] for p in parsed) / len(parsed)
    avg_w = sum(p[3] for p in parsed) / len(parsed)
    vertical = force_vertical or (avg_h > avg_w * 2.5)
    ordered = sort_reading_order(parsed, vertical)

    words = [
        {
            "text": t,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "confidence": c,
            # The detector's four corners. They carry the line's angle, which
            # the rect above cannot, and the angled-callout pass groups a
            # slanted value with its deviation using them.
            "polygon": poly,
        }
        for t, x, y, w, h, c, poly in ordered
    ]

    raw = (
        "".join(w["text"].strip() for w in words)
        if vertical
        else " ".join(w["text"].strip() for w in words)
    )

    conf = sum(confidences) / len(confidences) if confidences else 0.0
    return fix_engineering_symbols_light(raw), conf, words


# ── Drawing NOTES block ────────────────────────────────────────────────────
# One numbered note line: "1.MAT'L GRADE…", "2) HOT DIP…". The marker must be
# followed by a letter, so a decimal callout (".12R", "1.5") can never match.
# The "." or ")" separator is REQUIRED. Making it optional was tried and is a
# regression: it turns ordinary lines into false markers, which breaks the
# "markers must climb" guard below and the whole block is then rejected.
_NOTE_POINT_RE = re.compile(r"^\s*(\d{1,2})\s*[.)]\s*[A-Za-z]")
# The block's own heading. PaddleOCR reads the CAD "O" as a zero often enough
# on these sheets that both spellings are accepted.
_NOTES_HEADING_RE = re.compile(r"^\s*N[O0]TES?\s*[:.\-]?\s*$", re.I)
# The heading with whatever the detector ran it together with ("NOTES: 1.").
_NOTES_ANCHOR_RE = re.compile(r"^\s*N[O0]TES?\s*[:.\-]?(\s|$)", re.I)
# The column of point markers read as one object ("12345", "1.2.3.4.").
_NOTE_COLUMN_RE = re.compile(r"^\s*(?:\d\s*[.)]?\s*){3,}$")

# Words that belong to the title block or the revision table, never to the prose
# of a general note. On a real sheet the title block starts TEN PIXELS under the
# last note line, so no distance rule can separate them — the boundary has to be
# recognised by what the text says.
_TITLE_BLOCK_WORDS_RE = re.compile(
    r"\b(TOLERANCES?|DWG|DRAWN|CHECKED|APPROVED|SCALE|SHEET|UNITS?"
    r"|FRACTION|DECIMAL|REVISIONS|DESCRIPTION|ZONE|INCHES|MILLIMETERS)\b",
    re.I,
)


def _covers_digits(whole: str, part: str) -> bool:
    """
    True when ``part``'s digits all appear in ``whole``, in order.

    A subsequence rather than a substring: the upright pass may read a
    deviation pair in a different order from the levelled pass (``Ø33-0.05``
    against ``Ø33-0.1-0.05``), and that is still the same callout.
    """
    if not part:
        return False
    it = iter(whole)
    return all(c in it for c in part)


# One row of a limit dimension: an optional diameter mark and a decimal value.
_LIMIT_PART_RE = re.compile(r"^\s*([Øø⌀φ∅]?)\s*(\d*\.\d+)\s*$")
# A limit dimension read flat, both rows in one string: "Ø0.620φ0.612",
# "0.620/0.612", "Ø0.620 0.612". The separator is whatever the recogniser
# made of the row break (or of the Ø that sits between the rows).
_LIMIT_WHOLE_RE = re.compile(
    r"^\s*([Øø⌀φ∅]?)\s*(\d*\.\d+)\s*[Øø⌀φ∅/|\\ ]?\s*(\d*\.\d+)\s*$"
)
_DIAMETER_MARKS = "Øø⌀φ∅"


def parse_limit_pair(upper: str, lower: str) -> str | None:
    """
    Join the two rows of a limit dimension into one value, or ``None``.

    A limit dimension is drawn as its upper limit over its lower limit —
    ``Ø0.620`` over ``0.612`` — and is ONE callout, not two. Detection cuts
    the stack in every way but the right one (two columns, one row plus a
    stray, the whole read flat), so the rows are recognised as text and joined
    by this rule: two plain decimals of the same precision, no brackets or
    tolerance of their own, differing by a few percent at most. That last
    condition is what keeps a pair of independent stacked callouts apart —
    ``8.00`` over ``9.00`` on the same sheet is two lengths, and ``0.20`` over
    ``0.25`` two more.

    Returns ``"Ø0.620/0.612"``: upper limit first, exactly as printed.
    """
    m_up = _LIMIT_PART_RE.match(upper or "")
    m_lo = _LIMIT_PART_RE.match(lower or "")
    if not m_up or not m_lo:
        return None
    up_text, lo_text = m_up.group(2), m_lo.group(2)
    if len(up_text.split(".")[1]) != len(lo_text.split(".")[1]):
        return None
    try:
        up_val, lo_val = float(up_text), float(lo_text)
    except ValueError:
        return None
    if up_val == lo_val:
        return None
    spread = abs(up_val - lo_val)
    if spread > max(0.06 * max(up_val, lo_val), 0.02):
        return None
    symbol = "Ø" if (m_up.group(1) or m_lo.group(1)) else ""
    return f"{symbol}{up_text}/{lo_text}"


def limit_pair_from_read(text: str) -> str | None:
    """``"Ø0.620φ0.612"`` (a limit stack read flat) → ``"Ø0.620/0.612"``."""
    m = _LIMIT_WHOLE_RE.match(text or "")
    if not m:
        return None
    return parse_limit_pair(f"{m.group(1)}{m.group(2)}", m.group(3))


_LIMIT_VALUE_RE = re.compile(r"^([Øø⌀φ∅]?)(\d*\.\d+)/(\d*\.\d+)$")
_METRIC_LIMIT_RE = re.compile(r"^\[?(\d*\.\d+)/(\d*\.\d+)\]?$")


def join_dual_unit_limits(
    regions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    ``Ø0.620/0.612`` beside ``[15.75/15.54]`` → ``Ø0.620/0.612[15.75/15.54]``.

    On a dual-unit sheet a limit dimension carries its metric limits in
    brackets alongside, drawn as a second two-row stack. Each stack is read
    as one value by :meth:`OcrPipeline._read_limit_pair`; this joins the two
    when the right-hand one is the left one in millimetres (×25.4, within the
    rounding band the dual-unit repair uses), sits on the same rows, and is
    little further away than the stack is tall (the bracket sits in that gap).
    The bracketed half is removed.

    Returns ``(regions, removed)``.
    """
    from dual_unit import _tol

    out = list(regions)
    removed: list[dict[str, Any]] = []
    joined = True
    while joined:
        joined = False
        for i, left in enumerate(out):
            ml = _LIMIT_VALUE_RE.match(str(left.get("text") or "").strip())
            if not ml:
                continue
            lb = left["bbox"]
            for j, right in enumerate(out):
                if i == j:
                    continue
                mr = _METRIC_LIMIT_RE.match(str(right.get("text") or "").strip())
                if not mr:
                    continue
                rb = right["bbox"]
                gap = float(rb["x"]) - (float(lb["x"]) + float(lb["width"]))
                if gap < -0.2 * float(lb["height"]) or gap > 1.2 * float(lb["height"]):
                    continue
                y_overlap = min(
                    float(lb["y"]) + float(lb["height"]), float(rb["y"]) + float(rb["height"])
                ) - max(float(lb["y"]), float(rb["y"]))
                if y_overlap < 0.6 * min(float(lb["height"]), float(rb["height"])):
                    continue
                try:
                    metric = all(
                        abs(float(inch) * 25.4 - float(mm)) <= _tol(float(mm))
                        for inch, mm in (
                            (ml.group(2), mr.group(1)),
                            (ml.group(3), mr.group(2)),
                        )
                    )
                except ValueError:
                    continue
                if not metric:
                    continue
                x0 = min(float(lb["x"]), float(rb["x"]))
                y0 = min(float(lb["y"]), float(rb["y"]))
                x1 = max(float(lb["x"]) + float(lb["width"]), float(rb["x"]) + float(rb["width"]))
                y1 = max(float(lb["y"]) + float(lb["height"]), float(rb["y"]) + float(rb["height"]))
                left["text"] = (
                    f"{ml.group(1)}{ml.group(2)}/{ml.group(3)}[{mr.group(1)}/{mr.group(2)}]"
                )
                left["bbox"] = {
                    "x": round(x0, 1),
                    "y": round(y0, 1),
                    "width": round(x1 - x0, 1),
                    "height": round(y1 - y0, 1),
                }
                left["confidence"] = round(
                    min(float(left.get("confidence") or 0.0), float(right.get("confidence") or 0.0)), 4
                )
                # The joined text is a new value, so its category and balloon
                # label are re-derived rather than left on the inch half's.
                if "category" in left or "label" in left:
                    OcrPipeline._apply_feature_labels(left)
                removed.append(out.pop(j))
                joined = True
                break
            if joined:
                break
    return out, removed


class OcrPipeline:
    def __init__(self) -> None:
        self._configured_device = configured_ocr_device()
        self._paddle_runtime = PaddleRuntimeInfo()
        self._paddle = None
        self._paddle_available = False
        self._paddle_api = 0  # 3 = PaddleOCR 3.x (.predict), 2 = 2.x (.ocr)
        self._text_detector = None
        self._text_detector_available = False
        self._text_recognizer = None
        self._text_recognizer_available = False
        self._text_orientation = None
        self._text_orientation_available = False
        self._page_batch_recognition_available = False
        self._paddle_version = "unknown"
        self._init_errors: list[str] = []

    def load(self) -> None:
        self._load_paddle()

    def _device_kwargs(self) -> dict[str, str]:
        """Pass a device only when the operator explicitly selected one."""

        if self._configured_device is None:
            return {}
        return {"device": self._configured_device}

    def _load_paddle(self) -> None:
        try:
            import paddle  # type: ignore
            import paddleocr  # type: ignore
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._init_errors.append(f"PaddleOCR import: {exc}")
            return

        ver = str(getattr(paddleocr, "__version__", "0"))
        self._paddle_version = ver

        self._paddle_runtime = probe_paddle_runtime(paddle)
        device_error = validate_configured_device(
            self._configured_device,
            self._paddle_runtime,
        )
        if device_error:
            self._init_errors.append(device_error)
            return

        # Detect the API by VERSION, not by probing the constructor: PaddleOCR
        # 2.7.x silently swallows unknown 3.x kwargs, so a try/except on the
        # constructor would mis-detect 2.x as 3.x and then call .predict()
        # (which doesn't exist on 2.x) — yielding empty results.
        major_part = ver.split(".", 1)[0]
        major = int(major_part) if major_part.isdigit() else 0
        is_3x = major >= 3 and hasattr(PaddleOCR, "predict")

        if is_3x:
            try:
                
                self._paddle = PaddleOCR(
                    lang="en",
                    **self._device_kwargs(),

                    # The UI sends tightly cropped CAD dimensions, not full documents.
                    # Running these document-level models adds latency without helping OCR.
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    
                    # Handle crops rotated by 0°, 90°, 180°, or 270°.
                    #use_doc_orientation_classify=True,

                    # Retain text-line orientation for the current POC.
                    use_textline_orientation=True,
                    text_det_thresh=0.2,
                    text_det_box_thresh=0.4,
                )
                self._paddle_api = 3
                self._paddle_available = True
                self._load_text_detector(paddleocr)
                self._load_page_recognition_models(paddleocr)
                self._paddle_runtime = probe_paddle_runtime(paddle)
                return
            except Exception as exc:  # noqa: BLE001
                self._init_errors.append(f"PaddleOCR 3.x: {exc}")
                # Version detection already established that this is 3.x.
                # Never hide a bad GPU/device configuration by retrying the
                # same package with legacy 2.x constructor arguments.
                return

        # PaddleOCR 2.x.
        try:
            kwargs: dict[str, Any] = {
                "use_angle_cls": True,
                "lang": "en",
                "show_log": False,
            }
            if self._configured_device is not None:
                kwargs["use_gpu"] = is_gpu_device(self._configured_device)
                if is_gpu_device(self._configured_device):
                    device_parts = self._configured_device.split(":", 1)
                    if len(device_parts) == 2 and device_parts[1].isdigit():
                        kwargs["gpu_id"] = int(device_parts[1])
            try:
                self._paddle = PaddleOCR(
                    **kwargs,
                    det_db_thresh=0.2,
                    det_db_box_thresh=0.4,
                    rec_batch_num=6,
                )
            except TypeError:
                self._paddle = PaddleOCR(**kwargs)
            self._paddle_api = 2
            self._paddle_available = True
        except Exception as exc:  # noqa: BLE001
            self._init_errors.append(f"PaddleOCR 2.x: {exc}")

    def _load_text_detector(self, paddleocr_module: Any) -> None:
        """Load PaddleOCR 3.x's standalone detector for auto-balloon scans."""

        detector_class = getattr(paddleocr_module, "TextDetection", None)
        if detector_class is None:
            self._init_errors.append(
                "PaddleOCR TextDetection module is unavailable; "
                "auto-ballooning will use the reduced compatibility path"
            )
            return

        try:
            self._text_detector = detector_class(
                model_name="PP-OCRv5_mobile_det",
                thresh=0.2,
                box_thresh=0.4,
                **self._device_kwargs(),
            )
            self._text_detector_available = True
        except Exception as exc:  # noqa: BLE001
            self._init_errors.append(f"PaddleOCR TextDetection: {exc}")

    def _load_page_recognition_models(self, paddleocr_module: Any) -> None:
        """Load reusable recognition-only modules for whole-page batches."""

        recognizer_class = getattr(paddleocr_module, "TextRecognition", None)
        orientation_class = getattr(
            paddleocr_module,
            "TextLineOrientationClassification",
            None,
        )
        if recognizer_class is None or orientation_class is None:
            self._init_errors.append(
                "PaddleOCR standalone TextRecognition/TextLineOrientationClassification "
                "modules are unavailable"
            )
            return

        try:
            self._text_recognizer = recognizer_class(
                model_name="PP-OCRv6_medium_rec",
                **self._device_kwargs(),
            )
            self._text_recognizer_available = True
        except Exception as exc:  # noqa: BLE001
            self._init_errors.append(f"PaddleOCR TextRecognition: {exc}")

        try:
            self._text_orientation = orientation_class(
                model_name="PP-LCNet_x1_0_textline_ori",
                **self._device_kwargs(),
            )
            self._text_orientation_available = True
        except Exception as exc:  # noqa: BLE001
            self._init_errors.append(
                f"PaddleOCR TextLineOrientationClassification: {exc}"
            )

        self._page_batch_recognition_available = bool(
            self._text_recognizer_available
            and self._text_orientation_available
        )

    @property
    def status(self) -> dict[str, Any]:
        return {
            "paddleocr": self._paddle_available,
            "paddleocr_version": self._paddle_version,
            "paddleocr_api": self._paddle_api,
            "ocr_device_requested": configured_device_label(
                self._configured_device
            ),
            "ocr_device_effective": effective_device(
                self._configured_device,
                self._paddle_runtime,
            ),
            "paddle_global_device": self._paddle_runtime.global_device,
            "paddle_cuda_compiled": self._paddle_runtime.cuda_compiled,
            "paddle_available_devices": list(
                self._paddle_runtime.available_devices
            ),
            "paddle_device_probe_error": self._paddle_runtime.probe_error,
            "document_orientation_classify": False,
            "document_unwarping": False,
            "text_detector": self._text_detector_available,
            "text_recognizer": self._text_recognizer_available,
            "text_line_orientation": self._text_orientation_available,
            "page_batch_recognition": self._page_batch_recognition_available,
            "auto_balloon_detection_mode": (
                "detector_only"
                if self._text_detector_available
                else "reduced_ocr_compatibility"
            ),
            "trocr": False,
            "symbol_vision": True,
            "errors": self._init_errors,
        }

    def _predict_3x(self, arr: np.ndarray) -> Any:
        """Run PaddleOCR 3.x; paddle_parse normalizes its Result objects."""
        return self._paddle.predict(arr)

    def _predict_text_detector(self, arr: np.ndarray) -> Any:
        """Run the standalone PaddleOCR detector without recognition."""

        # The image has already been deliberately bounded/upscaled by the
        # adaptive caller. Override the model's generic document limit so it
        # does not silently downscale a 2400px engineering drawing again.
        return self._text_detector.predict(
            arr,
            batch_size=1,
            limit_side_len=max(arr.shape[:2]),
            limit_type="max",
        )

    def _run_text_detector(
        self,
        image: Image.Image,
    ) -> list[tuple[float, float, float, float, float]]:
        """Return detector-only boxes in the supplied image coordinates."""

        if not self._text_detector_available or self._text_detector is None:
            raise RuntimeError("The standalone PaddleOCR text detector is unavailable")
        try:
            result = self._predict_text_detector(np.asarray(image.convert("RGB")))
        except Exception as exc:
            raise RuntimeError(
                f"PaddleOCR {self._paddle_version} detector-only inference failed: {exc}"
            ) from exc
        return extract_paddle_detection_boxes(result)

    def _run_text_detector_regions(
        self,
        image: Image.Image,
    ) -> list[dict[str, Any]]:
        """Return standalone detector boxes with their source polygons."""

        if not self._text_detector_available or self._text_detector is None:
            raise RuntimeError("The standalone PaddleOCR text detector is unavailable")
        try:
            result = self._predict_text_detector(np.asarray(image.convert("RGB")))
        except Exception as exc:
            raise RuntimeError(
                f"PaddleOCR {self._paddle_version} detector-only inference failed: {exc}"
            ) from exc
        return extract_paddle_detection_regions(result)

    @staticmethod
    def _module_inputs(images: list[Image.Image]) -> list[np.ndarray]:
        return [np.asarray(image.convert("RGB")) for image in images]

    def _predict_text_orientations(
        self,
        images: list[Image.Image],
        *,
        batch_size: int,
    ) -> list[tuple[int, float]]:
        if (
            not getattr(self, "_text_orientation_available", False)
            or self._text_orientation is None
        ):
            raise RuntimeError(
                "Whole-page auto-ballooning requires PaddleOCR's standalone "
                "text-line orientation model"
            )
        try:
            output = list(
                self._text_orientation.predict(
                    input=self._module_inputs(images),
                    batch_size=batch_size,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR standalone text-line orientation inference failed: "
                f"{exc}"
            ) from exc
        if len(output) != len(images):
            raise RuntimeError(
                "PaddleOCR text-line orientation returned "
                f"{len(output)} results for {len(images)} crops"
            )
        return [extract_text_orientation_result(item) for item in output]

    def _predict_text_recognition(
        self,
        images: list[Image.Image],
        *,
        batch_size: int,
    ) -> list[tuple[str, float]]:
        if (
            not getattr(self, "_text_recognizer_available", False)
            or self._text_recognizer is None
        ):
            raise RuntimeError(
                "Whole-page auto-ballooning requires PaddleOCR's standalone "
                "text recognition model"
            )
        try:
            output = list(
                self._text_recognizer.predict(
                    input=self._module_inputs(images),
                    batch_size=batch_size,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR standalone text recognition inference failed: "
                f"{exc}"
            ) from exc
        if len(output) != len(images):
            raise RuntimeError(
                "PaddleOCR text recognition returned "
                f"{len(output)} results for {len(images)} crops"
            )
        return [extract_text_recognition_result(item) for item in output]

    def _recognize_page_batch(
        self,
        crops: list[Image.Image],
        *,
        batch_size: int,
        profile: str = "batch_recognition",
    ) -> list[dict[str, Any]]:
        """Orient and recognize a crop batch without running text detection."""

        if not getattr(self, "_page_batch_recognition_available", False):
            raise RuntimeError(
                "Whole-page auto-ballooning requires standalone PaddleOCR "
                "orientation and recognition models; the slow full-pipeline "
                "fallback is intentionally disabled"
            )
        if not crops:
            return []

        from dimension_digits import correct_numeric_confusables

        if profile == "recovery_expanded_sharp":
            prepared = [
                sharpen_rgb(
                    primary_oriented(upscale_min_edge(pad_image(crop)))
                )
                for crop in crops
            ]
        else:
            prepared = [prepare_primary_ocr_variant(crop)[1] for crop in crops]
        orientations = self._predict_text_orientations(
            prepared,
            batch_size=batch_size,
        )
        oriented_images = [
            image.rotate(180, expand=False, fillcolor=(255, 255, 255))
            if degrees == 180
            else image
            for image, (degrees, _score) in zip(prepared, orientations)
        ]
        recognized = self._predict_text_recognition(
            oriented_images,
            batch_size=batch_size,
        )

        results: list[dict[str, Any]] = []
        for crop, (raw_text, confidence), (degrees, orientation_score) in zip(
            crops,
            recognized,
            orientations,
        ):
            fixed, corrected = correct_numeric_confusables(raw_text)
            text_hints = detect_prefix_from_ocr_text(fixed)
            composed = compose_engineering_dimension(
                fixed,
                crop,
                text_hints,
            )
            text = fix_engineering_symbols_light(composed.text).strip()
            results.append(
                {
                    "text": text,
                    "raw_ocr": raw_text,
                    "confidence": float(confidence),
                    "confusable_corrected": bool(corrected),
                    "agreement": 1.0 if text else 0.0,
                    # Confidence and orientation quality help rank alternative
                    # reads; they are not approval gates for a usable value.
                    "needs_review": False,
                    "type": composed.kind,
                    "engine": "paddleocr+compose"
                    if composed.applied
                    else "paddleocr",
                    "orientation": (
                        "vertical" if is_vertical_dimension(crop) else "horizontal"
                    ),
                    "rotation": 0,
                    "symbols_detected": symbols_to_dict(text_hints),
                    "orientation_correction": degrees,
                    "orientation_confidence": float(orientation_score),
                    "ocr_profile": profile,
                }
            )
        return results

    def _run_paddle(
        self,
        image: Image.Image,
        det: bool = True,
        *,
        timing_label: str | None = None,
        timings: list[dict[str, Any]] | None = None,
    ) -> tuple[str, float, list]:
        """
        Run one PaddleOCR prediction.

        When ``timings`` is supplied, the complete wall-clock time for the pass is
        appended to it. This includes image conversion, inference and result parsing.
        """
        if not self._paddle_available or self._paddle is None:
            return "", 0.0, []

        started = perf_counter()

        try:
            arr = np.array(image.convert("RGB"))

            if self._paddle_api == 3:
                try:
                    result: Any = self._predict_3x(arr)
                except Exception as exc:
                    raise RuntimeError(
                        f"PaddleOCR {self._paddle_version} inference failed "
                        f"(API {self._paddle_api}, det={det}): {exc}"
                    ) from exc
            else:
                try:
                    result = (
                        self._paddle.ocr(arr, cls=True)
                        if det
                        else self._paddle.ocr(arr, det=False, cls=True)
                    )
                except TypeError:
                    try:
                        # Older 2.x builds do not accept every optional keyword.
                        result = self._paddle.ocr(arr)
                    except Exception as exc:
                        raise RuntimeError(
                            f"PaddleOCR {self._paddle_version} inference failed "
                            f"(API {self._paddle_api}, det={det}): {exc}"
                        ) from exc
                except Exception as exc:
                    raise RuntimeError(
                        f"PaddleOCR {self._paddle_version} inference failed "
                        f"(API {self._paddle_api}, det={det}): {exc}"
                    ) from exc

            try:
                return assemble_paddle_lines(result, force_vertical=False)
            except Exception as exc:
                raise RuntimeError(
                    f"PaddleOCR {self._paddle_version} result parsing failed "
                    f"(API {self._paddle_api}, det={det}): {exc}"
                ) from exc
        finally:
            if timings is not None:
                timings.append(
                    {
                        "stage": timing_label or "paddle",
                        "elapsed_ms": round(
                            (perf_counter() - started) * 1000,
                            2,
                        ),
                        "width": image.width,
                        "height": image.height,
                        "det": det,
                    }
                )

    def recognize_paddle(
        self,
        image: Image.Image,
        dumper: StepDumper | None = None,
        timings: list[dict[str, Any]] | None = None,
        *,
        max_predictions: int | None = None,
    ) -> tuple[str, float, list, float, bool]:
        """
        Run preprocessing variants and vote across the candidates.

        ``max_predictions=1`` selects the fast primary variant and does not
        construct retry variants. The default preserves the accuracy profile.

        Returns (text, confidence, words, agreement, corrected) where
        `agreement` is the fraction of candidates that agree with the winning
        value and `corrected` is True when a numeric-confusable fix was applied.
        """
        from collections import defaultdict

        from dimension_digits import correct_numeric_confusables
        from ocr_select import digit_quality_score

        if max_predictions is not None and max_predictions < 1:
            raise ValueError("max_predictions must be at least one")

        # 3.x .predict() always detects, so det=False is redundant there.
        det_modes = (True,) if self._paddle_api == 3 else (True, False)

        # Group candidates by their normalized text.
        groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"score": 0.0, "count": 0, "conf": 0.0,
                     "words": [], "corrected": False}
        )
        total = 0
        predictions_run = 0

        # A candidate above the strict confidence threshold wins immediately.
        # Variants still execute sequentially; no parallel model calls are introduced.
        early_winner: dict[str, Any] | None = None
        candidates_log: list[dict[str, Any]] = []

        variants = (
            [prepare_primary_ocr_variant(image)]
            if max_predictions == 1
            else prepare_ocr_variants(image)
        )
        for _name, variant in variants:
            for det in det_modes:
                if (
                    max_predictions is not None
                    and predictions_run >= max_predictions
                ):
                    break
                predictions_run += 1
                text, conf, words = self._run_paddle(
                    variant,
                    det=det,
                    timing_label=f"variant:{_name}",
                    timings=timings,
                )

                if not text:
                    candidates_log.append(
                        {
                            "variant": _name,
                            "det": det,
                            "raw": "",
                            "skipped": "empty",
                        }
                    )
                    continue

                fixed, corrected = correct_numeric_confusables(text)
                normalized = normalize_cad_number_string(fixed)
                score = digit_quality_score(normalized, conf)
                early_accepted = bool(
                    normalized.strip()
                    and conf > EARLY_ACCEPT_CONFIDENCE
                )

                total += 1
                g = groups[normalized]
                g["score"] += score
                g["count"] += 1
                g["corrected"] = g["corrected"] or corrected

                # Deskew variants read text in a rotated frame, so their word
                # boxes must not become the group's boxes (they'd misplace the
                # balloon). They still vote on the text value; bbox falls back.
                is_deskew = _name.startswith("dsk")
                if conf > g["conf"]:
                    g["conf"] = conf
                    if not is_deskew:
                        g["words"] = words

                candidates_log.append(
                    {
                        "variant": _name,
                        "det": det,
                        "raw": text,
                        "after_confusables": fixed,
                        "normalized": normalized,
                        "score": round(score, 2),
                        "confidence": conf,
                        "corrected": corrected,
                        "early_accepted": early_accepted,
                    }
                )

                if early_accepted:
                    early_winner = {
                        "variant": _name,
                        "det": det,
                        "text": normalized,
                        "confidence": conf,
                        # Same rule as the vote: a deskewed read's boxes are
                        # in the levelled frame, so hand back none and let the
                        # text-bbox fallback locate the value in the crop.
                        "words": [] if is_deskew else words,
                        "corrected": corrected,
                    }
                    break

            # Stop before constructing or running the next preprocessing variant.
            if (
                early_winner is not None
                or (
                    max_predictions is not None
                    and predictions_run >= max_predictions
                )
            ):
                break

        if dumper and dumper.active:
            dumper.stage(
                "paddle_candidates",
                {
                    "paddle_api": self._paddle_api,
                    "early_stop_threshold": EARLY_ACCEPT_CONFIDENCE,
                    "early_stop": (
                        {
                            "variant": early_winner["variant"],
                            "det": early_winner["det"],
                            "text": early_winner["text"],
                            "confidence": early_winner["confidence"],
                        }
                        if early_winner is not None
                        else None
                    ),
                    "total_passes": total,
                    "predictions_run": predictions_run,
                    "prediction_limit": max_predictions,
                    "candidates": candidates_log,
                    "groups": {
                        k: {
                            "votes": v["count"],
                            "total_score": round(v["score"], 2),
                            "confidence": v["conf"],
                            "corrected": v["corrected"],
                        }
                        for k, v in groups.items()
                    },
                },
            )

        if early_winner is not None:
            if dumper and dumper.active:
                dumper.stage(
                    "paddle_winner",
                    {
                        "text": early_winner["text"],
                        "confidence": early_winner["confidence"],
                        "agreement": 1.0,
                        "corrected": early_winner["corrected"],
                        "selection": "early_confidence",
                        "threshold": EARLY_ACCEPT_CONFIDENCE,
                    },
                )

            return (
                early_winner["text"],
                early_winner["confidence"],
                early_winner["words"],
                1.0,
                early_winner["corrected"],
            )

        if not groups:
            return "", 0.0, [], 0.0, False

        best_text = max(groups, key=lambda k: groups[k]["score"])
        g = groups[best_text]
        agreement = g["count"] / total if total else 0.0
        if dumper and dumper.active:
            dumper.stage(
                "paddle_winner",
                {
                    "text": best_text,
                    "confidence": g["conf"],
                    "agreement": round(agreement, 3),
                    "corrected": g["corrected"],
                },
            )
        return best_text, g["conf"], g["words"], round(agreement, 3), g["corrected"]

    @staticmethod
    def _text_bbox_from_main_words(
        words: list[dict[str, Any]],
        *,
        image: Image.Image,
        prep: Image.Image,
        oriented: Image.Image,
        vertical: bool,
    ) -> dict[str, float] | None:
        """
        Map the winning main OCR word boxes back to received-image pixels.

        Main OCR already detects word boxes. Reusing those coordinates avoids
        an additional full PaddleOCR prediction solely for balloon placement.
        """
        if not words:
            return None

        try:
            x0 = min(float(word["x"]) for word in words)
            y0 = min(float(word["y"]) for word in words)
            x1 = max(
                float(word["x"]) + float(word["width"])
                for word in words
            )
            y1 = max(
                float(word["y"]) + float(word["height"])
                for word in words
            )
        except (KeyError, TypeError, ValueError):
            return None

        # prepare_ocr_variants() adds a 36-pixel border around `oriented`.
        # `oriented` already has a long edge >= 280, so that inner image is
        # never enlarged again by upscale_min_edge().
        inner_pad = 36.0
        x0 -= inner_pad
        y0 -= inner_pad
        x1 -= inner_pad
        y1 -= inner_pad

        x0 = max(0.0, min(x0, float(oriented.width)))
        y0 = max(0.0, min(y0, float(oriented.height)))
        x1 = max(0.0, min(x1, float(oriented.width)))
        y1 = max(0.0, min(y1, float(oriented.height)))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None

        bbox = {
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
        }

        # Undo the clockwise orientation used for vertical dimensions.
        if vertical:
            bbox = bbox_from_oriented_to_original(
                bbox,
                prep.width,
                prep.height,
            )

        # Undo the outer upscale and 36-pixel padding added in recognize().
        outer_pad = 36.0
        padded_width = image.width + int(outer_pad * 2)
        padded_height = image.height + int(outer_pad * 2)
        scale_x = prep.width / max(padded_width, 1)
        scale_y = prep.height / max(padded_height, 1)

        x0 = bbox["x"] / scale_x - outer_pad
        y0 = bbox["y"] / scale_y - outer_pad
        x1 = (bbox["x"] + bbox["width"]) / scale_x - outer_pad
        y1 = (bbox["y"] + bbox["height"]) / scale_y - outer_pad

        x0 = max(0.0, min(x0, float(image.width)))
        y0 = max(0.0, min(y0, float(image.height)))
        x1 = max(0.0, min(x1, float(image.width)))
        y1 = max(0.0, min(y1, float(image.height)))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None

        return {
            "x": round(x0, 1),
            "y": round(y0, 1),
            "width": round(x1 - x0, 1),
            "height": round(y1 - y0, 1),
        }

    def _detect_text_bbox(
        self,
        image: Image.Image,
        timings: list[dict[str, Any]] | None = None,
    ) -> dict[str, float] | None:
        """
        Tight bounding box of the detected text in *received-image* pixels.

        Uses the same primary orientation as OCR (rotate vertical crops -90°)
        so detection boxes line up with the read value, then maps back into
        received-image space. Returns None if no text.
        """
        vertical = is_vertical_dimension(image)
        iw, ih = image.size
        det_image = primary_oriented(image) if vertical else image

        pad = 36
        padded = pad_image(det_image, px=pad)
        pw, ph = padded.size
        edge = max(pw, ph)
        factor = 1.0 if edge >= 280 else min(12.0, 280 / max(edge, 1))
        det_img = (
            padded.resize((int(pw * factor), int(ph * factor)),
                          Image.Resampling.LANCZOS)
            if factor > 1.0
            else padded
        )

        _, _, words = self._run_paddle(
            clahe_rgb(det_img),
            det=True,
            timing_label="text_bbox",
            timings=timings,
        )
        if not words:
            return None

        x0 = min(w["x"] for w in words)
        y0 = min(w["y"] for w in words)
        x1 = max(w["x"] + w["width"] for w in words)
        y1 = max(w["y"] + w["height"] for w in words)

        # Invert upscale, then padding -> oriented-image coordinates.
        x0, y0, x1, y1 = (v / factor for v in (x0, y0, x1, y1))
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 - pad, y1 - pad

        ow, oh = det_image.size
        x0, x1 = max(0.0, min(x0, ow)), max(0.0, min(x1, ow))
        y0, y1 = max(0.0, min(y0, oh)), max(0.0, min(y1, oh))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None

        bbox = {
            "x": round(x0, 1),
            "y": round(y0, 1),
            "width": round(x1 - x0, 1),
            "height": round(y1 - y0, 1),
        }
        if vertical:
            bbox = bbox_from_oriented_to_original(bbox, iw, ih)
        return bbox

    @staticmethod
    def _text_bbox_plausible(
        text_bbox: dict[str, float],
        union_bbox: dict[str, float],
    ) -> bool:
        """
        Reject tight boxes that sit outside or dwarf the cluster union — a
        common failure mode on vertical CAD text where det finds a stray tick.
        """
        tx0, ty0 = text_bbox["x"], text_bbox["y"]
        tx1 = tx0 + text_bbox["width"]
        ty1 = ty0 + text_bbox["height"]
        ux0, uy0 = union_bbox["x"], union_bbox["y"]
        ux1 = ux0 + union_bbox["width"]
        uy1 = uy0 + union_bbox["height"]

        ix0, iy0 = max(tx0, ux0), max(ty0, uy0)
        ix1, iy1 = min(tx1, ux1), min(ty1, uy1)
        if ix1 <= ix0 or iy1 <= iy0:
            return False

        inter = (ix1 - ix0) * (iy1 - iy0)
        tb_area = max(text_bbox["width"] * text_bbox["height"], 1.0)
        ub_area = max(union_bbox["width"] * union_bbox["height"], 1.0)
        if inter / tb_area < 0.45:
            return False
        if inter / ub_area < 0.08:
            return False
        if tb_area < 0.15 * ub_area and union_bbox["height"] > union_bbox["width"] * 1.3:
            return False
        return True

    def detect_regions(
        self,
        image: Image.Image,
        *,
        progress_callback: Callable[..., None] | None = None,
        thorough: bool = True,
    ) -> list[dict[str, Any]]:
        """Propose text boxes in the received image's original coordinates.

        A top-level auto-balloon scan uses the bounded Milestone 2B adaptive
        cascade. Oversized-cluster refinement can request the earlier quick
        path so one refinement does not recursively launch another page scan.
        """

        if not thorough:
            return self._detect_regions_quick(image)

        from detection_passes import (
            DETECTION_FALLBACK_MAX_REFINEMENT_REGIONS,
            DETECTION_MAX_REFINEMENT_REGIONS,
            DETECTION_REFINEMENT_TARGET_EDGE,
            build_detection_pass_plan,
            build_primary_detection_image,
            build_refinement_regions,
            map_deskewed_box_to_original,
            map_quarter_turn_box_to_source,
            prepare_detection_source,
            refinement_rotation,
            rotate_for_detection,
        )
        from region_detect import propose_text_regions

        prepared = prepare_detection_source(image)
        pass_plan = build_detection_pass_plan(prepared.image)
        primary_total = len(pass_plan)
        morphology_candidates: list[dict[str, Any]] = []
        primary_candidates: list[dict[str, Any]] = []

        def emit(
            *,
            phase: str,
            completed: int,
            total: int,
            state: str,
            label: str,
            pass_current: int,
            proposals: int,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    phase=phase,
                    completed=completed,
                    total=total,
                    state=state,
                    label=label,
                    proposals=proposals,
                    deskew_angle=prepared.correction_angle,
                    pass_current=pass_current,
                    pass_total=total,
                    tile_current=1,
                    tile_total=1,
                )

        # Morphology runs once at source resolution.  It is inexpensive enough
        # to preserve faint/small proposals that the bounded page detector may
        # miss, and it defines where local detector retries are useful.
        emit(
            phase="proposing",
            completed=0,
            total=1,
            state="running",
            label="source-resolution morphology proposals",
            pass_current=1,
            proposals=0,
        )
        for box in propose_text_regions(prepared.image):
            morphology_candidates.append(
                {
                    **box,
                    "text": "",
                    "conf": 0.0,
                    "_detection_pass": "morphology",
                }
            )
        emit(
            phase="proposing",
            completed=1,
            total=1,
            state="completed",
            label="source-resolution morphology proposals",
            pass_current=1,
            proposals=len(morphology_candidates),
        )

        primary_image = build_primary_detection_image(prepared.image)
        detector_mode = (
            "detector only"
            if getattr(self, "_text_detector_available", False)
            else "reduced OCR compatibility"
        )
        for plan_index, spec in enumerate(pass_plan, start=1):
            pass_current = plan_index
            label = f"{spec.label}, {detector_mode}"
            emit(
                phase="detecting",
                completed=plan_index - 1,
                total=primary_total,
                state="running",
                label=label,
                pass_current=pass_current,
                proposals=len(morphology_candidates) + len(primary_candidates),
            )
            rotated = rotate_for_detection(primary_image, spec.rotation_cw)
            for box in self._detector_only_boxes(
                rotated,
                target_long_edge=spec.target_long_edge,
            ):
                mapped = map_quarter_turn_box_to_source(
                    box,
                    spec.rotation_cw,
                    prepared.image.size,
                )
                if mapped is None:
                    continue
                mapped["_detection_pass"] = spec.label
                primary_candidates.append(mapped)
            emit(
                phase="detecting",
                completed=plan_index,
                total=primary_total,
                state="completed",
                label=label,
                pass_current=pass_current,
                proposals=len(morphology_candidates) + len(primary_candidates),
            )

        max_refinements = (
            DETECTION_MAX_REFINEMENT_REGIONS
            if getattr(self, "_text_detector_available", False)
            else DETECTION_FALLBACK_MAX_REFINEMENT_REGIONS
        )
        refinement_regions = build_refinement_regions(
            morphology_candidates,
            primary_candidates,
            source_size=prepared.image.size,
            max_regions=max_refinements,
        )
        refinement_candidates: list[dict[str, Any]] = []
        source_width, source_height = prepared.image.size
        for region_index, region in enumerate(refinement_regions, start=1):
            rotation = refinement_rotation(region)
            label = (
                f"local coverage gap {region_index}, {rotation} deg, "
                f"{DETECTION_REFINEMENT_TARGET_EDGE}px"
            )
            emit(
                phase="refining",
                completed=region_index - 1,
                total=len(refinement_regions),
                state="running",
                label=label,
                pass_current=region_index,
                proposals=(
                    len(morphology_candidates)
                    + len(primary_candidates)
                    + len(refinement_candidates)
                ),
            )

            x0 = max(0, int(region["x"]))
            y0 = max(0, int(region["y"]))
            x1 = min(
                source_width,
                max(x0 + 1, int(region["x"] + region["w"] + 0.999)),
            )
            y1 = min(
                source_height,
                max(y0 + 1, int(region["y"] + region["h"] + 0.999)),
            )
            crop = build_primary_detection_image(
                prepared.image.crop((x0, y0, x1, y1))
            )
            rotated = rotate_for_detection(crop, rotation)
            for box in self._detector_only_boxes(
                rotated,
                target_long_edge=DETECTION_REFINEMENT_TARGET_EDGE,
            ):
                local = map_quarter_turn_box_to_source(
                    box,
                    rotation,
                    crop.size,
                )
                if local is None:
                    continue
                local["x"] = float(local["x"]) + x0
                local["y"] = float(local["y"]) + y0
                local["_detection_pass"] = label
                refinement_candidates.append(local)

            emit(
                phase="refining",
                completed=region_index,
                total=len(refinement_regions),
                state="completed",
                label=label,
                pass_current=region_index,
                proposals=(
                    len(morphology_candidates)
                    + len(primary_candidates)
                    + len(refinement_candidates)
                ),
            )

        # All three proposal sources currently use deskewed coordinates. Restore
        # them once, after the adaptive plan is complete, then apply only coarse
        # same-position suppression. Logical-object deduplication is Milestone 3.
        # A morphology proposal is a connected blob, not a read line, so a
        # shaded region or a hatched area can come back as one box covering a
        # large part of the sheet. Left in, it bridges every callout it touches
        # into a single cluster: on a real drawing an 877x628 blob (9% of the
        # sheet) fused three dimensions and the notes block into one unusable
        # read. The detector passes are unaffected — they emit one box per text
        # line — so the guard is scoped to morphology and measured against the
        # detector's own line height.
        line_scale = OcrPipeline._text_scale(primary_candidates)
        if line_scale:
            blob_limit = line_scale * 6.0
            morphology_candidates = [
                b
                for b in morphology_candidates
                if min(b["w"], b["h"]) <= blob_limit
            ]

        restored: list[dict[str, Any]] = []
        for candidate in (
            morphology_candidates + primary_candidates + refinement_candidates
        ):
            mapped = map_deskewed_box_to_original(
                candidate,
                prepared.image.size,
                prepared.correction_angle,
            )
            if mapped is not None:
                restored.append(mapped)
        return self._dedupe_det_boxes(restored, overlap_thresh=0.62)

    def _detect_regions_quick(self, image: Image.Image) -> list[dict[str, Any]]:
        """Earlier two-orientation proposer used only for local refinement."""

        from region_detect import opencv_available, propose_text_regions

        paddle_boxes = self._detect_regions_paddle_ink(image)
        if paddle_boxes:
            return paddle_boxes

        boxes: list[dict[str, Any]] = []
        if opencv_available():
            cv_boxes = propose_text_regions(image)
            if cv_boxes:
                boxes = [{**b, "text": "", "conf": 0.0} for b in cv_boxes]

        ocr_boxes = self._detect_regions_ocr(image)
        if not boxes:
            return ocr_boxes

        # Add substantial Paddle det boxes only when morphology missed a value.
        return self._merge_region_proposals(boxes, ocr_boxes)

    def _detector_only_boxes(
        self,
        pil_img: Image.Image,
        *,
        target_long_edge: int,
    ) -> list[dict[str, Any]]:
        """Run bounded localization and restore boxes to ``pil_img`` pixels.

        PaddleOCR 3.x uses its standalone ``TextDetection`` model here, so no
        text recognition is performed.  Older compatible installations use
        the same bounded image with the general OCR pipeline, but the adaptive
        caller reduces their local retry cap separately.
        """

        pad = 24
        padded = pad_image(pil_img, px=pad)
        pw, ph = padded.size
        edge = max(pw, ph)
        factor = min(6.0, target_long_edge / max(edge, 1))
        if abs(factor - 1.0) > 0.01:
            detector_input = padded.resize(
                (
                    max(1, int(round(pw * factor))),
                    max(1, int(round(ph * factor))),
                ),
                Image.Resampling.LANCZOS,
            )
        else:
            factor = 1.0
            detector_input = padded

        detections: list[dict[str, Any]] = []
        if getattr(self, "_text_detector_available", False):
            detections = [
                {**region, "text": ""}
                for region in self._run_text_detector_regions(detector_input)
            ]
        else:
            _, _, words = self._run_paddle(detector_input, det=True)
            detections = [
                {
                    "x": float(word["x"]),
                    "y": float(word["y"]),
                    "width": float(word["width"]),
                    "height": float(word["height"]),
                    "confidence": float(word.get("confidence", 0.0)),
                    "text": str(word.get("text", "")),
                    "polygon": [
                        [float(word["x"]), float(word["y"])],
                        [
                            float(word["x"]) + float(word["width"]),
                            float(word["y"]),
                        ],
                        [
                            float(word["x"]) + float(word["width"]),
                            float(word["y"]) + float(word["height"]),
                        ],
                        [
                            float(word["x"]),
                            float(word["y"]) + float(word["height"]),
                        ],
                    ],
                }
                for word in words
            ]

        return [
            {
                "x": float(region["x"]) / factor - pad,
                "y": float(region["y"]) / factor - pad,
                "w": float(region["width"]) / factor,
                "h": float(region["height"]) / factor,
                "text": str(region.get("text", "")),
                "conf": float(region.get("confidence", 0.0)),
                "polygon": [
                    [
                        float(point[0]) / factor - pad,
                        float(point[1]) / factor - pad,
                    ]
                    for point in region.get("polygon", [])
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ],
            }
            for region in detections
        ]

    def _paddle_det_boxes(
        self,
        pil_img: Image.Image,
        *,
        target_long_edge: int = 1100,
        preprocess: bool = True,
    ) -> list[dict[str, Any]]:
        """Paddle detection with boxes restored to the supplied image pixels."""

        pad = 24
        padded = pad_image(pil_img, px=pad)
        pw, ph = padded.size
        edge = max(pw, ph)
        factor = min(12.0, max(1.0, target_long_edge / max(edge, 1)))
        det_img = (
            padded.resize((int(pw * factor), int(ph * factor)),
                          Image.Resampling.LANCZOS)
            if factor > 1.0
            else padded
        )
        # Legacy callers pass an ink image and expect this final CLAHE step.
        # Accuracy-first callers supply an already prepared pass variant.
        paddle_input = clahe_rgb(det_img) if preprocess else det_img
        _, _, words = self._run_paddle(paddle_input, det=True)
        out: list[dict[str, Any]] = []
        for w in words:
            # paddle_parse names a word's corners "polygon" on this pipeline;
            # this method was written against "quad". Same four points.
            quad = w.get("quad") or w.get("polygon")
            out.append(
                {
                    "x": w["x"] / factor - pad,
                    "y": w["y"] / factor - pad,
                    "w": w["width"] / factor,
                    "h": w["height"] / factor,
                    "text": w.get("text", ""),
                    "conf": float(w.get("confidence", 0.0)),
                    # The detector's own corners, back in this image's pixels.
                    # They carry the text's angle; the axis-aligned box above
                    # does not. The angled pass groups a slanted value with its
                    # deviation using them, so without this "Ø18H10 +0.070"
                    # comes back as two separate balloons.
                    "quad": (
                        [(px / factor - pad, py / factor - pad) for px, py in quad]
                        if quad
                        else None
                    ),
                }
            )
        return out

    def _detect_regions_paddle_ink(self, image: Image.Image) -> list[dict[str, Any]]:
        """
        Learned text-line proposer: two-pass PaddleOCR detection on the ink mask.

        ``cad_ink_to_gray`` isolates the (usually blue) dimension strokes from
        the black geometry. PaddleOCR's detector reads horizontal text well but
        misses 90°-rotated CAD dimensions, so we run it twice — on the upright
        image (horizontal text, e.g. angles) and on a 90°-clockwise rotation
        (vertical Ø dimensions) — and map the rotated boxes back into the upright
        frame. Returns {x, y, w, h, text, conf} dicts, deduped across passes.
        """
        from image_preprocess import cad_ink_to_gray

        ink = cad_ink_to_gray(image).convert("RGB")
        iw, ih = image.size

        # Collect candidates from both passes (keep every box + its read text).
        cand: list[dict[str, Any]] = list(self._paddle_det_boxes(ink))

        # Vertical pass: rotate 90° CW so upright-vertical text reads horizontally
        # (PaddleOCR's strength). A box (xr, yr, wr, hr) in the rotated frame maps
        # back to the upright frame as: x = yr, y = ih - (xr + wr), w = hr, h = wr.
        rotated = ink.transpose(Image.ROTATE_270)
        for b in self._paddle_det_boxes(rotated):
            cand.append(
                {
                    "x": b["y"],
                    "y": ih - (b["x"] + b["w"]),
                    "w": b["h"],
                    "h": b["w"],
                    "text": b["text"],
                    "conf": b["conf"],
                }
            )

        # Ink mask: the discriminator between a real detection (sits tightly on
        # its strokes) and a phantom (the other pass re-reads rotated text into a
        # displaced box). Higher ink-fill = the box actually covers text.
        ink_mask = None
        try:
            import cv2

            ink_gray = np.asarray(cad_ink_to_gray(image).convert("L"))
            _, ink_mask = cv2.threshold(
                ink_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        except Exception:  # noqa: BLE001 - ink check is best-effort
            ink_mask = None

        scored: list[dict[str, Any]] = []
        for b in cand:
            x0 = max(0.0, min(b["x"], iw))
            y0 = max(0.0, min(b["y"], ih))
            x1 = max(0.0, min(b["x"] + b["w"], iw))
            y1 = max(0.0, min(b["y"] + b["h"], ih))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            fill = 1.0
            if ink_mask is not None:
                roi = ink_mask[int(y0):int(y1), int(x0):int(x1)]
                fill = float((roi > 0).mean()) if roi.size else 0.0
                if fill < 0.025:
                    continue  # box sits on (near-)empty space -> phantom
            scored.append(
                {
                    "x": round(x0, 1),
                    "y": round(y0, 1),
                    "w": round(x1 - x0, 1),
                    "h": round(y1 - y0, 1),
                    "text": b["text"],
                    "conf": b["conf"],
                    "_fill": fill,
                }
            )
        kept = self._dedupe_det_boxes(scored)
        for b in kept:
            b.pop("_fill", None)
        return kept

    @staticmethod
    def _overlap_area(a: dict[str, Any], b: dict[str, Any]) -> float:
        ix0, iy0 = max(a["x"], b["x"]), max(a["y"], b["y"])
        ix1 = min(a["x"] + a["w"], b["x"] + b["w"])
        iy1 = min(a["y"] + a["h"], b["y"] + b["h"])
        return (ix1 - ix0) * (iy1 - iy0) if ix1 > ix0 and iy1 > iy0 else 0.0

    @classmethod
    def _dedupe_det_boxes(
        cls, boxes: list[dict[str, Any]], *, overlap_thresh: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Merge only *positional* duplicates from the two passes.

        We deliberately do NOT try to pick the "correct" box per value here — the
        two passes disagree (a value's correct box and a displaced phantom of it
        rarely overlap, and ink-fill can favour a phantom sitting on geometry).
        Instead we keep both and let the downstream ``recognize`` + worthiness
        filter drop the phantom (cropping its wrong location reads nothing) while
        keeping the real box. Only boxes at the *same* location are deduped, so a
        value detected twice in the same place isn't proposed twice.
        """
        ordered = sorted(boxes, key=lambda b: b.get(
            "_fill", 1.0), reverse=True)
        kept: list[dict[str, Any]] = []
        for b in ordered:
            barea = max(b["w"] * b["h"], 1.0)
            if any(
                cls._overlap_area(b, k) / min(barea, max(k["w"] * k["h"], 1.0))
                >= overlap_thresh
                for k in kept
            ):
                continue
            kept.append(b)
        return kept

    def _detect_regions_ocr(self, image: Image.Image) -> list[dict[str, Any]]:
        """
        Fallback proposer: a single geometry-preserving PaddleOCR detection pass
        (pad + upscale only, no rotation), coordinates inverted back to received
        image space — mirroring `_detect_text_bbox` but keeping each box.
        """
        pad = 36
        padded = pad_image(image, px=pad)
        pw, ph = padded.size
        edge = max(pw, ph)
        factor = 1.0 if edge >= 280 else min(12.0, 280 / max(edge, 1))
        det_img = (
            padded.resize((int(pw * factor), int(ph * factor)),
                          Image.Resampling.LANCZOS)
            if factor > 1.0
            else padded
        )

        _, _, words = self._run_paddle(clahe_rgb(det_img), det=True)
        iw, ih = image.size
        regions: list[dict[str, Any]] = []
        for w in words:
            # Invert upscale then padding -> received-image coordinates.
            x = w["x"] / factor - pad
            y = w["y"] / factor - pad
            bw = w["width"] / factor
            bh = w["height"] / factor
            # Clamp to the received image and drop degenerate boxes.
            x0 = max(0.0, min(x, iw))
            y0 = max(0.0, min(y, ih))
            x1 = max(0.0, min(x + bw, iw))
            y1 = max(0.0, min(y + bh, ih))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            regions.append(
                {
                    "x": round(x0, 1),
                    "y": round(y0, 1),
                    "w": round(x1 - x0, 1),
                    "h": round(y1 - y0, 1),
                    "text": w.get("text", ""),
                    "conf": float(w.get("confidence", 0.0)),
                }
            )
        return regions

    @staticmethod
    def _merge_region_proposals(
        vision: list[dict[str, Any]],
        ocr: list[dict[str, Any]],
        *,
        contain_ratio: float = 0.72,
    ) -> list[dict[str, Any]]:
        """
        Keep vision proposals and add OCR boxes that are not already covered.

        When morphology over-merges, Paddle det usually still returns separate
        tight boxes for each stacked dimension.
        """
        merged = list(vision)

        def covered(box: dict[str, Any]) -> bool:
            bx0, by0 = box["x"], box["y"]
            bx1, by1 = bx0 + box["w"], by0 + box["h"]
            b_area = max(box["w"] * box["h"], 1.0)
            for v in vision:
                vx0, vy0 = v["x"], v["y"]
                vx1, vy1 = vx0 + v["w"], vy0 + v["h"]
                ix0, iy0 = max(bx0, vx0), max(by0, vy0)
                ix1, iy1 = min(bx1, vx1), min(by1, vy1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / b_area >= contain_ratio:
                    return True
            return False

        for box in ocr:
            short = min(box["w"], box["h"])
            if short < 11 or box["w"] * box["h"] < 180:
                continue
            if not covered(box):
                merged.append(box)
        return merged

    def _expand_clusters(
        self,
        image: Image.Image,
        clusters: list[list[dict[str, Any]]],
        *,
        cluster_margin: float,
        max_refinements: int = GROUPING_MAX_REFINEMENTS,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Refine a bounded number of oversized clusters with detector-only calls.

        Clusters beyond the cap, and all clusters when the standalone detector
        is unavailable, remain in the result unchanged. Grouping therefore
        cannot silently launch the complete OCR pipeline or discard candidates.
        """
        from detection_passes import (
            DETECTION_REFINEMENT_TARGET_EDGE,
            map_quarter_turn_box_to_source,
            refinement_rotation,
            rotate_for_detection,
        )
        from region_cluster import (
            cluster_boxes,
            order_clusters,
            split_mixed_clusters,
            union_bbox,
        )

        iw, ih = image.size
        median_h = 0.0
        if clusters:
            heights = [union_bbox(c)["height"] for c in clusters]
            median_h = sorted(heights)[len(heights) // 2]

        height_limit = max(median_h * 2.8, ih * 0.28)
        width_limit = iw * 0.38
        area_limit = max(float(iw * ih) * 0.07, 1.0)
        oversized: list[tuple[float, int]] = []
        for index, cluster in enumerate(clusters):
            ub = union_bbox(cluster)
            score = max(
                ub["height"] / max(height_limit, 1.0),
                ub["width"] / max(width_limit, 1.0),
                (ub["width"] * ub["height"]) / area_limit,
            )
            if score > 1.0:
                oversized.append((score, index))

        detector_available = bool(
            getattr(self, "_text_detector_available", False)
        )
        selected = (
            {
                index
                for _score, index in sorted(
                    oversized,
                    key=lambda item: (-item[0], item[1]),
                )[: max(0, max_refinements)]
            }
            if detector_available
            else set()
        )
        refinement_total = len(selected)

        def emit(
            *,
            completed: int,
            state: str,
            label: str,
            candidate_count: int,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    completed=completed,
                    total=refinement_total,
                    state=state,
                    label=label,
                    candidate_count=candidate_count,
                )

        if not detector_available:
            emit(
                completed=0,
                state="skipped",
                label="Standalone detector unavailable; retaining oversized candidates",
                candidate_count=len(clusters),
            )
        elif refinement_total == 0:
            emit(
                completed=0,
                state="skipped",
                label="No oversized candidates require local refinement",
                candidate_count=len(clusters),
            )

        out: list[list[dict[str, Any]]] = []
        refined = 0
        for cluster_index, cluster in enumerate(clusters):
            ub = union_bbox(cluster)
            if cluster_index not in selected:
                out.append(cluster)
                continue

            refined += 1
            label = f"oversized candidate {refined} of {refinement_total}"
            emit(
                completed=refined - 1,
                state="running",
                label=label,
                candidate_count=len(out) + len(clusters) - cluster_index,
            )
            margin = 4
            cx0 = max(0, int(ub["x"] - margin))
            cy0 = max(0, int(ub["y"] - margin))
            cx1 = min(iw, int(ub["x"] + ub["width"] + margin))
            cy1 = min(ih, int(ub["y"] + ub["height"] + margin))
            sub = image.crop((cx0, cy0, cx1, cy1))
            rotation = refinement_rotation(
                {"w": float(sub.width), "h": float(sub.height)}
            )
            detector_input = rotate_for_detection(sub, rotation)
            sub_boxes: list[dict[str, Any]] = []
            for detected in self._detector_only_boxes(
                detector_input,
                target_long_edge=DETECTION_REFINEMENT_TARGET_EDGE,
            ):
                mapped = map_quarter_turn_box_to_source(
                    detected,
                    rotation,
                    sub.size,
                )
                if mapped is not None:
                    sub_boxes.append(mapped)
            if not sub_boxes:
                out.append(cluster)
                emit(
                    completed=refined,
                    state="completed",
                    label=label,
                    candidate_count=len(out) + len(clusters) - cluster_index - 1,
                )
                continue
            sub_clusters = split_mixed_clusters(
                cluster_boxes(
                    sub_boxes,
                    margin_ratio=cluster_margin,
                    img_w=cx1 - cx0,
                    img_h=cy1 - cy0,
                )
            )
            if len(sub_clusters) <= 1:
                out.append(cluster)
            else:
                for sub_cluster in sub_clusters:
                    for box in sub_cluster:
                        box["x"] = round(float(box["x"]) + cx0, 1)
                        box["y"] = round(float(box["y"]) + cy0, 1)
                out.extend(sub_clusters)
            emit(
                completed=refined,
                state="completed",
                label=label,
                candidate_count=len(out) + len(clusters) - cluster_index - 1,
            )
        return order_clusters(out)


    def title_fields(
        self,
        image: Image.Image,
        keywords: list[str],
    ) -> list[dict[str, Any]]:
        """
        Read the configured title-block fields from a WHOLE drawing sheet.

        Deliberately separate from `segment`: the title block is a fixed part of
        the sheet, but Auto-Segment only ever sees the rectangle the user drew
        around a cluster of dimensions. Running the keyword scan on that crop
        made the result depend on where the box happened to land — a selection
        covering the upper sheet found the revision table's REV and missed the
        DWG NO. printed at the bottom.

        Only the detection pass runs here, not the per-cluster recognition, so
        this costs a fraction of a full segment.
        """
        from title_fields import extract_title_fields

        if not keywords:
            return []

        def _read(region: Image.Image, dx: int, dy: int) -> list[dict[str, Any]]:
            """Recognise a region and return its boxes in page coordinates."""
            boxes = self._detect_regions_ocr(region)
            for box in boxes:
                box["x"] = float(box["x"]) + dx
                box["y"] = float(box["y"]) + dy
            return boxes

        # detect_regions is DETECTOR-ONLY on this pipeline: it returns geometry
        # with empty text, so matching a keyword against it can never succeed.
        # Recognition is required here, and it is run on the bottom-right
        # corner first — the title block's labels are small, and reading just
        # that corner resolves them far better than one pass over the sheet.
        width, height = image.size
        corner_x = int(width * 0.45)
        corner_y = int(height * 0.70)
        corner = image.crop((corner_x, corner_y, width, height))
        fields = extract_title_fields(_read(corner, corner_x, corner_y), keywords)

        # Some title-block labels are printed small enough that the detector
        # fuses them with their own value and the keyword has nothing to match
        # — "REV" over its number is the usual one. Re-read the corner enlarged
        # and fill in only the fields still blank, so a keyword that already
        # resolved is never second-guessed. Capped to stay inside the
        # detector's 4000px side limit.
        if any(not field["value"] for field in fields):
            limit = 4000 / max(corner.width, corner.height, 1)
            factor = min(2.5, max(limit, 1.0))
            if factor > 1.05:
                enlarged = corner.resize(
                    (int(corner.width * factor), int(corner.height * factor)),
                    Image.LANCZOS,
                )
                boxes = self._detect_regions_ocr(enlarged)
                for box in boxes:
                    box["x"] = float(box["x"]) / factor + corner_x
                    box["y"] = float(box["y"]) / factor + corner_y
                    box["w"] = float(box["w"]) / factor
                    box["h"] = float(box["h"]) / factor
                retry = {
                    field["keyword"]: field
                    for field in extract_title_fields(boxes, keywords)
                }
                fields = [
                    retry.get(field["keyword"], field)
                    if not field["value"] and retry.get(field["keyword"], {}).get("value")
                    else field
                    for field in fields
                ]

        if any(field["value"] for field in fields):
            return fields

        # Nothing in the corner: the sheet may place its title block elsewhere,
        # or the caller may have handed us a crop of the block itself.
        whole = extract_title_fields(_read(image, 0, 0), keywords)
        return whole if any(field["value"] for field in whole) else fields

    @staticmethod
    def _lines_from_boxes(
        boxes: list[dict[str, Any]], line_h: float
    ) -> list[str]:
        """
        Assemble detection boxes into text lines, in reading order.

        At a low render scale the detector breaks one printed line into several
        boxes whose ``y`` differ by a few pixels ("2" / "HOT" / "DIP GALVANIZE
        PER ASTM…"). Sorting on exact ``y`` interleaves those fragments and the
        paragraph comes out scrambled, so boxes are first grouped into bands one
        line high, then read left to right within each band.
        """
        if not boxes:
            return []

        # Walk the boxes by vertical centre and start a new line only where the
        # centres actually step down. Measuring against the first box of a row
        # instead would drop the third fragment of a line whose boxes drift by a
        # pixel or two each ("2" 337, "HOT" 339, "DIP GALVANIZE…" 330).
        # The band comes from the boxes themselves, not from the caller's
        # line_h: that is the median height of the numbered MARKERS, which on a
        # real sheet runs nearly twice the true line pitch and fuses note 1 into
        # note 2. A low percentile of the actual box heights is a safer floor
        # for one line, while ignoring the odd oversized box.
        heights = sorted(b["h"] for b in boxes)
        basis = heights[len(heights) // 4] if heights else line_h
        if basis <= 0:
            basis = max(line_h, 1.0)
        # Chaining alone can cascade — each box within the band of the last one,
        # so three printed lines fuse into a single row. The row's own extent is
        # therefore capped at about one line as well.
        band = max(basis * 0.6, 1.0)
        extent = max(basis * 0.9, band)
        ordered = sorted(boxes, key=lambda b: b["y"] + b["h"] / 2.0)
        rows: list[list[dict[str, Any]]] = []
        previous: float | None = None
        row_start: float = 0.0
        for b in ordered:
            centre = b["y"] + b["h"] / 2.0
            same_row = (
                previous is not None
                and centre - previous <= band
                and centre - row_start <= extent
            )
            if same_row:
                rows[-1].append(b)
            else:
                rows.append([b])
                row_start = centre
            previous = centre

        lines: list[str] = []
        for row in rows:
            row.sort(key=lambda b: b["x"])
            parts = [(b.get("text") or "").strip() for b in row]
            parts = [t for t in parts if t]
            if not parts:
                continue
            # A lone leading number is a point marker whose "." the recogniser
            # dropped ("2" + "HOT DIP…"). Put it back so the sheet can split on
            # it, rather than loosening the marker pattern for every line.
            if len(parts) > 1 and re.fullmatch(r"\d{1,2}", parts[0]):
                lines.append(f"{parts[0]}. " + " ".join(parts[1:]))
            else:
                lines.append(" ".join(parts))
        return lines

    def _reread_notes_block(
        self,
        image: Image.Image,
        bounds: tuple[float, float, float, float],
        line_h: float,
    ) -> list[str]:
        """
        Re-read the notes paragraph from an enlarged crop.

        A drawing is rendered for the viewer, not for OCR (the client uses a
        1.5× page scale), and at that size the detector loses whole note lines —
        on a real sheet it missed note 1 and the "NOTES" heading entirely. The
        block is small, so re-reading just that rectangle enlarged recovers them
        at negligible cost.

        The crop reaches well above the detected top on purpose: the block's top
        is only as high as the FIRST marker that was read, and the lines above it
        are exactly the ones that were missed. Anything picked up above the
        paragraph is discarded afterwards by trimming to the first marker.

        Returns [] when the re-read finds nothing better, so the caller keeps the
        text it already had.
        """
        x0, y0, x1, y1 = bounds
        height = max(y1 - y0, 1.0)
        pad_x = max(0.05 * (x1 - x0), 4.0)
        # Reach up by most of the block's own height, and a little below.
        crop_box = (
            max(0, int(x0 - pad_x)),
            max(0, int(y0 - 0.7 * height)),
            min(image.width, int(x1 + pad_x)),
            min(image.height, int(y1 + 0.15 * height)),
        )
        if crop_box[2] - crop_box[0] < 8 or crop_box[3] - crop_box[1] < 8:
            return []

        crop = image.crop(crop_box)
        # Enlarge until a line is around 36px tall, which is where the
        # recogniser reads this text reliably. Capped to stay inside the
        # detector's 4000px side limit.
        factor = 3.0 if line_h <= 0 else max(1.0, min(4.0, 36.0 / line_h))
        limit = 4000 / max(crop.width, crop.height, 1)
        factor = min(factor, max(limit, 1.0))
        # Nothing to gain at sheet resolution: the lines in hand were read at
        # this size already, and a second pass at the same size only trades
        # one recogniser's slips for another's ("MATERIAL:C", "ARIIIN").
        if factor < 1.2:
            return []
        if factor > 1.01:
            crop = crop.resize(
                (int(crop.width * factor), int(crop.height * factor)),
                Image.LANCZOS,
            )

        try:
            # A recognising pass: detect_regions on this pipeline is
            # detector-only and returns boxes with no text, so it could never
            # produce lines to compare with the ones already in hand.
            boxes = self._detect_regions_ocr(crop.convert("RGB"))
        except Exception:
            # A failed re-read must never lose the notes already in hand.
            return []
        if not boxes:
            return []

        scaled_line_h = line_h * factor if line_h > 0 else 0.0
        if scaled_line_h <= 0:
            heights = sorted(b["h"] for b in boxes)
            scaled_line_h = heights[len(heights) // 2]

        lines = OcrPipeline._lines_from_boxes(boxes, scaled_line_h)

        # Trim to the paragraph: drop everything above its heading, or above the
        # first numbered point when no heading was read.
        start = next(
            (i for i, ln in enumerate(lines) if _NOTES_HEADING_RE.match(ln)), None
        )
        if start is None:
            start = next(
                (i for i, ln in enumerate(lines) if _NOTE_POINT_RE.match(ln)), None
            )
        if start is None:
            return []
        lines = lines[start:]

        # The sheet writes its own NOTES row, and a low-res re-read leaves
        # single-glyph debris between the real lines ("E", "RS", a stray CJK
        # character the recogniser emits for a stroke it cannot resolve). A
        # line of a paragraph always carries several letters; a point marker is
        # kept whatever follows it.
        # NOTE: the section route's is_segment_worthy / has_dimension_value
        # gates were tried here twice and measurably hurt both times, before
        # and after the review-policy fix (56103-0182B 16/24 -> 11/24,
        # BS1801006.020 9/12 -> 7/12). The page route has its own value
        # grammar in page_value_filters; these two do not belong on top of it.
        from segment_quality import (
            count_dimension_values,
            dedupe_regions,
            strip_foreign_glyphs,
        )

        cleaned: list[str] = []
        for line in lines:
            line = strip_foreign_glyphs(line).strip()
            if not line or _NOTES_HEADING_RE.match(line):
                continue
            if not _NOTE_POINT_RE.match(line) and sum(
                c.isalpha() for c in line
            ) < 4:
                continue
            cleaned.append(line)

        markers = [ln for ln in cleaned if _NOTE_POINT_RE.match(ln)]
        # Only accept the re-read when it actually resolved a list.
        return cleaned if len(markers) >= 2 else []

    @staticmethod
    def _detect_notes_block(
        boxes: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[int]]:
        """
        Recover the drawing's NOTES paragraph as ONE region.

        Every line of a notes block fails ``has_dimension_value`` — it opens
        with a word and carries nothing measurable — so ``segment`` drops the
        whole paragraph. The notes are still wanted on the inspection sheet,
        just not as dimensions, so the block is assembled here and returned as
        a single General Note region.

        The text is taken from the detection boxes rather than by re-reading
        the crop: each box is already one line, and preserving that line
        structure is what lets the sheet split the paragraph back into its
        numbered points. Re-reading a whole paragraph through the dimension
        recogniser would also lose the line breaks.

        Returns ``(region, indices)`` — the region and the indices of the boxes
        it consumed — or ``(None, [])`` when the drawing carries no notes.
        """
        points = [
            (i, b)
            for i, b in enumerate(boxes)
            if _NOTE_POINT_RE.match(b.get("text") or "")
        ]
        # Two points is the smallest thing that is recognisably a list. A lone
        # "1. SOMETHING" is far more likely to be a stray callout.
        if len(points) < 2:
            return None, []

        points.sort(key=lambda p: p[1]["y"])
        numbers = [
            int(_NOTE_POINT_RE.match(b["text"]).group(1)) for _, b in points
        ]
        # The markers must climb as the eye moves down the block, and the list
        # has to start at its top. Without this a column of unrelated numbered
        # table rows would be swept up as "notes".
        if numbers[0] > 2 or any(
            b <= a for a, b in zip(numbers, numbers[1:])
        ):
            return None, []

        left = min(b["x"] for _, b in points)
        right = max(b["x"] + b["w"] for _, b in points)
        line_h = sorted(b["h"] for _, b in points)[len(points) // 2]
        top = min(b["y"] for _, b in points)
        bottom = max(b["y"] + b["h"] for _, b in points)

        chosen = {i for i, _ in points}

        # Wrapped continuation lines sit between and below the markers, and the
        # heading sits just above. Grow the block until it stops absorbing
        # lines, so the tail of the LAST point is not left behind.
        #
        # The bounds below are what stop the block walking off down the page and
        # swallowing whatever table is printed under the notes: the horizontal
        # span is FROZEN at the span of the numbered markers (widening it would
        # loosen the test that decides what joins next), the ceiling and floor
        # are absolute, and a line only joins if it reads like prose.
        span_left, span_right = left, right
        ceiling = top - 2.0 * line_h
        floor = bottom + 6.0 * line_h

        # The paragraph can never continue past the top of the title block, so
        # the first title-block word below the notes becomes the hard floor.
        for b in boxes:
            if b["y"] >= bottom and _TITLE_BLOCK_WORDS_RE.search(
                (b.get("text") or "")
            ):
                floor = min(floor, b["y"])

        def joins(b: dict[str, Any]) -> bool:
            text = (b.get("text") or "").strip()
            if not text:
                return False
            # A title-block cell is not part of the notes, wherever it sits.
            if _TITLE_BLOCK_WORDS_RE.search(text):
                return False
            # A wrapped note line is words. A stray dimension that happens to
            # sit under the block is not part of the paragraph.
            letters = sum(c.isalpha() for c in text)
            if letters < 3 or letters < 0.5 * len(text.replace(" ", "")):
                return False
            overlap = min(b["x"] + b["w"], span_right) - max(b["x"], span_left)
            return overlap > 0.5 * b["w"]

        grew = True
        while grew:
            grew = False
            for i, b in enumerate(boxes):
                if i in chosen or not joins(b):
                    continue
                if not (ceiling <= b["y"] and b["y"] + b["h"] <= bottom + 1.6 * line_h):
                    continue
                if b["y"] + b["h"] > floor:
                    continue
                chosen.add(i)
                top = min(top, b["y"])
                bottom = max(bottom, b["y"] + b["h"])
                grew = True

        members = [boxes[i] for i in chosen]
        lines = OcrPipeline._lines_from_boxes(members, line_h)
        # Keep the heading out of the text: the sheet writes its own NOTES row.
        lines = [ln for ln in lines if ln and not _NOTES_HEADING_RE.match(ln)]
        if not lines:
            return None, []

        confs = [float(b.get("conf") or 0.0) for b in members]
        region = {
            "bbox": {
                "x": round(left, 1),
                "y": round(top, 1),
                "width": round(right - left, 1),
                "height": round(bottom - top, 1),
            },
            "text": "\n".join(lines),
            "confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "type": "General Note",
            "category": "General Note",
            "subtype": None,
            "label": "General Notes",
            "orientation": "horizontal",
            "rotation": 0,
            "needs_review": False,
            "agreement": 0.0,
            "engine": "paddleocr",
            "symbols_detected": None,
        }
        # Carried for the enlarged re-read; stripped before the region is
        # returned to the client.
        region["_bounds"] = (left, top, right, bottom)
        region["_line_h"] = line_h
        return region, sorted(chosen)

    @staticmethod
    def _region_from_result(
        res: dict[str, Any],
        bbox: dict[str, float],
        text: str,
    ) -> dict[str, Any]:
        """Assemble one segment region dict from a `recognize` result."""
        return {
            "bbox": bbox,
            "text": text,
            "confidence": res.get("confidence", 0.0),
            "type": res.get("type"),
            "category": res.get("category"),
            "subtype": res.get("subtype"),
            "label": res.get("label"),
            "orientation": res.get("orientation", "horizontal"),
            "rotation": res.get("rotation", 0),
            "needs_review": res.get("needs_review", False),
            "agreement": res.get("agreement", 0.0),
            "engine": res.get("engine", "paddleocr"),
            "symbols_detected": res.get("symbols_detected"),
            "engineering_symbol": res.get("engineering_symbol"),
        }

    @staticmethod
    def _cluster_row_count(cluster: list[dict[str, Any]]) -> int:
        """
        Number of distinct horizontal text rows among a cluster's member boxes.

        Boxes that don't overlap on the y-axis sit on separate rows. Two stacked
        close-proximity callouts fuse into one cluster whose members span two y
        bands; this reports that so `segment` can attempt a content split even
        when the fused single-line read collapsed to one value.
        """
        spans = sorted((b["y"], b["y"] + b["h"]) for b in cluster)
        rows: list[list[float]] = []
        for y0, y1 in spans:
            for r in rows:
                if not (y1 <= r[0] or y0 >= r[1]):  # overlaps this band in y
                    r[0], r[1] = min(r[0], y0), max(r[1], y1)
                    break
            else:
                rows.append([y0, y1])
        return len(rows)

    @staticmethod
    def _region_from_result(
        res: dict[str, Any],
        bbox: dict[str, float],
        text: str,
    ) -> dict[str, Any]:
        """Assemble one segment region dict from a `recognize` result."""
        return {
            "bbox": bbox,
            "text": text,
            "confidence": res.get("confidence", 0.0),
            "type": res.get("type"),
            "category": res.get("category"),
            "subtype": res.get("subtype"),
            "label": res.get("label"),
            "orientation": res.get("orientation", "horizontal"),
            "rotation": res.get("rotation", 0),
            "needs_review": res.get("needs_review", False),
            "agreement": res.get("agreement", 0.0),
            "engine": res.get("engine", "paddleocr"),
            "symbols_detected": res.get("symbols_detected"),
        }

    @staticmethod
    def _prefer_detection_text(
        cluster: list[dict[str, Any]], res: dict[str, Any], text: str
    ) -> str:
        """
        Fall back to the detector's own read when the crop re-read lost a tail.

        The detection pass reads each box in full-crop context; the per-cluster
        ``recognize`` re-reads a tight crop that may clip a small trailing part
        (``45°±3°`` → ``45°``). When a single-box cluster's detection text is a
        confident superset of the re-read (same leading digits, more content),
        compose that instead. Never replaces a read with something that
        disagrees on the digits already read.
        """
        import re

        from segment_quality import is_segment_worthy, strip_foreign_glyphs

        if len(cluster) != 1:
            return text
        det = (cluster[0].get("text") or "").strip()
        conf = float(cluster[0].get("conf") or 0.0)
        if not det or conf < 0.85 or det == text:
            return text
        d_text = re.sub(r"\D", "", text)
        d_det = re.sub(r"\D", "", det)
        if not d_det:
            return text
        # The tight re-read failed outright ("四1", "E") while the detector read
        # a complete value ("R1") — take the detector's read.
        rescue = not is_segment_worthy(text) and is_segment_worthy(det)
        clipped_tail = bool(d_text) and d_det.startswith(d_text) and len(d_det) > len(d_text)
        # "/10.03B" vs detector "// 0.03 B": the re-read turned a frame glyph's
        # stroke into a leading "1". Same digits after that stroke, and the
        # detector saw a GD&T symbol where the stroke is.
        stroke_prefix = (
            d_text.endswith(d_det)
            and set(d_text[: len(d_text) - len(d_det)]) == {"1"}
            and any(g in det for g in ("//", "∥", "⟂", "⊥", "∠", "◎", "⌖", "○", "↗"))
        )
        if not (rescue or clipped_tail or stroke_prefix):
            return text
        symbols = DetectedSymbols(**(res.get("symbols_detected") or {}))
        composed = compose_engineering_dimension(det, None, symbols)
        new = strip_foreign_glyphs((composed.text or "").strip())
        if not new or re.sub(r"\D", "", new) != d_det or not is_segment_worthy(new):
            return text
        res["text"] = new
        res["type"] = composed.kind
        res["confidence"] = min(float(res.get("confidence") or 0.0), conf)
        feature = classify_feature(new, symbols=res.get("symbols_detected") or {})
        res["category"], res["subtype"], res["label"] = (
            feature.category, feature.subtype, feature.label,
        )
        return new


    def _split_stacked_cluster(
        self,
        image: Image.Image,
        coords: tuple[int, int, int, int],
        *,
        debug_dump: bool = False,
        debug_dump_force: bool = False,
        max_paddle_predictions: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Re-segment one cluster that read back as multiple stacked values.

        Detects text rows inside the cluster crop with a tighter merge margin so
        the stacked callouts separate, then runs the full `recognize` pipeline
        per row (so each gets its own dual-unit repair and balloon box). Returns
        the per-row regions, or ``None`` when the crop doesn't actually split
        into 2+ worthy values (leaving the original single region untouched).
        """
        from region_cluster import (
            cluster_boxes,
            order_clusters,
            split_mixed_clusters,
            union_bbox,
        )
        from segment_quality import (
            count_dimension_values,
            has_dimension_value,
            is_segment_worthy,
            strip_foreign_glyphs,
        )
        from stroke_filter import is_stray_line

        cx0, cy0, cx1, cy1 = coords
        sub = image.crop((cx0, cy0, cx1, cy1))
        # The quick proposer: one recognising detection pass over the crop,
        # which is what this split was written against. The adaptive cascade
        # (morphology, several passes, local retries) costs seconds per crop
        # and returns boxes without text, which also silences the
        # detection-text fallback below.
        sub_boxes = self.detect_regions(sub, thorough=False)
        if len(sub_boxes) < 2:
            return None

        sub_clusters = order_clusters(
            split_mixed_clusters(
                cluster_boxes(
                    sub_boxes,
                    margin_ratio=0.45,  # tighter than the top level -> splits rows
                    img_w=cx1 - cx0,
                    img_h=cy1 - cy0,
                )
            )
        )
        if len(sub_clusters) < 2:
            return None

        m = 4
        out: list[dict[str, Any]] = []
        for c in sub_clusters:
            u = union_bbox(c)
            rx0 = max(0, int(cx0 + u["x"] - m))
            ry0 = max(0, int(cy0 + u["y"] - m))
            rx1 = min(image.width, int(cx0 + u["x"] + u["width"] + m))
            ry1 = min(image.height, int(cy0 + u["y"] + u["height"] + m))
            if rx1 - rx0 < 1 or ry1 - ry0 < 1:
                continue
            crop = image.crop((rx0, ry0, rx1, ry1))
            res = self.recognize(
                crop,
                debug_dump=debug_dump,
                debug_dump_force=debug_dump_force,
                compute_text_bbox=False,
                max_paddle_predictions=max_paddle_predictions,
            )
            t = strip_foreign_glyphs((res.get("text") or "").strip())
            t = self._prefer_detection_text(c, res, t)
            if not t or not is_segment_worthy(t) or is_stray_line(crop, t):
                continue
            if not has_dimension_value(t):
                continue
            bbox = {
                "x": round(cx0 + u["x"], 1),
                "y": round(cy0 + u["y"], 1),
                "width": round(u["width"], 1),
                "height": round(u["height"], 1),
            }
            out.append(self._region_from_result(res, bbox, t))

        return out if len(out) >= 2 else None

    @staticmethod
    def _limit_group_bbox(
        bbox: dict[str, float],
        records: list[dict[str, Any]],
        line_h: float,
    ) -> dict[str, float]:
        """
        The whole callout a tall page object is a piece of.

        The page detector cuts a limit stack every way at once — the whole
        thing, each column, the top row — and dedupe keeps several of those as
        distinct objects. Whichever survived to publication, the stack is only
        readable as a whole, so the box is grown over every detected object
        that overlaps it or abuts it on the same rows. Boxes much larger than
        the seed are left out: they are a neighbour's fused read, not this
        stack.
        """
        x0, y0 = float(bbox["x"]), float(bbox["y"])
        x1, y1 = x0 + float(bbox["width"]), y0 + float(bbox["height"])
        seed_w, seed_h = x1 - x0, y1 - y0
        gap = 0.25 * max(line_h, 1.0)
        for _ in range(2):
            for record in records:
                if record.get("table_excluded"):
                    continue
                rb = record.get("bbox") or {}
                try:
                    rx0, ry0 = float(rb["x"]), float(rb["y"])
                    rx1, ry1 = rx0 + float(rb["width"]), ry0 + float(rb["height"])
                except (KeyError, TypeError, ValueError):
                    continue
                if ry1 - ry0 > 1.3 * seed_h or rx1 - rx0 > 3.0 * max(seed_w, seed_h):
                    continue
                y_overlap = min(y1, ry1) - max(y0, ry0)
                if y_overlap < 0.7 * min(y1 - y0, ry1 - ry0):
                    continue
                x_overlap = min(x1, rx1) - max(x0, rx0)
                if x_overlap < -gap:
                    continue
                x0, y0 = min(x0, rx0), min(y0, ry0)
                x1, y1 = max(x1, rx1), max(y1, ry1)
        return {
            "x": round(x0, 1),
            "y": round(y0, 1),
            "width": round(x1 - x0, 1),
            "height": round(y1 - y0, 1),
        }

    def _read_limit_pair(
        self,
        image: Image.Image,
        bbox: dict[str, float],
        *,
        max_paddle_predictions: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Read a two-row limit dimension as ONE value, or ``None``.

        ``Ø0.620`` over ``0.612`` is one callout. Read flat it comes back as
        ``6200.612±0.7`` or ``Ø0.620φ0.612``; cut by the detector it comes back
        as columns (``620612``, ``Ø1.00``). The rows are separated here on the
        ink itself: glyphs that span both rows — the Ø drawn centred between
        them, the brackets of a dual-unit stack — are set aside, the remaining
        glyph rows give the two bands, and each band is recognised on its own
        with the tall glyphs blanked out so half a Ø is not read as a digit.
        The two reads are then joined by :func:`parse_limit_pair`, which is
        strict enough that two independent stacked callouts never merge.
        """
        try:
            import cv2
        except ImportError:
            return None
        import numpy as np

        from image_preprocess import _suppress_long_lines, cad_ink_to_gray
        from segment_quality import strip_foreign_glyphs

        # A little wider than the detector's box: the bracket of a dual-unit
        # stack is thin and often just outside it.
        m = max(6, min(16, int(0.14 * float(bbox["height"]))))
        cx0 = max(0, int(bbox["x"] - m))
        cy0 = max(0, int(bbox["y"] - 6))
        cx1 = min(image.width, int(bbox["x"] + bbox["width"] + m))
        cy1 = min(image.height, int(bbox["y"] + bbox["height"] + 6))
        if cx1 - cx0 < 12 or cy1 - cy0 < 12:
            return None
        crop = image.crop((cx0, cy0, cx1, cy1)).convert("RGB")
        gray = np.asarray(cad_ink_to_gray(crop).convert("L"))
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = _suppress_long_lines(mask)
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps = [
            tuple(int(v) for v in stats[i][:4])
            for i in range(1, count)
            if stats[i][cv2.CC_STAT_HEIGHT] >= 4 and stats[i][cv2.CC_STAT_AREA] >= 6
        ]
        if len(comps) < 4:
            return None
        heights = sorted(h for _x, _y, _w, h in comps)
        glyph_h = float(heights[len(heights) // 2])
        if glyph_h < 6:
            return None
        # Digits: glyph-sized. Points and commas are smaller; a bracket that
        # spans both rows is taller.
        tall = [c for c in comps if c[3] >= 1.6 * glyph_h]
        digits = [c for c in comps if 0.6 * glyph_h <= c[3] < 1.6 * glyph_h]
        if len(digits) < 4:
            return None

        # Two rows: the upper row's centre is the median of the highest
        # digit centres, the lower row's of the lowest. (Not the widest gap
        # between centres: the Ø drawn centred BETWEEN the rows sits in that
        # gap and halves it.) A glyph well away from both centres straddles
        # the rows and is read with neither.
        centres = sorted(y + h / 2.0 for _x, y, _w, h in digits)
        edge = max(1, int(0.4 * len(centres) + 0.999))
        upper_c, lower_c = centres[:edge], centres[-edge:]
        row_centres = (
            upper_c[len(upper_c) // 2],
            lower_c[len(lower_c) // 2],
        )
        if row_centres[1] - row_centres[0] < 0.9 * glyph_h:
            return None
        rows: list[list[tuple[int, int, int, int]]] = [[], []]
        straddlers: list[tuple[int, int, int, int]] = list(tall)
        for c in digits:
            cy = c[1] + c[3] / 2.0
            distances = [abs(cy - rc) for rc in row_centres]
            nearest = min(range(2), key=lambda i: distances[i])
            if distances[nearest] > 0.35 * glyph_h:
                straddlers.append(c)
            else:
                rows[nearest].append(c)
        if any(len(row) < 2 for row in rows):
            return None
        bands = [
            (min(c[1] for c in row), max(c[1] + c[3] for c in row)) for row in rows
        ]
        if bands[0][1] > bands[1][0]:
            return None
        short = rows[0] + rows[1]
        tall = straddlers

        text_left = min(x for x, _y, _w, _h in short)
        text_right = max(x + w for x, _y, w, _h in short)
        clean = np.array(crop)
        for x, y, w, h in tall:
            clean[max(0, y - 1) : y + h + 1, max(0, x - 1) : x + w + 1] = 255
        clean_img = Image.fromarray(clean)
        pad = max(3, int(0.2 * glyph_h))
        reads: list[tuple[str, float, dict[str, Any]]] = []
        for top, bottom in bands:
            row_crop = clean_img.crop(
                (
                    max(0, text_left - pad),
                    max(0, top - pad),
                    min(crop.width, text_right + pad),
                    min(crop.height, bottom + pad),
                )
            )
            res = self.recognize(
                row_crop,
                compute_text_bbox=False,
                max_paddle_predictions=max_paddle_predictions,
            )
            text = strip_foreign_glyphs(str(res.get("text") or "")).strip()
            reads.append((text, float(res.get("confidence") or 0.0), res))
        pair = parse_limit_pair(reads[0][0], reads[1][0])
        if pair is None:
            return None

        # What the tall glyphs are: a ring at the reading start is the Ø, a
        # thin stroke at either end is a bracket of a dual-unit stack.
        ring = any(
            x + w <= text_left + 0.3 * glyph_h and 0.5 <= w / max(h, 1) <= 1.4
            for x, _y, w, h in tall
        )
        left_bracket = any(
            x + w <= text_left + 0.2 * glyph_h and w < 0.4 * h for x, _y, w, h in tall
        )
        right_bracket = any(
            x >= text_right - 0.2 * glyph_h and w < 0.4 * h for x, _y, w, h in tall
        )
        if ring and not pair.startswith("Ø"):
            pair = "Ø" + pair
        if left_bracket and right_bracket:
            pair = f"[{pair}]"

        # The box is the value's own glyphs: the rows, plus the Ø and the
        # brackets that were credited to it. Not every straddler — the wider
        # crop may hold the bracket of the callout next door.
        parts = list(short) + [
            c for c in tall
            if (c[0] + c[2] <= text_left + 0.3 * glyph_h)
            or (left_bracket and right_bracket and c[0] >= text_right - 0.2 * glyph_h)
        ]
        ex0 = min(c[0] for c in parts)
        ey0 = min(c[1] for c in parts)
        ex1 = max(c[0] + c[2] for c in parts)
        ey1 = max(c[1] + c[3] for c in parts)
        out_bbox = {
            "x": round(cx0 + ex0 - 2, 1),
            "y": round(cy0 + ey0 - 2, 1),
            "width": round(ex1 - ex0 + 4, 1),
            "height": round(ey1 - ey0 + 4, 1),
        }
        return {
            "text": pair,
            "bbox": out_bbox,
            "type": "diameter" if "Ø" in pair else "linear",
            "confidence": round(min(reads[0][1], reads[1][1]), 4),
            "result": reads[0][2],
        }

    @staticmethod
    def _apply_feature_labels(region: dict[str, Any]) -> dict[str, Any]:
        """
        Attach the rule engine's category, subtype and balloon label in place.

        The section route has always published these; the page route did not,
        so every page balloon reached the UI with no category and an empty
        label — the checksheet's Label column showed a dash for the whole
        sheet and the frontend fell back to its text-only classifier, losing
        every rule that reads the detected symbols. Classified from the
        PUBLISHED text, which is not always the recogniser's own (a review
        object's text is blanked, a limit stack is rewritten after the read).
        """
        feature = classify_feature(
            str(region.get("text") or ""),
            symbols=region.get("symbols_detected") or {},
            engineering_symbol=region.get("engineering_symbol") or {},
        )
        region["category"] = feature.category
        region["subtype"] = feature.subtype
        region["label"] = feature.label
        return region

    @staticmethod
    def _ink_at_left_edge(
        image: Image.Image, x0: int, y0: int, y1: int, scale: float
    ) -> bool:
        """Whether ink touches the left edge of a levelled cluster box."""
        import numpy as np

        from image_preprocess import cad_ink_to_gray

        band = max(2, int(0.15 * scale))
        left = max(0, x0 - band)
        if x0 - left < 1 or y1 - y0 < 2:
            return False
        strip = np.asarray(
            cad_ink_to_gray(image.crop((left, y0, x0, y1))).convert("L")
        )
        # Ink is bright after cad_ink_to_gray; a glyph sliver fills a good
        # part of the strip's height, a stray speck does not.
        rows_with_ink = (strip > 128).any(axis=1)
        return float(rows_with_ink.mean()) >= 0.25

    @staticmethod
    def _oriented_box_from_rot(
        bx: float, by: float, bw: float, bh: float, inv: Any
    ) -> dict[str, float]:
        """
        Map a rotated-frame axis box to a source-frame *oriented* rectangle.

        Unlike ``_bbox_rot_to_source`` (which loses orientation by taking the
        AABB of the 4 mapped corners — loose for diagonal text), this returns the
        tight rotated rectangle the frontend can draw with a Konva ``rotation``:
        the top-left corner mapped to source, the (rotation-preserved) width and
        height, and the clockwise screen-angle of the box's own x-axis.
        """
        import math

        ox = inv[0, 0] * bx + inv[0, 1] * by + inv[0, 2]
        oy = inv[1, 0] * bx + inv[1, 1] * by + inv[1, 2]
        # The box's local +x (its reading direction) maps to (inv[0,0], inv[1,0]);
        # atan2(dy, dx) with y-down is Konva's clockwise rotation.
        angle = math.degrees(math.atan2(inv[1, 0], inv[0, 0]))
        return {
            "x": round(ox, 1),
            "y": round(oy, 1),
            "width": round(bw, 1),
            "height": round(bh, 1),
            "rotation": round(angle, 1),
        }

    def _slant_neighbourhoods(
        self,
        image: Image.Image,
        det_boxes: list[dict[str, Any]],
        *,
        margin_frac: float = 1.0,
        expand_frac: float = 0.2,
        max_rois: int = 3,
    ) -> list[tuple[int, int, int, int]]:
        """
        Regions of the crop that contain diagonal text, from the detector's quads.

        Diagonal callouts sit in a few small neighbourhoods, and working on those
        rather than the whole selection matters for accuracy, not just speed:
        detection upscales a small image and not a large one, so the quad (and
        therefore the measured angle) is markedly more accurate on a local region
        than on a full sheet. Boxes are grouped when their inflated extents
        touch, and each group returns a padded, clamped bounding region.

        The padding is generous on purpose: a callout's detection box covers the
        value but not always its stacked deviation or its leader, and a region
        cropped tight to the boxes detects differently from one with room around
        it.
        """
        iw, ih = image.size
        slanted: list[dict[str, Any]] = []
        for box in det_boxes or []:
            # The branch's detector keeps a text line's corners under "polygon";
            # this pass was written against "quad". Same four points.
            angle = self._quad_angle(box.get("quad") or box.get("polygon"))
            if angle is None or abs(angle) < 8.0 or abs(angle) > 82.0:
                continue
            if box.get("w", 0) <= 1 or box.get("h", 0) <= 1:
                continue
            slanted.append(box)
        if not slanted:
            return []

        def extent(b: dict[str, Any]) -> tuple[float, float, float, float]:
            pad = max(b["w"], b["h"]) * margin_frac
            return (b["x"] - pad, b["y"] - pad,
                    b["x"] + b["w"] + pad, b["y"] + b["h"] + pad)

        parent = list(range(len(slanted)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        rects = [extent(b) for b in slanted]
        for i in range(len(slanted)):
            for j in range(i + 1, len(slanted)):
                a, b = rects[i], rects[j]
                if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        groups: dict[int, list[int]] = {}
        for i in range(len(slanted)):
            groups.setdefault(find(i), []).append(i)

        rois: list[tuple[float, tuple[int, int, int, int]]] = []
        for members in groups.values():
            xs0 = min(rects[i][0] for i in members)
            ys0 = min(rects[i][1] for i in members)
            xs1 = max(rects[i][2] for i in members)
            ys1 = max(rects[i][3] for i in members)
            # Widen by a fraction of the region's own size. The slant vote and
            # the detector both behave better with surrounding context than on a
            # region cropped to the ink, and a callout's leader and stacked
            # deviation often reach past the boxes that located it.
            grow_x = (xs1 - xs0) * expand_frac
            grow_y = (ys1 - ys0) * expand_frac
            x0 = max(0, int(xs0 - grow_x))
            y0 = max(0, int(ys0 - grow_y))
            x1 = min(iw, int(xs1 + grow_x))
            y1 = min(ih, int(ys1 + grow_y))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            weight = sum(max(slanted[i]["w"], slanted[i]["h"]) for i in members)
            rois.append((weight, (x0, y0, x1, y1)))
        rois.sort(key=lambda r: r[0], reverse=True)
        return [r[1] for r in rois[:max_rois]]

    def _neighbourhood_sign(
        self,
        det_boxes: list[dict[str, Any]],
        roi: tuple[int, int, int, int],
    ) -> float | None:
        """
        Which way the diagonal text in ``roi`` leans, from the detector's quads.

        A quad's magnitude drifts for steeply slanted text, but its sign does
        not, and knowing the sign halves the passes the caller has to make.
        """
        x0, y0, x1, y1 = roi
        total = 0.0
        for box in det_boxes:
            # The branch's detector keeps a text line's corners under "polygon";
            # this pass was written against "quad". Same four points.
            angle = self._quad_angle(box.get("quad") or box.get("polygon"))
            if angle is None or abs(angle) < 8.0 or abs(angle) > 82.0:
                continue
            cx = box["x"] + box["w"] / 2.0
            cy = box["y"] + box["h"] / 2.0
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            total += max(box["w"], box["h"]) * (1.0 if angle >= 0 else -1.0)
        if total == 0.0:
            return None
        return 1.0 if total > 0 else -1.0

    @staticmethod
    def _leader_angles_for_roi(
        lines: list[dict[str, float]],
        roi: tuple[int, int, int, int],
        *,
        max_angles: int = 2,
    ) -> list[float]:
        """
        Angles of the leader lines that run through ``roi``, longest first.

        A callout lettered along a leader shares that leader's direction, so
        this is the angle to level by — measured from a long straight stroke
        rather than voted from the text's own strokes. Unlike that vote it does
        not weaken as the selection grows, which is what made the two fit
        callouts unreadable on a full sheet.
        """
        x0, y0, x1, y1 = roi
        angles: list[float] = []
        for line in lines:
            lx0, lx1 = sorted((line["x1"], line["x2"]))
            ly0, ly1 = sorted((line["y1"], line["y2"]))
            # The leader must actually pass through this region.
            if lx1 < x0 or lx0 > x1 or ly1 < y0 or ly0 > y1:
                continue
            angle = float(line["angle"])
            if all(abs(angle - kept) > 6.0 for kept in angles):
                angles.append(angle)
            if len(angles) >= max_angles:
                break
        return angles

    def _levelled_det_boxes(self, image: Image.Image) -> list[dict[str, Any]]:
        """
        Text boxes for a levelled crop, detector-only when the standalone
        detector is loaded.

        The levelled pass only needs geometry (the boxes are read by
        ``recognize`` afterwards), and the standalone detector is several times
        cheaper than the full pipeline that ``_paddle_det_boxes`` runs. The
        stacked-deviation reader keeps the full pipeline: it needs each part's
        text and confidence.
        """
        if getattr(self, "_text_detector_available", False):
            try:
                return self._detector_only_boxes(image, target_long_edge=1100)
            except Exception:
                pass
        return self._paddle_det_boxes(image)

    def _fast_has_digit(self, crops: list[Image.Image]) -> list[bool]:
        """
        One batch of the fast recogniser over ``crops``: does each read a digit?

        A levelled neighbourhood yields many clusters that are not callouts —
        note words, specks, pieces of the part — and each one put through the
        full ``recognize`` costs about a second. The batch recogniser answers
        "is there a number here" in a few hundredths of a second per crop.
        Anything it cannot read at all is kept for the full read, which is the
        better of the two on small or faint text.
        """
        if not crops or not getattr(self, "_page_batch_recognition_available", False):
            return [True] * len(crops)
        try:
            fast = self._recognize_page_batch(crops, batch_size=16)
        except Exception:
            return [True] * len(crops)
        flags: list[bool] = []
        for result in fast:
            raw = str(result.get("raw_ocr") or result.get("text") or "")
            flags.append((not raw.strip()) or any(c.isdigit() for c in raw))
        return flags

    @staticmethod
    def _join_same_line(
        clusters: list[list[dict[str, Any]]],
        scale: float,
    ) -> list[list[dict[str, Any]]]:
        """
        Join clusters that sit on one text line with less than a character
        between them.

        The levelled detector sometimes breaks a value inside a word — at one
        crop scale "Ø20H10" came back as "Ø20H1" and "0", the "0" then read
        together with the deviation stack. The tight cluster margin does not
        close that gap, so it is closed here on the line's own geometry: the
        two boxes overlap by most of their height and the gap is under a
        character. A callout and its trailing radius ("0.2-0.3×45°" then "R1")
        join too, which the tail rule downstream already separates.
        """
        from region_cluster import union_bbox

        if scale <= 0 or len(clusters) < 2:
            return clusters
        changed = True
        while changed:
            changed = False
            boxes = [union_bbox(c) for c in clusters]
            order = sorted(range(len(boxes)), key=lambda k: boxes[k]["x"])
            for i_pos in range(len(order) - 1):
                i, j = order[i_pos], order[i_pos + 1]
                a, b = boxes[i], boxes[j]
                gap = b["x"] - (a["x"] + a["width"])
                top = max(a["y"], b["y"])
                bottom = min(a["y"] + a["height"], b["y"] + b["height"])
                overlap = bottom - top
                if (
                    -0.3 * scale <= gap <= 0.9 * scale
                    and overlap >= 0.6 * min(a["height"], b["height"])
                    and max(a["height"], b["height"]) <= 1.6 * scale
                    and (a["width"] + gap + b["width"]) <= 12.0 * scale
                ):
                    clusters[i] = clusters[i] + clusters[j]
                    del clusters[j]
                    changed = True
                    break
        return clusters

    @staticmethod
    def _adjoin_deviation_stacks(
        clusters: list[list[dict[str, Any]]],
        scale: float,
    ) -> list[list[dict[str, Any]]]:
        """
        Join a value cluster with the deviation stack drawn just after it.

        A fit callout is one value followed by two smaller numbers stacked to
        its right. Whether the levelled detector returns that as one box or as
        a value box plus a stack box depends on the crop's scale, and the two
        outcomes used to go different ways: the single box was split correctly
        by the stacked-deviation reader, the pair was published as a bare
        ``Ø18H10`` with its tolerances lost. Joining the pair here sends both
        shapes down the same path. The stack is narrower than its value, sits
        within about a line height of the value's end and shares its rows.
        """
        from region_cluster import union_bbox

        if scale <= 0 or len(clusters) < 2:
            return clusters
        boxes = [union_bbox(c) for c in clusters]
        merged_into: dict[int, int] = {}

        def is_deviation_of(a: dict[str, float], b: dict[str, float]) -> bool:
            a_right = a["x"] + a["width"]
            # (i) One deviation line: a short, narrow box in the right part of
            # the value's row band. The detector often fuses the value with
            # its UPPER deviation, so the lower one then sits inside the value
            # box's own x-range and must be allowed there.
            if (
                b["height"] <= 0.85 * a["height"]
                and b["width"] <= 0.6 * a["width"]
                and b["x"] + b["width"] / 2.0 > a["x"] + 0.55 * a["width"]
                and b["x"] <= a_right + 2.0 * scale
            ):
                top = max(a["y"] - 0.5 * a["height"], b["y"])
                bottom = min(a["y"] + 1.5 * a["height"], b["y"] + b["height"])
                if bottom - top > 0:
                    return True
            # (ii) The whole stack as one box: starts where the value ends, up
            # to two lines tall. It may be about as wide as the value when the
            # detector broke the value inside a word and glued its last digit
            # to the stack ("Ø20H1" + "0 +0.084/0"); the fused-read guards
            # downstream reject a genuine neighbour that slips through.
            if (
                a_right - 0.5 * scale <= b["x"] <= a_right + 2.0 * scale
                and b["height"] <= 2.2 * a["height"]
                and b["width"] <= 1.2 * a["width"]
            ):
                top = max(a["y"] - 0.6 * a["height"], b["y"])
                bottom = min(a["y"] + 1.6 * a["height"], b["y"] + b["height"])
                if bottom - top > 0:
                    return True
            return False

        # Widest first: the value is the long box, its deviations the short
        # ones, and a lower deviation that two values could both claim goes
        # to the wider (nearer, fused) one.
        for i in sorted(range(len(boxes)), key=lambda k: boxes[k]["width"], reverse=True):
            if i in merged_into:
                continue
            a = boxes[i]
            if a["width"] < 1.5 * scale:
                continue  # a fragment, not a value
            a_right = a["x"] + a["width"]
            members = sorted(
                (
                    j
                    for j, b in enumerate(boxes)
                    if j != i and j not in merged_into and is_deviation_of(a, b)
                ),
                key=lambda j: abs(boxes[j]["x"] - a_right),
            )[:3]
            # Members join one at a time, nearest the value's end first, and
            # only while the union stays about two lines tall: anything taller
            # has swept up a neighbour still slanted at this angle. Rejecting
            # the whole set for one such box lost the stack that belonged.
            joined = list(clusters[i])
            for j in members:
                candidate = joined + list(clusters[j])
                if union_bbox(candidate)["height"] > 2.6 * scale:
                    continue
                joined = candidate
                merged_into[j] = i
            clusters[i] = joined
        return [c for k, c in enumerate(clusters) if k not in merged_into]

    def _angled_in_roi(
        self,
        image: Image.Image,
        *,
        sign: float | None = None,
        max_magnitudes: int = 1,
        weight_floor: float | None = None,
        fallback_angles: list[float] | None = None,
        seed_box: tuple[float, float, float, float] | None = None,
        max_angles: int = 4,
        line_height: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Read the diagonal callouts in one image, levelling it by its own slant.

        The slant magnitude comes from the Hough vote over this image, so a
        tighter image gives a sharper answer. ``sign`` resolves the vote's
        sign ambiguity when the caller already knows which way the text leans
        (from the detector's quads), halving the number of passes and leaving
        room to try more magnitudes instead. ``weight_floor`` lowers the bar a
        magnitude must clear to be worth trying, which is safe once the caller
        has established that this region really does hold diagonal text.

        ``fallback_angles`` are tried BEFORE the vote's: the angles of the
        leader lines running through this region and the detector's own quad
        angle. A leader's angle is measured off the line the text sits on, so
        it is exact where the vote is a 1°-bucket estimate over ink.

        ``seed_box`` (x, y, w, h in this image's pixels) is the slanted text
        this neighbourhood was built around. Once an angle has produced a
        confident read over it, the remaining angles are skipped: they can only
        re-read the same ink, and every angle costs a detection and a read.
        """
        import re

        from image_preprocess import cad_ink_to_gray
        from region_cluster import cluster_boxes, union_bbox
        from region_detect import dominant_slant_angles
        from segment_quality import (
            contains_annotation_note,
            count_dimension_values,
            has_dimension_value,
            is_annotation_note,
            is_segment_worthy,
            strip_foreign_glyphs,
        )
        from stroke_filter import is_stray_line

        iw, ih = image.size
        # Preferred source: the angle each detection reports for itself. It is
        # exact, per callout, and free — the detector already computed it.
        vote_args: dict[str, Any] = {"max_magnitudes": max_magnitudes}
        if weight_floor is not None:
            vote_args["min_weight_frac"] = weight_floor
        magnitudes = dominant_slant_angles(image, **vote_args)

        angles: list[float] = []
        for m in magnitudes:
            if sign is None:
                # A slant bucket is sign-ambiguous (a line at +a and -a land in
                # the same bucket), so try both and let worthiness and dedupe
                # drop the wrong one.
                angles.extend((-m, m))
            else:
                angles.append(round(sign * m, 1))
        # Knowing the sign buys a third magnitude for the same pass count.
        voted = angles[: 3 if sign is not None else 2]
        # A leader's angle is measured from the line the text is written on, so
        # it is exact; the vote's is a 1°-bucket estimate over ink. That
        # difference decides whether a deviation stack reads as ``+0.070/0`` or
        # runs together as ``+0.0700``, so leaders are tried first and the vote
        # fills the remaining slots.
        angles = []
        for candidate in list(fallback_angles or []) + voted:
            if all(abs(candidate - kept) > 6.0 for kept in angles):
                angles.append(candidate)
        angles = angles[: max(1, max_angles)]
        if not angles:
            return []

        # Slanted callouts are thin and the client renders drawings for a
        # viewer, not for OCR. At that size "0.5×45°" came back as "6" and
        # "Ø20H10 +0.084" as "084H1000"; the same text read cleanly from a
        # sheet rendered twice as large. Levelling is already a resample, so
        # the enlargement is free here — it rides in the same affine.
        roi_edge = max(image.size)
        roi_scale = (
            1.0 if roi_edge >= _ANGLED_MIN_EDGE
            else min(3.0, _ANGLED_MIN_EDGE / max(roi_edge, 1))
        )

        def seed_read(region: dict[str, Any]) -> bool:
            """
            A confident, CLEAN read of the text this neighbourhood was built for.

            Confidence alone is not enough to stop trying angles: at a
            neighbour's angle the recogniser returned ``Ø10+0.0700`` at 0.99 —
            a value crossed with the next callout's deviation stack. A clean
            read has no stray letters, no deviation with four or more decimals
            (two deviations run together), and a fit class carries its stack
            as ``+0.084/0``.
            """
            if seed_box is None:
                return False
            if float(region.get("confidence") or 0.0) < 0.90:
                return False
            text = str(region.get("text") or "")
            if any(c.isalpha() and c not in "HRXhrx" for c in text):
                return False
            if re.search(r"[+\-−]\s*\d*[.,]\d{4,}", text):
                return False
            if re.search(r"\d[Hh]\d", text) and "/" not in text:
                return False
            sx, sy, sw, sh = seed_box
            b = region["bbox"]
            x0 = max(sx, b["x"])
            y0 = max(sy, b["y"])
            x1 = min(sx + sw, b["x"] + b["width"])
            y1 = min(sy + sh, b["y"] + b["height"])
            if x1 <= x0 or y1 <= y0:
                return False
            smaller = max(1.0, min(sw * sh, b["width"] * b["height"]))
            return (x1 - x0) * (y1 - y0) / smaller >= 0.3

        found: list[dict[str, Any]] = []
        for angle in angles:
            rimg, inv = self._rotate_expand(image, angle, scale=roi_scale)
            rw, rh = rimg.size
            # Detect again on the *levelled* ink. The quads told us the angle,
            # but a detector box for steeply slanted text is a poor fit — it
            # misses a stacked deviation and merges neighbours. Once the text is
            # horizontal the detector is accurate, which is what makes the
            # value-plus-deviation grouping below work. (Do NOT reuse the full
            # detect_regions/cluster pipeline here — its 90° pass and aggressive
            # merge fuse the now-diagonal axis text into giant blobs.)
            raw = self._levelled_det_boxes(cad_ink_to_gray(rimg).convert("RGB"))
            if not raw:
                continue
            boxes = [
                {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
                for b in raw
                if b["w"] > 1 and b["h"] > 1
                and not is_annotation_note(b.get("text", ""))
            ]
            if not boxes:
                continue
            scale = self._text_scale(boxes)
            # The caller may know the text height (a leader corridor does):
            # the median box height in a small crop is dragged down by the
            # lower deviation and stray marks, and every size guard below is
            # measured against it — "Ø18H10 +0.070/0" was rejected as a box
            # too large for its text at a scale of 29 px when the writing was
            # 55 px tall.
            if line_height and line_height > 0:
                scale = max(scale or 0.0, 0.8 * line_height * roi_scale)
            if scale:
                # Keep only what reads as a single line at THIS angle. Text
                # belonging to a different leader is still slanted here, so the
                # detector returns it as a tall blob — and that blob bridges the
                # two callouts into one cluster, which is how a levelled pass
                # ruins the very callout it was meant to read.
                upright_here = [b for b in boxes if b["h"] <= scale * 2.2]
                if upright_here:
                    boxes = upright_here
            # Tight merge only: join split fragments of one callout, never span
            # separate callouts. Cap the union so nothing balloons.
            # Pass the measured line height: this box set is one or two callouts
            # plus specks, where cluster_boxes' own median is unreliable.
            clusters = cluster_boxes(
                boxes, margin_ratio=0.3, img_w=rw, img_h=rh, text_scale=scale
            )
            clusters = self._join_same_line(clusters, scale)
            clusters = self._adjoin_deviation_stacks(clusters, scale)

            margin = 6
            pending: list[tuple[list[dict[str, Any]], dict[str, float], tuple[int, int, int, int], Image.Image]] = []
            for cluster in clusters:
                ub = union_bbox(cluster)
                cx0 = max(0, int(ub["x"] - margin))
                cy0 = max(0, int(ub["y"] - margin))
                cx1 = min(rw, int(ub["x"] + ub["width"] + margin))
                cy1 = min(rh, int(ub["y"] + ub["height"] + margin))
                if cx1 - cx0 < 1 or cy1 - cy0 < 1:
                    continue
                # A callout is at most a few lines across (value + tolerance
                # stack) and a dozen or so long. Judge that against the text
                # size in this frame, not the crop: a crop-fraction cap threw
                # away clean reads on a zoomed-in selection, and rejected the
                # larger of two fit callouts purely because it sat in a small
                # rotated canvas.
                if scale and (
                    min(ub["width"], ub["height"]) > scale * 4.0
                    or max(ub["width"], ub["height"]) > scale * 12.0
                ):
                    continue
                # A single slanted callout is ONE line of text, optionally with
                # its tolerance stacked above/below it: more rows than that is
                # a piece of the drawing, not worth a read.
                if self._cluster_row_count(cluster) > 3:
                    continue
                # A leader that runs into the first glyph makes the levelled
                # detector start its box AFTER that glyph: "0.5×45°" came back
                # as "1.5×45°" (the sliver of the 0 read as a 1) or "6.5×45°".
                # When ink is cut at the reading start, the crop reaches back
                # one text height, stopping short of any box to its left.
                cut_start = False
                if scale and self._ink_at_left_edge(rimg, cx0, cy0, cy1, scale):
                    reach = int(cx0 - scale)
                    for other in boxes:
                        if other["x"] + other["w"] <= cx0 and (
                            min(other["y"] + other["h"], cy1) - max(other["y"], cy0)
                            > 0.3 * (cy1 - cy0)
                        ):
                            reach = max(reach, int(other["x"] + other["w"] + margin))
                    if reach < cx0:
                        cx0 = max(0, reach)
                        cut_start = True
                pending.append(
                    (cluster, ub, (cx0, cy0, cx1, cy1), rimg.crop((cx0, cy0, cx1, cy1)), cut_start)
                )

            # Cheap first look: only clusters that read a digit get the full,
            # voting recogniser.
            keep_flags = self._fast_has_digit([item[3] for item in pending])
            for (cluster, ub, (cx0, cy0, cx1, cy1), sub, cut_start), keep in zip(pending, keep_flags):
                if not keep:
                    continue
                # The crop is already levelled, so the deskew variants inside
                # recognize can add nothing here; three upright variants is
                # the full vote.
                res = self.recognize(
                    sub, compute_text_bbox=False, max_paddle_predictions=3
                )
                if cut_start:
                    # The recovered glyph has the leader through it, and on
                    # the raw crop the recogniser reads the pair as one digit
                    # ("6.5×45°"). On ink-normalised input it does not, so
                    # that read is taken unless the raw one is more confident.
                    alt = self.recognize(
                        cad_ink_to_gray(sub).convert("RGB"),
                        compute_text_bbox=False,
                        max_paddle_predictions=3,
                    )
                    if float(alt.get("confidence") or 0.0) >= float(res.get("confidence") or 0.0):
                        res = alt
                text = strip_foreign_glyphs((res.get("text") or "").strip())
                # A value with stacked deviations reads back interleaved. Only
                # worth a finer pass when the text is long enough to hold one
                # and looks like one: a fit class ("18H10") or two signed
                # numbers. A chamfer "0.2-0.3×45°" has the digits but neither,
                # and its re-detection cost a second on every neighbourhood.
                if sum(c.isdigit() for c in text) >= 5 and (
                    re.search(r"\d[A-Za-z]\d", text)
                    or text.count("+") + text.count("-") + text.count("−") >= 2
                ):
                    members_in_crop = [
                        {
                            "x": float(b["x"]) - cx0,
                            "y": float(b["y"]) - cy0,
                            "w": float(b["w"]),
                            "h": float(b["h"]),
                        }
                        for b in cluster
                    ]
                    text = (
                        self._stacked_deviation_read(
                            sub, res, member_boxes=members_in_crop
                        )
                        or text
                    )
                # "0.2-0.3×45°R1": a radius callout drawn right after the
                # chamfer is a second value; keep the chamfer.
                m_tail = re.match(r"^(.*\d°)\s*[Rr]\d[\d.]*$", text)
                if m_tail:
                    text = m_tail.group(1)
                # A fit class and a chamfer are never the same callout: "Ø18H10"
                # tolerances a bore, "0.5×45°" breaks an edge. Both in one
                # levelled read means this crop bridged two callouts on
                # neighbouring leaders and ran their text together
                # ("115H10.5×45°"). count_dimension_values scores that as one
                # value because it is a single unbroken token, so it needs its
                # own guard. The per-leader passes read each callout properly,
                # so the fusion is dropped rather than balloonned.
                if re.search(r"\d\s*[Hh]\d", text) and re.search(
                    r"[xX×]\s*\d{1,3}\s*°", text
                ):
                    continue
                if not text or not is_segment_worthy(text):
                    continue
                if not has_dimension_value(text):
                    continue
                # A rotated re-read is speculative: the upright passes already
                # had their turn at this ink. An unconfident one is noise, and
                # on a wide selection that noise is what puts a tilted box over
                # a feature-control frame.
                if float(res.get("confidence") or 0.0) < 0.80:
                    continue
                # The box has to fit what it reports. A levelled cluster covering
                # far more area than its own characters occupy is a box drawn
                # around mostly empty drawing, which is what a stray balloon over
                # a watermark or a title block looks like on screen.
                if scale:
                    content = 0.6 * len(text) * scale * scale
                    if ub["width"] * ub["height"] > max(2.5 * content, 3.0 * scale * scale):
                        continue
                # Note wording mixed into digits means this rotated crop fused a
                # callout with an annotation; the upright pass already has the
                # values, so drop it rather than emit the blob.
                if contains_annotation_note(text):
                    continue
                if is_stray_line(sub, text):
                    continue
                # Judge fusion by structure — rows in the levelled frame — rather
                # than by a raw digit count, which a legitimate fit callout
                # (``Ø20H10 +0.084/0``, 9 digits) trips.
                if count_dimension_values(text) >= 2:
                    continue
                if sum(c.isdigit() for c in text) > 14:
                    continue
                # Shrink the cluster box onto the ink it actually contains, so
                # the balloon hugs the writing instead of the hull plus margin.
                box_x, box_y = ub["x"], ub["y"]
                box_w, box_h = ub["width"], ub["height"]
                ink = self._ink_extent(sub)
                if ink is not None:
                    ix0, iy0, ix1, iy1 = ink
                    if ix1 - ix0 >= 4 and iy1 - iy0 >= 4:
                        box_x, box_y = cx0 + ix0, cy0 + iy0
                        box_w, box_h = ix1 - ix0, iy1 - iy0
                        # Writing along a leader is elongated once the box is on
                        # the ink. A square patch is not a line of text, it is a
                        # piece of the part caught at this angle.
                        if max(box_w, box_h) < min(box_w, box_h) * 1.8:
                            continue
                bbox = self._bbox_rot_to_source(
                    box_x, box_y, box_w, box_h, inv, iw, ih
                )
                if bbox is None:
                    continue
                region = self._region_from_result(res, bbox, text)
                region["rotation"] = round(float(angle), 1)
                # Tight rotated rectangle for the frontend to draw; bbox stays the
                # loose AABB so dedupe/anchor logic remains axis-aligned.
                region["oriented_box"] = self._oriented_box_from_rot(
                    box_x, box_y, box_w, box_h, inv
                )
                found.append(region)

            # The text this neighbourhood exists for has been read confidently
            # at this angle; the other candidate angles would only re-read it.
            if any(seed_read(r) for r in found):
                break

        return found

    @staticmethod
    def _dedupe_angled(
        angled: list[dict[str, Any]], base_regions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Keep angled reads that add something new.

        A diagonal callout's upright AABB is loose, so area-overlap dedupe would
        wrongly clobber neighbouring upright regions. Instead we drop an angled
        region only when its own centre sits inside a base region whose text
        already contains the same digit string (i.e. the value is genuinely a
        re-read), and drop angled-vs-angled duplicates the same way — never
        removing a base region.
        """
        import re

        import math

        from geom_utils import overlap_frac as geom_overlap_frac
        from geom_utils import polygon_area, region_polygon

        def digits(t: str) -> str:
            return re.sub(r"\D", "", t or "")

        def center(r: dict[str, Any]) -> tuple[float, float]:
            # The tight oriented box centre is exact; the AABB of a diagonal
            # box is loose and its centre can fall outside a neighbour it
            # actually re-read.
            ob = r.get("oriented_box")
            if ob:
                a = math.radians(ob["rotation"])
                hx, hy = ob["width"] / 2.0, ob["height"] / 2.0
                return (
                    ob["x"] + hx * math.cos(a) - hy * math.sin(a),
                    ob["y"] + hx * math.sin(a) + hy * math.cos(a),
                )
            b = r["bbox"]
            return (b["x"] + b["width"] / 2.0, b["y"] + b["height"] / 2.0)

        def inside(pt: tuple[float, float], r: dict[str, Any]) -> bool:
            b = r["bbox"]
            return (
                b["x"] <= pt[0] <= b["x"] + b["width"]
                and b["y"] <= pt[1] <= b["y"] + b["height"]
            )

        def same_value(d: str, other: str) -> bool:
            """Digit strings of one value read twice (one may carry junk)."""
            if not d or not other:
                return False
            if d in other or other in d:
                return True
            # Same length, at most one digit differs: "753" vs "153" is the
            # same callout with one stroke-confused glyph.
            if len(d) == len(other) and len(d) >= 3:
                return sum(a != b for a, b in zip(d, other)) <= 1
            return False

        def overlaps(r: dict[str, Any], other: dict[str, Any]) -> bool:
            # Tight-rectangle overlap first: two callouts on parallel leaders
            # share almost no ink even where their axis-aligned hulls do.
            if geom_overlap_frac(r, other) >= 0.4:
                return True
            return inside(center(r), other) or inside(center(other), r)

        def symbol_count(t: str) -> int:
            return sum(t.count(ch) for ch in "Ø°±×/")

        def junk_letters(t: str) -> int:
            # Letters that belong on a dimension: the H of a fit class, R for a
            # radius, X as a multiplier. Anything else in a numeric callout is a
            # stray stroke the recogniser turned into a letter.
            return sum(1 for c in t or "" if c.isalpha() and c not in "HRXhrx")

        def quality(r: dict[str, Any]) -> float:
            """
            How well a candidate read came out.

            The same callout levelled well reads ``Ø18H10+0.070/0`` and levelled
            poorly reads ``Ø118H10V+0.070M0`` — the poor one is *longer*, so
            ranking by length alone keeps the wrong one. Resolved symbols count
            for more than length, and stray letters count against.
            """
            text = r.get("text", "")
            # A tie on text and confidence is broken by how the box sits on the
            # writing: the same callout levelled at the right angle comes back
            # in a tighter, longer box than one levelled a dozen degrees off,
            # and the off-angle box is what ends up crossing its neighbour.
            oriented = r.get("oriented_box")
            aspect = 0.0
            if oriented:
                side_a = float(oriented["width"])
                side_b = float(oriented["height"])
                aspect = max(side_a, side_b) / max(min(side_a, side_b), 1.0)
            return (
                symbol_count(text) * 2.0
                - junk_letters(text) * 2.0
                + len(digits(text)) * 0.5
                + float(r.get("confidence") or 0.0)
                + min(aspect, 8.0) * 0.15
            )

        kept: list[dict[str, Any]] = []
        # Best reads first, so a duplicate keeps the better one.
        order = sorted(angled, key=quality, reverse=True)
        for r in order:
            d = digits(r.get("text", ""))
            if len(d) < 2:
                continue
            # Two angled reads whose tight rectangles coincide are the same ink
            # read at two candidate angles, whatever their digits say. Only the
            # better-levelled one is worth keeping, and separate callouts never
            # coincide — that is what the oriented rectangle buys us.
            if any(geom_overlap_frac(r, k) >= 0.5 for k in kept):
                continue
            # An angled read lying across a region the upright pass already
            # produced is a second box over the same ink. Only one may be drawn.
            # It is a straight swap of one read for a better one, so it applies
            # to a single region of comparable size: overlap is measured against
            # the smaller of the two, so a long angled box contains a small
            # upright one completely, and letting that count as a swap once
            # deleted eight good callouts on one sheet. Anything broader is
            # dropped, and replacing several regions at once stays the job of
            # _supersede_fused_base.
            # Sitting on top of a region the upright pass already produced has to
            # be earned. The angled read is kept only if it is the better read;
            # otherwise it is a second box over ink that is already accounted
            # for. Quality decides this rather than size, because a long chamfer
            # box legitimately passes over a small neighbouring callout, while a
            # bloated re-read of one value does not.
            # A third of a callout covered is already a visible double box, and
            # a bloated re-read covers only part of the region it duplicates.
            overlapped = [b for b in base_regions if geom_overlap_frac(r, b) >= 0.3]
            swap_for: list[dict[str, Any]] = []
            if overlapped:
                # The upright read wins unless this one is plainly better.
                if quality(r) <= max(quality(b) for b in overlapped):
                    continue
                # Better, and now: does it account for what it covers? The
                # upright pass often splits a leader callout into its value and
                # its deviation stack, and the levelled read is those fragments
                # put back together — its digits contain each of theirs, in
                # order. Then it stands in for them. Covering a value it cannot
                # account for means it ran into a neighbour, and drawing it
                # would stack a second box over that neighbour, so it is
                # dropped instead. Fragments of one or no digits are ignored:
                # an R1 that a chamfer box merely passes over is not something
                # the chamfer has to explain.
                area_r = polygon_area(region_polygon(r))
                accounted: list[dict[str, Any]] = []
                blocked = False
                for b in overlapped:
                    bd = digits(b.get("text", ""))
                    if len(bd) < 2:
                        continue  # nothing to account for
                    if _covers_digits(d, bd):
                        accounted.append(b)
                        continue
                    # Not accounted for. It may still be a garbled fragment of
                    # this same callout, which the levelled read got right — but
                    # only if it is a decisively worse read of a box this size.
                    # Without the size test a long box overrules every small
                    # region it happens to lie across, which once deleted eight
                    # good callouts on one sheet.
                    area_b = polygon_area(region_polygon(b))
                    comparable = max(area_r, area_b) <= min(area_r, area_b) * 2.5
                    if comparable and quality(r) >= quality(b) + 2.0:
                        accounted.append(b)
                    else:
                        blocked = True
                        break
                if blocked:
                    continue
                swap_for = accounted
            # A base region the swap logic just accounted for as a FRAGMENT of
            # this callout is not a re-read of it. ``same_value`` is a containment
            # test, so a value's own deviation stack ("+0.0700" sits inside
            # "Ø18H10+0.070/0") reads as the same value twice — and treating it
            # that way dropped the levelled read and published the fragment on
            # its own, which is the upright pass's failure, not a fix for it.
            # An exact digit match is still a genuine re-read and stays on the
            # enrich path below, which keeps the base region's own identity.
            fragments = {
                id(b) for b in swap_for if digits(b.get("text", "")) != d
            }
            dup = [
                b for b in base_regions
                if id(b) not in fragments
                and overlaps(r, b) and same_value(d, digits(b.get("text", "")))
            ]
            if dup:
                # The upright read of a slanted callout keeps its digits but
                # drops the symbols only the levelled view resolves (the ° of
                # ``0.5×45°``). Same digits + more symbols → enrich the base
                # region's text in place; its box is kept.
                for b in dup:
                    bt = b.get("text", "")
                    # Same value, better read: hand the upright region the
                    # levelled text AND its box, so the balloon ends up along
                    # the line. Judged on overall quality — comparing symbol
                    # counts alone left ``V V0.5×45°`` in place, because the
                    # stray V's do not change how many symbols it has.
                    if digits(bt) == d and quality(r) > quality(b):
                        for key in ("text", "type", "category", "subtype", "label"):
                            b[key] = r.get(key)
                        b["bbox"] = r["bbox"]
                        b["oriented_box"] = r.get("oriented_box")
                        b["rotation"] = r.get("rotation", 0)
                        b["confidence"] = r.get("confidence", b.get("confidence"))
                        b["enriched_from"] = "angled"
                continue
            if any(
                overlaps(r, k) and same_value(d, digits(k.get("text", "")))
                for k in kept
            ):
                continue
            # Only now that this read is definitely being kept does the region
            # it replaces step aside; marking earlier could delete a region and
            # then drop its replacement further down.
            for replaced in swap_for:
                replaced["superseded"] = True
            kept.append(r)
        return kept

    @staticmethod
    def _supersede_fused_base(
        angled: list[dict[str, Any]], base_regions: list[dict[str, Any]]
    ) -> set[int]:
        """
        Retire an upright region that fused several along-the-line callouts.

        When two or more *distinct* slanted reads sit inside one base region,
        that region is the axis-aligned hull of both leaders and its text is the
        garbled concatenation of them (``118H10+0.070-20H10``). The levelled
        view resolved what the upright view could not, so the base region is
        marked ``superseded`` — excluded from the redundancy check below, and
        dropped by ``segment`` — and the per-leader reads stand in its place.

        Returns the ids of the superseded base regions. Conservative: a base
        region holding a single slanted value keeps its authority, so an
        ordinary upright read is never displaced by a rotated re-read.
        """
        import re

        from geom_utils import contains_point, overlap_frac, region_center

        superseded: set[int] = set()
        for base in base_regions:
            inside = [a for a in angled if contains_point(base, region_center(a))]
            if len(inside) < 2:
                continue
            digit_sets = {re.sub(r"\D", "", a.get("text", "")) for a in inside}
            digit_sets.discard("")
            if len(digit_sets) < 2:
                continue  # the same value read twice, not a fusion
            # The two reads must occupy separate ink: overlapping rotated boxes
            # are one callout read twice, not two callouts fused.
            separate = any(
                overlap_frac(inside[i], inside[j]) < 0.3
                for i in range(len(inside))
                for j in range(i + 1, len(inside))
            )
            if not separate:
                continue
            base["superseded"] = True
            superseded.add(id(base))
        return superseded

    @staticmethod
    def _quad_sides(quad: Any) -> tuple[float, float] | None:
        """(long side, short side) of a four-point detection quad."""
        import math

        try:
            pts = [(float(p[0]), float(p[1])) for p in quad][:4]
        except (TypeError, ValueError, IndexError):
            return None
        if len(pts) != 4:
            return None
        sides = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        return max(sides), min(sides)

    def _callout_neighbourhoods(
        self,
        image: Image.Image,
        det_boxes: list[dict[str, Any]],
        *,
        max_rois: int = 8,
        with_quads: bool = False,
    ) -> list[Any]:
        """
        One neighbourhood per slanted callout, from the detector's quads.

        ``_slant_neighbourhoods`` grouped every slanted box whose padded extent
        touched another's, with the padding taken from the box's axis-aligned
        size. A 45° callout 350 px long has a 250 px square hull, so on a full
        sheet the part view became ONE 1400 px neighbourhood that took two
        minutes to level and read, and text on two different leaders was
        levelled at one angle. Here each slanted quad is its own seed: the
        padding is its text THICKNESS (so the deviation stack beside it is
        included but the next callout is not), only boxes at the same angle
        whose centre falls inside are absorbed, and the region is grown to a
        minimum edge so the levelled view keeps some context.

        Returns ``(roi, quad_angle, seed_box)`` per neighbourhood, largest text
        first; ``seed_box`` is the seed's hull in ROI pixels.
        """
        iw, ih = image.size
        seeds: list[dict[str, Any]] = []
        for box in det_boxes or []:
            quad = box.get("quad") or box.get("polygon")
            angle = self._quad_angle(quad)
            if angle is None or abs(angle) < 8.0 or abs(angle) > 82.0:
                continue
            if box.get("w", 0) <= 1 or box.get("h", 0) <= 1:
                continue
            sides = self._quad_sides(quad)
            if sides is None:
                continue
            length, thickness = sides
            # A speck or a square blob is not a line of text.
            if thickness < 8.0 or length < 30.0 or length < 2.2 * thickness:
                continue
            seeds.append(
                {
                    **box,
                    "angle": angle,
                    "length": length,
                    "thickness": thickness,
                    "cx": box["x"] + box["w"] / 2.0,
                    "cy": box["y"] + box["h"] / 2.0,
                    "points": [(float(pt[0]), float(pt[1])) for pt in quad][:4],
                }
            )
        if not seeds:
            return []

        def padded(b: dict[str, Any]) -> list[float]:
            # Room for the deviation stack after the value (half its length)
            # and about a line height around it. The thickness of a two-line
            # quad ("0.5×45°" over "(BOTH SIDES)") is already two lines, so
            # it is not doubled — that made a 900 px neighbourhood.
            pad = max(1.2 * b["thickness"], 0.5 * b["length"], 40.0)
            return [b["x"] - pad, b["y"] - pad, b["x"] + b["w"] + pad, b["y"] + b["h"] + pad]

        seeds.sort(key=lambda b: b["length"], reverse=True)
        used = [False] * len(seeds)
        rois: list[tuple[float, tuple[int, int, int, int], float, tuple[float, float, float, float]]] = []
        for i, seed in enumerate(seeds):
            if used[i]:
                continue
            used[i] = True
            extent = padded(seed)
            weight = seed["length"]
            for j, other in enumerate(seeds):
                if used[j] or abs(other["angle"] - seed["angle"]) > 12.0:
                    continue
                if extent[0] <= other["cx"] <= extent[2] and extent[1] <= other["cy"] <= extent[3]:
                    used[j] = True
                    weight += other["length"]
                    o = padded(other)
                    extent = [
                        min(extent[0], o[0]), min(extent[1], o[1]),
                        max(extent[2], o[2]), max(extent[3], o[3]),
                    ]
            # Grow to a minimum edge, centred, so a short callout still gets a
            # levelled view with context and only a moderate enlargement.
            for axis in (0, 1):
                size = extent[axis + 2] - extent[axis]
                if size < _ANGLED_ROI_MIN_EDGE:
                    grow = (_ANGLED_ROI_MIN_EDGE - size) / 2.0
                    extent[axis] -= grow
                    extent[axis + 2] += grow
            x0 = max(0, int(extent[0]))
            y0 = max(0, int(extent[1]))
            x1 = min(iw, int(extent[2] + 0.999))
            y1 = min(ih, int(extent[3] + 0.999))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            seed_box = (
                float(seed["x"]) - x0,
                float(seed["y"]) - y0,
                float(seed["w"]),
                float(seed["h"]),
            )
            rois.append((weight, (x0, y0, x1, y1), float(seed["angle"]), seed_box, seed["points"]))
        rois.sort(key=lambda r: r[0], reverse=True)
        if with_quads:
            return [(roi, angle, seed_box, quad) for _w, roi, angle, seed_box, quad in rois[:max_rois]]
        return [(roi, angle, seed_box) for _w, roi, angle, seed_box, _quad in rois[:max_rois]]

    @staticmethod
    def _leader_frame(line: dict[str, float]) -> tuple[float, float, float, float, float, float, float]:
        """(x1, y1, dx, dy, nx, ny, length) with the start ordered along the reading direction."""
        import math

        x1, y1, x2, y2 = (float(line[k]) for k in ("x1", "y1", "x2", "y2"))
        theta = math.radians(float(line["angle"]))
        dx, dy = math.cos(theta), math.sin(theta)
        if x2 * dx + y2 * dy < x1 * dx + y1 * dy:
            x1, y1, x2, y2 = x2, y2, x1, y1
        length = (x2 - x1) * dx + (y2 - y1) * dy
        return x1, y1, dx, dy, -dy, dx, length

    def _leader_text_side(
        self, image: Image.Image, line: dict[str, float], line_h: float
    ) -> float:
        """
        +1 or -1: which side of the leader its callout is written on.

        The writing hugs the line — its baseline sits on it — so the ink in a
        narrow band right against the line is the callout's own; a wider band
        picks up the next leader's text as well and chose the wrong side on
        the fit pair.
        """
        x1, y1, dx, dy, nx, ny, length = self._leader_frame(line)
        iw, ih = image.size
        corners = [
            (x1 + t * dx + n * nx, y1 + t * dy + n * ny)
            for t in (0.0, length + 1.5 * line_h)
            for n in (-1.2 * line_h, 1.2 * line_h)
        ]
        bx0 = max(0, int(min(c[0] for c in corners)))
        by0 = max(0, int(min(c[1] for c in corners)))
        bx1 = min(iw, int(max(c[0] for c in corners)) + 1)
        by1 = min(ih, int(max(c[1] for c in corners)) + 1)
        if bx1 - bx0 < 2 or by1 - by0 < 2:
            return 1.0
        ys, xs = np.mgrid[by0:by1, bx0:bx1]
        xs = xs.astype(np.float32) - x1
        ys = ys.astype(np.float32) - y1
        t = xs * dx + ys * dy
        n = xs * nx + ys * ny
        dark = np.asarray(image.crop((bx0, by0, bx1, by1)).convert("L")) < 128
        band = dark & (t >= 0) & (t <= length + 1.5 * line_h) & (np.abs(n) >= 0.15 * line_h) & (np.abs(n) <= 1.1 * line_h)
        return 1.0 if np.count_nonzero(band & (n > 0)) >= np.count_nonzero(band & (n < 0)) else -1.0

    def _leader_corridor(
        self,
        image: Image.Image,
        line: dict[str, float],
        line_h: float,
        other_lines: list[dict[str, float]] | None = None,
        *,
        side: float | None = None,
        other_sides: list[float] | None = None,
    ) -> tuple[tuple[int, int, int, int], Image.Image, list[tuple[float, float]]] | None:
        """
        The strip of drawing a callout written along ``line`` can occupy.

        A slanted callout is lettered along its leader: the value sits just
        above the line and its deviation stack right after the value. So the
        text of ONE callout lies in a corridor along the leader, on the side
        that carries ink, about two and a half lines deep. Where two leaders
        run close together (the fit pair is 22° apart and shares an origin)
        every pixel is given to the leader it is NEAREST, so the next leader's
        text never enters this corridor. Returns the corridor's bounding box,
        the crop with everything outside the corridor painted white, and the
        corridor polygon in image pixels.
        """
        import math

        iw, ih = image.size
        x1, y1, x2, y2 = (float(line[k]) for k in ("x1", "y1", "x2", "y2"))
        angle = float(line["angle"])
        theta = math.radians(angle)
        dx, dy = math.cos(theta), math.sin(theta)
        nx, ny = -dy, dx
        t1 = x1 * dx + y1 * dy
        t2 = x2 * dx + y2 * dy
        if t2 < t1:
            x1, y1, x2, y2 = x2, y2, x1, y1
            t1, t2 = t2, t1
        length = t2 - t1
        if length < 3.0 * line_h:
            return None

        def at(t: float, n: float) -> tuple[float, float]:
            return (x1 + t * dx + n * nx, y1 + t * dy + n * ny)

        t_lo, t_hi = -1.0 * line_h, length + 1.5 * line_h
        n_far, n_near = 3.2 * line_h, 0.2 * line_h
        corners = [at(t, n) for t in (t_lo, t_hi) for n in (-n_far, n_far)]
        bx0 = max(0, int(min(c[0] for c in corners)) - 4)
        by0 = max(0, int(min(c[1] for c in corners)) - 4)
        bx1 = min(iw, int(max(c[0] for c in corners)) + 5)
        by1 = min(ih, int(max(c[1] for c in corners)) + 5)
        if bx1 - bx0 < 16 or by1 - by0 < 16:
            return None

        # Pixel geometry over the bounding box.
        ys, xs = np.mgrid[by0:by1, bx0:bx1]
        xs = xs.astype(np.float32) - x1
        ys = ys.astype(np.float32) - y1
        t = xs * dx + ys * dy
        n = xs * nx + ys * ny
        along = (t >= t_lo) & (t <= t_hi)
        if side is None:
            side = self._leader_text_side(image, line, line_h)
        nearest = np.ones_like(t, dtype=bool)
        for index, other in enumerate(other_lines or []):
            ox1, oy1, odx, ody, onx, ony, olen = self._leader_frame(other)
            o_side = (other_sides or [])[index] if other_sides and index < len(other_sides) else None
            if o_side is None:
                o_side = self._leader_text_side(image, other, line_h)
            oxs = (xs + x1) - ox1
            oys = (ys + y1) - oy1
            ot = oxs * odx + oys * ody
            on_signed = oxs * onx + oys * ony
            # The other leader claims a pixel only on ITS text side, within its
            # own extent (its text does not continue past its arrow), and only
            # when clearly nearer. Distance alone is not enough: the upper
            # deviation of the -34.5° fit is nearly equidistant from the -57°
            # leader's tip — but it lies on the -57° leader's blank side.
            o_along = (ot >= 0.0) & (ot <= olen)
            nearest &= ~(o_along & (on_signed * o_side > 0) & (np.abs(on_signed) < 0.8 * np.abs(n)))

        keep = along & nearest & (n * side >= -n_near) & (n * side <= n_far)

        roi = np.asarray(image.crop((bx0, by0, bx1, by1)).convert("RGB")).copy()
        roi[~keep] = 255
        polygon = [
            at(t_lo, -n_near * side),
            at(t_hi, -n_near * side),
            at(t_hi, n_far * side),
            at(t_lo, n_far * side),
        ]
        return (bx0, by0, bx1, by1), Image.fromarray(roi), polygon

    def _detect_angled_regions(
        self,
        image: Image.Image,
        base_regions: list[dict[str, Any]],
        det_boxes: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recover slanted callouts (chamfers ``0.5×45°``, angled fits) that the
        upright/90° detector misses.

        Runs per callout *neighbourhood* rather than over the whole selection,
        so the result does not depend on how much of the drawing was selected —
        the failure that left the two ``H10`` fit callouts unread on a full
        sheet while they read correctly on a zoomed crop. Each neighbourhood is
        levelled by its own angles — the leader lines through it first, then
        the detector's quad angle, then the slant vote — re-detected, read, and
        mapped back. Results are deduped against ``base_regions`` so a value
        already read upright is not reported twice.
        """
        from region_detect import detect_leader_lines

        found: list[dict[str, Any]] = []
        # The whole selection first, which is what a zoomed-in crop needs. A
        # page-sized image gets no such pass: its vote names one angle for the
        # whole sheet and levelling the full page costs more than every
        # neighbourhood together.
        if max(image.size) <= _ANGLED_WHOLE_IMAGE_MAX_EDGE:
            found = self._angled_in_roi(image)

        # The slanted quads at OTHER angles: text on a neighbouring leader. It
        # is erased from each neighbourhood before levelling, because levelled
        # for this callout it is still slanted and the detector fuses it with
        # the callout's own text ("Ø20H10+0.0018H10+0.07") or crosses the two
        # ("Ø10+0.0700"). It has a neighbourhood of its own.
        slanted_quads: list[tuple[float, float, list[tuple[float, float]]]] = []
        for box in det_boxes or []:
            quad = box.get("quad") or box.get("polygon")
            angle = self._quad_angle(quad)
            if angle is None or abs(angle) < 8.0 or abs(angle) > 82.0:
                continue
            sides = self._quad_sides(quad)
            try:
                points = [(float(pt[0]), float(pt[1])) for pt in quad][:4]
            except (TypeError, ValueError, IndexError):
                continue
            if len(points) == 4 and sides is not None:
                slanted_quads.append((angle, sides[0], points))

        def inside(poly: list[tuple[float, float]], pt: tuple[float, float]) -> bool:
            x, y = pt
            hit = False
            for i in range(4):
                ax, ay = poly[i]
                bx, by = poly[(i + 1) % 4]
                if (ay > y) != (by > y):
                    cross = ax + (y - ay) * (bx - ax) / ((by - ay) or 1e-9)
                    if x < cross:
                        hit = not hit
            return hit

        def sample_points(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
            # A 4x4 grid of interior points in the quad's own frame.
            (ax, ay), (bx, by), (cx, cy), (dx, dy) = poly
            pts = []
            for u in (0.15, 0.4, 0.6, 0.85):
                for v in (0.15, 0.4, 0.6, 0.85):
                    top = (ax + (bx - ax) * u, ay + (by - ay) * u)
                    bottom = (dx + (cx - dx) * u, dy + (cy - dy) * u)
                    pts.append((top[0] + (bottom[0] - top[0]) * v, top[1] + (bottom[1] - top[1]) * v))
            return pts

        def share_ink(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
            ab = sum(1 for pt in sample_points(a) if inside(b, pt)) / 16.0
            ba = sum(1 for pt in sample_points(b) if inside(a, pt)) / 16.0
            return max(ab, ba) >= 0.2

        jobs: list[tuple[tuple[int, int, int, int], Image.Image, dict[str, Any]]] = []

        # Leaders first. A leader's angle is exact and its position says where
        # the callout's text is, which the detector's slanted quads do not:
        # on the fit pair the value quad measured 8° off its leader and the
        # stack quad 16°, and the longest "seed" quad was a box around the
        # surface-finish triangles. Each leader gets one levelled pass over
        # its own corridor, with the neighbouring leader's text painted out.
        try:
            page_leaders = detect_leader_lines(image)
        except Exception:
            page_leaders = []
        # The sheet's slanted line height: the median thickness of quads that
        # look like a line of writing (long, thin), never of every slanted
        # speck — on a full sheet those dragged it to 35 px for 55 px text
        # and the corridor was too shallow to hold the deviation stack. The
        # upright text height is the floor: callouts are lettered at one size.
        thicknesses = sorted(
            sides[1]
            for _angle, _length, points in slanted_quads
            for sides in [self._quad_sides(points)]
            if sides is not None
            and sides[1] >= 8.0
            and sides[0] >= 30.0
            and sides[0] >= 2.2 * sides[1]
        )
        upright_heights = sorted(
            float(b.get("h", 0))
            for b in det_boxes or []
            if float(b.get("w", 0)) >= 1.5 * float(b.get("h", 0)) > 0
            and self._quad_angle(b.get("quad") or b.get("polygon")) is not None
            and abs(self._quad_angle(b.get("quad") or b.get("polygon")) or 0.0) < 8.0
        )
        upright_h = upright_heights[len(upright_heights) // 2] if upright_heights else 0.0
        slanted_h = thicknesses[len(thicknesses) // 2] if thicknesses else 0.0
        line_h = max(slanted_h, upright_h, 30.0)
        line_h = max(12.0, min(line_h, 160.0))
        corridors: list[list[tuple[float, float]]] = []
        page_leaders = page_leaders[:8]
        leader_sides = [self._leader_text_side(image, line, line_h) for line in page_leaders]
        for index, line in enumerate(page_leaders):
            built = self._leader_corridor(
                image,
                line,
                line_h,
                other_lines=[o for k, o in enumerate(page_leaders) if k != index],
                side=leader_sides[index],
                other_sides=[sd for k, sd in enumerate(leader_sides) if k != index],
            )
            if built is None:
                continue
            (bx0, by0, bx1, by1), corridor_roi, polygon = built
            corridors.append(polygon)
            leader_angle = round(float(line["angle"]), 1)
            jobs.append(
                (
                    (bx0, by0, bx1, by1),
                    corridor_roi,
                    {
                        "sign": 1.0 if leader_angle >= 0 else -1.0,
                        "max_magnitudes": 1,
                        "weight_floor": 0.08,
                        "fallback_angles": [leader_angle],
                        "seed_box": None,
                        "max_angles": 1,
                        "line_height": line_h,
                    },
                )
            )

        def in_polygon(poly: list[tuple[float, float]], pt: tuple[float, float]) -> bool:
            return inside(poly, pt)

        for (x0, y0, x1, y1), quad_angle, seed_box, seed_quad in self._callout_neighbourhoods(
            image, det_boxes or [], with_quads=True
        ):
            # A seed whose text lies in a leader's corridor was read there.
            seed_centre = (
                sum(px for px, _ in seed_quad) / 4.0,
                sum(py for _, py in seed_quad) / 4.0,
            )
            if any(in_polygon(poly, seed_centre) for poly in corridors):
                continue
            roi = image.crop((x0, y0, x1, y1))
            seed_length, seed_thickness = self._quad_sides(seed_quad) or (0.0, 1.0)
            # Where a quad sits in the SEED'S OWN FRAME decides whether it is
            # this callout's or a neighbour's. Reading runs along the seed's
            # angle; its deviation stack sits ahead of the value's end and
            # within about a line of its centre line. A quad elsewhere that is
            # long enough to be a value, and shares no ink with the seed, is
            # text on another leader. Quad angles are too noisy to use here:
            # a stack's own quad came back 16° off its value, while the
            # neighbouring leader was only 22° away.
            import math

            theta = math.radians(quad_angle)
            direction = (math.cos(theta), math.sin(theta))
            normal = (-direction[1], direction[0])
            seed_t = [px * direction[0] + py * direction[1] for px, py in seed_quad]
            seed_n = [px * normal[0] + py * normal[1] for px, py in seed_quad]
            t_start, t_end = min(seed_t), max(seed_t)
            n_mid = (min(seed_n) + max(seed_n)) / 2.0

            def in_stack_zone(points: list[tuple[float, float]]) -> bool:
                cx = sum(px for px, _ in points) / 4.0
                cy = sum(py for _, py in points) / 4.0
                t = cx * direction[0] + cy * direction[1]
                n = cx * normal[0] + cy * normal[1]
                return (
                    t_start + 0.5 * (t_end - t_start) < t < t_end + 4.0 * seed_thickness
                    and abs(n - n_mid) < 1.5 * seed_thickness
                )

            def is_second_line(angle: float, points: list[tuple[float, float]]) -> bool:
                """
                A parallel line of text directly above or below the seed.

                "(BOTH SIDES)" under "0.5×45°" is the same callout, and so is
                the value when the note line happened to be picked as the
                seed. Blanking it as a neighbour erased the value itself, and
                the neighbourhood then read only the note.
                """
                if abs(angle - quad_angle) > 10.0:
                    return False
                cx = sum(px for px, _ in points) / 4.0
                cy = sum(py for _, py in points) / 4.0
                t = cx * direction[0] + cy * direction[1]
                n = cx * normal[0] + cy * normal[1]
                return (
                    t_start - seed_thickness < t < t_end + seed_thickness
                    and abs(n - n_mid) <= 2.5 * seed_thickness
                )

            others = [
                [(px - x0, py - y0) for px, py in points]
                for angle, length, points in slanted_quads
                if length >= 0.5 * seed_length
                and not share_ink(points, seed_quad)
                and not in_stack_zone(points)
                and not is_second_line(angle, points)
            ]
            if others:
                from PIL import ImageDraw

                roi = roi.copy()
                draw = ImageDraw.Draw(roi)
                for polygon in others:
                    # Grown by a few pixels so the strokes' anti-aliased edges go too.
                    cx = sum(px for px, _ in polygon) / 4.0
                    cy = sum(py for _, py in polygon) / 4.0
                    grown = [
                        (px + (4.0 if px > cx else -4.0), py + (4.0 if py > cy else -4.0))
                        for px, py in polygon
                    ]
                    draw.polygon(grown, fill=(255, 255, 255))
            sign = 1.0 if quad_angle >= 0 else -1.0
            # Leaders are measured on the neighbourhood itself: cheap, and the
            # lines found are the ones this text can sit on. Only those leaning
            # the seed's way are candidates; the one nearest the quad's angle
            # is tried first because it is almost certainly the seed's own.
            try:
                leaders = detect_leader_lines(roi)
            except Exception:
                leaders = []
            leader_angles = [
                a
                for a in self._leader_angles_for_roi(
                    leaders, (0, 0, x1 - x0, y1 - y0), max_angles=3
                )
                if (a >= 0) == (quad_angle >= 0)
            ]
            nearest = [a for a in leader_angles if abs(a - quad_angle) <= 8.0][:1]
            others = [a for a in leader_angles if a not in nearest][:2]
            candidate_angles = nearest + [round(quad_angle, 1)] + others
            jobs.append(
                (
                    (x0, y0, x1, y1),
                    roi,
                    {
                        "sign": sign,
                        "max_magnitudes": 1,
                        # This region is known to hold diagonal text, so a
                        # magnitude needs less of the vote to be worth a try.
                        "weight_floor": 0.08,
                        "fallback_angles": candidate_angles,
                        "seed_box": tuple(float(v) for v in seed_box),
                        "max_angles": 3,
                    },
                )
            )

        # Each neighbourhood is independent, so they are dealt across the OCR
        # workers when the server has them; otherwise read here, in order.
        pool = self._worker_pool() if len(jobs) > 1 else None
        if pool is not None:
            from ocr_workers import image_to_png, run_task

            payloads = [
                {"png": image_to_png(roi), "kwargs": kwargs} for _box, roi, kwargs in jobs
            ]
            per_roi = pool.map(
                "angled_roi",
                payloads,
                local=lambda payload: run_task(self, "angled_roi", payload),
            )
        else:
            per_roi = [self._angled_in_roi(roi, **kwargs) for _box, roi, kwargs in jobs]
        for ((x0, y0, _x1, _y1), _roi, _kwargs), regions_here in zip(jobs, per_roi):
            for region in regions_here or []:
                for box in (region["bbox"], region.get("oriented_box")):
                    if not box:
                        continue
                    box["x"] = round(box["x"] + x0, 1)
                    box["y"] = round(box["y"] + y0, 1)
                found.append(region)

        superseded = self._supersede_fused_base(found, base_regions)
        return self._dedupe_angled(
            found, [b for b in base_regions if id(b) not in superseded]
        )

    @staticmethod
    def _bbox_rot_to_source(
        bx: float, by: float, bw: float, bh: float, inv: Any, iw: int, ih: int
    ) -> dict[str, float] | None:
        """Map a rotated-frame AABB to a clamped upright AABB via ``inv``."""
        corners = [
            (bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh),
        ]
        xs, ys = [], []
        for px, py in corners:
            sx = inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]
            sy = inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]
            xs.append(sx)
            ys.append(sy)
        x0 = max(0.0, min(xs))
        y0 = max(0.0, min(ys))
        x1 = min(float(iw), max(xs))
        y1 = min(float(ih), max(ys))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        return {
            "x": round(x0, 1),
            "y": round(y0, 1),
            "width": round(x1 - x0, 1),
            "height": round(y1 - y0, 1),
        }

    @staticmethod
    def _ink_extent(image: Image.Image) -> tuple[int, int, int, int] | None:
        """
        Tight bounds of the *text* ink in a levelled crop, or ``None``.

        A cluster's union box is the hull of its detection boxes plus a margin,
        which on a diagonal callout leaves a box noticeably larger than the
        writing — that is what makes a balloon look like it is drawn over empty
        drawing rather than along the text. Long straight strokes are removed
        first so the leader the text sits on does not stretch the bounds back
        out to the whole crop.
        """
        try:
            import cv2
        except ImportError:
            return None
        import numpy as np

        from image_preprocess import _suppress_long_lines, cad_ink_to_gray

        gray = np.asarray(cad_ink_to_gray(image).convert("L"))
        if gray.size == 0:
            return None
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = _suppress_long_lines(mask)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _quad_angle(quad: Any) -> float | None:
        """
        Screen angle of a detection quad's reading direction, in degrees.

        The detector returns the four corners of a text line, so its longer edge
        *is* the direction the text runs — no estimation needed. Returns degrees
        in (-90, 90], clockwise positive (y axis down), or ``None`` when the quad
        is missing or degenerate.
        """
        import math

        if not quad or len(quad) < 4:
            return None
        (x0, y0), (x1, y1), _, (x3, y3) = quad[:4]
        top = (x1 - x0, y1 - y0)
        side = (x3 - x0, y3 - y0)
        len_top = math.hypot(*top)
        len_side = math.hypot(*side)
        if max(len_top, len_side) < 2.0:
            return None
        dx, dy = top if len_top >= len_side else side
        angle = math.degrees(math.atan2(dy, dx))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        return angle

    @staticmethod
    def _rotate_expand(
        image: Image.Image,
        angle: float,
        *,
        scale: float = 1.0,
        with_forward: bool = False,
    ) -> tuple[Image.Image, Any] | tuple[Image.Image, Any, Any]:
        """
        Rotate ``image`` CCW by ``angle`` degrees onto an expanded white canvas.

        Returns the rotated PIL image plus the inverse 2×3 affine that maps a
        point in the *rotated* frame back to the original (used to place a
        rotated-frame detection's balloon in received-image coordinates).
        With ``with_forward`` the forward matrix comes too, so source-frame
        detections can be projected into the rotated frame without re-detecting.
        """
        import cv2

        arr = np.asarray(image.convert("RGB"))
        h, w = arr.shape[:2]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        # The scale rides in the same affine, so the inverse returned below
        # maps an enlarged, levelled detection straight back to source pixels
        # with no extra bookkeeping at the call sites.
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        M[0, 2] += (nw - w) / 2.0
        M[1, 2] += (nh - h) / 2.0
        rot = cv2.warpAffine(
            arr, M, (nw, nh),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        inv = cv2.invertAffineTransform(M)
        if with_forward:
            return Image.fromarray(rot), inv, M
        return Image.fromarray(rot), inv

    def _stacked_deviation_read(
        self,
        crop: Image.Image,
        res: dict[str, Any],
        *,
        member_boxes: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """
        Re-read a levelled callout as ``value upper/lower``, or ``None``.

        A fit callout carries its two deviations stacked beside the value, and
        flat reading order interleaves them (``Ø18H10 +0.070/0`` comes back as
        ``1+0.07018H100``). One finer detection pass inside the crop separates
        the parts, and their geometry says which is which. Only the narrow
        stacked pattern with confidently-read parts is accepted, and the result
        is composed and classified like any other read, so ``None`` simply
        leaves the recogniser's own text in place.
        """
        from image_preprocess import cad_ink_to_gray
        from segment_quality import is_segment_worthy
        from stacked_tolerance import stacked_deviation_text

        parts = self._paddle_det_boxes(cad_ink_to_gray(crop).convert("RGB"))
        raw = stacked_deviation_text(parts)
        # The detector inside the crop tends to lose the small lower deviation
        # ("0") that the levelled pass had found as a box of its own. Those
        # member boxes are read with the fast recogniser and added wherever
        # the crop's detection left a gap, then the members alone are tried.
        if not raw and member_boxes and getattr(
            self, "_page_batch_recognition_available", False
        ):
            try:
                part_crops = []
                usable_members = []
                for box in member_boxes:
                    x0 = max(0, int(box["x"]) - 2)
                    y0 = max(0, int(box["y"]) - 2)
                    x1 = min(crop.width, int(box["x"] + box["w"]) + 2)
                    y1 = min(crop.height, int(box["y"] + box["h"]) + 2)
                    if x1 - x0 < 2 or y1 - y0 < 2:
                        continue
                    part_crops.append(crop.crop((x0, y0, x1, y1)))
                    usable_members.append(box)
                fast = self._recognize_page_batch(part_crops, batch_size=16)
                member_parts = [
                    {
                        **box,
                        "text": str(item.get("raw_ocr") or item.get("text") or ""),
                        "conf": float(item.get("confidence") or 0.0),
                    }
                    for box, item in zip(usable_members, fast)
                ]

                def overlaps_a_part(box: dict[str, Any]) -> bool:
                    for part in parts:
                        ox = min(box["x"] + box["w"], part["x"] + part["w"]) - max(box["x"], part["x"])
                        oy = min(box["y"] + box["h"], part["y"] + part["h"]) - max(box["y"], part["y"])
                        if ox > 0 and oy > 0 and ox * oy >= 0.3 * box["w"] * box["h"]:
                            return True
                    return False

                combined = list(parts) + [m for m in member_parts if not overlaps_a_part(m)]
                raw = stacked_deviation_text(combined) or stacked_deviation_text(member_parts)
            except Exception:
                raw = None
        if not raw:
            # Last resort: the interleaved string itself. Reading order puts
            # the upper deviation first, then the value, then the lower
            # deviation — "+0.07018H100" is "+0.070" | "18H10" | "0". A
            # deviation has two or three decimals and a fit class is one to
            # three digits, a letter and one or two digits, which pins the
            # split. Stray letters from a finish mark at either end are
            # dropped first.
            import re as _re

            text = str(res.get("text") or "")
            text = _re.sub(r"^[^+\-−\dØø]+|[^0-9]+$", "", text.replace(" ", ""))
            m = _re.fullmatch(
                r"([+\-−]\d[.,]\d{2,3})(\d{1,3}[A-Za-z]\d{1,2})([+\-−]?\d(?:[.,]\d{1,3})?)",
                text,
            )
            if m:
                raw = f"{m.group(2)} {m.group(1)}/{m.group(3)}"
        if not raw:
            return None
        # The value part sometimes carries the upper deviation's sign (and a
        # stray "0") when the detector cut between them: "18H10+0 +0.070/0".
        # A value never ends in a bare sign, so it is dropped.
        import re as _re2

        raw = _re2.sub(r"[+\-−]0?\s+(?=[+\-−])", " ", raw)
        symbols = DetectedSymbols(**(res.get("symbols_detected") or {}))
        composed = compose_engineering_dimension(raw, crop, symbols)
        text = (composed.text or "").strip()
        if not text or not is_segment_worthy(text):
            return None
        res["text"] = text
        res["type"] = composed.kind
        feature = classify_feature(text, symbols=res.get("symbols_detected") or {})
        res["category"], res["subtype"], res["label"] = (
            feature.category, feature.subtype, feature.label,
        )
        return text

    @staticmethod
    def _text_scale(boxes: list[dict[str, Any]]) -> float:
        """
        Median short side of the line-shaped detection boxes (≈ one line height).

        Near-square boxes (diagonal text, symbol frames) are excluded: their
        short side says nothing about the line height. Returns 0.0 when fewer
        than two line-shaped boxes exist, and callers then skip scale checks.
        """
        shorts = sorted(
            min(b["w"], b["h"])
            for b in boxes
            if max(b["w"], b["h"]) >= 1.5 * max(min(b["w"], b["h"]), 1.0)
        )
        if len(shorts) < 2:
            return 0.0
        return float(shorts[len(shorts) // 2])

    @staticmethod
    def _record_text_boxes(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Detection-style boxes (x, y, w, h, text, conf) from page records."""
        boxes: list[dict[str, Any]] = []
        for record in records:
            if record.get("table_excluded"):
                continue
            result = record.get("result") or {}
            text = str(result.get("raw_ocr") or record.get("text") or "").strip()
            if not text:
                continue
            bbox = record.get("bbox") or {}
            try:
                boxes.append(
                    {
                        "x": float(bbox["x"]),
                        "y": float(bbox["y"]),
                        "w": float(bbox["width"]),
                        "h": float(bbox["height"]),
                        "text": text,
                        "conf": float(result.get("confidence") or 0.0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return boxes

    def _notes_boxes_near_anchor(
        self,
        image: Image.Image,
        boxes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Line boxes from a recognising pass over the corner the notes sit in.

        The corner is found from what was already read: the "NOTES:" heading,
        or failing that the column of point markers ("12345") that the
        detector returns as one tall object. The crop runs from there to the
        right edge of the sheet and down to the first title-block word, so
        one small OCR pass replaces the whole-page pass that cost 75 s.
        Returns [] when neither anchor is present.
        """
        heading = None
        column = None
        for b in boxes:
            text = str(b.get("text") or "").strip()
            if heading is None and _NOTES_ANCHOR_RE.match(text):
                heading = b
            elif (
                column is None
                and _NOTE_COLUMN_RE.match(text)
                and b["h"] >= 2.5 * b["w"]
            ):
                digits = re.sub(r"\D", "", text)
                if digits == "".join(str(i) for i in range(1, len(digits) + 1)):
                    column = b
        iw, ih = image.size
        if heading is not None:
            line_h = float(heading["h"])
            x0 = heading["x"] - 0.5 * line_h
            y0 = heading["y"] - 0.5 * line_h
            y1 = heading["y"] + 14.0 * line_h
            anchor_bottom = heading["y"] + heading["h"]
        elif column is not None:
            count = len(re.sub(r"\D", "", str(column.get("text") or "")))
            line_h = float(column["h"]) / max(count, 1)
            x0 = column["x"] - 0.5 * line_h
            y0 = column["y"] - 2.0 * line_h
            y1 = column["y"] + column["h"] + 2.0 * line_h
            anchor_bottom = column["y"] + column["h"]
        else:
            return []
        # The paragraph never continues past the top of the title block.
        for b in boxes:
            if (
                b["y"] >= anchor_bottom
                and b["x"] < x0 + 0.6 * iw
                and _TITLE_BLOCK_WORDS_RE.search(str(b.get("text") or ""))
            ):
                y1 = min(y1, b["y"])
        crop_box = (
            max(0, int(x0)),
            max(0, int(y0)),
            iw,
            min(ih, int(y1)),
        )
        if crop_box[2] - crop_box[0] < 8 or crop_box[3] - crop_box[1] < 8:
            return []
        try:
            found = self._detect_regions_ocr(image.crop(crop_box).convert("RGB"))
        except Exception:
            return []
        for b in found:
            b["x"] = round(b["x"] + crop_box[0], 1)
            b["y"] = round(b["y"] + crop_box[1], 1)
        return self._join_note_markers(found)

    @staticmethod
    def _join_note_markers(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Put each point marker back on its line.

        At sheet resolution the detector returns "1." and "MATERIAL: …" as two
        boxes, and two markers drawn close together as one ("45."). The block
        finder recognises a point by its marker AND its first word, so a lone
        marker is joined to the line box beside it on the same row, and a
        fused marker column is cut into one box per digit first.
        """
        lone = re.compile(r"^\s*(\d{1,2})\s*[.)]?\s*$")
        expanded: list[dict[str, Any]] = []
        for b in boxes:
            text = str(b.get("text") or "").strip()
            digits = re.sub(r"\D", "", text)
            if (
                _NOTE_COLUMN_RE.match(text) or (lone.match(text) and len(digits) >= 2)
            ) and len(digits) >= 2 and b["h"] >= 1.5 * b["w"]:
                pitch = b["h"] / len(digits)
                for i, d in enumerate(digits):
                    expanded.append(
                        {**b, "y": round(b["y"] + i * pitch, 1), "h": round(pitch, 1), "text": f"{d}."}
                    )
            else:
                expanded.append(b)

        markers = [b for b in expanded if lone.match(str(b.get("text") or ""))]
        lines = [b for b in expanded if b not in markers]
        used: set[int] = set()
        out: list[dict[str, Any]] = []
        for m in markers:
            m_text = str(m.get("text") or "").strip()
            digit = lone.match(m_text).group(1)
            best = None
            best_gap = None
            for i, ln in enumerate(lines):
                if i in used:
                    continue
                text = str(ln.get("text") or "")
                if sum(c.isalpha() for c in text) < 3:
                    continue
                gap = ln["x"] - (m["x"] + m["w"])
                if gap < -0.3 * m["h"] or gap > 2.0 * m["h"]:
                    continue
                overlap = min(m["y"] + m["h"], ln["y"] + ln["h"]) - max(m["y"], ln["y"])
                if overlap < 0.5 * min(m["h"], ln["h"]):
                    continue
                if best is None or gap < best_gap:
                    best, best_gap = i, gap
            if best is None:
                out.append(m)
                continue
            used.add(best)
            ln = lines[best]
            top = min(m["y"], ln["y"])
            bottom = max(m["y"] + m["h"], ln["y"] + ln["h"])
            out.append(
                {
                    "x": m["x"],
                    "y": top,
                    "w": round(ln["x"] + ln["w"] - m["x"], 1),
                    "h": round(bottom - top, 1),
                    "text": f"{digit}. {str(ln.get('text') or '').strip()}",
                    "conf": min(float(m.get("conf") or 0.0), float(ln.get("conf") or 0.0)),
                }
            )
        out.extend(ln for i, ln in enumerate(lines) if i not in used)
        return out

    def _notes_region(
        self,
        image: Image.Image,
        *,
        boxes: list[dict[str, Any]] | None = None,
        debug_dump: bool = False,
        debug_dump_force: bool = False,
    ) -> dict[str, Any] | None:
        """
        Detect the NOTES paragraph and read it as one region.

        Deliberately runs its own recognition pass: detect_regions on this
        pipeline is DETECTOR-ONLY and returns geometry with empty text, so the
        numbered-point markers this looks for are simply not there. With
        recognised boxes the same detector reads all five notes on a real
        sheet; without them it finds nothing and the note lines leak into
        neighbouring dimension clusters, which is how a balloon came out as
        "AN.2WTOOPING.".
        """
        if boxes is None:
            try:
                boxes = self._detect_regions_ocr(image)
            except Exception:
                return None
        region, _ = self._detect_notes_block(boxes)
        if region is None:
            # The page route's objects are not lines: it hands over the
            # marker column as one tall "12345" and each long note line
            # read as debris, so the markers this looks for are missing.
            # The heading (or the column) still says where the block IS, so
            # just that corner is read again, line by line.
            near = self._notes_boxes_near_anchor(image, boxes)
            if near:
                region, _ = self._detect_notes_block(near)
        if region is None:
            return None
        bounds = region.pop("_bounds", None)
        line_h = region.pop("_line_h", 0.0)
        if bounds is not None:
            better = self._reread_notes_block(image, bounds, line_h)
            if better:
                region["text"] = "\n".join(better)
        return region

    def segment(
        self,
        image: Image.Image,
        *,
        debug_dump: bool = False,
        debug_dump_force: bool = False,
        cluster_margin: float = 0.72,
        progress_callback: Callable[..., None] | None = None,
        detection_only: bool = False,
        existing_value_boxes: Sequence[Mapping[str, float]] = (),
    ) -> dict[str, Any]:
        """
        Auto-segment a multi-value selection into individual dimensions.

        Detects all text regions, clusters fragments that belong to the same
        dimension, then runs the full single-value `recognize` pipeline on each
        cluster's crop. ``detection_only`` is reserved for whole-page
        orchestration: it returns those same final clusters without OCR so
        overlapping tile duplicates can be removed before recognition.

        Boxes are returned in received-image pixel coordinates (same space
        `recognize`'s `text_bbox` uses) so the client can reuse its existing
        value-box mapping.
        """
        from region_cluster import (
            cluster_boxes,
            drop_bridge_boxes,
            merge_fragment_clusters,
            merge_overlapping_clusters,
            order_clusters,
            split_mixed_clusters,
            union_bbox,
        )
        from segment_quality import (
            count_dimension_values,
            dedupe_regions,
            has_dimension_value,
            is_segment_worthy,
        )
        from stroke_filter import is_stray_line
        from page_scan import overlaps_existing_value
        from page_value_filters import (
            PageValueCandidate,
            is_complete_engineering_value,
            evaluate_scan_value,
            normalize_page_value_text,
        )
        from engineering_object_assembly import classify_assembly_role
        from engineering_value_parser import (
            engineering_parse_statistics,
            parse_engineering_value,
        )
        from engineering_disposition import (
            disposition_engineering_object,
            engineering_disposition_statistics,
        )
        from structured_symbol_vision import structured_symbol_statistics

        def report(
            *,
            stage: str,
            message: str,
            percent: int,
            completed: int = 0,
            total: int = 0,
            pass_current: int = 0,
            pass_total: int = 0,
            tile_current: int = 0,
            tile_total: int = 0,
            object_current: int = 0,
            object_total: int = 0,
            operation_label: str = "",
            candidate_count: int = 0,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    stage=stage,
                    message=message,
                    percent=percent,
                    completed=completed,
                    total=total,
                    pass_current=pass_current,
                    pass_total=pass_total,
                    tile_current=tile_current,
                    tile_total=tile_total,
                    object_current=object_current,
                    object_total=object_total,
                    operation_label=operation_label,
                    candidate_count=candidate_count,
                )

        report(
            stage="preparing",
            message="Preparing source-resolution scan and deskew",
            percent=4,
        )
        report(
            stage="detecting",
            message="Starting bounded detector-only localization",
            percent=7,
        )
        detection_state = {"completed": 0, "total": 0}

        def report_detection_pass(
            *,
            phase: str,
            completed: int,
            total: int,
            state: str,
            label: str,
            proposals: int,
            deskew_angle: float,
            pass_current: int,
            pass_total: int,
            tile_current: int,
            tile_total: int,
        ) -> None:
            detection_state["completed"] = completed
            detection_state["total"] = total
            deskew = (
                f"; deskew {deskew_angle:+.2f} deg"
                if deskew_angle != 0.0
                else ""
            )
            tile_text = (
                f"; tile {tile_current} of {tile_total}"
                if tile_total > 1
                else ""
            )
            action = "Running" if state == "running" else "Completed"
            phase_progress = {
                "proposing": (7, 3, "proposal"),
                "detecting": (10, 12, "detection"),
                "refining": (22, 8, "refinement"),
            }
            percent_start, percent_span, pass_name = phase_progress.get(
                phase,
                (7, 23, "detection"),
            )
            report(
                stage=phase,
                message=(
                    f"{action} {pass_name} pass {pass_current} of {pass_total}"
                    f"{tile_text}: {label} ({proposals} proposals{deskew})"
                ),
                percent=(
                    percent_start
                    + int(percent_span * completed / max(total, 1))
                ),
                completed=completed,
                total=total,
                pass_current=pass_current,
                pass_total=pass_total,
                tile_current=tile_current,
                tile_total=tile_total,
                operation_label=label,
                candidate_count=proposals,
            )

        boxes = self.detect_regions(
            image,
            progress_callback=report_detection_pass,
        )
        # The NOTES paragraph is lifted out BEFORE clustering: its lines are not
        # dimensions (they would all be dropped downstream), and leaving them in
        # lets a note line bridge into a neighbouring callout's cluster. The
        # region is added back untouched just before returning. Boxes are
        # excluded by geometry because the notes pass runs its own recognition
        # and its box list does not index into this one.
        notes_region = self._notes_region(
            image, debug_dump=debug_dump, debug_dump_force=debug_dump_force
        )
        if notes_region is not None:
            nb = notes_region["bbox"]
            nx0, ny0 = nb["x"], nb["y"]
            nx1, ny1 = nx0 + nb["width"], ny0 + nb["height"]
            boxes = [
                b
                for b in boxes
                if not (
                    b["x"] + b["w"] > nx0
                    and b["x"] < nx1
                    and b["y"] + b["h"] > ny0
                    and b["y"] < ny1
                )
            ]
        report(
            stage="detecting",
            message=f"Detected {len(boxes)} region proposals",
            percent=30,
            completed=detection_state["completed"],
            total=detection_state["total"],
            candidate_count=len(boxes),
        )

        grouping_units = 5
        # Drop cross-column bridge boxes before clustering so separate
        # dimensions (e.g. Ø174,07 and Ø175,32) don't fuse into one cluster.
        report(
            stage="grouping",
            message=f"Filtering bridge boxes from {len(boxes)} proposals",
            percent=30,
            completed=0,
            total=grouping_units,
            operation_label="Bridge filtering",
            candidate_count=len(boxes),
        )
        boxes = drop_bridge_boxes(boxes)
        report(
            stage="grouping",
            message=f"Kept {len(boxes)} proposals after bridge filtering",
            percent=32,
            completed=1,
            total=grouping_units,
            operation_label="Bridge filtering",
            candidate_count=len(boxes),
        )
        iw, ih = image.size
        report(
            stage="grouping",
            message=f"Spatially grouping {len(boxes)} nearby proposals",
            percent=32,
            completed=1,
            total=grouping_units,
            operation_label="Spatial grouping",
            candidate_count=len(boxes),
        )
        clustered = cluster_boxes(
            boxes,
            margin_ratio=cluster_margin,
            img_w=iw,
            img_h=ih,
        )
        report(
            stage="grouping",
            message=f"Built {len(clustered)} initial candidate objects",
            percent=34,
            completed=2,
            total=grouping_units,
            operation_label="Spatial grouping",
            candidate_count=len(clustered),
        )
        report(
            stage="grouping",
            message=f"Merging small fragments into {len(clustered)} candidates",
            percent=34,
            completed=2,
            total=grouping_units,
            operation_label="Fragment merging",
            candidate_count=len(clustered),
        )
        clustered = merge_fragment_clusters(clustered)
        report(
            stage="grouping",
            message=f"Kept {len(clustered)} candidates after fragment merging",
            percent=37,
            completed=3,
            total=grouping_units,
            operation_label="Fragment merging",
            candidate_count=len(clustered),
        )
        report(
            stage="grouping",
            message=f"Separating mixed orientations in {len(clustered)} candidates",
            percent=37,
            completed=3,
            total=grouping_units,
            operation_label="Orientation splitting",
            candidate_count=len(clustered),
        )
        clusters = order_clusters(split_mixed_clusters(clustered))

        report(
            stage="grouping",
            message=f"Prepared {len(clusters)} candidates for bounded refinement",
            percent=39,
            completed=4,
            total=grouping_units,
            operation_label="Orientation splitting",
            candidate_count=len(clusters),
        )

        def report_cluster_refinement(
            *,
            completed: int,
            total: int,
            state: str,
            label: str,
            candidate_count: int,
        ) -> None:
            action = "Refining" if state == "running" else "Completed"
            if state == "skipped":
                action = "Skipped"
            pass_current = (
                completed + 1
                if state == "running"
                else completed
            )
            report(
                stage="grouping",
                message=f"{action} detector-only {label}",
                percent=(
                    39 + int(2 * completed / max(total, 1))
                    if total > 0
                    else 39
                ),
                completed=4,
                total=grouping_units,
                pass_current=(min(pass_current, total) if total > 0 else 0),
                pass_total=total,
                operation_label="Detector-only cluster refinement",
                candidate_count=candidate_count,
            )

        clusters = self._expand_clusters(
            image,
            clusters,
            cluster_margin=cluster_margin,
            progress_callback=report_cluster_refinement,
        )
        # Re-join any fragments of one dimension that landed in overlapping
        # boxes (e.g. a value split from its REF. tag onto a perpendicular axis).
        report(
            stage="grouping",
            message=f"Merging overlaps among {len(clusters)} candidate objects",
            percent=41,
            completed=4,
            total=grouping_units,
            operation_label="Overlap merging",
            candidate_count=len(clusters),
        )
        clusters = order_clusters(merge_overlapping_clusters(clusters))
        clusters_before_existing = len(clusters)
        if existing_value_boxes:
            clusters = [
                cluster
                for cluster in clusters
                if not overlaps_existing_value(
                    union_bbox(cluster),
                    existing_value_boxes,
                )
            ]
        skipped_existing_count = clusters_before_existing - len(clusters)

        def cluster_children(
            cluster: Sequence[Mapping[str, Any]],
            *,
            cluster_index: int,
        ) -> list[dict[str, Any]]:
            """Serialize every detector fragment owned by a section object."""

            children: list[dict[str, Any]] = []
            for child_index, box in enumerate(cluster, start=1):
                bbox = {
                    "x": round(float(box.get("x", 0.0)), 1),
                    "y": round(float(box.get("y", 0.0)), 1),
                    "width": round(float(box.get("w", 0.0)), 1),
                    "height": round(float(box.get("h", 0.0)), 1),
                }
                text = str(box.get("text") or "")
                children.append(
                    {
                        "candidate_id": (
                            f"SECTION:{cluster_index:04d}:D{child_index:03d}"
                        ),
                        "text": text,
                        "raw_text": text,
                        "bbox": bbox,
                        "polygon": [
                            [bbox["x"], bbox["y"]],
                            [bbox["x"] + bbox["width"], bbox["y"]],
                            [
                                bbox["x"] + bbox["width"],
                                bbox["y"] + bbox["height"],
                            ],
                            [bbox["x"], bbox["y"] + bbox["height"]],
                        ],
                        "confidence": float(box.get("conf") or 0.0),
                        "orientation": (
                            "vertical"
                            if bbox["height"] > 1.35 * max(bbox["width"], 1.0)
                            else "horizontal"
                        ),
                        "rotation": 0.0,
                        "role": classify_assembly_role(text),
                        "recognition_source": "ocr",
                        "recognition_evidence": {},
                        "source_conflict": False,
                    }
                )
            return children
        report(
            stage="grouping",
            message=(
                f"Prepared {len(clusters)} candidate objects; "
                f"skipped {skipped_existing_count} existing balloons"
            ),
            percent=42,
            completed=grouping_units,
            total=grouping_units,
            operation_label="Grouping complete",
            candidate_count=len(clusters),
        )

        if should_dump(request_override=debug_dump):
            from debug_dump import dump_segment

            dump_segment(
                image,
                boxes,
                [union_bbox(c) for c in clusters],
                enabled=True,
                force=dump_force() or debug_dump_force,
            )

        if detection_only:
            detected_regions: list[dict[str, Any]] = []
            for cluster_index, cluster in enumerate(clusters, start=1):
                ub = union_bbox(cluster)
                children = cluster_children(
                    cluster,
                    cluster_index=cluster_index,
                )
                detected_regions.append(
                    {
                        "bbox": {
                            "x": round(ub["x"], 1),
                            "y": round(ub["y"], 1),
                            "width": round(ub["width"], 1),
                            "height": round(ub["height"], 1),
                        },
                        "detection_confidence": max(
                            (
                                float(box.get("conf", 0.0))
                                for box in cluster
                            ),
                            default=0.0,
                        ),
                        "object_id": f"SECTION:{cluster_index:04d}",
                        "assembly_id": f"SECTION:{cluster_index:04d}",
                        "assembly_rule": (
                            "section_geometry_cluster"
                            if len(children) > 1
                            else "atomic"
                        ),
                        "assembly_conflict": False,
                        "assembly_review_reason": "",
                        "assembly_children": children,
                    }
                )
            return {
                "count": len(detected_regions),
                "skipped_existing_count": skipped_existing_count,
                "regions": detected_regions,
            }

        margin = 6  # a few px of context around each cluster crop
        regions: list[dict[str, Any]] = []
        cluster_total = len(clusters)
        for cluster_index, cluster in enumerate(clusters, start=1):
            report(
                stage="recognizing",
                message=f"Recognizing object {cluster_index} of {cluster_total}",
                percent=42 + int(48 * (cluster_index - 1) / max(cluster_total, 1)),
                completed=cluster_index - 1,
                total=cluster_total,
                object_current=cluster_index,
                object_total=cluster_total,
                operation_label=f"Object {cluster_index}",
                candidate_count=cluster_total,
            )
            ub = union_bbox(cluster)
            children = cluster_children(
                cluster,
                cluster_index=cluster_index,
            )
            cx0 = max(0, int(ub["x"] - margin))
            cy0 = max(0, int(ub["y"] - margin))
            cx1 = min(iw, int(ub["x"] + ub["width"] + margin))
            cy1 = min(ih, int(ub["y"] + ub["height"] + margin))
            if cx1 - cx0 < 1 or cy1 - cy0 < 1:
                continue

            sub = image.crop((cx0, cy0, cx1, cy1))
            res = self.recognize(
                sub,
                debug_dump=debug_dump,
                debug_dump_force=debug_dump_force,
                compute_text_bbox=False,  # balloon snaps to cluster union, not text_bbox
            )
            text = (res.get("text") or "").strip()
            if not text:
                continue
            engineering_symbol = res.get("engineering_symbol") or {}
            has_structured_evidence = bool(engineering_symbol.get("kind"))

            # A balloon is for something a person measures. These three gates
            # drop what merely looks numeric: zone letters and balloon numbers
            # ("4", "33"), revision dates ("DATE3/17/2011..."), and leader or
            # extension lines that OCR'd as a phantom digit.
            canonical_short_nominal = bool(
                re.fullmatch(r"[1-9]\d?", text.strip())
            )
            if (not has_structured_evidence and not is_segment_worthy(text)
                    and not canonical_short_nominal):
                continue
            if not has_structured_evidence and not has_dimension_value(text):
                continue
            if not has_structured_evidence and is_stray_line(sub, text):
                continue

            # Content-aware split: close-proximity callouts that fused into one
            # cluster. Trigger when the fused read still shows 2+ complete values
            # OR the cluster's boxes span 2+ horizontal rows (the fused single
            # read often collapses to one value, hiding the multiplicity). The
            # split only replaces this region when it actually yields 2+ worthy
            # values, so a false trigger falls back to the single region.
            multivalue = count_dimension_values(text) >= 2
            multirow = (
                self._cluster_row_count(cluster) >= 2
                and ub["width"] >= ub["height"]
            )
            if multirow:
                # A limit dimension is one callout drawn on two rows; read it
                # as one value before the split has a chance to cut it in two.
                limit = self._read_limit_pair(
                    image,
                    {"x": cx0, "y": cy0, "width": cx1 - cx0, "height": cy1 - cy0},
                )
                if limit is not None:
                    limit_res = dict(limit["result"])
                    limit_res["type"] = limit["type"]
                    limit_res["confidence"] = limit["confidence"]
                    feature = classify_feature(
                        limit["text"], symbols=limit_res.get("symbols_detected") or {}
                    )
                    limit_res["category"], limit_res["subtype"], limit_res["label"] = (
                        feature.category, feature.subtype, feature.label,
                    )
                    limit_region = self._region_from_result(
                        limit_res, limit["bbox"], limit["text"]
                    )
                    limit_region.update(
                        {
                            "assembly_rule": "section_limit_stack",
                            "assembly_conflict": False,
                            "assembly_review_reason": "",
                            "assembly_children": children,
                        }
                    )
                    regions.append(limit_region)
                    continue
            if multivalue or multirow:
                split_regions = self._split_stacked_cluster(
                    image, (cx0, cy0, cx1, cy1),
                    debug_dump=debug_dump, debug_dump_force=debug_dump_force,
                )
                if split_regions:
                    regions.extend(split_regions)
                    continue

            # Snap balloon to the full detected cluster, not a tight OCR sliver.
            bbox = {
                "x": round(ub["x"], 1),
                "y": round(ub["y"], 1),
                "width": round(ub["width"], 1),
                "height": round(ub["height"], 1),
            }

            regions.append(
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": res.get("confidence", 0.0),
                    "type": res.get("type"),
                    # Rule-engine output, so the balloon can show the feature
                    # category and its suggested label rather than the
                    # frontend's text-only guess.
                    "category": res.get("category"),
                    "subtype": res.get("subtype"),
                    "label": res.get("label"),
                    "orientation": res.get("orientation", "horizontal"),
                    "rotation": res.get("rotation", 0),
                    "needs_review": res.get("needs_review", False),
                    "agreement": res.get("agreement", 0.0),
                    "engine": res.get("engine", "paddleocr"),
                    "symbols_detected": res.get("symbols_detected"),
                    "engineering_symbol": res.get("engineering_symbol"),
                    "assembly_rule": (
                        "section_geometry_cluster"
                        if len(children) > 1
                        else "atomic"
                    ),
                    "assembly_conflict": False,
                    "assembly_review_reason": "",
                    "assembly_children": children,
                }
            )

            report(
                stage="recognizing",
                message=f"Recognized object {cluster_index} of {cluster_total}",
                percent=42 + int(48 * cluster_index / max(cluster_total, 1)),
                completed=cluster_index,
                total=cluster_total,
                object_current=cluster_index,
                object_total=cluster_total,
                operation_label=f"Object {cluster_index}",
                candidate_count=cluster_total,
            )

        report(
            stage="finalizing",
            message="Checking and merging scan results",
            percent=94,
            completed=cluster_total,
            total=cluster_total,
            candidate_count=cluster_total,
        )
        regions = self._complete_angle_regions(image, regions)
        # Inch limits beside their bracketed metric limits are one callout.
        regions, _joined_metric = join_dual_unit_limits(regions)
        regions = dedupe_regions(regions)

        # Recover slanted callouts (chamfers, angled fits) the upright/90°
        # detector misses — the branch's detection only ever levels by a
        # quarter-turn, so text drawn along a leader line goes unread. Gated on
        # slant presence, so non-diagonal sheets pay only a single Hough call.
        # Added AFTER the base set is deduped and treated as authoritative:
        # angled reads only ADD values not already present, never displace a
        # clean upright region.
        regions.extend(self._detect_angled_regions(image, regions, det_boxes=boxes))
        # A base region the angled pass resolved into its separate callouts is
        # dropped here (it was flagged while those reads were being placed).
        regions = [r for r in regions if not r.pop("superseded", False)]
        recognized_count = len(regions)
        filtered_regions: list[dict[str, Any]] = []
        review_candidates: list[dict[str, Any]] = []
        candidate_outcomes: list[dict[str, Any]] = []
        filter_rule_counts: dict[str, int] = {}
        for candidate_index, region in enumerate(regions, start=1):
            text = normalize_page_value_text(str(region.get("text") or ""))
            classified_region = {**region, "text": text}
            self._apply_feature_labels(classified_region)
            candidate_id = f"S{candidate_index:04d}"
            assembly_children = list(region.get("assembly_children") or ())
            if not assembly_children:
                assembly_children = [
                    {
                        "candidate_id": candidate_id,
                        "text": text,
                        "raw_text": str(
                            region.get("raw_ocr") or region.get("text") or ""
                        ),
                        "bbox": dict(region["bbox"]),
                        "polygon": [],
                        "confidence": float(region.get("confidence") or 0.0),
                        "orientation": region.get("orientation", "horizontal"),
                        "rotation": float(region.get("rotation") or 0.0),
                        "role": classify_assembly_role(text),
                        "recognition_source": str(
                            region.get("recognition_source") or "ocr"
                        ),
                        "recognition_evidence": dict(
                            region.get("recognition_evidence") or {}
                        ),
                        "source_conflict": bool(region.get("source_conflict")),
                    }
                ]
            classified_region.update(
                {
                    "object_id": candidate_id,
                    "assembly_id": candidate_id,
                    "assembly_rule": region.get("assembly_rule", "atomic"),
                    "assembly_conflict": bool(
                        region.get("assembly_conflict")
                    ),
                    "assembly_review_reason": str(
                        region.get("assembly_review_reason") or ""
                    ),
                    "assembly_children": assembly_children,
                }
            )
            decision = evaluate_scan_value(
                PageValueCandidate(text=text, bbox=region["bbox"]),
                scope_kind="section",
                table_masks=(),
                page_size=image.size,
            )
            filter_rule_counts[decision.rule_name] = (
                filter_rule_counts.get(decision.rule_name, 0) + 1
            )
            engineering_parse = parse_engineering_value(
                text,
                feature_category=str(classified_region.get("category") or "") or None,
                engineering_symbol=classified_region.get("engineering_symbol") or {},
            )
            disposition = disposition_engineering_object(
                engineering_parse,
                recognized=True,
                authoritative=True,
                context_rule=decision.rule_name,
                context_reason=decision.reason,
                assembly_conflict=bool(
                    classified_region.get("assembly_conflict")
                ),
                assembly_review_reason=str(
                    classified_region.get("assembly_review_reason") or ""
                ),
                source_conflict=bool(classified_region.get("source_conflict")),
                numeric_conflict=bool(classified_region.get("numeric_conflict")),
                recognition_review_required=bool(
                    classified_region.get("authoritative_review_required")
                ),
                recognition_needs_review=bool(
                    classified_region.get("needs_review")
                ),
                recognition_review_reason=str(
                    classified_region.get("review_reason") or ""
                ),
            )
            state = (
                "excluded" if disposition.state == "other" else disposition.state
            )
            classified_region.update(
                {
                    "recognized": True,
                    "needs_review": state == "review",
                    "page_filter_rule": decision.rule_name,
                    "page_filter_reason": decision.reason,
                    "review_reason": (
                        disposition.reason if state == "review" else ""
                    ),
                    "engineering_parse": engineering_parse.to_dict(),
                    "engineering_disposition": disposition.to_dict(),
                }
            )
            candidate_outcomes.append(
                {
                    "candidate_id": candidate_id,
                    "object_id": candidate_id,
                    "bbox": dict(region["bbox"]),
                    "state": state,
                    "text": text,
                    "raw_text": str(
                        region.get("raw_ocr") or region.get("text") or ""
                    ),
                    "preliminary_text": str(region.get("text") or ""),
                    "confidence": float(region.get("confidence") or 0.0),
                    "recognized": True,
                    "reason": disposition.reason,
                    "rule": disposition.rule,
                    "page_filter_rule": decision.rule_name,
                    "page_filter_reason": decision.reason,
                    "type": classified_region.get("type"),
                    "category": classified_region.get("category"),
                    "subtype": classified_region.get("subtype"),
                    "label": classified_region.get("label"),
                    "engineering_symbol": classified_region.get("engineering_symbol"),
                    "orientation": region.get("orientation", "horizontal"),
                    "rotation": float(region.get("rotation") or 0.0),
                    "oriented_box": region.get("oriented_box"),
                    "recovery_attempted": bool(
                        region.get("recovery_attempted")
                    ),
                    "authoritative_reread": bool(
                        region.get("authoritative_reread")
                    ),
                    "assembly_id": candidate_id,
                    "assembly_rule": classified_region.get(
                        "assembly_rule", "atomic"
                    ),
                    "assembly_conflict": bool(
                        classified_region.get("assembly_conflict")
                    ),
                    "assembly_review_reason": str(
                        classified_region.get("assembly_review_reason") or ""
                    ),
                    "assembly_children": [
                        dict(child) for child in assembly_children
                    ],
                    "engineering_parse": engineering_parse.to_dict(),
                    "engineering_disposition": disposition.to_dict(),
                }
            )
            published_region = {
                **classified_region,
                "candidate_id": candidate_id,
            }
            if state == "eligible":
                filtered_regions.append(published_region)
            elif state == "review":
                review_candidates.append(published_region)
        regions = filtered_regions
        report(
            stage="finalizing",
            message=f"Prepared {len(regions)} balloon candidates",
            percent=99,
            completed=len(regions),
            total=len(regions),
            candidate_count=len(regions),
        )
        # The notes block remains visible, but it is non-inspection content and
        # therefore belongs in Other instead of receiving a balloon.
        if notes_region is not None:
            note_text = str(notes_region.get("text") or "")
            note_parse = parse_engineering_value(note_text)
            note_disposition = disposition_engineering_object(
                note_parse,
                recognized=bool(note_text.strip()),
                authoritative=True,
                context_rule="note_information",
                context_reason="General drawing notes",
            )
            note_id = "S-NOTES"
            candidate_outcomes.append(
                {
                    "candidate_id": note_id,
                    "object_id": note_id,
                    "bbox": dict(notes_region["bbox"]),
                    "state": "excluded",
                    "text": note_text,
                    "raw_text": note_text,
                    "preliminary_text": note_text,
                    "confidence": float(notes_region.get("confidence") or 0.0),
                    "recognized": bool(note_text.strip()),
                    "reason": note_disposition.reason,
                    "rule": note_disposition.rule,
                    "page_filter_rule": "note_information",
                    "page_filter_reason": "General drawing notes",
                    "type": notes_region.get("type"),
                    "category": notes_region.get("category"),
                    "subtype": notes_region.get("subtype"),
                    "label": notes_region.get("label"),
                    "orientation": notes_region.get("orientation", "horizontal"),
                    "rotation": float(notes_region.get("rotation") or 0.0),
                    "recovery_attempted": False,
                    "authoritative_reread": True,
                    "assembly_id": note_id,
                    "assembly_rule": "notes_block",
                    "assembly_conflict": False,
                    "assembly_review_reason": "",
                    "assembly_children": [],
                    "engineering_parse": note_parse.to_dict(),
                    "engineering_disposition": note_disposition.to_dict(),
                }
            )
            recognized_count += 1

        # M6 runs after assembly and M5 parsing, so every published, review,
        # and Other object carries the exact evidence behind its state.
        parsed_objects = [
            item["engineering_parse"] for item in candidate_outcomes
        ]
        dispositions = [
            item["engineering_disposition"] for item in candidate_outcomes
        ]
        candidate_ids = {
            str(item.get("candidate_id") or "") for item in candidate_outcomes
        }
        parsed_objects.extend(
            item["engineering_parse"]
            for item in (*regions, *review_candidates)
            if not item.get("candidate_id")
            or str(item.get("candidate_id")) not in candidate_ids
        )

        excluded_count = sum(
            1 for item in candidate_outcomes if item["state"] == "excluded"
        )
        review_count = len(review_candidates)
        return {
            "count": len(regions),
            "detected_count": cluster_total + (1 if notes_region is not None else 0),
            "recognized_count": recognized_count,
            "eligible_count": len(regions),
            "excluded_count": excluded_count,
            "review_count": review_count,
            "unread_count": max(
                0,
                cluster_total + (1 if notes_region is not None else 0)
                - recognized_count,
            ),
            "skipped_existing_count": skipped_existing_count,
            "filter_rule_counts": dict(sorted(filter_rule_counts.items())),
            "assembly_stats": {
                "atomic_detection_count": sum(len(cluster) for cluster in clusters),
                "object_count": cluster_total,
                "assembled_object_count": sum(
                    1 for cluster in clusters if len(cluster) > 1
                ),
                "absorbed_fragment_count": max(
                    0,
                    sum(len(cluster) for cluster in clusters) - cluster_total,
                ),
                "conflict_count": 0,
                "rule_counts": {},
            },
            "structured_symbol_stats": structured_symbol_statistics(
                [item for item in candidate_outcomes if item.get("engineering_symbol")]
            ),
            "engineering_parse_stats": engineering_parse_statistics(
                parsed_objects
            ),
            "engineering_disposition_stats": engineering_disposition_statistics(
                dispositions
            ),
            "regions": regions,
            "review_candidates": review_candidates,
            "candidate_outcomes": candidate_outcomes,
        }

    def _worker_pool(self) -> Any | None:
        """
        The started OCR worker pool, or ``None`` to run everything in-process.

        Only the server starts the pool (in its start-up hook); tests and
        scripts never spawn workers unless they ask to, and every pooled stage
        has an in-process path that is exactly the sequential code.
        """
        try:
            from ocr_workers import get_worker_pool

            pool = get_worker_pool(start=False)
        except Exception:  # noqa: BLE001 - no pool is a slower scan, not a failure
            return None
        try:
            return pool if pool.available else None
        except Exception:  # noqa: BLE001
            return None

    def _authoritative_attempts(
        self,
        tight_crop: Any,
        padded_crop: Any,
        *,
        debug_dump: bool = False,
        debug_dump_force: bool = False,
        max_paddle_predictions: int | None = None,
        preliminary_text: str = "",
    ) -> list[dict[str, Any]]:
        """
        The full Draw Value OCR read of one detected object: the tight crop,
        then the padded crop only when the tight read cannot be accepted.

        ``max_paddle_predictions`` bounds the voting recogniser per crop. The
        page route passes 3: its crops are rectified upright from the detector
        polygon, so the two deskew variants can add nothing, and on the junk
        objects that reach this stage (specks, hatching, a datum letter) they
        doubled a read that was never going to be a value.

        The padded retry exists to complete a value the tight crop clipped.
        When neither the tight read nor the preliminary read contains a single
        digit there is no value to complete, so the retry is skipped.
        """
        from page_candidate_recovery import (
            assess_authoritative_result,
            authoritative_result_needs_retry,
        )

        tight_result = self.recognize(
            tight_crop.image,
            debug_dump=debug_dump,
            debug_dump_force=debug_dump_force,
            compute_text_bbox=True,
            max_paddle_predictions=max_paddle_predictions,
        )
        attempts = [assess_authoritative_result(tight_result, tight_crop)]
        if authoritative_result_needs_retry(attempts[0]):
            digit_seen = any(
                c.isdigit()
                for c in (
                    str(attempts[0].get("text") or "")
                    + str(tight_result.get("raw_ocr") or "")
                    + str(preliminary_text or "")
                )
            )
            if digit_seen:
                padded_result = self.recognize(
                    padded_crop.image,
                    debug_dump=debug_dump,
                    debug_dump_force=debug_dump_force,
                    compute_text_bbox=True,
                    max_paddle_predictions=max_paddle_predictions,
                )
                attempts.append(
                    assess_authoritative_result(padded_result, padded_crop)
                )
        return attempts

    def segment_page(
        self,
        image: Image.Image,
        *,
        layout: Any | None = None,
        debug_dump: bool = False,
        debug_dump_force: bool = False,
        cluster_margin: float = 0.72,
        progress_callback: Callable[..., None] | None = None,
        existing_value_boxes: Sequence[Mapping[str, float]] = (),
        native_evidence: Any | None = None,
    ) -> dict[str, Any]:
        """Scan a whole page through the bounded technical-value route.

        Section scanning intentionally remains in :meth:`segment`.  This page
        path runs only standalone detection (0/90 degrees) on adaptive panels,
        deduplicates atomic boxes without grouping neighbours, recognizes crops
        in standalone model batches, performs bounded recognition-only recovery,
        applies ``page_value_filters.py`` as a cost gate, then rereads every
        eligible/review object through the complete Draw Value OCR pipeline
        before final filtering and publication.
        """

        from collections import Counter

        from detection_passes import (
            DetectionTile,
            build_primary_detection_image,
            detection_primary_target_edge,
            detection_tile_target_edge,
            map_quarter_turn_box_to_source,
            rotate_for_detection,
        )
        from page_layout import (
            PageLayout,
            analyze_page_layout,
            mask_table_regions,
        )
        from page_scan import (
            PAGE_SCAN_DETECTOR_MIN_LONG_EDGE,
            PAGE_SCAN_MAX_DETECTOR_CALLS,
            PAGE_SCAN_MAX_TILES,
            PAGE_SCAN_RECOGNITION_BATCH_SIZE,
            PAGE_SCAN_ROTATIONS_CW,
            assign_candidate_ids,
            deduplicate_page_candidates,
            map_tile_candidate,
            overlaps_existing_value,
        )
        from page_candidate_recovery import (
            CONTEXT_MAX_CANDIDATES,
            RECOVERY_MAX_CANDIDATES,
            assess_authoritative_result,
            authoritative_result_needs_retry,
            build_authoritative_crops,
            build_context_crop,
            build_recovery_crops,
            reconstruct_line_context,
            resolve_authoritative_hypotheses,
            resolve_recovery_consensus,
            result_needs_recovery,
            result_needs_second_recovery,
            select_recovery_record_indexes,
        )
        from page_value_filters import (
            EXCLUDE_TABLE_REGIONS,
            PageValueCandidate,
            candidate_is_in_table,
            is_complete_engineering_value,
            evaluate_page_value,
            needs_expanded_filter_context,
            normalize_page_value_text,
        )
        from pdf_evidence import (
            PdfPageEvidence,
            fuse_native_with_ocr,
            match_native_spans,
            native_match_is_authoritative,
            native_result,
            ocr_evidence,
        )
        from segment_quality import (
            count_dimension_values,
            dedupe_regions,
            strip_foreign_glyphs,
        )
        from engineering_object_assembly import (
            assembly_statistics,
            atomic_object_evidence,
            plan_engineering_object_assemblies,
        )
        from engineering_value_parser import (
            engineering_parse_statistics,
            parse_engineering_value,
        )
        from engineering_disposition import (
            disposition_engineering_object,
            engineering_disposition_statistics,
        )
        from structured_symbol_vision import (
            plan_structured_engineering_symbols,
            structured_symbol_statistics,
        )

        def report(
            *,
            stage: str,
            message: str,
            percent: int,
            completed: int = 0,
            total: int = 0,
            pass_current: int = 0,
            pass_total: int = 0,
            tile_current: int = 0,
            tile_total: int = 0,
            object_current: int = 0,
            object_total: int = 0,
            batch_current: int = 0,
            batch_total: int = 0,
            operation_label: str = "",
            candidate_count: int = 0,
            overlay: dict[str, object] | None = None,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    stage=stage,
                    message=message,
                    percent=percent,
                    completed=completed,
                    total=total,
                    pass_current=pass_current,
                    pass_total=pass_total,
                    tile_current=tile_current,
                    tile_total=tile_total,
                    object_current=object_current,
                    object_total=object_total,
                    batch_current=batch_current,
                    batch_total=batch_total,
                    operation_label=operation_label,
                    candidate_count=candidate_count,
                    overlay=overlay,
                )

        if not getattr(self, "_text_detector_available", False):
            raise RuntimeError(
                "Whole-page auto-ballooning requires the standalone "
                "PaddleOCR text detector; section scanning remains available"
            )
        if not getattr(self, "_page_batch_recognition_available", False):
            raise RuntimeError(
                "Whole-page auto-ballooning requires standalone PaddleOCR "
                "orientation and recognition models; the slow per-object "
                "fallback is intentionally disabled"
            )

        # ``cluster_margin`` remains in the public signature for compatibility
        # with existing callers. Page mode deliberately performs no grouping.
        _ = (cluster_margin, debug_dump, debug_dump_force)
        page_size = image.size
        if native_evidence is not None and not isinstance(
            native_evidence, PdfPageEvidence
        ):
            raise TypeError("segment_page native_evidence must be PdfPageEvidence")
        source_profile = (
            native_evidence.profile_dict()
            if native_evidence is not None
            else {
                "kind": "raster",
                "page": 0,
                "page_count": 0,
                "native_text_available": False,
                "native_span_count": 0,
                "native_character_count": 0,
                "raster_coverage": 1.0,
                "vector_object_count": 0,
            }
        )
        native_spans = native_evidence.spans if native_evidence is not None else ()
        if layout is None:
            layout = analyze_page_layout(image)
        if not isinstance(layout, PageLayout):
            raise TypeError("segment_page layout must be a PageLayout")
        if (layout.width, layout.height) != page_size:
            raise ValueError("Page layout dimensions do not match the scan image")

        masked_page = mask_table_regions(image, layout.table_masks)
        tiles = [
            DetectionTile(
                x=panel.bbox.x,
                y=panel.bbox.y,
                width=panel.bbox.width,
                height=panel.bbox.height,
            )
            for panel in layout.panels
        ]
        tile_total = len(tiles)
        if tile_total < 1 or tile_total > PAGE_SCAN_MAX_TILES:
            raise AssertionError("Whole-page layout exceeded eight panels")
        detector_pass_total = tile_total * len(PAGE_SCAN_ROTATIONS_CW)
        if detector_pass_total > PAGE_SCAN_MAX_DETECTOR_CALLS:
            raise AssertionError("Whole-page scan exceeded sixteen detector calls")
        page_candidates = []
        panel_states = {
            panel.panel_id: "pending" for panel in layout.panels
        }
        report(
            stage="preparing",
            message=(
                f"Preparing {tile_total} adaptive panel"
                f"{'s' if tile_total != 1 else ''}, "
                f"{len(layout.table_masks)} table masks, and "
                f"{detector_pass_total} detector-only passes"
            ),
            percent=10,
            completed=0,
            total=detector_pass_total,
            tile_total=tile_total,
            pass_total=detector_pass_total,
            operation_label="Adaptive page layout",
            overlay=layout.overlay(
                scope_kind="page",
                panel_states=panel_states,
            ),
        )

        detector_pass_index = 0
        detection_pool = self._worker_pool() if detector_pass_total > 2 else None
        if detection_pool is not None:
            # Every (panel, rotation) pass is independent, so they are dealt
            # across the OCR workers and this process; boxes are mapped back
            # to the page here, in pass order, exactly as the loop below does.
            from ocr_workers import image_to_png, png_to_image

            full_target_edge = max(
                PAGE_SCAN_DETECTOR_MIN_LONG_EDGE,
                detection_primary_target_edge(masked_page),
            )
            pass_specs: list[tuple[int, Any, int, int, tuple[int, int]]] = []
            payloads: list[dict[str, Any]] = []
            for tile_index, tile in enumerate(tiles, start=1):
                tile_image = masked_page.crop(tile.box)
                primary_image = build_primary_detection_image(tile_image)
                target_long_edge = detection_tile_target_edge(
                    tile,
                    full_size=page_size,
                    pass_target_edge=full_target_edge,
                )
                for rotation_cw in PAGE_SCAN_ROTATIONS_CW:
                    pass_specs.append(
                        (tile_index, tile, rotation_cw, target_long_edge, tile_image.size)
                    )
                    payloads.append(
                        {
                            "png": image_to_png(
                                rotate_for_detection(primary_image, rotation_cw)
                            ),
                            "target_long_edge": target_long_edge,
                        }
                    )
            report(
                stage="detecting",
                message=(
                    f"Running {detector_pass_total} detector-only passes across "
                    "the OCR workers"
                ),
                percent=12,
                completed=0,
                total=detector_pass_total,
                pass_current=1,
                pass_total=detector_pass_total,
                tile_current=1,
                tile_total=tile_total,
                operation_label="Adaptive panels · detector only",
                candidate_count=0,
                overlay=layout.overlay(scope_kind="page", panel_states=panel_states),
            )
            passes_done = 0

            def on_pass_done(index: int, detected_boxes: Any) -> None:
                nonlocal passes_done
                passes_done += 1
                tile_index, tile, rotation_cw, _target, tile_size = pass_specs[index]
                for detected in detected_boxes or []:
                    restored = map_quarter_turn_box_to_source(
                        detected, rotation_cw, tile_size
                    )
                    if restored is None:
                        continue
                    mapped = map_tile_candidate(
                        {
                            "x": restored["x"],
                            "y": restored["y"],
                            "width": restored["w"],
                            "height": restored["h"],
                        },
                        tile,
                        tile_index=tile_index,
                        page_size=page_size,
                        detection_confidence=float(restored.get("conf", 0.0)),
                        pass_index=index + 1,
                        rotation_cw=rotation_cw,
                        polygon=restored.get("polygon"),
                    )
                    if mapped is not None:
                        page_candidates.append(mapped)
                panel_states[layout.panels[tile_index - 1].panel_id] = "completed"
                report(
                    stage="detecting",
                    message=(
                        f"Completed detector-only pass {passes_done} of "
                        f"{detector_pass_total}; collected "
                        f"{len(page_candidates)} raw candidates"
                    ),
                    percent=12 + int(40 * passes_done / max(detector_pass_total, 1)),
                    completed=passes_done,
                    total=detector_pass_total,
                    pass_current=passes_done,
                    pass_total=detector_pass_total,
                    tile_current=tile_index,
                    tile_total=tile_total,
                    operation_label=f"Panel {tile_index} · {rotation_cw} deg complete",
                    candidate_count=len(page_candidates),
                    overlay=layout.overlay(scope_kind="page", panel_states=panel_states),
                )

            detection_pool.map(
                "detect_tile",
                payloads,
                local=lambda payload: self._detector_only_boxes(
                    png_to_image(payload["png"]),
                    target_long_edge=int(payload["target_long_edge"]),
                ),
                on_item_done=on_pass_done,
            )
            detector_pass_index = detector_pass_total
            tiles = []
        for tile_index, tile in enumerate(tiles, start=1):
            panel_id = layout.panels[tile_index - 1].panel_id
            panel_states[panel_id] = "active"
            tile_image = masked_page.crop(tile.box)
            primary_image = build_primary_detection_image(tile_image)
            full_target_edge = max(
                PAGE_SCAN_DETECTOR_MIN_LONG_EDGE,
                detection_primary_target_edge(masked_page),
            )
            target_long_edge = detection_tile_target_edge(
                tile,
                full_size=page_size,
                pass_target_edge=full_target_edge,
            )
            for rotation_cw in PAGE_SCAN_ROTATIONS_CW:
                detector_pass_index += 1
                report(
                    stage="detecting",
                    message=(
                        f"Running detector-only pass {detector_pass_index} of "
                        f"{detector_pass_total}: panel {tile_index} of "
                        f"{tile_total}, {rotation_cw} deg"
                    ),
                    percent=(
                        12
                        + int(
                            40
                            * (detector_pass_index - 1)
                            / max(detector_pass_total, 1)
                        )
                    ),
                    completed=detector_pass_index - 1,
                    total=detector_pass_total,
                    pass_current=detector_pass_index,
                    pass_total=detector_pass_total,
                    tile_current=tile_index,
                    tile_total=tile_total,
                    operation_label=(
                        f"Panel {panel_id} · {rotation_cw} deg · "
                        f"{target_long_edge}px"
                    ),
                    candidate_count=len(page_candidates),
                    overlay=layout.overlay(
                        scope_kind="page",
                        panel_states=panel_states,
                    ),
                )
                rotated = rotate_for_detection(primary_image, rotation_cw)
                detected_boxes = self._detector_only_boxes(
                    rotated,
                    target_long_edge=target_long_edge,
                )
                for detected in detected_boxes:
                    restored = map_quarter_turn_box_to_source(
                        detected,
                        rotation_cw,
                        tile_image.size,
                    )
                    if restored is None:
                        continue
                    mapped = map_tile_candidate(
                        {
                            "x": restored["x"],
                            "y": restored["y"],
                            "width": restored["w"],
                            "height": restored["h"],
                        },
                        tile,
                        tile_index=tile_index,
                        page_size=page_size,
                        detection_confidence=float(
                            restored.get("conf", 0.0)
                        ),
                        pass_index=detector_pass_index,
                        rotation_cw=rotation_cw,
                        polygon=restored.get("polygon"),
                    )
                    if mapped is not None:
                        page_candidates.append(mapped)

                report(
                    stage="detecting",
                    message=(
                        f"Completed detector-only pass {detector_pass_index} "
                        f"of {detector_pass_total}; collected "
                        f"{len(page_candidates)} raw candidates"
                    ),
                    percent=(
                        12
                        + int(
                            40
                            * detector_pass_index
                            / max(detector_pass_total, 1)
                        )
                    ),
                    completed=detector_pass_index,
                    total=detector_pass_total,
                    pass_current=detector_pass_index,
                    pass_total=detector_pass_total,
                    tile_current=tile_index,
                    tile_total=tile_total,
                    operation_label=(
                        f"Panel {panel_id} · {rotation_cw} deg complete"
                    ),
                    candidate_count=len(page_candidates),
                )

            panel_states[panel_id] = "completed"
            report(
                stage="detecting",
                message=f"Completed panel {tile_index} of {tile_total}",
                percent=(
                    12
                    + int(40 * detector_pass_index / max(detector_pass_total, 1))
                ),
                completed=detector_pass_index,
                total=detector_pass_total,
                pass_current=detector_pass_index,
                pass_total=detector_pass_total,
                tile_current=tile_index,
                tile_total=tile_total,
                operation_label=f"Panel {panel_id} complete",
                candidate_count=len(page_candidates),
                overlay=layout.overlay(
                    scope_kind="page",
                    panel_states=panel_states,
                ),
            )

        report(
            stage="grouping",
            message=(
                f"Strictly deduplicating {len(page_candidates)} atomic "
                "detector candidates"
            ),
            percent=54,
            completed=0,
            total=1,
            tile_current=tile_total,
            tile_total=tile_total,
            pass_current=detector_pass_total,
            pass_total=detector_pass_total,
            operation_label="Atomic candidate deduplication",
            candidate_count=len(page_candidates),
        )
        deduplicated_candidates = deduplicate_page_candidates(page_candidates)
        duplicates_removed = len(page_candidates) - len(deduplicated_candidates)
        new_candidates = [
            candidate
            for candidate in deduplicated_candidates
            if not overlaps_existing_value(
                candidate.bbox,
                existing_value_boxes,
            )
        ]
        skipped_existing_count = (
            len(deduplicated_candidates) - len(new_candidates)
        )
        final_candidates = assign_candidate_ids(new_candidates)
        neutral_overlay_candidates = [
            {
                "id": candidate.candidate_id,
                "bbox": dict(candidate.bbox),
                "state": "detected",
            }
            for candidate in final_candidates
        ]
        report(
            stage="grouping",
            message=(
                f"Prepared {len(final_candidates)} page objects; "
                f"removed {duplicates_removed} same-object duplicates and "
                f"skipped {skipped_existing_count} existing balloons"
            ),
            percent=57,
            completed=1,
            total=1,
            tile_current=tile_total,
            tile_total=tile_total,
            pass_current=detector_pass_total,
            pass_total=detector_pass_total,
            operation_label="Atomic candidate deduplication complete",
            candidate_count=len(final_candidates),
            overlay=layout.overlay(
                scope_kind="page",
                panel_states=panel_states,
                candidates=neutral_overlay_candidates,
            ),
        )

        table_masks = [mask.to_dict() for mask in layout.table_masks]
        margin = 6
        page_width, page_height = page_size
        detected_count = len(final_candidates)
        primary_recognized_count = 0
        # An object inside a detected table is excluded by the page policy
        # whatever it reads — the table rule is checked before the text — so
        # recognising it first is pure cost. On a real sheet the title block
        # and the tolerance tables held 60 of 204 detected objects. They keep
        # their record (the state invariant counts every detection) but skip
        # every recognition, recovery and reread batch.
        native_matches = {
            index: match
            for index, candidate in enumerate(final_candidates)
            if (
                match := match_native_spans(candidate.bbox, native_spans)
            ) is not None
        }
        native_primary_indexes = {
            index
            for index, match in native_matches.items()
            if not (
                EXCLUDE_TABLE_REGIONS
                and candidate_is_in_table(
                    final_candidates[index].bbox,
                    table_masks,
                )
            )
            and native_match_is_authoritative(
                match,
                validator=lambda text: is_complete_engineering_value(
                    normalize_page_value_text(text)
                ),
            )
        }
        recognition_indexes = [
            index
            for index, candidate in enumerate(final_candidates)
            if index not in native_primary_indexes
            and not (
                EXCLUDE_TABLE_REGIONS
                and candidate_is_in_table(candidate.bbox, table_masks)
            )
        ]
        recognition_total = len(recognition_indexes)
        table_skipped_count = detected_count - recognition_total - len(
            native_primary_indexes
        )
        ocr_records: list[dict[str, Any]] = []
        for index, candidate in enumerate(final_candidates):
            table_excluded = bool(
                EXCLUDE_TABLE_REGIONS
                and candidate_is_in_table(candidate.bbox, table_masks)
            )
            native_authoritative = (
                index in native_primary_indexes and not table_excluded
            )
            if native_authoritative:
                result = native_result(native_matches[index])
                text = normalize_page_value_text(str(result.get("text") or ""))
                result["text"] = text
                primary_recognized_count += int(bool(text))
            else:
                result = {
                    "text": "",
                    "raw_ocr": "",
                    "confidence": 0.0,
                    "agreement": 0.0,
                    "needs_review": False,
                    "type": None,
                    "engine": "paddleocr",
                    "orientation": "horizontal",
                    "rotation": 0,
                    "symbols_detected": None,
                    "ocr_profile": (
                        "table_region_skipped"
                        if table_excluded
                        else "batch_recognition"
                    ),
                }
                text = ""
            ocr_records.append(
                {
                    "candidate": candidate,
                    "candidate_id": candidate.candidate_id,
                    "bbox": candidate.bbox,
                    "polygon": [list(point) for point in candidate.polygon],
                    "text": text,
                    "result": result,
                    "recognized": bool(text),
                    "context_text": "",
                    "table_excluded": table_excluded,
                    "native_match": native_matches.get(index),
                    "native_authoritative": native_authoritative,
                }
            )
        candidate_crops: dict[int, Image.Image] = {}
        for index in recognition_indexes:
            bbox = final_candidates[index].bbox
            x0 = max(0, int(bbox["x"] - margin))
            y0 = max(0, int(bbox["y"] - margin))
            x1 = min(
                page_width,
                int(bbox["x"] + bbox["width"] + margin + 0.999),
            )
            y1 = min(
                page_height,
                int(bbox["y"] + bbox["height"] + margin + 0.999),
            )
            candidate_crops[index] = (
                image.crop((x0, y0, x1, y1))
                if x1 > x0 and y1 > y0
                else Image.new("RGB", (1, 1), (255, 255, 255))
            )

        primary_batch_total = (
            (recognition_total + PAGE_SCAN_RECOGNITION_BATCH_SIZE - 1)
            // PAGE_SCAN_RECOGNITION_BATCH_SIZE
        )
        pool = self._worker_pool()

        def apply_primary_batch(
            batch_record_indexes: list[int], batch_results: list[dict[str, Any]]
        ) -> None:
            nonlocal primary_recognized_count
            if len(batch_results) != len(batch_record_indexes):
                raise RuntimeError(
                    "Page recognition batch returned an unexpected result count"
                )
            for record_index, result in zip(batch_record_indexes, batch_results):
                record = ocr_records[record_index]
                match = record.get("native_match")
                if match is not None:
                    result = fuse_native_with_ocr(
                        match,
                        result,
                        native_authoritative=False,
                    )
                else:
                    result = ocr_evidence(result)
                text = normalize_page_value_text(
                    strip_foreign_glyphs(str(result.get("text") or ""))
                )
                result["text"] = text
                if text:
                    primary_recognized_count += 1
                record["text"] = text
                record["result"] = result
                record["recognized"] = bool(text)
                record["table_excluded"] = False

        def run_pooled_batches(
            *,
            stage: str,
            label: str,
            batches: list[list[Image.Image]],
            profile: str,
            percent_from: int,
            percent_span: int,
            overlay: dict[str, object] | None,
            on_batch: Callable[[int, list[dict[str, Any]]], None],
        ) -> None:
            """Run recognition batches across the worker pool, reporting as each lands."""
            from ocr_workers import image_to_png, png_to_image

            total = len(batches)
            report(
                stage=stage,
                message=f"{label}: {total} batches across the OCR workers",
                percent=percent_from,
                completed=0,
                total=total,
                batch_current=1,
                batch_total=total,
                operation_label=f"{label} 1/{total}",
                candidate_count=sum(len(b) for b in batches),
                overlay=overlay,
            )
            payloads = [
                {
                    "pngs": [image_to_png(crop) for crop in batch],
                    "batch_size": PAGE_SCAN_RECOGNITION_BATCH_SIZE,
                    "profile": profile,
                }
                for batch in batches
            ]
            finished = 0

            def on_done(index: int, results: Any) -> None:
                nonlocal finished
                finished += 1
                on_batch(index, list(results))
                report(
                    stage=stage,
                    message=f"Completed {label.lower()} batch {finished} of {total}",
                    percent=percent_from + int(percent_span * finished / max(total, 1)),
                    completed=finished,
                    total=total,
                    batch_current=finished,
                    batch_total=total,
                    operation_label=f"{label} {finished}/{total}",
                    candidate_count=sum(len(b) for b in batches),
                )

            pool.map(
                "batch",
                payloads,
                local=lambda payload: self._recognize_page_batch(
                    [png_to_image(data) for data in payload["pngs"]],
                    batch_size=PAGE_SCAN_RECOGNITION_BATCH_SIZE,
                    profile=profile,
                ),
                on_item_done=on_done,
            )

        primary_batches = [
            recognition_indexes[start : start + PAGE_SCAN_RECOGNITION_BATCH_SIZE]
            for start in range(0, recognition_total, PAGE_SCAN_RECOGNITION_BATCH_SIZE)
        ]
        if pool is not None and len(primary_batches) > 1:
            run_pooled_batches(
                stage="recognizing",
                label="Primary recognition",
                batches=[[candidate_crops[i] for i in b] for b in primary_batches],
                profile="batch_recognition",
                percent_from=59,
                percent_span=17,
                overlay=layout.overlay(
                    scope_kind="page",
                    panel_states=panel_states,
                    candidates=neutral_overlay_candidates,
                ),
                on_batch=lambda index, results: apply_primary_batch(
                    primary_batches[index], results
                ),
            )
            primary_batches = []
        for batch_index, batch_record_indexes in enumerate(primary_batches, start=1):
            batch_start = (batch_index - 1) * PAGE_SCAN_RECOGNITION_BATCH_SIZE
            batch_end = min(
                recognition_total,
                batch_start + PAGE_SCAN_RECOGNITION_BATCH_SIZE,
            )
            report(
                stage="recognizing",
                message=(
                    f"Primary recognition batch {batch_index} of "
                    f"{primary_batch_total} ({batch_start + 1}–{batch_end} "
                    f"of {recognition_total} objects; "
                    f"{table_skipped_count} table objects skipped)"
                ),
                percent=(
                    59
                    + int(
                        17
                        * (batch_index - 1)
                        / max(primary_batch_total, 1)
                    )
                ),
                completed=batch_index - 1,
                total=primary_batch_total,
                batch_current=batch_index,
                batch_total=primary_batch_total,
                operation_label=(
                    f"Primary recognition {batch_index}/{primary_batch_total}"
                ),
                candidate_count=detected_count,
                overlay=layout.overlay(
                    scope_kind="page",
                    panel_states=panel_states,
                    candidates=neutral_overlay_candidates,
                ),
            )
            batch_results = self._recognize_page_batch(
                [candidate_crops[index] for index in batch_record_indexes],
                batch_size=PAGE_SCAN_RECOGNITION_BATCH_SIZE,
            )
            apply_primary_batch(batch_record_indexes, batch_results)
            report(
                stage="recognizing",
                message=(
                    f"Completed primary batch {batch_index} of "
                    f"{primary_batch_total}; read {primary_recognized_count}"
                ),
                percent=(
                    59
                    + int(
                        17 * batch_index / max(primary_batch_total, 1)
                    )
                ),
                completed=batch_index,
                total=primary_batch_total,
                batch_current=batch_index,
                batch_total=primary_batch_total,
                operation_label=(
                    f"Primary recognition {batch_index}/{primary_batch_total}"
                ),
                candidate_count=detected_count,
            )

        # Table-skipped objects read as empty, which is exactly what selects a
        # record for recovery, so they are kept out of the pool (and its
        # budget) rather than filtered after selection.
        recoverable_indexes = [
            index
            for index, record in enumerate(ocr_records)
            if not record.get("table_excluded")
            and not record.get("native_authoritative")
        ]
        selected_recovery_indexes = [
            recoverable_indexes[position]
            for position in select_recovery_record_indexes(
                [ocr_records[index] for index in recoverable_indexes],
                maximum=RECOVERY_MAX_CANDIDATES,
            )
        ]
        selected_recovery_set = set(selected_recovery_indexes)
        all_doubtful_indexes = {
            index
            for index in recoverable_indexes
            if result_needs_recovery(ocr_records[index]["result"])
        }
        budget_exhausted_indexes = (
            all_doubtful_indexes - selected_recovery_set
        )
        recovery_attempts: dict[int, list[dict[str, Any]]] = {
            index: [] for index in selected_recovery_indexes
        }
        recovery_crops = {
            index: build_recovery_crops(
                image,
                ocr_records[index]["bbox"],
                ocr_records[index]["polygon"],
            )
            for index in selected_recovery_indexes
        }
        recovery_overlay_candidates = [
            {
                "id": record["candidate_id"],
                "bbox": dict(record["bbox"]),
                "state": (
                    "recovering"
                    if index in selected_recovery_set
                    else "detected"
                ),
                "text": record["text"],
            }
            for index, record in enumerate(ocr_records)
        ]
        first_recovery_batch_total = (
            (
                len(selected_recovery_indexes)
                + PAGE_SCAN_RECOGNITION_BATCH_SIZE
                - 1
            )
            // PAGE_SCAN_RECOGNITION_BATCH_SIZE
        )
        first_recovery_batches = [
            selected_recovery_indexes[
                batch_start : batch_start + PAGE_SCAN_RECOGNITION_BATCH_SIZE
            ]
            for batch_start in range(
                0, len(selected_recovery_indexes), PAGE_SCAN_RECOGNITION_BATCH_SIZE
            )
        ]
        if pool is not None and len(first_recovery_batches) > 1:
            def apply_first_recovery(index: int, results: list[dict[str, Any]]) -> None:
                batch_indexes = first_recovery_batches[index]
                if len(results) != len(batch_indexes):
                    raise RuntimeError(
                        "Page recovery batch returned an unexpected result count"
                    )
                for record_index, result in zip(batch_indexes, results):
                    recovery_attempts[record_index].append(result)

            run_pooled_batches(
                stage="recovering",
                label="Recovery pass 1",
                batches=[
                    [recovery_crops[i][0].image for i in b] for b in first_recovery_batches
                ],
                profile=recovery_crops[first_recovery_batches[0][0]][0].profile,
                percent_from=77,
                percent_span=6,
                overlay=layout.overlay(
                    scope_kind="page",
                    panel_states=panel_states,
                    candidates=recovery_overlay_candidates,
                ),
                on_batch=apply_first_recovery,
            )
            first_recovery_batches = []
        for first_batch_index, batch_indexes in enumerate(first_recovery_batches, start=1):
            batch_profile = recovery_crops[batch_indexes[0]][0].profile
            report(
                stage="recovering",
                message=(
                    f"Recovery pass 1 batch {first_batch_index} of "
                    f"{first_recovery_batch_total}: "
                    f"{batch_profile.replace('_', ' ')}"
                ),
                percent=(
                    77
                    + int(
                        6
                        * (first_batch_index - 1)
                        / max(first_recovery_batch_total, 1)
                    )
                ),
                completed=first_batch_index - 1,
                total=first_recovery_batch_total,
                batch_current=first_batch_index,
                batch_total=first_recovery_batch_total,
                operation_label=(
                    f"Recovery pass 1 "
                    f"{first_batch_index}/{first_recovery_batch_total}"
                ),
                candidate_count=len(selected_recovery_indexes),
                overlay=layout.overlay(
                    scope_kind="page",
                    panel_states=panel_states,
                    candidates=recovery_overlay_candidates,
                ),
            )
            results = self._recognize_page_batch(
                [recovery_crops[index][0].image for index in batch_indexes],
                batch_size=PAGE_SCAN_RECOGNITION_BATCH_SIZE,
                profile=batch_profile,
            )
            if len(results) != len(batch_indexes):
                raise RuntimeError(
                    "Page recovery batch returned an unexpected result count"
                )
            for record_index, result in zip(batch_indexes, results):
                recovery_attempts[record_index].append(result)
            report(
                stage="recovering",
                message=(
                    f"Completed recovery pass 1 batch {first_batch_index} of "
                    f"{first_recovery_batch_total}"
                ),
                percent=(
                    77
                    + int(
                        6
                        * first_batch_index
                        / max(first_recovery_batch_total, 1)
                    )
                ),
                completed=first_batch_index,
                total=first_recovery_batch_total,
                batch_current=first_batch_index,
                batch_total=first_recovery_batch_total,
                operation_label=(
                    f"Recovery pass 1 "
                    f"{first_batch_index}/{first_recovery_batch_total}"
                ),
                candidate_count=len(selected_recovery_indexes),
            )

        second_recovery_indexes = [
            index
            for index in selected_recovery_indexes
            if recovery_attempts[index]
            and result_needs_second_recovery(
                ocr_records[index]["result"],
                recovery_attempts[index][0],
            )
        ]
        second_recovery_batch_total = (
            (
                len(second_recovery_indexes)
                + PAGE_SCAN_RECOGNITION_BATCH_SIZE
                - 1
            )
            // PAGE_SCAN_RECOGNITION_BATCH_SIZE
        )
        recovery_batch_total = (
            first_recovery_batch_total + second_recovery_batch_total
        )
        second_recovery_batches = [
            second_recovery_indexes[
                batch_start : batch_start + PAGE_SCAN_RECOGNITION_BATCH_SIZE
            ]
            for batch_start in range(
                0, len(second_recovery_indexes), PAGE_SCAN_RECOGNITION_BATCH_SIZE
            )
        ]
        if pool is not None and len(second_recovery_batches) > 1:
            def apply_second_recovery(index: int, results: list[dict[str, Any]]) -> None:
                batch_indexes = second_recovery_batches[index]
                if len(results) != len(batch_indexes):
                    raise RuntimeError(
                        "Page recovery batch returned an unexpected result count"
                    )
                for record_index, result in zip(batch_indexes, results):
                    recovery_attempts[record_index].append(result)

            run_pooled_batches(
                stage="recovering",
                label="Recovery pass 2",
                batches=[
                    [recovery_crops[i][1].image for i in b] for b in second_recovery_batches
                ],
                profile=recovery_crops[second_recovery_batches[0][0]][1].profile,
                percent_from=83,
                percent_span=6,
                overlay=layout.overlay(
                    scope_kind="page",
                    panel_states=panel_states,
                    candidates=recovery_overlay_candidates,
                ),
                on_batch=apply_second_recovery,
            )
            second_recovery_batches = []
        for second_batch_index, batch_indexes in enumerate(second_recovery_batches, start=1):
            recovery_batch_index = (
                first_recovery_batch_total + second_batch_index
            )
            batch_profile = recovery_crops[batch_indexes[0]][1].profile
            report(
                stage="recovering",
                message=(
                    f"Recovery pass 2 batch {second_batch_index} of "
                    f"{second_recovery_batch_total}: "
                    f"{batch_profile.replace('_', ' ')}"
                ),
                percent=(
                    83
                    + int(
                        6
                        * (second_batch_index - 1)
                        / max(second_recovery_batch_total, 1)
                    )
                ),
                completed=recovery_batch_index - 1,
                total=recovery_batch_total,
                batch_current=recovery_batch_index,
                batch_total=recovery_batch_total,
                operation_label=(
                    f"Recovery pass 2 "
                    f"{second_batch_index}/{second_recovery_batch_total}"
                ),
                candidate_count=len(second_recovery_indexes),
                overlay=layout.overlay(
                    scope_kind="page",
                    panel_states=panel_states,
                    candidates=recovery_overlay_candidates,
                ),
            )
            results = self._recognize_page_batch(
                [recovery_crops[index][1].image for index in batch_indexes],
                batch_size=PAGE_SCAN_RECOGNITION_BATCH_SIZE,
                profile=batch_profile,
            )
            if len(results) != len(batch_indexes):
                raise RuntimeError(
                    "Page recovery batch returned an unexpected result count"
                )
            for record_index, result in zip(batch_indexes, results):
                recovery_attempts[record_index].append(result)
            report(
                stage="recovering",
                message=(
                    f"Completed recovery pass 2 batch {second_batch_index} of "
                    f"{second_recovery_batch_total}"
                ),
                percent=(
                    83
                    + int(
                        6
                        * second_batch_index
                        / max(second_recovery_batch_total, 1)
                    )
                ),
                completed=recovery_batch_index,
                total=recovery_batch_total,
                batch_current=recovery_batch_index,
                batch_total=recovery_batch_total,
                operation_label=(
                    f"Recovery pass 2 "
                    f"{second_batch_index}/{second_recovery_batch_total}"
                ),
                candidate_count=len(second_recovery_indexes),
            )

        for index, record in enumerate(ocr_records):
            if record.get("table_excluded") or record.get(
                "native_authoritative"
            ):
                continue
            resolved = resolve_recovery_consensus(
                record["result"],
                recovery_attempts.get(index, ()),
                attempted=index in selected_recovery_set,
                budget_exhausted=index in budget_exhausted_indexes,
            )
            # The multilingual recogniser emits a CJK character for a stroke
            # cluster it cannot resolve, so "R1.4" can arrive as "月1.4". The
            # section route has always stripped these; the page route did not,
            # and the residue then failed every downstream value check.
            normalized_text = normalize_page_value_text(
                strip_foreign_glyphs(str(resolved.get("text") or ""))
            )
            resolved["text"] = normalized_text
            match = record.get("native_match")
            if match is not None:
                resolved = fuse_native_with_ocr(
                    match,
                    resolved,
                    native_authoritative=False,
                )
            else:
                resolved = ocr_evidence(resolved)
            normalized_text = normalize_page_value_text(
                strip_foreign_glyphs(str(resolved.get("text") or ""))
            )
            resolved["text"] = normalized_text
            record["result"] = resolved
            record["text"] = normalized_text
            record["recognized"] = bool(record["text"])

        # Recover graphical engineering objects before ordinary text assembly.
        from dataclasses import replace

        atomic_detection_count = len(ocr_records)
        structured_plans = plan_structured_engineering_symbols(image, ocr_records)
        structured_by_first = {min(plan.member_indexes): plan for plan in structured_plans}
        structured_absorbed = {i for plan in structured_plans for i in plan.member_indexes}
        structured_records: list[dict[str, Any]] = []
        for record_index, record in enumerate(ocr_records):
            plan = structured_by_first.get(record_index)
            if plan is None:
                if record_index not in structured_absorbed:
                    structured_records.append(record)
                continue
            members = [ocr_records[i] for i in plan.member_indexes]
            anchor = dict(ocr_records[plan.anchor_index])
            box = plan.bbox
            polygon = ((box["x"], box["y"]), (box["x"]+box["width"], box["y"]),
                       (box["x"]+box["width"], box["y"]+box["height"]),
                       (box["x"], box["y"]+box["height"]))
            candidate = replace(
                anchor["candidate"], bbox=dict(box), polygon=polygon,
                candidate_id=plan.object_id,
                boundary_review=any(bool(m["candidate"].boundary_review) for m in members),
                detection_confidence=max(float(m["candidate"].detection_confidence) for m in members),
            )
            evidence = plan.evidence.to_dict()
            result = dict(anchor.get("result") or {})
            result.update({"text": plan.text, "confidence": plan.confidence,
                           "needs_review": plan.needs_review,
                           "review_reason": plan.review_reason,
                           "engineering_symbol": evidence,
                           "ocr_profile": "structured_symbol_vision"})
            structured_records.append({**anchor, "candidate": candidate,
                "candidate_id": plan.object_id, "object_id": plan.object_id,
                "bbox": dict(box), "polygon": [list(point) for point in polygon],
                "text": plan.text, "result": result, "recognized": bool(plan.text),
                "context_text": "", "table_excluded": False,
                "native_match": None, "native_authoritative": False,
                "engineering_symbol": evidence, "assembly_id": plan.object_id,
                "assembly_rule": f"structured_{plan.evidence.kind}",
                "assembly_conflict": False,
                "assembly_review_reason": plan.review_reason,
                "assembly_children": [atomic_object_evidence(m) for m in members]})
        ocr_records = structured_records
        structured_stats = structured_symbol_statistics(structured_plans)

        # The detector deliberately returns atomic text boxes.  Build the
        # engineering objects a balloon actually belongs to before context
        # filtering and target-locked OCR: the final reread then sees the
        # union crop for ``4X`` + ``Ø10`` + ``THRU`` or a nominal with its
        # stacked deviations.  Every absorbed box remains on the object as
        # serializable child evidence; nothing disappears from the audit
        # trail merely because it was assembled.
        assembly_plans = plan_engineering_object_assemblies(ocr_records)
        plans_by_first = {
            min(plan.member_indexes): plan for plan in assembly_plans
        }
        absorbed_indexes = {
            index
            for plan in assembly_plans
            for index in plan.member_indexes
        }
        assembled_records: list[dict[str, Any]] = []
        for record_index, record in enumerate(ocr_records):
            plan = plans_by_first.get(record_index)
            if plan is None:
                if record_index in absorbed_indexes:
                    continue
                atomic = dict(record)
                atomic["object_id"] = str(record.get("object_id") or record.get("candidate_id") or "")
                atomic.setdefault("assembly_id", atomic["object_id"])
                atomic.setdefault("assembly_rule", "atomic")
                atomic.setdefault("assembly_conflict", bool((record.get("result") or {}).get("source_conflict")))
                atomic.setdefault("assembly_review_reason", "")
                atomic.setdefault("assembly_children", [atomic_object_evidence(record)])
                assembled_records.append(atomic)
                continue

            members = [ocr_records[index] for index in plan.member_indexes]
            anchor = dict(ocr_records[plan.anchor_index])
            anchor_candidate = anchor["candidate"]
            merged_candidate = replace(
                anchor_candidate,
                bbox=dict(plan.bbox),
                polygon=tuple(plan.polygon),
                candidate_id=plan.object_id,
                boundary_review=any(
                    bool(member["candidate"].boundary_review)
                    for member in members
                ),
                detection_confidence=max(
                    float(member["candidate"].detection_confidence)
                    for member in members
                ),
            )
            merged_result = dict(anchor.get("result") or {})
            merged_result.update(
                {
                    "text": plan.text,
                    "raw_ocr": " ".join(
                        str(
                            (member.get("result") or {}).get("raw_ocr")
                            or (member.get("result") or {}).get("text")
                            or member.get("text")
                            or ""
                        ).strip()
                        for member in members
                        if str(
                            (member.get("result") or {}).get("raw_ocr")
                            or (member.get("result") or {}).get("text")
                            or member.get("text")
                            or ""
                        ).strip()
                    ),
                    "confidence": plan.confidence,
                    "orientation": plan.orientation,
                    "rotation": plan.rotation,
                    "needs_review": plan.needs_review,
                    "review_reason": plan.review_reason,
                    "source_conflict": plan.conflict,
                    "recognition_source": plan.recognition_source,
                    "recognition_evidence": dict(plan.recognition_evidence),
                    "ocr_profile": "engineering_object_assembly",
                }
            )
            all_native = bool(members) and all(
                bool(member.get("native_authoritative")) for member in members
            )
            assembled_records.append(
                {
                    **anchor,
                    "candidate": merged_candidate,
                    "candidate_id": plan.object_id,
                    "object_id": plan.object_id,
                    "bbox": dict(plan.bbox),
                    "polygon": [list(point) for point in plan.polygon],
                    "text": plan.text,
                    "result": merged_result,
                    "recognized": bool(plan.text),
                    "context_text": "",
                    "table_excluded": False,
                    # A multi-fragment match has its own aggregated source
                    # evidence.  Reusing one child's native match would fuse
                    # the full-object reread against only that child.
                    "native_match": None,
                    "native_authoritative": all_native,
                    "assembly_id": plan.object_id,
                    "assembly_rule": plan.rule,
                    "assembly_conflict": plan.conflict,
                    "assembly_review_reason": plan.review_reason,
                    "assembly_children": [dict(child) for child in plan.children],
                }
            )

        ocr_records = assembled_records
        detected_count = len(ocr_records)
        assembly_stats = assembly_statistics(
            atomic_detection_count,
            detected_count,
            assembly_plans,
        )
        assembly_stats["structured_object_count"] = len(structured_plans)
        assembly_stats["assembled_object_count"] += len(structured_plans)
        assembly_stats["rule_counts"].update({
            f"structured_{kind}": count
            for kind, count in structured_stats["kind_counts"].items()
        })
        report(
            stage="assembling",
            message=(
                f"Built {detected_count} engineering objects from "
                f"{atomic_detection_count} atomic detections; preserved "
                f"{assembly_stats['absorbed_fragment_count']} absorbed fragments"
            ),
            percent=89,
            completed=detected_count,
            total=detected_count,
            object_total=detected_count,
            operation_label="Engineering-object assembly",
            candidate_count=detected_count,
        )

        context_indexes = [
            index
            for index, record in enumerate(ocr_records)
            if record["recognized"]
            and needs_expanded_filter_context(
                record["text"],
                record["bbox"],
            )
        ]
        for index in context_indexes:
            ocr_records[index]["context_text"] = reconstruct_line_context(
                ocr_records[index],
                ocr_records,
            )
        context_ocr_indexes = [
            index
            for index in context_indexes
            if not ocr_records[index].get("native_authoritative")
        ][:CONTEXT_MAX_CANDIDATES]
        context_batch_total = (
            (
                len(context_ocr_indexes)
                + PAGE_SCAN_RECOGNITION_BATCH_SIZE
                - 1
            )
            // PAGE_SCAN_RECOGNITION_BATCH_SIZE
        )
        for context_batch_index, batch_start in enumerate(
            range(
                0,
                len(context_ocr_indexes),
                PAGE_SCAN_RECOGNITION_BATCH_SIZE,
            ),
            start=1,
        ):
            batch_indexes = context_ocr_indexes[
                batch_start : batch_start + PAGE_SCAN_RECOGNITION_BATCH_SIZE
            ]
            report(
                stage="context",
                message=(
                    f"Context batch {context_batch_index} of "
                    f"{context_batch_total} for exclusion-prone values"
                ),
                percent=(
                    90
                    + int(
                        4
                        * (context_batch_index - 1)
                        / max(context_batch_total, 1)
                    )
                ),
                completed=context_batch_index - 1,
                total=context_batch_total,
                batch_current=context_batch_index,
                batch_total=context_batch_total,
                operation_label=(
                    f"Filter context {context_batch_index}/{context_batch_total}"
                ),
                candidate_count=len(context_ocr_indexes),
            )
            results = self._recognize_page_batch(
                [
                    build_context_crop(image, ocr_records[index]["bbox"])
                    for index in batch_indexes
                ],
                batch_size=PAGE_SCAN_RECOGNITION_BATCH_SIZE,
                profile="context_recognition",
            )
            if len(results) != len(batch_indexes):
                raise RuntimeError(
                    "Page context batch returned an unexpected result count"
                )
            for record_index, context_result in zip(batch_indexes, results):
                parts = [
                    ocr_records[record_index]["context_text"],
                    str(context_result.get("raw_ocr") or "").strip(),
                    str(context_result.get("text") or "").strip(),
                ]
                ocr_records[record_index]["context_text"] = " ".join(
                    dict.fromkeys(part for part in parts if part)
                )
            report(
                stage="context",
                message=(
                    f"Completed context batch {context_batch_index} of "
                    f"{context_batch_total}"
                ),
                percent=(
                    90
                    + int(
                        4
                        * context_batch_index
                        / max(context_batch_total, 1)
                    )
                ),
                completed=context_batch_index,
                total=context_batch_total,
                batch_current=context_batch_index,
                batch_total=context_batch_total,
                operation_label=(
                    f"Filter context {context_batch_index}/{context_batch_total}"
                ),
                candidate_count=len(context_ocr_indexes),
            )

        hard_exclusion_rules = {
            "table_region",
            # A sheet-frame zone label is not a value a human can confirm, so
            # it is dropped outright. Left out of this set it fell through to
            # the review queue instead, which filled the queue with the very
            # "1" ... "8" border digits the rule exists to remove.
            "sheet_frame_label",
            "detail_view_section",
            "scale_information",
            "date",
            "revision_history",
            "note_information",
            "document_metadata",
        }

        def build_filter_candidates() -> list[PageValueCandidate]:
            return [
                PageValueCandidate(
                    text=record["text"],
                    bbox=record["bbox"],
                    context_text=record["context_text"],
                )
                for record in ocr_records
            ]

        def resolve_candidate_state(
            record: dict[str, Any],
            decision: Any,
            *,
            authoritative: bool,
        ) -> tuple[str, str]:
            result = record["result"]
            review_reason = str(result.get("review_reason") or "").strip()
            if decision.rule_name in hard_exclusion_rules:
                return "excluded", decision.reason

            if record.get("assembly_conflict"):
                return (
                    "review",
                    str(record.get("assembly_review_reason") or "")
                    or "Assembled fragments contain conflicting recognition evidence",
                )


            if not record["recognized"]:
                # No text at all. Once the AUTHORITATIVE pass has also come
                # back empty, the detector fired on ink that is not writing —
                # hatching, a leader line, an arrowhead — and there is nothing
                # for anyone to confirm. On a real sheet that was 27 of 57
                # queued items, all blank, which is the noise that teaches an
                # inspector to stop reading the queue.
                #
                # Before that pass runs, an empty read must stay a review: this
                # state is what SELECTS an object for the expensive re-read, so
                # excluding here would deny it its second chance. Doing that
                # cost seven published values on one sheet.
                if authoritative and result.get(
                    "authoritative_review_required"
                ):
                    return (
                        "review",
                        review_reason
                        or "Authoritative OCR could not confirm this detector target",
                    )
                if authoritative:
                    return (
                        "excluded",
                        review_reason
                        or "Recognition found no text at this object",
                    )
                if record.get("speck"):
                    return (
                        "excluded",
                        "No text was read at this object and it is too small "
                        "to hold a value",
                    )
                return (
                    "review",
                    review_reason or "Accurate recognition returned no text",
                )
            if result.get("authoritative_review_required"):
                return (
                    "review",
                    review_reason
                    or "Authoritative OCR requires manual confirmation",
                )
            if not decision.accepted:
                if (
                    decision.rule_name == "no_numeric_component"
                    and (
                        result.get("stable_alpha")
                        or not result.get("needs_review", False)
                    )
                ):
                    return "excluded", decision.reason
                return (
                    "review",
                    review_reason
                    or "The detected object could not be read as an engineering value",
                )
            # Full Draw Value OCR is authoritative. Its confidence/retry flag
            # is diagnostic and must not demote a structurally valid value.
            if not authoritative and result.get("needs_review", False):
                return (
                    "review",
                    review_reason
                    or "Recognition remains genuinely ambiguous after recovery",
                )
            return "eligible", decision.reason

        # An object that read as nothing through the primary pass AND both
        # recovery passes, and whose box could not hold even two characters of
        # this sheet's text, is a speck — a tick, an arrowhead, a piece of
        # hatching. On a real sheet 28 of 86 authoritative rereads were such
        # objects and every one came back empty. Boxes large enough to hold a
        # value keep their reread: an empty batch read of a real value is
        # exactly what that pass exists to rescue.
        recognized_heights = sorted(
            float(record["bbox"]["height"])
            for record in ocr_records
            if record.get("recognized")
            and float(record["bbox"]["width"]) >= float(record["bbox"]["height"])
        )
        speck_line_height = (
            recognized_heights[len(recognized_heights) // 2]
            if recognized_heights
            else 0.0
        )
        if speck_line_height > 0:
            for record in ocr_records:
                if record.get("table_excluded") or record.get("recognized"):
                    continue
                width = float(record["bbox"]["width"])
                height = float(record["bbox"]["height"])
                if (
                    max(width, height) < 1.2 * speck_line_height
                    or min(width, height) < 0.45 * speck_line_height
                ):
                    record["speck"] = True

        # Keep the current fast page policy as the cost gate. Only objects that
        # would be published or reviewed receive the expensive Draw Value OCR.
        preliminary_filter_candidates = build_filter_candidates()
        preliminary_decisions: list[Any] = []
        authoritative_indexes: list[int] = []
        for filter_index, (record, filter_candidate) in enumerate(
            zip(ocr_records, preliminary_filter_candidates),
            start=1,
        ):
            report(
                stage="filtering",
                message=(
                    f"Checking page policy for object {filter_index} of "
                    f"{detected_count}"
                ),
                percent=94 + int(filter_index / max(detected_count, 1)),
                completed=filter_index,
                total=detected_count,
                object_current=filter_index,
                object_total=detected_count,
                operation_label="Preliminary page value filters",
                candidate_count=detected_count,
            )
            decision = evaluate_page_value(
                filter_candidate,
                page_candidates=preliminary_filter_candidates,
                table_masks=table_masks,
                page_size=image.size,
            )
            preliminary_decisions.append(decision)
            preliminary_state, preliminary_reason = resolve_candidate_state(
                record,
                decision,
                authoritative=False,
            )
            record["preliminary_state"] = preliminary_state
            record["preliminary_reason"] = preliminary_reason
            if preliminary_state != "excluded" and not record.get(
                "native_authoritative"
            ):
                authoritative_indexes.append(filter_index - 1)

        authoritative_total = len(authoritative_indexes)

        def apply_authoritative(
            record: dict[str, Any], attempts: list[dict[str, Any]]
        ) -> None:
            previous_result = dict(record["result"])
            record["preliminary_result"] = previous_result
            record["preliminary_text"] = str(previous_result.get("text") or "")
            accurate_result = resolve_authoritative_hypotheses(
                previous_result,
                attempts,
            )
            match = record.get("native_match")
            if match is not None:
                accurate_result = fuse_native_with_ocr(
                    match,
                    accurate_result,
                    native_authoritative=False,
                    final=True,
                )
            else:
                accurate_result = ocr_evidence(accurate_result)
            accurate_text = normalize_page_value_text(
                strip_foreign_glyphs(str(accurate_result.get("text") or ""))
            )
            structured = record.get("engineering_symbol") or {}
            previous_text = str(record.get("text") or "").strip()
            if structured.get("kind") == "feature_control_frame":
                symbol = str(structured.get("completed_symbol") or "").strip()
                if accurate_text and symbol and symbol not in accurate_text:
                    accurate_text = f"{symbol} | {accurate_text}"
                elif not accurate_text:
                    accurate_text = previous_text
            elif structured.get("kind") == "datum":
                accurate_text = str(structured.get("subtype") or accurate_text or previous_text)
            elif structured.get("kind") == "surface_finish" and not accurate_text:
                accurate_text = previous_text
            accurate_result["text"] = accurate_text
            if structured:
                accurate_result["engineering_symbol"] = dict(structured)
            accurate_result["authoritative_reread"] = True
            record["result"] = accurate_result
            record["text"] = accurate_text
            record["recognized"] = bool(accurate_text)
            record["authoritative_reread"] = True

        if pool is not None and authoritative_total > 1:
            # Every object's two crops are built up front (cheap, pure image
            # work); the reads — the expensive part — are dealt across the
            # workers and this process. Results are applied in order.
            from ocr_workers import image_to_png, run_task

            report(
                stage="rereading",
                message=(
                    f"Reading {authoritative_total} final values with Draw "
                    "Value OCR across the OCR workers"
                ),
                percent=95,
                completed=0,
                total=authoritative_total,
                object_current=1,
                object_total=authoritative_total,
                operation_label="Authoritative value OCR",
                candidate_count=authoritative_total,
            )
            payloads = []
            for record_index in authoritative_indexes:
                record = ocr_records[record_index]
                crops = build_authoritative_crops(
                    image, record["bbox"], record["polygon"]
                )
                payloads.append(
                    {
                        "crops": [
                            {
                                "profile": crop.profile,
                                "png": image_to_png(crop.image),
                                "target_bbox": dict(crop.target_bbox),
                                "target_polygon": [
                                    list(point) for point in crop.target_polygon
                                ],
                                "polygon_usable": bool(crop.polygon_usable),
                            }
                            for crop in crops
                        ],
                        "debug_dump": debug_dump,
                        "debug_dump_force": debug_dump_force,
                        "max_paddle_predictions": _PAGE_REREAD_MAX_PREDICTIONS,
                        "preliminary_text": str(record.get("text") or ""),
                    }
                )
            reread_done = 0

            def on_reread_done(index: int, attempts: Any) -> None:
                nonlocal reread_done
                reread_done += 1
                apply_authoritative(
                    ocr_records[authoritative_indexes[index]], list(attempts)
                )
                report(
                    stage="rereading",
                    message=(
                        f"Read final value {reread_done} of {authoritative_total}"
                    ),
                    percent=95 + int(3 * reread_done / max(authoritative_total, 1)),
                    completed=reread_done,
                    total=authoritative_total,
                    object_current=reread_done,
                    object_total=authoritative_total,
                    operation_label="Authoritative value OCR",
                    candidate_count=authoritative_total,
                )

            pool.map(
                "authoritative",
                payloads,
                local=lambda payload: run_task(self, "authoritative", payload),
                on_item_done=on_reread_done,
            )
            authoritative_indexes = []
        for reread_index, record_index in enumerate(
            authoritative_indexes,
            start=1,
        ):
            record = ocr_records[record_index]
            report(
                stage="rereading",
                message=(
                    f"Reading final value {reread_index} of "
                    f"{authoritative_total} with Draw Value OCR"
                ),
                percent=(
                    95
                    + int(
                        3
                        * (reread_index - 1)
                        / max(authoritative_total, 1)
                    )
                ),
                completed=reread_index - 1,
                total=authoritative_total,
                object_current=reread_index,
                object_total=authoritative_total,
                operation_label="Authoritative value OCR",
                candidate_count=authoritative_total,
            )
            authoritative_crops = build_authoritative_crops(
                image,
                record["bbox"],
                record["polygon"],
            )
            apply_authoritative(
                record,
                self._authoritative_attempts(
                    authoritative_crops[0],
                    authoritative_crops[1],
                    debug_dump=debug_dump,
                    debug_dump_force=debug_dump_force,
                    max_paddle_predictions=_PAGE_REREAD_MAX_PREDICTIONS,
                    preliminary_text=str(record.get("text") or ""),
                ),
            )
            report(
                stage="rereading",
                message=(
                    f"Read final value {reread_index} of "
                    f"{authoritative_total}"
                ),
                percent=(
                    95
                    + int(
                        3 * reread_index / max(authoritative_total, 1)
                    )
                ),
                completed=reread_index,
                total=authoritative_total,
                object_current=reread_index,
                object_total=authoritative_total,
                operation_label="Authoritative value OCR",
                candidate_count=authoritative_total,
            )

        # Refresh cheap neighbouring-token context after authoritative text has
        # replaced preliminary OCR. Existing expanded context reads are kept.
        for record in ocr_records:
            if not record["recognized"] or not needs_expanded_filter_context(
                record["text"],
                record["bbox"],
            ):
                continue
            reconstructed = reconstruct_line_context(record, ocr_records)
            record["context_text"] = " ".join(
                dict.fromkeys(
                    part
                    for part in (record["context_text"], reconstructed)
                    if part
                )
            )

        filter_candidates = build_filter_candidates()
        filter_rule_counts: Counter[str] = Counter()
        regions: list[dict[str, Any]] = []
        review_candidates: list[dict[str, Any]] = []
        candidate_outcomes: list[dict[str, Any]] = []
        filtered_overlay_candidates: list[dict[str, object]] = []

        for filter_index, (record, filter_candidate) in enumerate(
            zip(ocr_records, filter_candidates),
            start=1,
        ):
            report(
                stage="filtering",
                message=(
                    f"Resolving final state for object {filter_index} of "
                    f"{detected_count}"
                ),
                percent=98 + int(filter_index / max(detected_count, 1)),
                completed=filter_index,
                total=detected_count,
                object_current=filter_index,
                object_total=detected_count,
                operation_label="Final page value filters",
                candidate_count=detected_count,
            )
            if (
                record.get("preliminary_state") == "excluded"
                and not record.get("authoritative_reread", False)
            ):
                decision = preliminary_decisions[filter_index - 1]
            else:
                decision = evaluate_page_value(
                    filter_candidate,
                    page_candidates=filter_candidates,
                    table_masks=table_masks,
                    page_size=image.size,
                )
            if not record["recognized"] and decision.rule_name != "table_region":
                filter_rule_counts["unread"] += 1
            else:
                filter_rule_counts[decision.rule_name] += 1

            candidate = record["candidate"]
            result = record["result"]
            published_text = record["text"]
            engineering_parse = parse_engineering_value(
                published_text,
                feature_category=str(result.get("category") or "") or None,
                engineering_symbol=record.get("engineering_symbol") or {},
            )
            disposition = disposition_engineering_object(
                engineering_parse,
                recognized=bool(record["recognized"]),
                authoritative=bool(
                    record.get("authoritative_reread")
                    or record.get("native_authoritative")
                ),
                context_rule=decision.rule_name,
                context_reason=decision.reason,
                assembly_conflict=bool(record.get("assembly_conflict")),
                assembly_review_reason=str(
                    record.get("assembly_review_reason") or ""
                ),
                source_conflict=bool(result.get("source_conflict")),
                numeric_conflict=bool(result.get("numeric_conflict")),
                recognition_review_required=bool(
                    result.get("authoritative_review_required")
                ),
                recognition_needs_review=bool(result.get("needs_review")),
                recognition_review_reason=str(
                    result.get("review_reason") or ""
                ),
                speck=bool(record.get("speck")),
            )
            final_state = (
                "excluded" if disposition.state == "other" else disposition.state
            )
            final_reason = disposition.reason

            common_region = {
                "candidate_id": record["candidate_id"],
                "object_id": record.get("object_id", record["candidate_id"]),
                "bbox": record["bbox"],
                "text": published_text,
                "confidence": result.get("confidence", 0.0),
                "type": result.get("type"),
                "orientation": result.get("orientation", "horizontal"),
                "rotation": result.get("rotation", 0),
                "needs_review": final_state == "review",
                "recognized": record["recognized"],
                "boundary_review": candidate.boundary_review,
                "agreement": result.get("agreement", 0.0),
                "engine": result.get("engine", "paddleocr"),
                "symbols_detected": result.get("symbols_detected"),
                "engineering_symbol": dict(record.get("engineering_symbol") or {}) or None,
                "ocr_profile": result.get("ocr_profile", "batch_recognition"),
                "page_filter_rule": decision.rule_name,
                "page_filter_reason": decision.reason,
                "review_reason": final_reason if final_state == "review" else "",
                "authoritative_review_required": bool(
                    result.get("authoritative_review_required")
                ),
                "recovery_attempted": bool(result.get("recovery_attempted")),
                "authoritative_reread": bool(
                    result.get("authoritative_reread")
                ),
                "authoritative_target_owned": bool(
                    result.get("authoritative_target_owned")
                ),
                "numeric_conflict": bool(result.get("numeric_conflict")),
                "recognition_source": result.get(
                    "recognition_source", "ocr"
                ),
                "recognition_evidence": dict(
                    result.get("recognition_evidence") or {}
                ),
                "source_conflict": bool(result.get("source_conflict")),
                "assembly_id": record.get("assembly_id", record["candidate_id"]),
                "assembly_rule": record.get("assembly_rule", "atomic"),
                "assembly_conflict": bool(record.get("assembly_conflict")),
                "assembly_review_reason": str(
                    record.get("assembly_review_reason") or ""
                ),
                "assembly_children": [
                    dict(child) for child in record.get("assembly_children", ())
                ],
                "engineering_parse": engineering_parse.to_dict(),
                "engineering_disposition": disposition.to_dict(),
            }
            oriented_box = result.get("oriented_box") or record.get(
                "oriented_box"
            )
            if isinstance(oriented_box, dict):
                common_region["oriented_box"] = dict(oriented_box)
            self._apply_feature_labels(common_region)
            if final_state == "eligible":
                regions.append(common_region)
            elif final_state == "review":
                review_candidates.append(common_region)

            outcome = {
                "candidate_id": record["candidate_id"],
                "object_id": record.get("object_id", record["candidate_id"]),
                "bbox": dict(record["bbox"]),
                "polygon": [list(point) for point in candidate.polygon],
                "state": final_state,
                "text": published_text,
                "raw_text": str(
                    result.get("raw_ocr")
                    or result.get("text")
                    or record.get("text")
                    or ""
                ),
                "preliminary_text": str(
                    (record.get("preliminary_result") or {}).get("raw_ocr")
                    or (record.get("preliminary_result") or {}).get("text")
                    or record.get("preliminary_text")
                    or ""
                ),
                "confidence": float(result.get("confidence") or 0.0),
                "recognized": record["recognized"],
                "reason": final_reason or decision.reason,
                "rule": disposition.rule,
                "page_filter_rule": decision.rule_name,
                "page_filter_reason": decision.reason,
                "type": common_region.get("type"),
                "category": common_region.get("category"),
                "subtype": common_region.get("subtype"),
                "label": common_region.get("label"),
                "engineering_symbol": common_region.get("engineering_symbol"),
                "orientation": common_region.get("orientation", "horizontal"),
                "rotation": float(common_region.get("rotation") or 0.0),
                "oriented_box": common_region.get("oriented_box"),
                "recovery_attempted": bool(result.get("recovery_attempted")),
                "authoritative_reread": bool(
                    result.get("authoritative_reread")
                ),
                "authoritative_review_required": bool(
                    result.get("authoritative_review_required")
                ),
                "recognition_source": result.get(
                    "recognition_source", "ocr"
                ),
                "recognition_evidence": dict(
                    result.get("recognition_evidence") or {}
                ),
                "source_conflict": bool(result.get("source_conflict")),
                "assembly_id": record.get("assembly_id", record["candidate_id"]),
                "assembly_rule": record.get("assembly_rule", "atomic"),
                "assembly_conflict": bool(record.get("assembly_conflict")),
                "assembly_review_reason": str(
                    record.get("assembly_review_reason") or ""
                ),
                "assembly_children": [
                    dict(child) for child in record.get("assembly_children", ())
                ],
                "engineering_parse": engineering_parse.to_dict(),
                "engineering_disposition": disposition.to_dict(),
            }
            candidate_outcomes.append(outcome)
            filtered_overlay_candidates.append(
                {
                    "id": record["candidate_id"],
                    "bbox": dict(record["bbox"]),
                    "state": final_state,
                    "text": published_text,
                    "reason": final_reason,
                    "rule": disposition.rule,
                }
            )

        # Two callouts stacked one above the other read back as a single
        # garbled string here — "Ø0.620 / 0.612" came out as "0.020D0.612" and
        # "8.00[203.20] / 9.00[228.60]" as "8.007oo" — because this route
        # recognises each detected object once and never looks inside a fused
        # one. The section route has always re-segmented such a read; the same
        # pass runs here, replacing a region only when the finer look actually
        # resolves two or more worthy values.
        report(
            stage="filtering",
            message="Re-segmenting fused or stacked candidates",
            percent=99,
            completed=detected_count,
            total=detected_count,
            candidate_count=len(regions),
            operation_label="Stacked-callout split",
        )
        split_regions: list[dict[str, Any]] = []
        split_jobs: list[tuple[dict[str, Any], tuple[int, int, int, int]]] = []
        # Limit dimensions resolved as one value, and the objects that own them.
        limit_boxes: list[dict[str, float]] = []
        limit_owners: set[str] = set()

        def retire_inside(
            boxes: list[dict[str, float]], reason: str, *, keep: set[str]
        ) -> None:
            """
            Retire published regions and review candidates lying inside ``boxes``.

            Their outcome moves to excluded, so the state invariant below
            still balances. Regions this route did not detect itself (the
            notes block, slanted reads) carry no candidate id and are never
            retired.
            """
            nonlocal regions, review_candidates
            from geom_utils import contains_point, region_center

            def inside(item: dict[str, Any]) -> bool:
                try:
                    centre = region_center(item)
                except Exception:
                    return False
                return any(contains_point({"bbox": lb}, centre) for lb in boxes)

            retired: set[str] = set()
            kept_regions: list[dict[str, Any]] = []
            for item in regions:
                cid = str(item.get("candidate_id") or "")
                if cid and cid not in keep and inside(item):
                    retired.add(cid)
                    continue
                kept_regions.append(item)
            regions = kept_regions
            kept_reviews: list[dict[str, Any]] = []
            for item in review_candidates:
                cid = str(item.get("candidate_id") or "")
                if cid and cid not in keep and inside(item):
                    retired.add(cid)
                    continue
                kept_reviews.append(item)
            review_candidates = kept_reviews
            for outcome in candidate_outcomes:
                if outcome["candidate_id"] in retired:
                    outcome["state"] = "excluded"
                    outcome["reason"] = reason
            for overlay_item in filtered_overlay_candidates:
                if overlay_item.get("id") in retired:
                    overlay_item["state"] = "excluded"
                    overlay_item["reason"] = reason

        # The sheet's own line height, from the upright objects it detected:
        # a box has to be at least two of those tall to hold a stacked pair.
        upright_heights = sorted(
            float(record["bbox"]["height"])
            for record in ocr_records
            if record.get("recognized")
            and float(record["bbox"]["width"]) >= float(record["bbox"]["height"])
        )
        page_line_height = (
            upright_heights[len(upright_heights) // 2] if upright_heights else 0.0
        )
        for region in regions:
            text = str(region.get("text") or "")
            box = region["bbox"]
            # A tall box is two stacked lines only when the text runs across
            # it; a vertical dimension is tall because it is one line turned
            # through 90°. The section route has always made this distinction
            # (its multirow split requires width >= height); without it every
            # vertical value on a sheet paid for a detection cascade here.
            vertical_text = str(region.get("orientation") or "") == "vertical"
            # A limit dimension — "Ø0.620" over "0.612" — is ONE callout, and
            # the one stacked thing the split below must not cut in two. Any
            # published piece of it (the whole read flat, a column, a row) is
            # at least two lines tall, so a tall object is first read as a
            # limit stack over the whole callout; only when that fails does
            # the two-callout split get its turn. A second piece of a stack
            # already resolved is left for retirement below.
            if page_line_height > 0 and box["height"] >= 1.7 * page_line_height and not vertical_text:
                from geom_utils import contains_point, region_center

                centre = region_center(region)
                if any(contains_point({"bbox": lb}, centre) for lb in limit_boxes):
                    split_regions.append(region)
                    continue
                group = self._limit_group_bbox(box, ocr_records, page_line_height)
                limit = None
                try:
                    limit = self._read_limit_pair(
                        image, group, max_paddle_predictions=_PAGE_REREAD_MAX_PREDICTIONS
                    )
                except Exception:
                    limit = None
                if limit is not None:
                    region["text"] = limit["text"]
                    region["bbox"] = dict(limit["bbox"])
                    region["type"] = limit["type"]
                    region["confidence"] = limit["confidence"]
                    region["needs_review"] = False
                    region["orientation"] = "horizontal"
                    region["rotation"] = 0
                    region.pop("oriented_box", None)
                    self._apply_feature_labels(region)
                    limit_boxes.append(dict(limit["bbox"]))
                    limit_owners.add(str(region.get("candidate_id") or ""))
                    split_regions.append(region)
                    continue
            tall = (
                box["height"] >= 1.6 * _PAGE_SPLIT_LINE_RATIO * max(box["width"], 1.0)
                and box["height"] >= 1.7 * page_line_height
                and not vertical_text
            )
            # Stacked pairs are drawn one above the other in reading direction;
            # a vertical dimension is one line, however many numbers it holds.
            if vertical_text or (count_dimension_values(text) < 2 and not tall):
                split_regions.append(region)
                continue
            margin = 6
            coords = (
                max(0, int(box["x"] - margin)),
                max(0, int(box["y"] - margin)),
                min(image.width, int(box["x"] + box["width"] + margin)),
                min(image.height, int(box["y"] + box["height"] + margin)),
            )
            split_jobs.append((region, coords))
            split_regions.append(region)  # placeholder, replaced below if the split is usable

        def run_split(coords: tuple[int, int, int, int]) -> list[dict[str, Any]] | None:
            try:
                return self._split_stacked_cluster(
                    image, coords, max_paddle_predictions=_PAGE_REREAD_MAX_PREDICTIONS
                )
            except Exception:
                return None

        split_pool = self._worker_pool() if len(split_jobs) > 1 else None
        if split_pool is not None:
            # Each fused candidate is re-segmented independently, so the crops
            # are dealt across the OCR workers; pieces come back in crop
            # pixels and are moved to page pixels here.
            from ocr_workers import image_to_png, run_task

            payloads = [
                {"png": image_to_png(image.crop(coords)), "max_paddle_predictions": _PAGE_REREAD_MAX_PREDICTIONS}
                for _region, coords in split_jobs
            ]
            raw_pieces = split_pool.map(
                "split", payloads, local=lambda payload: run_task(self, "split", payload)
            )
            split_results: list[list[dict[str, Any]] | None] = []
            for (_region, (cx0, cy0, _cx1, _cy1)), pieces in zip(split_jobs, raw_pieces):
                if pieces:
                    for piece in pieces:
                        piece["bbox"]["x"] = round(float(piece["bbox"]["x"]) + cx0, 1)
                        piece["bbox"]["y"] = round(float(piece["bbox"]["y"]) + cy0, 1)
                split_results.append(list(pieces) if pieces else None)
        else:
            split_results = [run_split(coords) for _region, coords in split_jobs]

        for (region, _coords), pieces in zip(split_jobs, split_results):
            # Splitting a garbled read just yields garbled pieces, so the
            # finer look is only taken when every piece it produced is itself
            # a complete engineering value. That keeps the recovered
            # "8.00[203.20]" / "9.00[228.60]" pair and discards the fragments
            # a failed split leaves behind.
            usable = bool(pieces) and len(pieces) >= 2 and all(
                is_complete_engineering_value(
                    normalize_page_value_text(str(piece.get("text") or ""))
                )
                for piece in pieces
            )
            if not usable:
                continue
            position = next(
                (k for k, item in enumerate(split_regions) if item is region), None
            )
            if position is None:
                continue
            split_regions[position : position + 1] = list(pieces)
            detected_count += len(pieces) - 1
        regions = split_regions
        if limit_boxes:
            # The other cuts of a resolved stack — "620612", "Ø1.00", the top
            # row alone — would each have carried a balloon of their own.
            retire_inside(
                limit_boxes, "Part of a limit dimension read as one value", keep=limit_owners
            )
            # A dual-unit sheet draws the metric limits in brackets beside the
            # inch ones; the two stacks are one callout.
            regions, joined_metric = join_dual_unit_limits(regions)
            joined_ids = {str(item.get("candidate_id") or "") for item in joined_metric}
            joined_ids.discard("")
            for outcome in candidate_outcomes:
                if outcome["candidate_id"] in joined_ids:
                    outcome["state"] = "excluded"
                    outcome["reason"] = "Metric half of a dual-unit limit dimension"
            for overlay_item in filtered_overlay_candidates:
                if overlay_item.get("id") in joined_ids:
                    overlay_item["state"] = "excluded"
                    overlay_item["reason"] = "Metric half of a dual-unit limit dimension"

        # Slanted callouts (chamfers, angled fits) that this route cannot read:
        # its detection only ever levels by a quarter turn, so text drawn along
        # a leader line — "Ø20H10 +0.084" on a real sheet — is never recovered.
        # The pass needs raw detection boxes to find the slant neighbourhoods,
        # and runs only when a slant is actually present, so an orthogonal sheet
        # pays one Hough call. Appended after the page policy, so an angled read
        # only ADDS a value the upright passes missed; detected_count moves with
        # it so the state invariant below still balances.
        # The neighbourhoods come from the quads this route already detected;
        # a second whole-page detection purely to find them cost 17 s.
        angled_seed_boxes = [
            {
                "x": candidate.bbox["x"],
                "y": candidate.bbox["y"],
                "w": candidate.bbox["width"],
                "h": candidate.bbox["height"],
                "quad": [list(point) for point in candidate.polygon],
                "text": "",
            }
            for candidate in final_candidates
        ]
        report(
            stage="angled",
            message="Reading slanted callouts along their leader lines",
            percent=99,
            completed=detected_count,
            total=detected_count,
            candidate_count=len(regions),
            operation_label="Slanted callouts",
        )
        try:
            angled_regions = self._detect_angled_regions(
                image, regions, det_boxes=angled_seed_boxes
            )
        except Exception:
            # The upright balloons are the point of the scan; a failed angled
            # recovery must never discard them.
            angled_regions = []
        if angled_regions:
            regions.extend(angled_regions)
            detected_count += len(angled_regions)
        # A base region the levelled read stood in for — the value's deviation
        # stack read on its own, or a garbled fragment of it — is dropped here.
        # The section route has always done this; this one did not, so both the
        # whole callout and the fragment it replaced were balloonned, leaving
        # "Ø20H10+0.084/0" next to a second balloon reading just "+0.0840".
        # detected_count drops with them so the state invariant still balances.
        before_supersede = len(regions)
        regions = [r for r in regions if not r.pop("superseded", False)]
        detected_count -= before_supersede - len(regions)

        # A review candidate lying under an accepted slanted read is the
        # upright pass's failed attempt at the same ink — "6.5×0" under
        # "0.5×45°", "2CH10" under "Ø20H10+0.084/0". Shown together they put a
        # grey box on top of the balloon that resolved it, so the candidate is
        # retired as excluded (its outcome moves from review to excluded, so
        # the state invariant below still balances).
        if angled_regions:
            from geom_utils import contains_point, overlap_frac, region_center

            retired_ids: set[str] = set()
            kept_reviews: list[dict[str, Any]] = []
            for candidate in review_candidates:
                covered = any(
                    overlap_frac(candidate, angled) >= 0.3
                    or contains_point(angled, region_center(candidate))
                    for angled in angled_regions
                )
                if covered:
                    retired_ids.add(str(candidate.get("candidate_id")))
                else:
                    kept_reviews.append(candidate)
            review_candidates = kept_reviews
            for outcome in candidate_outcomes:
                if outcome["candidate_id"] in retired_ids:
                    outcome["state"] = "excluded"
                    outcome["reason"] = "Resolved by a slanted read of the same callout"
            for overlay_item in filtered_overlay_candidates:
                if overlay_item.get("id") in retired_ids:
                    overlay_item["state"] = "excluded"
                    overlay_item["reason"] = "Resolved by a slanted read of the same callout"

        # The NOTES paragraph, read as one region rather than left to leak into
        # neighbouring clusters as fragments. Same accounting as the angled
        # pass, so the state invariant below still balances.
        # Built from the lines this route has ALREADY read: a second whole-page
        # OCR pass purely to find the NOTES markers cost 75 s on one sheet and
        # found nothing the panel passes had not. The raw recogniser text is
        # used because the value normaliser rewrites prose.
        notes_region = self._notes_region(
            image,
            boxes=self._record_text_boxes(ocr_records),
            debug_dump=debug_dump,
            debug_dump_force=debug_dump_force,
        )
        if notes_region is not None:
            # The fragments the block was assembled from — the marker column,
            # a phrase out of one line — must not keep balloons of their own.
            retire_inside([dict(notes_region["bbox"])], "Part of the general notes", keep=set())
            note_text = str(notes_region.get("text") or "")
            note_parse = parse_engineering_value(note_text)
            note_disposition = disposition_engineering_object(
                note_parse,
                recognized=bool(note_text.strip()),
                authoritative=True,
                context_rule="note_information",
                context_reason="General drawing notes",
            )
            note_id = "P-NOTES"
            candidate_outcomes.append(
                {
                    "candidate_id": note_id,
                    "object_id": note_id,
                    "bbox": dict(notes_region["bbox"]),
                    "polygon": [],
                    "state": "excluded",
                    "text": note_text,
                    "raw_text": note_text,
                    "preliminary_text": note_text,
                    "confidence": float(notes_region.get("confidence") or 0.0),
                    "recognized": bool(note_text.strip()),
                    "reason": note_disposition.reason,
                    "rule": note_disposition.rule,
                    "page_filter_rule": "note_information",
                    "page_filter_reason": "General drawing notes",
                    "type": notes_region.get("type"),
                    "category": notes_region.get("category"),
                    "subtype": notes_region.get("subtype"),
                    "label": notes_region.get("label"),
                    "orientation": notes_region.get("orientation", "horizontal"),
                    "rotation": float(notes_region.get("rotation") or 0.0),
                    "oriented_box": notes_region.get("oriented_box"),
                    "recovery_attempted": False,
                    "authoritative_reread": True,
                    "authoritative_review_required": False,
                    "recognition_source": "ocr",
                    "recognition_evidence": {},
                    "source_conflict": False,
                    "assembly_id": note_id,
                    "assembly_rule": "notes_block",
                    "assembly_conflict": False,
                    "assembly_review_reason": "",
                    "assembly_children": [],
                    "engineering_parse": note_parse.to_dict(),
                    "engineering_disposition": note_disposition.to_dict(),
                }
            )
            filtered_overlay_candidates.append(
                {
                    "id": note_id,
                    "bbox": dict(notes_region["bbox"]),
                    "state": "excluded",
                    "text": note_text,
                    "reason": note_disposition.reason,
                    "rule": note_disposition.rule,
                }
            )
            detected_count += 1

        # Two detection passes can resolve the same callout and both survive to
        # publication, which puts a second balloon exactly on top of the first:
        # on a real sheet "Ø30-0.2" came out twice with an identical box. The
        # section route has always deduped its regions; this route did not.
        # detected_count drops with them so the state invariant still balances.
        before_dedupe = len(regions)
        regions = dedupe_regions(regions)
        detected_count -= before_dedupe - len(regions)

        # Refresh parse evidence after the final limit-stack, dual-unit,
        # slanted, notes, and deduplication passes. Earlier text can be
        # replaced by those passes. M6 disposition remains attached to the
        # same logical outcome, with parse status/kind refreshed when needed.
        for collection in (regions, review_candidates, candidate_outcomes):
            for item in collection:
                parsed = parse_engineering_value(
                    str(item.get("text") or ""),
                    feature_category=str(item.get("category") or "") or None,
                    engineering_symbol=item.get("engineering_symbol") or {},
                )
                item["engineering_parse"] = parsed.to_dict()
                existing_disposition = dict(
                    item.get("engineering_disposition") or {}
                )
                if existing_disposition:
                    existing_disposition["parse_status"] = parsed.status
                    existing_disposition["parse_kind"] = parsed.kind
                    item["engineering_disposition"] = existing_disposition
                    continue
                disposition = disposition_engineering_object(
                    parsed,
                    recognized=bool(item.get("recognized", True)),
                    authoritative=True,
                    context_rule=str(item.get("page_filter_rule") or ""),
                    context_reason=str(item.get("page_filter_reason") or ""),
                    assembly_conflict=bool(item.get("assembly_conflict")),
                    assembly_review_reason=str(
                        item.get("assembly_review_reason") or ""
                    ),
                    source_conflict=bool(item.get("source_conflict")),
                    numeric_conflict=bool(item.get("numeric_conflict")),
                    recognition_review_required=bool(
                        item.get("authoritative_review_required")
                    ),
                    recognition_needs_review=bool(item.get("needs_review")),
                    recognition_review_reason=str(
                        item.get("review_reason") or ""
                    ),
                )
                item["engineering_disposition"] = disposition.to_dict()

        # Post-processing can retire an otherwise eligible object when a
        # higher-quality limit, dual-unit, slanted, or notes object owns the
        # same ink. Keep that evidence as an explicit Other disposition.
        for item in candidate_outcomes:
            disposition = dict(item.get("engineering_disposition") or {})
            mapped_state = (
                "excluded" if disposition.get("state") == "other"
                else disposition.get("state")
            )
            if item.get("state") == mapped_state:
                continue
            parsed = item["engineering_parse"]
            if item.get("state") == "excluded":
                item["engineering_disposition"] = {
                    "schema_version": 1,
                    "state": "other",
                    "rule": "post_processing_retirement",
                    "reason": str(item.get("reason") or "Superseded object"),
                    "parse_status": str(parsed.get("status") or "unparsed"),
                    "parse_kind": str(parsed.get("kind") or "unknown"),
                    "hard_context": False,
                }

        parsed_objects = [
            item["engineering_parse"] for item in candidate_outcomes
        ]
        disposition_objects = [
            item["engineering_disposition"] for item in candidate_outcomes
        ]
        candidate_ids = {
            str(item.get("candidate_id") or "") for item in candidate_outcomes
        }
        parsed_objects.extend(
            item["engineering_parse"]
            for item in regions
            if not item.get("candidate_id")
            or str(item.get("candidate_id")) not in candidate_ids
        )
        parsed_objects.extend(
            item["engineering_parse"]
            for item in review_candidates
            if not item.get("candidate_id")
            or str(item.get("candidate_id")) not in candidate_ids
        )
        disposition_objects.extend(
            item["engineering_disposition"]
            for item in (*regions, *review_candidates)
            if not item.get("candidate_id")
            or str(item.get("candidate_id")) not in candidate_ids
        )

        recognized_count = sum(
            1 for record in ocr_records if record["recognized"]
        ) + (1 if notes_region is not None else 0)
        recognition_source_counts: Counter[str] = Counter(
            str(
                record["result"].get("recognition_source")
                or ("native_pdf" if record.get("native_authoritative") else "ocr")
            )
            for record in ocr_records
            if record.get("recognized") and not record.get("table_excluded")
        )
        unread_count = detected_count - recognized_count
        eligible_count = len(regions)
        excluded_count = sum(
            1
            for outcome in candidate_outcomes
            if outcome["state"] == "excluded"
        )
        review_count = len(review_candidates)
        if detected_count != eligible_count + excluded_count + review_count:
            raise AssertionError("Every detected page object must have one final state")

        report(
            stage="finalizing",
            message=(
                f"Prepared {eligible_count} balloons from {detected_count} "
                f"detected: {excluded_count} excluded, {review_count} review"
            ),
            percent=99,
            completed=detected_count,
            total=detected_count,
            candidate_count=eligible_count,
            operation_label="Page scan complete",
            overlay=layout.overlay(
                scope_kind="page",
                panel_states=panel_states,
                candidates=filtered_overlay_candidates,
            ),
        )
        return {
            "count": eligible_count,
            "detected_count": detected_count,
            "recognized_count": recognized_count,
            "eligible_count": eligible_count,
            "excluded_count": excluded_count,
            "review_count": review_count,
            "unread_count": unread_count,
            "duplicates_removed": duplicates_removed,
            "skipped_existing_count": skipped_existing_count,
            "filter_rule_counts": dict(sorted(filter_rule_counts.items())),
            "source_profile": source_profile,
            "recognition_source_counts": dict(
                sorted(recognition_source_counts.items())
            ),
            "assembly_stats": assembly_stats,
            "structured_symbol_stats": structured_stats,
            "engineering_parse_stats": engineering_parse_statistics(
                parsed_objects
            ),
            "engineering_disposition_stats": engineering_disposition_statistics(
                disposition_objects
            ),
            "regions": regions,
            "review_candidates": review_candidates,
            "candidate_outcomes": candidate_outcomes,
        }

    def _complete_angle_regions(
        self, image: Image.Image, regions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Re-read angle regions whose box clipped the trailing minute/second marks.

        Detection frequently undershoots the *end* of a left-to-right angle, so
        ``32°20'40"`` reads as just ``32°``. Expanding the crop in the reading
        direction — clipped so it cannot reach a neighbouring dimension — and
        re-OCRing recovers the full value. Strictly safe: only angle-typed
        regions are touched, and the wider read is accepted only when it parses
        as a richer DMS angle, so a correct read (or any non-angle) is untouched.
        """
        from angle_utils import parse_dms

        iw, ih = image.size
        for r in regions:
            if r.get("type") != "angle":
                continue
            old = (r.get("text") or "").strip()
            if parse_dms(old) and "'" in old and '"' in old:
                continue  # already a complete DMS angle
            b = r["bbox"]
            bx, by, bw, bh = b["x"], b["y"], b["width"], b["height"]

            # Don't let rightward growth reach a vertically-overlapping neighbour.
            right_limit = float(iw)
            for o in regions:
                if o is r:
                    continue
                ob = o["bbox"]
                oy0, oy1 = ob["y"], ob["y"] + ob["height"]
                if ob["x"] >= bx + bw and not (oy1 < by or oy0 > by + bh):
                    right_limit = min(right_limit, ob["x"] - 8)

            x0 = max(0, int(bx))
            x1 = int(min(bx + bw + max(bw, 50.0), right_limit, iw))
            y0 = max(0, int(by - 6))
            y1 = min(ih, int(by + bh + 8))
            if x1 - x0 < bw + 8:
                continue  # no room to grow

            res = self.recognize(image.crop((x0, y0, x1, y1)))
            new = (res.get("text") or "").strip()
            new_dms = parse_dms(new)
            if not new_dms:
                continue
            old_digits = sum(c.isdigit() for c in old)
            if sum(c.isdigit() for c in new_dms) <= old_digits:
                continue  # not richer than what we already have

            r["text"] = new_dms
            r["confidence"] = res.get("confidence", r.get("confidence", 0.0))
            tb = res.get("text_bbox")
            if tb:
                r["bbox"] = {
                    "x": round(x0 + tb["x"], 1),
                    "y": round(y0 + tb["y"], 1),
                    "width": round(tb["width"], 1),
                    "height": round(tb["height"], 1),
                }
        return regions

    def recognize(
        self,
        image: Image.Image,
        image_bytes: bytes | None = None,
        *,
        debug_dump: bool = False,
        debug_dump_force: bool = False,
        compute_text_bbox: bool = True,
        max_paddle_predictions: int | None = None,
        allow_prefix_ocr: bool = True,
    ) -> dict[str, Any]:
        pipeline_started = perf_counter()
        paddle_timings: list[dict[str, Any]] = []
        # DEBUG_DUMP=1 on server, or ?debug_dump=1 per request.
        enabled = should_dump(request_override=debug_dump)
        force = dump_force() or debug_dump_force
        key_bytes = image_bytes if image_bytes is not None else image.tobytes()
        dumper = StepDumper.from_bytes(key_bytes, enabled=enabled, force=force)

        padded = pad_image(image)
        prep = upscale_min_edge(padded)
        oriented = primary_oriented(prep)
        vertical = is_vertical_dimension(image)

        dumper.stage(
            "00_meta",
            {
                "debug_dump": enabled,
                "debug_dump_force": force,
                "debug_dump_per_request": debug_dump,
                "image_size": list(image.size),
                "vertical": vertical,
                "dump_dir": str(dumper.dir) if enabled else None,
                "already_done": dumper.already_done,
                "skip_reason": dumper.skip_reason,
            },
        )
        dumper.image("input", image)
        dumper.image("padded", padded)
        dumper.image("upscaled", prep)
        dumper.image("oriented", oriented)
        # Recomputing every OCR variant purely to dump it is pure overhead on the
        # hot path (recognize_paddle already built them); only do it when dumping.
        if dumper.active:
            for vname, variant in prepare_ocr_variants(image):
                dumper.image(f"variant_{vname}", variant)

        raw_text, confidence, words, agreement, corrected = self.recognize_paddle(
            oriented,
            dumper=dumper,
            timings=paddle_timings,
            max_predictions=max_paddle_predictions,
        )

        symbols, symbol_debug = detect_symbols(prep)
        symbols_before_merge = symbols_to_dict(symbols)

        # Reuse any explicit symbol information already present in the main
        # OCR result before deciding whether a dedicated prefix pass is useful.
        main_text_hints = detect_prefix_from_ocr_text(raw_text)
        symbols = merge_symbol_scores(symbols, main_text_hints)

        compact_raw = "".join(raw_text.split())
        ambiguous_leading = bool(
            len(compact_raw) > 1
            and compact_raw[0] in {"O", "0", "Q", "©", "¢", "C"}
        )
        borderline_phi = bool(
            PREFIX_RECHECK_PHI_SCORE
            <= float(symbols_before_merge["diameter_score"])
            < 0.32
        )
        reusable_symbol_hints = any(
            (
                symbols.diameter,
                symbols.plus_minus,
                symbols.degree,
                symbols.radius,
            )
        )

        # High-confidence plain values do not need a second Paddle prediction.
        # Keep the prefix pass for uncertain or symbol-ambiguous cases.
        prefix_ocr_used = bool(
            allow_prefix_ocr
            and raw_text.strip()
            and not reusable_symbol_hints
            and (
                confidence <= EARLY_ACCEPT_CONFIDENCE
                or ambiguous_leading
                or borderline_phi
            )
        )

        if not allow_prefix_ocr:
            prefix_ocr_reason = "disabled_by_scan_profile"
        elif not raw_text.strip():
            prefix_ocr_reason = "empty_main_result"
        elif reusable_symbol_hints:
            prefix_ocr_reason = "reused_main_or_visual_symbol"
        elif confidence <= EARLY_ACCEPT_CONFIDENCE:
            prefix_ocr_reason = "main_confidence_not_above_95"
        elif ambiguous_leading:
            prefix_ocr_reason = "ambiguous_leading_character"
        elif borderline_phi:
            prefix_ocr_reason = "borderline_phi_score"
        else:
            prefix_ocr_reason = "high_confidence_main_result"

        prefix_text = ""
        prefix_conf = 0.0
        prefix_words: list[dict[str, Any]] = []

        # Prefix images are constructed only for an OCR pass or an active dump.
        if prefix_ocr_used or dumper.active:
            zones = split_symbol_zones(
                oriented,
                from_vertical_rotated=vertical,
            )
            prefix_img = enlarge_zone(zones["prefix"], 3.5)
            prefix_clahe = clahe_rgb(prefix_img)

            if dumper.active:
                zone_layout = (
                    "right ~28% of oriented image "
                    "(-90° rot; Ø was at bottom of vertical)"
                    if vertical
                    else "left ~28% of crop (horizontal reading start)"
                )
                body_layout = (
                    "left ~72% of oriented image "
                    "(digits above Ø in original vertical)"
                    if vertical
                    else "right ~72% of crop (main numbers)"
                )
                dumper.image(
                    "zone_prefix",
                    zones["prefix"],
                    summary={
                        "layout": zone_layout,
                        "purpose": "Ø / R / leading ± detection",
                    },
                )
                dumper.image(
                    "zone_body",
                    zones["body"],
                    summary={
                        "layout": body_layout,
                        "purpose": "nominal value + tolerance digits",
                    },
                )
                dumper.image("prefix_enlarged", prefix_img)
                dumper.image("prefix_clahe", prefix_clahe)

            if prefix_ocr_used:
                prefix_text, prefix_conf, prefix_words = self._run_paddle(
                    prefix_clahe,
                    det=False,
                    timing_label="prefix",
                    timings=paddle_timings,
                )

        dedicated_prefix_hints = detect_prefix_from_ocr_text(prefix_text)
        symbols = merge_symbol_scores(symbols, dedicated_prefix_hints)

        dumper.stage(
            "symbol_vision",
            {
                **symbol_debug,
                "prefix_ocr": prefix_text,
                "prefix_confidence": prefix_conf,
                "prefix_ocr_used": prefix_ocr_used,
                "prefix_ocr_reason": prefix_ocr_reason,
                "main_text_hints": symbols_to_dict(main_text_hints),
                "prefix_hints": symbols_to_dict(dedicated_prefix_hints),
                "symbols_before_merge": symbols_before_merge,
                "symbols_merged": symbols_to_dict(symbols),
            },
        )

        compose_input = {
            "raw_ocr": raw_text,
            "prefix_ocr": prefix_text,
            "symbols": symbols_to_dict(symbols),
            "vertical": vertical,
        }
        dumper.stage("compose_input", compose_input)

        composed = compose_engineering_dimension(
            raw_text, prep, symbols, prefix_ocr=prefix_text
        )
        text = composed.text
        engine = "paddleocr+compose" if composed.applied else "paddleocr"

        # Dual-unit (inch [mm]) cross-check: the bracket is a redundant ×25.4
        # encoding of the primary, so it validates — and, when a digit was
        # misread, repairs — the read against the ratio a human would use. A
        # no-op unless the text actually carries a numeric [bracket] pair.
        from dual_unit import repair_dual

        dual = repair_dual(text)
        if dual.status == "repaired":
            text = dual.text
            engine = f"{engine}+dual" if "+" in engine else "paddleocr+dual"
        dumper.stage(
            "dual_unit",
            {
                "status": dual.status,
                "text": dual.text,
                "direction": dual.direction,
                "expected_mm": dual.expected_mm,
                "edits": dual.edits,
            },
        )

        dumper.stage(
            "compose_output",
            {
                "text": composed.text,
                "kind": composed.kind,
                "applied": composed.applied,
            },
        )

        needs_review = bool(
            text.strip()
            and (confidence < 0.9 or agreement < 0.6 or corrected)
            and dual.status != "consistent"
        ) or dual.needs_review or dual.corrected

        text_bbox: dict[str, float] | None = None
        text_bbox_source: str | None = None

        if compute_text_bbox and text.strip():
            text_bbox = self._text_bbox_from_main_words(
                words,
                image=image,
                prep=prep,
                oriented=oriented,
                vertical=vertical,
            )

            if text_bbox is not None:
                text_bbox_source = "main_words"
            else:
                # Preserve the old detection path for unusual results without
                # usable word coordinates. Only these cases pay for another
                # full Paddle prediction.
                text_bbox = self._detect_text_bbox(
                    image,
                    timings=paddle_timings,
                )
                if text_bbox is not None:
                    text_bbox_source = "fallback_paddle"

        if text_bbox:
            dumper.stage(
                "text_bbox",
                {
                    "source": text_bbox_source,
                    **text_bbox,
                },
            )

        engineering_symbol: dict[str, Any] | None = None
        if text.strip():
            from structured_symbol_vision import plan_structured_engineering_symbols

            symbol_bbox = text_bbox or self._text_bbox_from_main_words(
                words, image=image, prep=prep, oriented=oriented, vertical=vertical,
            )
            if symbol_bbox is not None:
                plans = plan_structured_engineering_symbols(image, [{
                    "candidate_id": "MANUAL:0001", "bbox": symbol_bbox,
                    "text": text, "confidence": confidence,
                    "result": {"text": text, "confidence": confidence},
                }])
                if plans:
                    plan = plans[0]
                    text = plan.text
                    engineering_symbol = plan.evidence.to_dict()
                    needs_review = needs_review or plan.needs_review

        timings_ms = {
            # Pipeline time intentionally excludes debug-dump file writing.
            "pipeline_total": round(
                (perf_counter() - pipeline_started) * 1000,
                2,
            ),
            "paddle_total": round(
                sum(float(item["elapsed_ms"])
                    for item in paddle_timings),
                2,
            ),
            "paddle_pass_count": len(paddle_timings),
            "paddle_passes": paddle_timings,
        }

        dumper.stage("timings_ms", timings_ms)

        result: dict[str, Any] = {
            "text": text,
            "raw_ocr": raw_text,
            "confidence": confidence,
            "agreement": agreement,
            "needs_review": needs_review,
            "text_bbox": text_bbox,
            "text_bbox_source": text_bbox_source,
            "type": composed.kind,
            "engine": engine,
            "orientation": "vertical" if vertical else "horizontal",
            "rotation": 0,
            "words": words,
            "compose_steps": composed.applied,
            "symbols_detected": symbols_to_dict(symbols),
            "engineering_symbol": engineering_symbol,
            "prefix_ocr": prefix_text,
            "prefix_ocr_used": prefix_ocr_used,
            "ocr_profile": (
                "single_pass"
                if max_paddle_predictions == 1 and not allow_prefix_ocr
                else "accuracy"
            ),
            "timings_ms": timings_ms,
        }

        # GD&T rule engine (feature_rules / feature_dictionary): assign the
        # feature category, subtype and the suggested balloon label. Without
        # this a region reaches the UI with no category and the frontend falls
        # back to its text-only classifier, losing every rule that reads the
        # detected symbols.
        feature = classify_feature(
            text, symbols=symbols_to_dict(symbols),
            engineering_symbol=engineering_symbol or {},
        )
        result["category"] = feature.category
        result["subtype"] = feature.subtype
        result["label"] = feature.label

        dump_path = dumper.finalize(result)
        result["debug_dump"] = {
            **dump_status(),
            "active": dumper.active,
            "skipped_reason": dumper.skip_reason,
            "dir": dump_path,
        }
        if dump_path:
            result["debug_dump_dir"] = dump_path
        return result


_pipeline: OcrPipeline | None = None


def get_pipeline() -> OcrPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = OcrPipeline()
        _pipeline.load()
    return _pipeline
