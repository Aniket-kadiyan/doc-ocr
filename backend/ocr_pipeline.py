"""
PaddleOCR pipeline: best digit read + balanced symbol compose.
"""

from __future__ import annotations
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from debug_dump import StepDumper, dump_force, dump_status, should_dump
from dimension_compose import compose_engineering_dimension
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
    extract_paddle_detection_boxes,
    extract_paddle_detection_regions,
    extract_paddle_lines,
    extract_text_orientation_result,
    extract_text_recognition_result,
)
from symbol_normalize import fix_engineering_symbols_light
from symbol_regions import enlarge_zone, split_symbol_zones
from symbol_vision import (
    detect_diameter_symbol,
    detect_prefix_from_ocr_text,
    detect_symbols,
    merge_symbol_scores,
    symbols_to_dict,
)

LINE_THRESHOLD = 15
# Paddle confidence is expressed from 0.0 to 1.0.
# This comparison is intentionally strict: exactly 0.95 continues.
EARLY_ACCEPT_CONFIDENCE = 0.95
OCR_CHANGESET_ID = "m1-p02r1-field-gated-phi"

# Oversized-cluster refinement is a best-effort accuracy improvement. It must
# never expand into unbounded OCR work during a whole-page scan.
GROUPING_MAX_REFINEMENTS = 8


def sort_reading_order(
    items: list[tuple[str, float, float, float, float, float]],
    vertical: bool,
) -> list[tuple[str, float, float, float, float, float]]:
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
        {"text": t, "x": x, "y": y, "width": w, "height": h, "confidence": c}
        for t, x, y, w, h, c in ordered
    ]

    raw = (
        "".join(w["text"].strip() for w in words)
        if vertical
        else " ".join(w["text"].strip() for w in words)
    )

    conf = sum(confidences) / len(confidences) if confidences else 0.0
    return fix_engineering_symbols_light(raw), conf, words


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
            "ocr_changeset": OCR_CHANGESET_ID,
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
            visual_diameter, _diameter_debug = detect_diameter_symbol(crop)
            raw_text_hints = detect_prefix_from_ocr_text(raw_text)
            symbols = merge_symbol_scores(visual_diameter, raw_text_hints)
            fixed, corrected = correct_numeric_confusables(raw_text)
            fixed_text_hints = detect_prefix_from_ocr_text(fixed)
            symbols = merge_symbol_scores(symbols, fixed_text_hints)
            composed = compose_engineering_dimension(
                fixed,
                crop,
                symbols,
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
                    "symbols_detected": symbols_to_dict(symbols),
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

                if conf > g["conf"]:
                    g["conf"] = conf
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
                        "words": words,
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
            out.append(
                {
                    "x": w["x"] / factor - pad,
                    "y": w["y"] / factor - pad,
                    "w": w["width"] / factor,
                    "h": w["height"] / factor,
                    "text": w.get("text", ""),
                    "conf": float(w.get("confidence", 0.0)),
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
        from segment_quality import dedupe_regions
        from page_scan import overlaps_existing_value
        from page_value_filters import (
            PageValueCandidate,
            evaluate_scan_value,
            normalize_page_value_text,
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
            for cluster in clusters:
                ub = union_bbox(cluster)
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
                    "orientation": res.get("orientation", "horizontal"),
                    "rotation": res.get("rotation", 0),
                    "needs_review": res.get("needs_review", False),
                    "agreement": res.get("agreement", 0.0),
                    "engine": res.get("engine", "paddleocr"),
                    "symbols_detected": res.get("symbols_detected"),
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
        regions = dedupe_regions(regions)
        recognized_count = len(regions)
        filtered_regions: list[dict[str, Any]] = []
        candidate_outcomes: list[dict[str, Any]] = []
        filter_rule_counts: dict[str, int] = {}
        for region in regions:
            text = normalize_page_value_text(str(region.get("text") or ""))
            decision = evaluate_scan_value(
                PageValueCandidate(text=text, bbox=region["bbox"]),
                scope_kind="section",
                table_masks=(),
            )
            filter_rule_counts[decision.rule_name] = (
                filter_rule_counts.get(decision.rule_name, 0) + 1
            )
            state = "eligible" if decision.accepted else "excluded"
            candidate_outcomes.append(
                {
                    "bbox": dict(region["bbox"]),
                    "state": state,
                    "text": text,
                    "recognized": True,
                    "reason": decision.reason,
                    "rule": decision.rule_name,
                }
            )
            if decision.accepted:
                filtered_regions.append(
                    {
                        **region,
                        "text": text,
                        "recognized": True,
                        "page_filter_rule": decision.rule_name,
                        "page_filter_reason": decision.reason,
                    }
                )
        regions = filtered_regions
        report(
            stage="finalizing",
            message=f"Prepared {len(regions)} balloon candidates",
            percent=99,
            completed=len(regions),
            total=len(regions),
            candidate_count=len(regions),
        )
        excluded_count = recognized_count - len(regions)
        return {
            "count": len(regions),
            "detected_count": cluster_total,
            "recognized_count": recognized_count,
            "eligible_count": len(regions),
            "excluded_count": excluded_count,
            "review_count": 0,
            "unread_count": max(0, cluster_total - recognized_count),
            "skipped_existing_count": skipped_existing_count,
            "filter_rule_counts": dict(sorted(filter_rule_counts.items())),
            "regions": regions,
            "review_candidates": [],
            "candidate_outcomes": candidate_outcomes,
        }

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
            PageValueCandidate,
            evaluate_page_value,
            needs_expanded_filter_context,
            normalize_page_value_text,
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

        margin = 6
        page_width, page_height = page_size
        detected_count = len(final_candidates)
        primary_recognized_count = 0
        ocr_records: list[dict[str, Any]] = []
        candidate_crops: list[Image.Image] = []
        for candidate in final_candidates:
            bbox = candidate.bbox
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
            candidate_crops.append(
                image.crop((x0, y0, x1, y1))
                if x1 > x0 and y1 > y0
                else Image.new("RGB", (1, 1), (255, 255, 255))
            )

        primary_batch_total = (
            (detected_count + PAGE_SCAN_RECOGNITION_BATCH_SIZE - 1)
            // PAGE_SCAN_RECOGNITION_BATCH_SIZE
        )
        for batch_index, batch_start in enumerate(
            range(0, detected_count, PAGE_SCAN_RECOGNITION_BATCH_SIZE),
            start=1,
        ):
            batch_end = min(
                detected_count,
                batch_start + PAGE_SCAN_RECOGNITION_BATCH_SIZE,
            )
            report(
                stage="recognizing",
                message=(
                    f"Primary recognition batch {batch_index} of "
                    f"{primary_batch_total} ({batch_start + 1}–{batch_end} "
                    f"of {detected_count} objects)"
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
                candidate_crops[batch_start:batch_end],
                batch_size=PAGE_SCAN_RECOGNITION_BATCH_SIZE,
            )
            if len(batch_results) != batch_end - batch_start:
                raise RuntimeError(
                    "Page recognition batch returned an unexpected result count"
                )
            for candidate, result in zip(
                final_candidates[batch_start:batch_end],
                batch_results,
            ):
                text = str(result.get("text") or "").strip()
                if text:
                    primary_recognized_count += 1
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
                    }
                )
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

        selected_recovery_indexes = select_recovery_record_indexes(
            ocr_records,
            maximum=RECOVERY_MAX_CANDIDATES,
        )
        selected_recovery_set = set(selected_recovery_indexes)
        all_doubtful_indexes = {
            index
            for index, record in enumerate(ocr_records)
            if result_needs_recovery(record["result"])
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
        for first_batch_index, batch_start in enumerate(
            range(
                0,
                len(selected_recovery_indexes),
                PAGE_SCAN_RECOGNITION_BATCH_SIZE,
            ),
            start=1,
        ):
            batch_indexes = selected_recovery_indexes[
                batch_start : batch_start + PAGE_SCAN_RECOGNITION_BATCH_SIZE
            ]
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
        for second_batch_index, batch_start in enumerate(
            range(
                0,
                len(second_recovery_indexes),
                PAGE_SCAN_RECOGNITION_BATCH_SIZE,
            ),
            start=1,
        ):
            recovery_batch_index = (
                first_recovery_batch_total + second_batch_index
            )
            batch_indexes = second_recovery_indexes[
                batch_start : batch_start + PAGE_SCAN_RECOGNITION_BATCH_SIZE
            ]
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
            resolved = resolve_recovery_consensus(
                record["result"],
                recovery_attempts.get(index, ()),
                attempted=index in selected_recovery_set,
                budget_exhausted=index in budget_exhausted_indexes,
            )
            record["result"] = resolved
            normalized_text = normalize_page_value_text(
                str(resolved.get("text") or "")
            )
            resolved["text"] = normalized_text
            record["text"] = normalized_text
            record["recognized"] = bool(record["text"])

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
        context_ocr_indexes = context_indexes[:CONTEXT_MAX_CANDIDATES]
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

        table_masks = [mask.to_dict() for mask in layout.table_masks]
        hard_exclusion_rules = {
            "table_region",
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
            if not record["recognized"]:
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
            )
            preliminary_decisions.append(decision)
            preliminary_state, preliminary_reason = resolve_candidate_state(
                record,
                decision,
                authoritative=False,
            )
            record["preliminary_state"] = preliminary_state
            record["preliminary_reason"] = preliminary_reason
            if preliminary_state != "excluded":
                authoritative_indexes.append(filter_index - 1)

        authoritative_total = len(authoritative_indexes)
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
            previous_result = dict(record["result"])
            record["preliminary_result"] = previous_result
            authoritative_crops = build_authoritative_crops(
                image,
                record["bbox"],
                record["polygon"],
            )
            tight_crop = authoritative_crops[0]
            tight_result = self.recognize(
                tight_crop.image,
                debug_dump=debug_dump,
                debug_dump_force=debug_dump_force,
                compute_text_bbox=True,
            )
            authoritative_attempts = [
                assess_authoritative_result(tight_result, tight_crop)
            ]
            if authoritative_result_needs_retry(authoritative_attempts[0]):
                padded_crop = authoritative_crops[1]
                padded_result = self.recognize(
                    padded_crop.image,
                    debug_dump=debug_dump,
                    debug_dump_force=debug_dump_force,
                    compute_text_bbox=True,
                )
                authoritative_attempts.append(
                    assess_authoritative_result(padded_result, padded_crop)
                )
            accurate_result = resolve_authoritative_hypotheses(
                previous_result,
                authoritative_attempts,
            )
            accurate_text = str(accurate_result.get("text") or "")
            record["result"] = accurate_result
            record["text"] = accurate_text
            record["recognized"] = bool(accurate_text)
            record["authoritative_reread"] = True
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
                final_state = "excluded"
                final_reason = str(record.get("preliminary_reason") or "")
            else:
                decision = evaluate_page_value(
                    filter_candidate,
                    page_candidates=filter_candidates,
                    table_masks=table_masks,
                )
                final_state, final_reason = resolve_candidate_state(
                    record,
                    decision,
                    authoritative=bool(record.get("authoritative_reread")),
                )
            if not record["recognized"] and decision.rule_name != "table_region":
                filter_rule_counts["unread"] += 1
            else:
                filter_rule_counts[decision.rule_name] += 1

            candidate = record["candidate"]
            result = record["result"]
            # A non-numeric review proposal is safer as a blank editable value
            # than misleading OCR such as "rat". Raw text remains diagnostic.
            published_text = record["text"]
            if (
                final_state == "review"
                and decision.rule_name == "no_numeric_component"
            ):
                published_text = ""

            common_region = {
                "candidate_id": record["candidate_id"],
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
                "ocr_profile": result.get("ocr_profile", "batch_recognition"),
                "page_filter_rule": decision.rule_name,
                "page_filter_reason": decision.reason,
                "review_reason": final_reason if final_state == "review" else "",
                "recovery_attempted": bool(result.get("recovery_attempted")),
                "authoritative_reread": bool(
                    result.get("authoritative_reread")
                ),
                "authoritative_target_owned": bool(
                    result.get("authoritative_target_owned")
                ),
                "numeric_conflict": bool(result.get("numeric_conflict")),
            }
            if final_state == "eligible":
                regions.append(common_region)
            elif final_state == "review":
                review_candidates.append(common_region)

            outcome = {
                "candidate_id": record["candidate_id"],
                "bbox": dict(record["bbox"]),
                "polygon": [list(point) for point in candidate.polygon],
                "state": final_state,
                "text": published_text,
                "confidence": float(result.get("confidence") or 0.0),
                "recognized": record["recognized"],
                "reason": final_reason,
                "rule": decision.rule_name,
                "recovery_attempted": bool(result.get("recovery_attempted")),
                "authoritative_reread": bool(
                    result.get("authoritative_reread")
                ),
            }
            candidate_outcomes.append(outcome)
            filtered_overlay_candidates.append(
                {
                    "id": record["candidate_id"],
                    "bbox": dict(record["bbox"]),
                    "state": final_state,
                    "text": published_text,
                    "reason": final_reason,
                    "rule": decision.rule_name,
                }
            )

        recognized_count = sum(
            1 for record in ocr_records if record["recognized"]
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
            not bool(symbols_before_merge["diameter"])
            and bool(symbols_before_merge["diameter_visual_candidate"])
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
        )

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
            "prefix_ocr": prefix_text,
            "prefix_ocr_used": prefix_ocr_used,
            "ocr_profile": (
                "single_pass"
                if max_paddle_predictions == 1 and not allow_prefix_ocr
                else "accuracy"
            ),
            "timings_ms": timings_ms,
        }

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
