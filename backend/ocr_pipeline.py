"""
PaddleOCR pipeline: best digit read + balanced symbol compose.
"""

from __future__ import annotations
from time import perf_counter
from typing import Any, Callable

import numpy as np
from PIL import Image

from debug_dump import StepDumper, dump_force, dump_status, should_dump
from dimension_compose import compose_engineering_dimension
from dimension_digits import normalize_cad_number_string
from image_preprocess import (
    bbox_from_oriented_to_original,
    is_vertical_dimension,
    pad_image,
    prepare_ocr_variants,
    primary_oriented,
    upscale_min_edge,
    clahe_rgb,
)
from paddle_parse import extract_paddle_detection_boxes, extract_paddle_lines
from symbol_normalize import fix_engineering_symbols_light
from symbol_regions import enlarge_zone, split_symbol_zones
from symbol_vision import (
    detect_prefix_from_ocr_text,
    detect_symbols,
    merge_symbol_scores,
    symbols_to_dict,
)

LINE_THRESHOLD = 15
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
        self._paddle = None
        self._paddle_available = False
        self._paddle_api = 0  # 3 = PaddleOCR 3.x (.predict), 2 = 2.x (.ocr)
        self._text_detector = None
        self._text_detector_available = False
        self._paddle_version = "unknown"
        self._init_errors: list[str] = []

    def load(self) -> None:
        self._load_paddle()

    def _load_paddle(self) -> None:
        try:
            import paddleocr  # type: ignore
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._init_errors.append(f"PaddleOCR import: {exc}")
            return

        # Detect the API by VERSION, not by probing the constructor: PaddleOCR
        # 2.7.x silently swallows unknown 3.x kwargs, so a try/except on the
        # constructor would mis-detect 2.x as 3.x and then call .predict()
        # (which doesn't exist on 2.x) — yielding empty results.
        ver = str(getattr(paddleocr, "__version__", "0"))
        self._paddle_version = ver
        major_part = ver.split(".", 1)[0]
        major = int(major_part) if major_part.isdigit() else 0
        is_3x = major >= 3 and hasattr(PaddleOCR, "predict")

        if is_3x:
            try:
                
                self._paddle = PaddleOCR(
                    lang="en",

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
                return
            except Exception as exc:  # noqa: BLE001
                self._init_errors.append(f"PaddleOCR 3.x: {exc}")

        # PaddleOCR 2.x.
        try:
            kwargs: dict[str, Any] = {
                "use_angle_cls": True,
                "lang": "en",
                "show_log": False,
            }
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
            )
            self._text_detector_available = True
        except Exception as exc:  # noqa: BLE001
            self._init_errors.append(f"PaddleOCR TextDetection: {exc}")

    @property
    def status(self) -> dict[str, Any]:
        return {
            "paddleocr": self._paddle_available,
            "paddleocr_version": self._paddle_version,
            "paddleocr_api": self._paddle_api,
            "text_detector": self._text_detector_available,
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
    ) -> tuple[str, float, list, float, bool]:
        """
        Run every preprocessing variant and vote across the candidates.

        Returns (text, confidence, words, agreement, corrected) where
        `agreement` is the fraction of candidates that agree with the winning
        value and `corrected` is True when a numeric-confusable fix was applied.
        """
        from collections import defaultdict

        from dimension_digits import correct_numeric_confusables
        from ocr_select import digit_quality_score

        # 3.x .predict() always detects, so det=False is redundant there.
        det_modes = (True,) if self._paddle_api == 3 else (True, False)

        # Group candidates by their normalized text.
        groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"score": 0.0, "count": 0, "conf": 0.0,
                     "words": [], "corrected": False}
        )
        total = 0

        # A candidate above the strict confidence threshold wins immediately.
        # Variants still execute sequentially; no parallel model calls are introduced.
        early_winner: dict[str, Any] | None = None
        candidates_log: list[dict[str, Any]] = []

        for _name, variant in prepare_ocr_variants(image):
            for det in det_modes:
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
            if early_winner is not None:
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

        detections: list[tuple[float, float, float, float, float, str]] = []
        if getattr(self, "_text_detector_available", False):
            detections = [
                (x, y, width, height, confidence, "")
                for x, y, width, height, confidence in self._run_text_detector(
                    detector_input
                )
            ]
        else:
            _, _, words = self._run_paddle(detector_input, det=True)
            detections = [
                (
                    float(word["x"]),
                    float(word["y"]),
                    float(word["width"]),
                    float(word["height"]),
                    float(word.get("confidence", 0.0)),
                    str(word.get("text", "")),
                )
                for word in words
            ]

        return [
            {
                "x": x / factor - pad,
                "y": y / factor - pad,
                "w": width / factor,
                "h": height / factor,
                "text": text,
                "conf": confidence,
            }
            for x, y, width, height, confidence, text in detections
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
    ) -> dict[str, Any]:
        """
        Auto-segment a multi-value selection into individual dimensions.

        Detects all text regions, clusters fragments that belong to the same
        dimension, then runs the full single-value `recognize` pipeline on each
        cluster's crop. Boxes are returned in received-image pixel coordinates
        (same space `recognize`'s `text_bbox` uses) so the client can reuse its
        existing value-box mapping.
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
        from segment_quality import dedupe_regions, is_segment_worthy

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
        report(
            stage="grouping",
            message=f"Prepared {len(clusters)} candidate objects",
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
            if not text or not is_segment_worthy(text):
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
        report(
            stage="finalizing",
            message=f"Prepared {len(regions)} balloon candidates",
            percent=99,
            completed=len(regions),
            total=len(regions),
            candidate_count=len(regions),
        )
        return {"count": len(regions), "regions": regions}

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
            raw_text.strip()
            and not reusable_symbol_hints
            and (
                confidence <= EARLY_ACCEPT_CONFIDENCE
                or ambiguous_leading
                or borderline_phi
            )
        )

        if not raw_text.strip():
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
