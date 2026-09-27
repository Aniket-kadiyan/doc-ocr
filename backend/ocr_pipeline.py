"""
PaddleOCR pipeline: best digit read + balanced symbol compose.
"""

from __future__ import annotations
from time import perf_counter

from typing import Any
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
    prepare_ocr_variants,
    primary_oriented,
    upscale_min_edge,
    clahe_rgb,
)
from paddle_parse import extract_paddle_records
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
# Paddle confidence is expressed from 0.0 to 1.0.
# This comparison is intentionally strict: exactly 0.95 continues.
EARLY_ACCEPT_CONFIDENCE = 0.95

# If visual Ø detection is close to its 0.32 acceptance threshold, use the
# dedicated prefix OCR as a second opinion.
PREFIX_RECHECK_PHI_SCORE = 0.20


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


def sort_reading_order(
    items: list[dict[str, Any]],
    vertical: bool,
) -> list[dict[str, Any]]:
    """Order recognized lines top-to-bottom (vertical) or by row then column."""
    if vertical:
        return sorted(items, key=lambda r: r["y"])
    return sorted(
        items,
        key=lambda r: (round(r["y"] / LINE_THRESHOLD), r["x"]),
    )


def assemble_paddle_lines(
    result: Any,
    force_vertical: bool = False,
) -> tuple[str, float, list[dict[str, Any]]]:
    parsed = extract_paddle_records(result)
    if not parsed:
        return "", 0.0, []

    confidences = [p["confidence"] for p in parsed]
    avg_h = sum(p["height"] for p in parsed) / len(parsed)
    avg_w = sum(p["width"] for p in parsed) / len(parsed)
    vertical = force_vertical or (avg_h > avg_w * 2.5)
    # Records already carry the word keys (plus the detector's ``quad``), so
    # they are the word dicts.
    words = sort_reading_order(parsed, vertical)

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

# Words that belong to the title block or the revision table, never to the prose
# of a general note. On a real sheet the title block starts TEN PIXELS under the
# last note line, so no distance rule can separate them — the boundary has to be
# recognised by what the text says.
_TITLE_BLOCK_WORDS_RE = re.compile(
    r"\b(TOLERANCES?|DWG|DRAWN|CHECKED|APPROVED|SCALE|SHEET|UNITS?"
    r"|FRACTION|DECIMAL|REVISIONS|DESCRIPTION|ZONE|INCHES|MILLIMETERS)\b",
    re.I,
)


class OcrPipeline:
    def __init__(self) -> None:
        self._paddle = None
        self._paddle_available = False
        self._paddle_api = 0  # 3 = PaddleOCR 3.x (.predict), 2 = 2.x (.ocr)
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

    @property
    def status(self) -> dict[str, Any]:
        return {
            "paddleocr": self._paddle_available,
            "paddleocr_version": self._paddle_version,
            "paddleocr_api": self._paddle_api,
            "trocr": False,
            "symbol_vision": True,
            "errors": self._init_errors,
        }

    def _predict_3x(self, arr: np.ndarray) -> Any:
        """Run PaddleOCR 3.x; paddle_parse normalizes its Result objects."""
        return self._paddle.predict(arr)

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

    def detect_regions(self, image: Image.Image) -> list[dict[str, Any]]:
        """
        Propose every text-region box inside ``image`` (received-image pixels).

        Primary path is PaddleOCR's learned text detector run on the
        blue-ink-emphasized image (``cad_ink_to_gray`` + upscale): on CAD sheets
        this reliably returns one tight box per dimension line — including faint
        angles and values wedged against geometry that the morphology proposer
        drops. Falls back to the morphology + OCR-supplement proposer only when
        the detector finds nothing (e.g. a blank or non-text crop). Returns
        dicts: {x, y, w, h, text, conf}.
        """
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

    def _paddle_det_boxes(self, pil_img: Image.Image) -> list[dict[str, Any]]:
        """PaddleOCR detection on one image; boxes in that image's own pixels."""
        pad = 24
        padded = pad_image(pil_img, px=pad)
        pw, ph = padded.size
        edge = max(pw, ph)
        # Target ~1100 px on the long edge — enough for det on thin strokes.
        factor = min(12.0, max(1.0, 1100.0 / max(edge, 1)))
        det_img = (
            padded.resize((int(pw * factor), int(ph * factor)),
                          Image.Resampling.LANCZOS)
            if factor > 1.0
            else padded
        )
        _, _, words = self._run_paddle(clahe_rgb(det_img), det=True)
        out: list[dict[str, Any]] = []
        for w in words:
            quad = w.get("quad")
            out.append(
                {
                    "x": w["x"] / factor - pad,
                    "y": w["y"] / factor - pad,
                    "w": w["width"] / factor,
                    "h": w["height"] / factor,
                    "text": w.get("text", ""),
                    "conf": float(w.get("confidence", 0.0)),
                    # The detector's own corners, back in this image's pixels.
                    # They carry the text's angle; the box above does not.
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
            quad = b.get("quad")
            cand.append(
                {
                    "x": b["y"],
                    "y": ih - (b["x"] + b["w"]),
                    "w": b["h"],
                    "h": b["w"],
                    "text": b["text"],
                    "conf": b["conf"],
                    # Same mapping, point by point: (xr, yr) -> (yr, ih - xr).
                    "quad": (
                        [(py, ih - px) for px, py in quad] if quad else None
                    ),
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
                    # Unclamped on purpose: clipping corners would skew the angle.
                    "quad": b.get("quad"),
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
    ) -> list[list[dict[str, Any]]]:
        """
        Re-detect inside oversized clusters and replace them with finer groups.
        """
        from region_cluster import cluster_boxes, order_clusters, split_mixed_clusters, union_bbox
        from segment_quality import is_segment_worthy

        iw, ih = image.size
        median_h = 0.0
        if clusters:
            heights = [union_bbox(c)["height"] for c in clusters]
            median_h = sorted(heights)[len(heights) // 2]
        # Text scale = median short side of the member detection boxes (≈ one
        # line height). The image-fraction tests below scale with the CROP, so
        # on a small selection a single ordinary callout (136 px wide in a
        # 340 px crop) tripped them and was shredded into unworthy fragments.
        # Floor every test by a text-relative size so only clusters that are
        # genuinely many lines across can be re-split.
        # Only line-shaped boxes (aspect ≥ 1.5) vote: diagonal text and symbol
        # frames detect as near-square boxes whose short side is nothing like a
        # line height and would inflate the floor until nothing re-splits.
        shorts = sorted(
            min(b["w"], b["h"])
            for c in clusters
            for b in c
            if max(b["w"], b["h"]) >= 1.5 * max(min(b["w"], b["h"]), 1.0)
        )
        text_scale = shorts[len(shorts) // 2] if shorts else 0.0

        out: list[list[dict[str, Any]]] = []
        for cluster in clusters:
            ub = union_bbox(cluster)
            oversized = (
                ub["height"] > max(median_h * 2.8, ih * 0.28, text_scale * 6.0)
                or ub["width"] > max(iw * 0.38, text_scale * 12.0)
                or ub["width"] * ub["height"]
                > max(iw * ih * 0.07, text_scale * text_scale * 40.0)
            )
            if not oversized:
                out.append(cluster)
                continue

            margin = 4
            cx0 = max(0, int(ub["x"] - margin))
            cy0 = max(0, int(ub["y"] - margin))
            cx1 = min(iw, int(ub["x"] + ub["width"] + margin))
            cy1 = min(ih, int(ub["y"] + ub["height"] + margin))
            sub = image.crop((cx0, cy0, cx1, cy1))
            sub_boxes = self.detect_regions(sub)
            if not sub_boxes:
                out.append(cluster)
                continue
            for b in sub_boxes:
                b["x"] = round(b["x"] + cx0, 1)
                b["y"] = round(b["y"] + cy0, 1)
            sub_clusters = split_mixed_clusters(
                cluster_boxes(
                    sub_boxes,
                    margin_ratio=cluster_margin,
                    img_w=cx1 - cx0,
                    img_h=cy1 - cy0,
                )
            )
            # Accept the finer split only when it actually separates 2+ real
            # values (by their detection text). A re-detect that shatters one
            # callout into "45" / "±3°" / "7" would otherwise replace a clean
            # region with fragments that fail the worthiness filter.
            worthy = sum(
                1
                for sc in sub_clusters
                if any(is_segment_worthy(b.get("text", "")) for b in sc)
            )
            if len(sub_clusters) <= 1 or worthy < 2:
                out.append(cluster)
            else:
                out.extend(sub_clusters)
        return order_clusters(out)

    @staticmethod
    def _rotate_expand(
        image: Image.Image, angle: float, *, with_forward: bool = False
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
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
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

    @classmethod
    def _slant_angle_groups(
        cls,
        boxes: list[dict[str, Any]],
        *,
        min_slant: float = 8.0,
        tolerance: float = 12.0,
        max_groups: int = 2,
    ) -> list[float]:
        """
        Angles at which this crop's text actually runs, from the detector.

        Each detection carries its own angle, so the result is local to the text
        rather than a sheet-wide vote whose outcome depended on how much of the
        drawing the user selected. Boxes are gathered into groups within
        ``tolerance`` degrees, heaviest box seeding each group, and each group
        reports its size-weighted mean, heaviest first.

        Near-horizontal and near-vertical text is skipped: the upright and 90°
        detection passes already read those, and on a CAD sheet the vertical
        dimensions would otherwise outvote the diagonal callouts entirely.
        """
        upright_guard = max(min_slant, 1.0)
        candidates: list[tuple[float, float]] = []
        for box in boxes:
            angle = cls._quad_angle(box.get("quad"))
            if angle is None:
                continue
            if abs(angle) < upright_guard or abs(angle) > 90.0 - upright_guard:
                continue
            weight = max(float(box.get("w", 0.0)), float(box.get("h", 0.0)))
            if weight > 0:
                candidates.append((angle, weight))

        # Heaviest first, so a group is seeded by its most substantial box and
        # smaller neighbouring reads merge into it rather than the reverse.
        candidates.sort(key=lambda t: t[1], reverse=True)
        groups: list[dict[str, float]] = []
        for angle, weight in candidates:
            for group in groups:
                if abs(group["angle"] - angle) <= tolerance:
                    total = group["weight"] + weight
                    group["angle"] = (
                        group["angle"] * group["weight"] + angle * weight
                    ) / total
                    group["weight"] = total
                    break
            else:
                groups.append({"angle": angle, "weight": weight})
        groups.sort(key=lambda g: g["weight"], reverse=True)
        return [round(g["angle"], 1) for g in groups[:max_groups]]

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
            angle = self._quad_angle(box.get("quad"))
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
            angle = self._quad_angle(box.get("quad"))
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

    def _detect_angled_regions(
        self,
        image: Image.Image,
        base_regions: list[dict[str, Any]],
        det_boxes: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recover slanted callouts (chamfers ``0.5×45°``, angled fits) that the
        upright/90° detector misses.

        Runs per diagonal *neighbourhood* rather than over the whole selection,
        so the result no longer depends on how much of the drawing was selected —
        the failure that left the two ``H10`` fit callouts unread on a full
        sheet while they read correctly on a zoomed crop. Each neighbourhood is
        levelled by its own measured angle, re-detected, read, and mapped back.
        Results are deduped against ``base_regions`` so a value already read
        upright is not reported twice.
        """
        # First the whole selection, which is what a zoomed-in crop needs.
        found = self._angled_in_roi(image)
        # Then each diagonal neighbourhood, always — not only when the pass
        # above came back empty. On a large selection the slant vote is diluted
        # by the rest of the drawing and names one angle, so it recovers one
        # leader's callout and misses its neighbour on the next leader. Working
        # neighbourhood by neighbourhood picks up the rest; anything found
        # twice is reconciled by the dedupe below.
        from region_detect import detect_leader_lines

        leaders = detect_leader_lines(image)
        for x0, y0, x1, y1 in self._slant_neighbourhoods(image, det_boxes or []):
            sign = self._neighbourhood_sign(det_boxes or [], (x0, y0, x1, y1))
            roi = image.crop((x0, y0, x1, y1))
            leader_angles = self._leader_angles_for_roi(leaders, (x0, y0, x1, y1))
            # Two zoom levels. Detection resolves a callout's parts at one
            # zoom and its neighbour's at another — on the two fit callouts
            # each is read cleanly at a different one — so gather both and
            # let the quality ranking in ``_dedupe_angled`` choose.
            for zoom in (1.0, 1.5):
                if zoom == 1.0:
                    view = roi
                else:
                    view = roi.resize(
                        (int(roi.width * zoom), int(roi.height * zoom)),
                        Image.Resampling.LANCZOS,
                    )
                for region in self._angled_in_roi(
                    view,
                    sign=sign,
                    max_magnitudes=3,
                    # Only consulted when no leader runs through the region:
                    # this region is known to hold diagonal text, so a
                    # magnitude needs less of the vote to be worth a try. At
                    # sheet scale the correct angle is rarely the top bucket.
                    weight_floor=0.08,
                    fallback_angles=leader_angles,
                ):
                    for box in (region["bbox"], region.get("oriented_box")):
                        if not box:
                            continue
                        box["x"] = round(box["x"] / zoom + x0, 1)
                        box["y"] = round(box["y"] / zoom + y0, 1)
                        box["width"] = round(box["width"] / zoom, 1)
                        box["height"] = round(box["height"] / zoom, 1)
                    found.append(region)

        superseded = self._supersede_fused_base(found, base_regions)
        return self._dedupe_angled(
            found, [b for b in base_regions if id(b) not in superseded]
        )

    def _angled_in_roi(
        self,
        image: Image.Image,
        *,
        sign: float | None = None,
        max_magnitudes: int = 1,
        weight_floor: float | None = None,
        fallback_angles: list[float] | None = None,
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

        ``fallback_angles`` is used only when the vote finds nothing at all —
        the angles of the leader lines running through this region. Measuring a
        long straight stroke works at any scale, so it covers regions where the
        vote over text strokes is too weak to name an angle. It does not
        override the vote, which measured better wherever both had an opinion.
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
        angles = angles[:4]
        if not angles:
            return []

        found: list[dict[str, Any]] = []
        for angle in angles:
            rimg, inv = self._rotate_expand(image, angle)
            rw, rh = rimg.size
            # Detect again on the *levelled* ink. The quads told us the angle,
            # but a detector box for steeply slanted text is a poor fit — it
            # misses a stacked deviation and merges neighbours. Once the text is
            # horizontal the detector is accurate, which is what makes the
            # value-plus-deviation grouping below work. (Do NOT reuse the full
            # detect_regions/cluster pipeline here — its 90° pass and aggressive
            # merge fuse the now-diagonal axis text into giant blobs.)
            raw = self._paddle_det_boxes(cad_ink_to_gray(rimg).convert("RGB"))
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
            clusters = cluster_boxes(boxes, margin_ratio=0.3, img_w=rw, img_h=rh)

            margin = 6
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
                sub = rimg.crop((cx0, cy0, cx1, cy1))
                res = self.recognize(sub, compute_text_bbox=False)
                text = strip_foreign_glyphs((res.get("text") or "").strip())
                # A value with stacked deviations reads back interleaved. Only
                # worth a finer pass when the text is long enough to hold one.
                if sum(c.isdigit() for c in text) >= 5:
                    text = self._stacked_deviation_read(sub, res) or text
                # "0.2-0.3×45°R1": a radius callout drawn right after the
                # chamfer is a second value; keep the chamfer.
                m_tail = re.match(r"^(.*\d°)\s*[Rr]\d[\d.]*$", text)
                if m_tail:
                    text = m_tail.group(1)
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
                # A single slanted callout is ONE line of text, optionally with
                # its tolerance stacked above/below it. Judge fusion by that
                # structure — rows in the levelled frame — rather than by a raw
                # digit count, which a legitimate fit callout
                # (``Ø20H10 +0.084/0``, 9 digits) trips.
                if count_dimension_values(text) >= 2:
                    continue
                if self._cluster_row_count(cluster) > 3:
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

        return found

    def _stacked_deviation_read(
        self, crop: Image.Image, res: dict[str, Any]
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
        if not raw:
            return None
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
            dup = [
                b for b in base_regions
                if overlaps(r, b) and same_value(d, digits(b.get("text", "")))
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
        return extract_title_fields(self.detect_regions(image), keywords)

    def segment(
        self,
        image: Image.Image,
        *,
        debug_dump: bool = False,
        debug_dump_force: bool = False,
        cluster_margin: float = 0.72,
        title_keywords: list[str] | None = None,
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
        from segment_quality import (
            count_dimension_values,
            dedupe_regions,
            has_dimension_value,
            is_annotation_note,
            is_segment_worthy,
            strip_foreign_glyphs,
        )
        from stroke_filter import is_stray_line
        from title_fields import extract_title_fields

        boxes = self.detect_regions(image)
        # The title-block pass needs every detected box, including the label
        # cells the dimension filters strip out below.
        det_boxes = list(boxes)
        # The NOTES paragraph is lifted out BEFORE clustering: its lines are not
        # dimensions (they would all be dropped downstream), and leaving them in
        # lets a note line bridge into a neighbouring callout's cluster. The
        # region is added back untouched just before returning.
        notes_region, notes_idx = self._detect_notes_block(boxes)
        if notes_idx:
            consumed = set(notes_idx)
            boxes = [b for i, b in enumerate(boxes) if i not in consumed]
        # Drop free-text annotation notes ("(BOTH SIDES)", "TYP") while their
        # detection text is still clean — before they cluster into a neighbour or
        # the composer injects a bogus Ø. Guarded on digit count in the helper.
        boxes = [b for b in boxes if not is_annotation_note(b.get("text", ""))]
        boxes = self._drop_speck_boxes(boxes)
        # Drop cross-column bridge boxes before clustering so separate
        # dimensions (e.g. Ø174,07 and Ø175,32) don't fuse into one cluster.
        boxes = drop_bridge_boxes(boxes)
        iw, ih = image.size
        clustered = cluster_boxes(
            boxes,
            margin_ratio=cluster_margin,
            img_w=iw,
            img_h=ih,
        )
        clustered = merge_fragment_clusters(clustered)
        clusters = order_clusters(split_mixed_clusters(clustered))
        clusters = self._expand_clusters(
            image, clusters, cluster_margin=cluster_margin
        )
        # Re-join any fragments of one dimension that landed in overlapping
        # boxes (e.g. a value split from its REF. tag onto a perpendicular axis).
        clusters = order_clusters(merge_overlapping_clusters(clusters))

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
        for cluster in clusters:
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
            text = strip_foreign_glyphs((res.get("text") or "").strip())
            text = self._prefer_detection_text(cluster, res, text)
            if not text or not is_segment_worthy(text):
                continue
            # A balloon is for something a person measures. Table rows, dates
            # and part numbers read like values but are not dimensions.
            if not has_dimension_value(text):
                continue
            # Drop leader/extension/tick lines that OCR'd as a phantom "1".
            if is_stray_line(sub, text):
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
            if multivalue or multirow:
                split = self._split_stacked_cluster(
                    image, (cx0, cy0, cx1, cy1),
                    debug_dump=debug_dump, debug_dump_force=debug_dump_force,
                )
                if split:
                    regions.extend(split)
                    continue

            # Snap balloon to the full detected cluster, not a tight OCR sliver.
            bbox = {
                "x": round(ub["x"], 1),
                "y": round(ub["y"], 1),
                "width": round(ub["width"], 1),
                "height": round(ub["height"], 1),
            }
            regions.append(self._region_from_result(res, bbox, text))

        regions = self._complete_angle_regions(image, regions)
        regions = dedupe_regions(regions)

        # Recover slanted callouts (chamfers, angled fits) the upright/90°
        # detector misses. Gated on slant presence, so non-diagonal sheets are
        # unaffected and pay only a single Hough call. Added AFTER the base set
        # is deduped and treated as authoritative: angled reads only ADD values
        # not already present, never displace a clean upright region.
        regions.extend(self._detect_angled_regions(image, regions, det_boxes=boxes))
        # A base region the angled pass resolved into its separate callouts is
        # dropped here (it was flagged while those reads were being placed).
        regions = [r for r in regions if not r.pop("superseded", False)]

        # Appended last so the dimension-oriented dedupe and angled passes,
        # which reason about values, never discard or reshape the notes block.
        if notes_region is not None:
            bounds = notes_region.pop("_bounds", None)
            block_line_h = notes_region.pop("_line_h", 0.0)
            if bounds is not None:
                better = self._reread_notes_block(image, bounds, block_line_h)
                if better:
                    notes_region["text"] = "\n".join(better)
            regions.append(notes_region)

        # Title-block fields ("DWG NO.", "REV") the user asked for by keyword.
        # Read from the ORIGINAL detection boxes: the label cells are stripped
        # out of `boxes` above as non-dimensions, and a field only becomes a
        # region when a value was actually read for it.
        for field in extract_title_fields(det_boxes, title_keywords or []):
            if not field["value"] or not field["bbox"]:
                continue
            fb = field["bbox"]
            regions.append(
                {
                    "bbox": {
                        "x": round(fb["x"], 1),
                        "y": round(fb["y"], 1),
                        "width": round(fb["w"], 1),
                        "height": round(fb["h"], 1),
                    },
                    "text": field["value"],
                    "confidence": round(float(field["confidence"]), 4),
                    "type": "Title Block",
                    "category": "Title Block",
                    "subtype": None,
                    "label": field["label"],
                    "orientation": "horizontal",
                    "rotation": 0,
                    "needs_review": False,
                    "agreement": 0.0,
                    "engine": "paddleocr",
                    "symbols_detected": None,
                }
            )

        return {"count": len(regions), "regions": regions}

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
    def _drop_speck_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Drop low-confidence specks far below text size (arrowheads, line
        crossings, hatch fragments the detector read as a lone "0"/"7"/"A").

        Such a speck next to a real callout clusters into it and the union
        crop then re-reads with the stroke as an extra digit (``R1`` → ``12R1``).
        Text scale is the median short side of the line-shaped boxes, so a
        legitimately small "°" or "." box that sits inside its own line is
        unaffected (it is part of a larger box, not standalone).
        """
        scale = OcrPipeline._text_scale(boxes)
        if not scale:
            return boxes
        limit = scale * 0.4
        out: list[dict[str, Any]] = []
        for b in boxes:
            speck = (
                max(b["w"], b["h"]) < limit
                and float(b.get("conf") or 0.0) < 0.7
                and sum(c.isdigit() for c in b.get("text", "")) <= 1
            )
            if not speck:
                out.append(b)
        return out

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
        if factor > 1.01:
            crop = crop.resize(
                (int(crop.width * factor), int(crop.height * factor)),
                Image.LANCZOS,
            )

        try:
            boxes = self.detect_regions(crop.convert("RGB"))
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
        from segment_quality import strip_foreign_glyphs

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

    def _split_stacked_cluster(
        self,
        image: Image.Image,
        coords: tuple[int, int, int, int],
        *,
        debug_dump: bool = False,
        debug_dump_force: bool = False,
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
            has_dimension_value,
            is_segment_worthy,
            strip_foreign_glyphs,
        )
        from stroke_filter import is_stray_line

        cx0, cy0, cx1, cy1 = coords
        sub = image.crop((cx0, cy0, cx1, cy1))
        sub_boxes = self.detect_regions(sub)
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

        # GD&T rule engine: classify into a feature category and derive a
        # balloon label from OCR text + detected symbols (geometry optional).
        feature = classify_feature(text, symbols=symbols_to_dict(symbols))

        dumper.stage(
            "compose_output",
            {
                "text": composed.text,
                "kind": composed.kind,
                "applied": composed.applied,
            },
        )

        # A dual pair that violates 25.4 and can't be repaired is a hard
        # signal the read is wrong; a repaired pair changed a digit, so a human
        # should confirm it. A pair that already agrees is, conversely, strong
        # evidence the read is right — trust it even at lower raw confidence.
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
            "category": feature.category,
            "subtype": feature.subtype,
            "label": feature.label,
            "feature": feature.to_dict(),
            "engine": engine,
            "orientation": "vertical" if vertical else "horizontal",
            "rotation": 0,
            "words": words,
            "compose_steps": composed.applied,
            "symbols_detected": symbols_to_dict(symbols),
            "prefix_ocr": prefix_text,
            "prefix_ocr_used": prefix_ocr_used,
            "dual_unit": {
                "status": dual.status,
                "direction": dual.direction,
                "expected_mm": dual.expected_mm,
                "edits": dual.edits,
            },
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
