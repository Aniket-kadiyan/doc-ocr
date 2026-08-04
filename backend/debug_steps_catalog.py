"""
Human-readable guide for each DEBUG_DUMP artifact.

Written into ``steps.json`` so each file's purpose and expected values are
documented next to the dump.
"""

from __future__ import annotations

import re
from typing import Any

# Static catalog keyed by step ``name`` (filename without NN_ prefix).
STEP_CATALOG: dict[str, dict[str, Any]] = {
    "00_meta": {
        "module": "debug_dump / ocr_pipeline",
        "use_case": "Run metadata for this crop — sizes, vertical flag, dump path.",
        "values": {
            "image_size": "[width, height] of uploaded crop in pixels",
            "vertical": "true when height > 1.35× width (CAD text runs vertically)",
            "debug_dump": "whether DEBUG_DUMP was on for this run",
            "already_done": "true if this crop was dumped before (skipped rewrite)",
        },
    },
    "input": {
        "module": "ocr_pipeline.recognize",
        "use_case": "Exact PNG the browser sent (your drawn box crop). Baseline reference.",
        "values": {"pixels": "RGB image as received from POST /ocr/recognize"},
    },
    "padded": {
        "module": "image_preprocess.pad_image",
        "use_case": "White border added so edge glyphs (Ø at crop edge) are not clipped.",
        "values": {"padding_px": "36px on each side"},
    },
    "upscaled": {
        "module": "image_preprocess.upscale_min_edge",
        "use_case": "Small crops enlarged so PaddleOCR has enough pixels for digits/symbols.",
        "values": {"min_edge": "280px on longest side (up to 12× scale)"},
    },
    "oriented": {
        "module": "image_preprocess.primary_oriented",
        "use_case": "Vertical CAD text rotated 90° CCW so OCR reads left-to-right.",
        "values": {
            "vertical_crop": "rotated -90° (CW)",
            "horizontal_crop": "unchanged",
        },
    },
    "zone_prefix": {
        "module": "symbol_regions.split_symbol_zones",
        "use_case": (
            "Leading symbol strip — where Ø (phi), R, or ± often sit before the "
            "nominal value. Used for dedicated phi vision + prefix-only OCR."
        ),
        "values": {
            "horizontal": "left ~28% of oriented crop (reading starts left)",
            "vertical": "bottom ~32% of original crop (reading starts bottom)",
            "look_for": "circle-with-slash Ø glyph, lone R, or ± in this image",
        },
    },
    "zone_body": {
        "module": "symbol_regions.split_symbol_zones",
        "use_case": (
            "Main numeric strip — nominal dimension and tolerance digits (e.g. "
            "215.37±0.05). ± template matching also runs here."
        ),
        "values": {
            "horizontal": "right ~72% after prefix strip",
            "vertical": "top ~68% above prefix strip",
            "look_for": "digits, decimal point, ± symbol",
        },
    },
    "prefix_enlarged": {
        "module": "symbol_regions.enlarge_zone",
        "use_case": "Prefix strip upscaled 3.5× before prefix OCR (tiny Ø reads better).",
        "values": {"scale": "3.5× Lanczos resize of zone_prefix"},
    },
    "prefix_clahe": {
        "module": "image_preprocess.clahe_rgb",
        "use_case": "Contrast-enhanced prefix fed to PaddleOCR for single-glyph read.",
        "values": {"input": "prefix_enlarged", "output": "text in symbol_vision.prefix_ocr"},
    },
    "symbol_vision": {
        "module": "symbol_vision.detect_symbols + phi_detector + prefix OCR",
        "use_case": (
            "Pixel + OCR symbol detection: Ø (phi), ±, °, R. Merges vision scores "
            "with prefix_ocr hints. Drives whether compose adds Ø/±."
        ),
        "values": {
            "phi_detected": "true when diameter_score ≥ 0.32",
            "phi_score": "0..1 confidence from circle+slash template/Hough (best strip)",
            "phi_strips": "per-strip scores (vertical_bottom/top, oriented_*_prefix)",
            "pm_scores": "± template match scores on prefix vs body (threshold 0.34)",
            "degree_detected": "small ring in upper band without slash (° not Ø)",
            "prefix_ocr": "Paddle read of prefix_clahe only (often 'O', '0', or empty)",
            "prefix_hints": "symbols inferred from prefix_ocr text",
            "symbols_before_merge": "vision-only result",
            "symbols_merged": "final symbols passed to dimension_compose",
        },
    },
    "paddle_candidates": {
        "module": "ocr_pipeline.recognize_paddle",
        "use_case": (
            "Every preprocess variant × det pass. Shows raw Paddle text, "
            "confusable fixes, normalized form, and vote score."
        ),
        "values": {
            "candidates[].variant": "e.g. ccw_clahe = rotated CCW + CLAHE",
            "candidates[].raw": "Paddle output before fixes",
            "candidates[].normalized": "after comma/± glue + confusable fixes",
            "candidates[].score": "digit_quality_score — highest wins",
            "groups": "vote totals per normalized string",
        },
    },
    "paddle_winner": {
        "module": "ocr_select.digit_quality_score",
        "use_case": "Chosen OCR string from all candidates (becomes raw_ocr).",
        "values": {
            "text": "winning normalized string",
            "agreement": "fraction of passes that agreed with winner",
            "corrected": "true if letter→digit confusable fix applied",
        },
    },
    "compose_input": {
        "module": "dimension_compose.compose_engineering_dimension",
        "use_case": "Inputs to final string assembly: raw OCR + symbols + prefix OCR.",
        "values": {
            "raw_ocr": "paddle_winner.text",
            "prefix_ocr": "symbol_vision.prefix_ocr",
            "symbols": "symbol_vision.symbols_merged",
            "vertical": "enables vertical_dia_tol rule (Ø + NN.NN±0.0N)",
        },
    },
    "compose_output": {
        "module": "dimension_compose",
        "use_case": "Final engineering string and which rules fired.",
        "values": {
            "text": "value shown in UI",
            "kind": "diameter | tolerance | linear | angle | radius",
            "applied": "e.g. cad_digit_fix, vertical_dia_tol, phi_prefix, explicit_pm",
        },
    },
    "text_bbox": {
        "module": "ocr_pipeline._detect_text_bbox",
        "use_case": "Tight box around detected text in original crop coords (balloon snap).",
        "values": {"x,y,width,height": "pixels relative to input.png"},
    },
    "result": {
        "module": "ocr_pipeline.recognize",
        "use_case": "Full API JSON returned to the browser.",
        "values": {
            "text": "final dimension",
            "raw_ocr": "before compose",
            "compose_steps": "same as compose_output.applied",
            "symbols_detected": "same as symbol_vision.symbols_merged",
        },
    },
}

_VARIANT_RE = re.compile(r"^variant_(.+)$")


def _variant_info(suffix: str) -> dict[str, Any]:
    parts = suffix.split("_", 1)
    orient = parts[0] if len(parts) > 1 else suffix
    filt = parts[1] if len(parts) > 1 else "raw"
    orient_map = {
        "ccw": "vertical crop rotated 90° CCW",
        "cw": "vertical crop rotated 90° CW",
        "h": "horizontal (no rotation)",
    }
    filt_map = {
        "clahe": "CAD blue ink + CLAHE contrast",
        "sharp": "CAD ink + sharpen",
        "cad": "CAD blue/cyan ink extraction only",
    }
    return {
        "module": "image_preprocess.prepare_ocr_variants",
        "use_case": "One OCR preprocess pass fed to Paddle (multi-pass voting).",
        "values": {
            "orientation": orient_map.get(orient, orient),
            "filter": filt_map.get(filt, filt),
            "orientation_code": orient,
            "filter_code": filt,
        },
    }


def lookup_step(name: str) -> dict[str, Any]:
    """Resolve catalog entry for a step name (supports variant_* patterns)."""
    if name in STEP_CATALOG:
        return dict(STEP_CATALOG[name])

    m = _VARIANT_RE.match(name)
    if m:
        return _variant_info(m.group(1))

    return {
        "module": "unknown",
        "use_case": "Pipeline artifact",
        "values": {},
    }


GLOSSARY: dict[str, str] = {
    "zone_prefix": (
        "Crop slice at the **start of reading order** where engineering symbols live. "
        "Horizontal: left edge. Vertical: bottom edge (text reads bottom-to-top). "
        "Purpose: find Ø/phi without digit noise."
    ),
    "zone_body": (
        "Crop slice with the **main dimension numbers** after the symbol strip. "
        "Purpose: OCR the nominal value and tolerance; secondary ± detection."
    ),
    "symbol_vision": (
        "Combined **OpenCV + Paddle** symbol pass on the crop. Detects Ø (phi) via "
        "circle/slash matching on multiple strips, ± via template match on prefix/body, "
        "° via small-ring heuristic. Merges with prefix_ocr. Output feeds "
        "dimension_compose (whether to prepend Ø, format ±)."
    ),
}
