"""Native PDF text evidence for the auto-balloon OCR pipeline.

The browser still renders the drawing page to an image and the existing raster
detector still owns candidate geometry.  When the original source is a digital
PDF, this module extracts its positioned text into those same rendered-page
coordinates.  A candidate may then use a complete native reading directly or
retain it beside OCR as an independently sourced hypothesis.

No page rotation, deskew, or unwarping is performed here.  The only direction
recorded is the local writing direction already stored in the PDF text line.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import math
import re
from statistics import median
from typing import Any, Callable, Iterable, Literal, Mapping


SourceProfileKind = Literal["vector", "raster", "hybrid"]
RecognitionSource = Literal["native_pdf", "ocr", "native_pdf+ocr"]

_MAX_NATIVE_SPANS = 20_000
_SIGNIFICANT_RASTER_COVERAGE = 0.15
_MIN_SPAN_COVERAGE = 0.55
_BAD_NATIVE_GLYPHS = {"\ufffd", "\x00"}
_ENGINEERING_MARKERS = set("Ø⌀∅Φφ±°º˚×xX")


def _clean_text(value: object) -> str:
    text = str(value or "").replace("\u00a0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _finite_box(box: Mapping[str, object]) -> dict[str, float] | None:
    try:
        values = {
            key: float(box[key])
            for key in ("x", "y", "width", "height")
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    if values["width"] <= 0 or values["height"] <= 0:
        return None
    return values


def _intersection_area(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    x0 = max(float(left["x"]), float(right["x"]))
    y0 = max(float(left["y"]), float(right["y"]))
    x1 = min(
        float(left["x"]) + float(left["width"]),
        float(right["x"]) + float(right["width"]),
    )
    y1 = min(
        float(left["y"]) + float(left["height"]),
        float(right["y"]) + float(right["height"]),
    )
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _union_box(boxes: Iterable[Mapping[str, float]]) -> dict[str, float]:
    items = list(boxes)
    x0 = min(float(box["x"]) for box in items)
    y0 = min(float(box["y"]) for box in items)
    x1 = max(float(box["x"]) + float(box["width"]) for box in items)
    y1 = max(float(box["y"]) + float(box["height"]) for box in items)
    return {
        "x": round(x0, 1),
        "y": round(y0, 1),
        "width": round(x1 - x0, 1),
        "height": round(y1 - y0, 1),
    }


def _normalise_direction(value: object) -> tuple[float, float]:
    try:
        x, y = value  # type: ignore[misc]
        dx, dy = float(x), float(y)
    except (TypeError, ValueError):
        return (1.0, 0.0)
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length < 1e-6:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def _orientation(direction: tuple[float, float]) -> str:
    dx, dy = direction
    if abs(dy) <= 0.25:
        return "horizontal"
    if abs(dx) <= 0.25:
        return "vertical"
    return "rotated"


@dataclass(frozen=True)
class NativePdfSpan:
    """One positioned text span extracted from a PDF content stream."""

    span_id: str
    text: str
    bbox: dict[str, float]
    direction: tuple[float, float] = (1.0, 0.0)
    rotation: float = 0.0
    font_size: float = 0.0

    @property
    def orientation(self) -> str:
        return _orientation(self.direction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.span_id,
            "text": self.text,
            "bbox": dict(self.bbox),
            "direction": [self.direction[0], self.direction[1]],
            "rotation": self.rotation,
            "orientation": self.orientation,
            "font_size": self.font_size,
        }


@dataclass(frozen=True)
class PdfPageEvidence:
    """Native text and page-level acquisition facts for one rendered page."""

    page_number: int
    page_count: int
    target_width: int
    target_height: int
    profile: SourceProfileKind
    spans: tuple[NativePdfSpan, ...] = ()
    native_character_count: int = 0
    raster_coverage: float = 0.0
    vector_object_count: int = 0
    extraction_error: str | None = None

    @property
    def native_text_available(self) -> bool:
        return bool(self.spans and self.native_character_count)

    def profile_dict(self) -> dict[str, Any]:
        return {
            "kind": self.profile,
            "page": self.page_number,
            "page_count": self.page_count,
            "native_text_available": self.native_text_available,
            "native_span_count": len(self.spans),
            "native_character_count": self.native_character_count,
            "raster_coverage": round(self.raster_coverage, 4),
            "vector_object_count": self.vector_object_count,
            **(
                {"fallback_reason": self.extraction_error}
                if self.extraction_error
                else {}
            ),
        }

    @classmethod
    def raster_fallback(
        cls,
        *,
        page_number: int,
        target_size: tuple[int, int],
        error: str | None = None,
    ) -> "PdfPageEvidence":
        return cls(
            page_number=page_number,
            page_count=0,
            target_width=int(target_size[0]),
            target_height=int(target_size[1]),
            profile="raster",
            extraction_error=error,
        )


@dataclass(frozen=True)
class NativeCandidateMatch:
    """Native spans that are geometrically owned by one raster candidate."""

    text: str
    spans: tuple[NativePdfSpan, ...]
    bbox: dict[str, float]
    coverage: float
    unexplained_margin_ratio: float
    internal_gap_ratio: float
    direction: tuple[float, float]
    orientation: str
    rotation: float
    has_bad_glyphs: bool

    @property
    def span_ids(self) -> tuple[str, ...]:
        return tuple(span.span_id for span in self.spans)


def extract_pdf_page_evidence(
    pdf_bytes: bytes,
    *,
    page_number: int,
    target_size: tuple[int, int],
) -> PdfPageEvidence:
    """Extract one PDF page and map its spans into rendered-image pixels."""

    if not pdf_bytes:
        raise ValueError("The source PDF is empty")
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    if target_width < 1 or target_height < 1:
        raise ValueError("The target page size must be positive")

    # Imported lazily so an image-only deployment can still start and use the
    # existing OCR route if its environment has not yet installed PyMuPDF.
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised in deployment
        raise RuntimeError("PyMuPDF is not installed") from exc

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs cannot provide native text")
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("The requested PDF page does not exist")

        page = document.load_page(page_number - 1)
        page_rect = page.rect
        if page_rect.width <= 0 or page_rect.height <= 0:
            raise ValueError("The PDF page has invalid dimensions")
        scale_x = target_width / float(page_rect.width)
        scale_y = target_height / float(page_rect.height)

        flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE
        extracted = page.get_text("dict", flags=flags)
        spans: list[NativePdfSpan] = []
        span_index = 0
        for block in extracted.get("blocks", ()):  # image blocks have no lines
            for line in block.get("lines", ()):
                raw_direction = _normalise_direction(line.get("dir", (1.0, 0.0)))
                mapped_direction = _normalise_direction(
                    (raw_direction[0] * scale_x, raw_direction[1] * scale_y)
                )
                rotation = round(
                    math.degrees(
                        math.atan2(mapped_direction[1], mapped_direction[0])
                    ),
                    2,
                )
                for raw_span in line.get("spans", ()):
                    text = _clean_text(raw_span.get("text"))
                    if not text:
                        continue
                    raw_bbox = raw_span.get("bbox")
                    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
                        continue
                    try:
                        x0, y0, x1, y1 = (float(value) for value in raw_bbox[:4])
                    except (TypeError, ValueError):
                        continue
                    x0 = max(float(page_rect.x0), min(float(page_rect.x1), x0))
                    y0 = max(float(page_rect.y0), min(float(page_rect.y1), y0))
                    x1 = max(float(page_rect.x0), min(float(page_rect.x1), x1))
                    y1 = max(float(page_rect.y0), min(float(page_rect.y1), y1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    span_index += 1
                    spans.append(
                        NativePdfSpan(
                            span_id=f"P{page_number}:S{span_index:05d}",
                            text=text,
                            bbox={
                                "x": round((x0 - page_rect.x0) * scale_x, 1),
                                "y": round((y0 - page_rect.y0) * scale_y, 1),
                                "width": round((x1 - x0) * scale_x, 1),
                                "height": round((y1 - y0) * scale_y, 1),
                            },
                            direction=mapped_direction,
                            rotation=rotation,
                            font_size=round(
                                float(raw_span.get("size") or 0.0)
                                * (scale_x + scale_y)
                                / 2,
                                2,
                            ),
                        )
                    )
                    if len(spans) >= _MAX_NATIVE_SPANS:
                        break
                if len(spans) >= _MAX_NATIVE_SPANS:
                    break
            if len(spans) >= _MAX_NATIVE_SPANS:
                break

        page_area = max(float(page_rect.width * page_rect.height), 1.0)
        raster_area = 0.0
        try:
            for image_info in page.get_image_info(xrefs=True):
                image_bbox = image_info.get("bbox")
                if not isinstance(image_bbox, (list, tuple)) or len(image_bbox) < 4:
                    continue
                image_rect = fitz.Rect(image_bbox) & page_rect
                if image_rect.is_empty:
                    continue
                raster_area += max(0.0, float(image_rect.get_area()))
        except (AttributeError, RuntimeError, ValueError):
            raster_area = 0.0
        raster_coverage = min(1.0, raster_area / page_area)

        try:
            vector_object_count = len(page.get_drawings())
        except (RuntimeError, ValueError):
            vector_object_count = 0

        native_character_count = sum(
            1 for span in spans for char in span.text if not char.isspace()
        )
        has_vector_content = bool(spans or vector_object_count)
        has_significant_raster = (
            raster_coverage >= _SIGNIFICANT_RASTER_COVERAGE
        )
        if has_vector_content and has_significant_raster:
            profile: SourceProfileKind = "hybrid"
        elif has_vector_content:
            profile = "vector"
        else:
            profile = "raster"

        return PdfPageEvidence(
            page_number=page_number,
            page_count=document.page_count,
            target_width=target_width,
            target_height=target_height,
            profile=profile,
            spans=tuple(spans),
            native_character_count=native_character_count,
            raster_coverage=raster_coverage,
            vector_object_count=vector_object_count,
        )


def crop_pdf_page_evidence(
    evidence: PdfPageEvidence,
    bbox: Mapping[str, float],
) -> PdfPageEvidence:
    """Clip page evidence to a section and translate it into section pixels."""

    scope = _finite_box(bbox)
    if scope is None:
        raise ValueError("The PDF evidence crop has invalid bounds")
    clipped: list[NativePdfSpan] = []
    scope_x1 = scope["x"] + scope["width"]
    scope_y1 = scope["y"] + scope["height"]
    for span in evidence.spans:
        span_area = max(
            1.0,
            float(span.bbox["width"]) * float(span.bbox["height"]),
        )
        span_x1 = span.bbox["x"] + span.bbox["width"]
        span_y1 = span.bbox["y"] + span.bbox["height"]
        x0 = max(scope["x"], span.bbox["x"])
        y0 = max(scope["y"], span.bbox["y"])
        x1 = min(scope_x1, span_x1)
        y1 = min(scope_y1, span_y1)
        if x1 <= x0 or y1 <= y0:
            continue
        # A PDF span carries one complete text string. If a user-drawn section
        # clips that span, retaining the full string would let a small visible
        # fragment inherit characters outside the selection. Leave it to OCR.
        if ((x1 - x0) * (y1 - y0)) / span_area < 0.98:
            continue
        clipped.append(
            replace(
                span,
                bbox={
                    "x": round(x0 - scope["x"], 1),
                    "y": round(y0 - scope["y"], 1),
                    "width": round(x1 - x0, 1),
                    "height": round(y1 - y0, 1),
                },
            )
        )
    return replace(
        evidence,
        target_width=max(1, int(round(scope["width"]))),
        target_height=max(1, int(round(scope["height"]))),
        spans=tuple(clipped),
        native_character_count=sum(
            1 for span in clipped for char in span.text if not char.isspace()
        ),
    )


def _projected_interval(
    bbox: Mapping[str, float], direction: tuple[float, float]
) -> tuple[float, float]:
    x0 = float(bbox["x"])
    y0 = float(bbox["y"])
    x1 = x0 + float(bbox["width"])
    y1 = y0 + float(bbox["height"])
    dx, dy = direction
    values = (
        x0 * dx + y0 * dy,
        x1 * dx + y0 * dy,
        x1 * dx + y1 * dy,
        x0 * dx + y1 * dy,
    )
    return (min(values), max(values))


def _perpendicular_extent(
    bbox: Mapping[str, float], direction: tuple[float, float]
) -> float:
    perpendicular = (-direction[1], direction[0])
    start, end = _projected_interval(bbox, perpendicular)
    return max(1.0, end - start)


def match_native_spans(
    candidate_bbox: Mapping[str, float],
    spans: Iterable[NativePdfSpan],
    *,
    minimum_span_coverage: float = _MIN_SPAN_COVERAGE,
) -> NativeCandidateMatch | None:
    """Find spans substantially contained by one detector candidate.

    Requiring the candidate to cover most of each span is important.  A small
    raster fragment inside a larger native span (for example the isolated ``5``
    inside vertical ``Rz 12.5``) must not inherit the complete native callout.
    """

    candidate = _finite_box(candidate_bbox)
    if candidate is None:
        return None
    selected: list[tuple[NativePdfSpan, float]] = []
    for span in spans:
        span_box = _finite_box(span.bbox)
        if span_box is None:
            continue
        span_area = span_box["width"] * span_box["height"]
        intersection = _intersection_area(candidate, span_box)
        coverage = intersection / max(span_area, 1.0)
        if coverage >= minimum_span_coverage:
            selected.append((span, coverage))
    if not selected:
        return None

    # Use the direction supported by the most characters, not merely the first
    # span returned by the PDF library.
    direction_weights: dict[tuple[float, float], int] = {}
    for span, _coverage in selected:
        key = (round(span.direction[0], 2), round(span.direction[1], 2))
        direction_weights[key] = direction_weights.get(key, 0) + max(
            1, len(span.text.strip())
        )
    dominant_key = max(direction_weights, key=direction_weights.get)
    dominant = _normalise_direction(dominant_key)

    ordered = sorted(
        (span for span, _coverage in selected),
        key=lambda span: (
            _projected_interval(span.bbox, dominant)[0],
            _projected_interval(
                span.bbox, (-dominant[1], dominant[0])
            )[0],
        ),
    )
    line_sizes = [
        _perpendicular_extent(span.bbox, dominant) for span in ordered
    ]
    line_size = max(1.0, median(line_sizes))

    parts: list[str] = []
    maximum_gap = 0.0
    previous_end: float | None = None
    for span in ordered:
        current_start, current_end = _projected_interval(span.bbox, dominant)
        gap = max(0.0, current_start - previous_end) if previous_end is not None else 0.0
        maximum_gap = max(maximum_gap, gap)
        cleaned = _clean_text(span.text)
        if not cleaned:
            previous_end = current_end
            continue
        if parts and gap > 0.45 * line_size:
            parts.append(" ")
        parts.append(cleaned)
        previous_end = current_end
    text = "".join(parts).strip()
    if not text:
        return None

    native_bbox = _union_box(span.bbox for span in ordered)
    candidate_start, candidate_end = _projected_interval(candidate, dominant)
    native_start, native_end = _projected_interval(native_bbox, dominant)
    unexplained_margin_ratio = max(
        max(0.0, native_start - candidate_start),
        max(0.0, candidate_end - native_end),
    ) / line_size
    weighted_coverage = sum(
        coverage * max(1.0, span.bbox["width"] * span.bbox["height"])
        for span, coverage in selected
    ) / sum(
        max(1.0, span.bbox["width"] * span.bbox["height"])
        for span, _coverage in selected
    )
    has_bad_glyphs = any(
        bad in text for bad in _BAD_NATIVE_GLYPHS
    ) or any(ord(char) < 32 and not char.isspace() for char in text)

    return NativeCandidateMatch(
        text=text,
        spans=tuple(ordered),
        bbox=native_bbox,
        coverage=round(weighted_coverage, 4),
        unexplained_margin_ratio=round(unexplained_margin_ratio, 4),
        internal_gap_ratio=round(maximum_gap / line_size, 4),
        direction=dominant,
        orientation=_orientation(dominant),
        rotation=round(math.degrees(math.atan2(dominant[1], dominant[0])), 2),
        has_bad_glyphs=has_bad_glyphs,
    )


def native_match_is_authoritative(
    match: NativeCandidateMatch | None,
    *,
    validator: Callable[[str], bool],
) -> bool:
    """Whether native text is complete enough to skip OCR for this object."""

    if match is None or match.has_bad_glyphs:
        return False
    if match.coverage < 0.82:
        return False
    # Parentheses can conceal a custom-font degree glyph: the benchmark PDF's
    # ``(67.1°)`` text layer contains only ``(67.1)`` and its closing bracket
    # makes the geometry look complete. OCR must independently verify these.
    if "(" in match.text or ")" in match.text:
        return False
    # Large blank space before/after/between extracted spans usually means the
    # PDF font omitted a diameter, degree, multiplication, or GD&T symbol.
    if match.unexplained_margin_ratio > 0.60:
        return False
    if match.internal_gap_ratio > 0.60:
        return False
    return validator(match.text)


def _canonical(value: str) -> str:
    translations = str.maketrans(
        {
            "⌀": "Ø",
            "∅": "Ø",
            "ø": "Ø",
            "Φ": "Ø",
            "φ": "Ø",
            "º": "°",
            "˚": "°",
            "−": "-",
            "–": "-",
            "—": "-",
        }
    )
    return re.sub(r"\s+", "", (value or "").translate(translations).upper())


def _digits(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def _markers(value: str) -> set[str]:
    canonical = _canonical(value)
    markers = {char for char in canonical if char in _ENGINEERING_MARKERS}
    if canonical.startswith("R"):
        markers.add("R")
    if "RZ" in canonical:
        markers.add("RZ")
    return markers


def _equivalent_readings(native_text: str, ocr_text: str) -> bool:
    native = _canonical(native_text)
    ocr = _canonical(ocr_text)
    if not native or not ocr:
        return False
    if native == ocr:
        return True
    # A custom PDF font often omits a symbol while leaving every digit in the
    # correct order.  Treat that as equivalent only when the two readings also
    # differ in engineering-marker evidence; ordinary decimal disagreements do
    # not get hidden by a digits-only comparison.
    return (
        bool(_digits(native))
        and _digits(native) == _digits(ocr)
        and _markers(native) != _markers(ocr)
    )


def _merge_equivalent_readings(native_text: str, ocr_text: str) -> str:
    native = _clean_text(native_text)
    ocr = _clean_text(ocr_text)
    native_canonical = _canonical(native)
    ocr_canonical = _canonical(ocr)
    merged = native

    if ocr_canonical.startswith("Ø") and not native_canonical.startswith("Ø"):
        merged = "Ø" + merged.lstrip()
    elif ocr_canonical.startswith("R") and not native_canonical.startswith("R"):
        merged = "R" + merged.lstrip()

    if "×" in ocr_canonical and "×" not in _canonical(merged):
        # OCR owns the missing multiplication sign's exact position.
        merged = ocr
    if "°" in ocr_canonical and "°" not in _canonical(merged):
        if re.search(r"[±+-]", merged):
            merged = re.sub(r"(?=\s*[±+-])", "°", merged, count=1)
        else:
            merged = merged.rstrip() + "°"
    return _clean_text(merged)


def _evidence_payload(
    match: NativeCandidateMatch,
    *,
    ocr_text: str,
    selected_source: RecognitionSource,
    agreement: float,
    conflict: bool,
) -> dict[str, Any]:
    return {
        "selected_source": selected_source,
        "sources": [
            "native_pdf",
            *(["ocr"] if ocr_text else []),
        ],
        "native_text": match.text,
        "ocr_text": ocr_text,
        "agreement": round(agreement, 4),
        "conflict": conflict,
        "native_span_ids": list(match.span_ids),
        "native_bbox": dict(match.bbox),
    }


def native_result(match: NativeCandidateMatch) -> dict[str, Any]:
    """Build a recognition result when complete native text needs no OCR."""

    return {
        "text": match.text,
        "raw_ocr": "",
        "confidence": 1.0,
        "agreement": 1.0,
        "needs_review": False,
        "type": None,
        "orientation": match.orientation,
        "orientation_confidence": 1.0,
        "rotation": match.rotation,
        "engine": "native_pdf",
        "symbols_detected": None,
        "ocr_profile": "native_pdf",
        "recognition_source": "native_pdf",
        "native_authoritative": True,
        "recognition_evidence": _evidence_payload(
            match,
            ocr_text="",
            selected_source="native_pdf",
            agreement=1.0,
            conflict=False,
        ),
    }


def fuse_native_with_ocr(
    match: NativeCandidateMatch,
    ocr_result: Mapping[str, Any],
    *,
    native_authoritative: bool,
    final: bool = False,
) -> dict[str, Any]:
    """Retain both hypotheses and select a safe published reading."""

    result = dict(ocr_result)
    ocr_text = _clean_text(result.get("text") or result.get("raw_ocr"))
    native_text = match.text
    canonical_native = _canonical(native_text)
    canonical_ocr = _canonical(ocr_text)
    agreement = (
        SequenceMatcher(None, canonical_native, canonical_ocr).ratio()
        if canonical_native and canonical_ocr
        else 0.0
    )

    if not ocr_text:
        chosen = native_text
        selected_source: RecognitionSource = "native_pdf"
        conflict = False
    elif _equivalent_readings(native_text, ocr_text):
        chosen = _merge_equivalent_readings(native_text, ocr_text)
        selected_source = "native_pdf+ocr"
        conflict = False
    else:
        chosen = native_text if native_authoritative else ocr_text
        selected_source = "native_pdf" if native_authoritative else "ocr"
        conflict = bool(canonical_native and canonical_ocr)

    result["text"] = chosen
    result["recognition_source"] = selected_source
    result["native_authoritative"] = native_authoritative
    result["source_conflict"] = conflict
    result["recognition_evidence"] = _evidence_payload(
        match,
        ocr_text=ocr_text,
        selected_source=selected_source,
        agreement=agreement,
        conflict=conflict,
    )
    if selected_source == "native_pdf":
        result["confidence"] = 1.0
        result["orientation"] = match.orientation
        result["orientation_confidence"] = 1.0
        result["rotation"] = match.rotation
    if conflict:
        result["needs_review"] = True
        result["review_reason"] = "Native PDF text and OCR disagree"
        if final:
            result["authoritative_review_required"] = True
    return result


def ocr_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    """Add explicit provenance to a result with no matching native span."""

    enriched = dict(result)
    text = _clean_text(enriched.get("text") or enriched.get("raw_ocr"))
    enriched["recognition_source"] = "ocr"
    enriched["recognition_evidence"] = {
        "selected_source": "ocr",
        "sources": ["ocr"],
        "native_text": "",
        "ocr_text": text,
        "agreement": 0.0,
        "conflict": False,
        "native_span_ids": [],
    }
    return enriched
