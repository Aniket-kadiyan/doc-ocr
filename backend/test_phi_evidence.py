from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

import phi_detector
from dimension_compose import compose_engineering_dimension
from ocr_pipeline import OCR_CHANGESET_ID, OcrPipeline
from phi_detector import (
    _fuse_phi_scores,
    _prepare_phi_gray,
    detect_phi_multi_strip,
)
from symbol_vision import (
    DetectedSymbols,
    detect_diameter_symbol,
    detect_prefix_from_ocr_text,
    merge_symbol_scores,
)


class PhiEvidenceRulesTests(unittest.TestCase):
    def test_runtime_changeset_identifies_this_package(self) -> None:
        self.assertEqual(OCR_CHANGESET_ID, "m1-p02r1-field-gated-phi")

    def test_explicit_diameter_glyphs_are_decisive(self) -> None:
        for text in ("Ø20", "ø20", "φ20", "Φ20", "⌀20", "∅20"):
            with self.subTest(text=text):
                symbols = detect_prefix_from_ocr_text(text)
                self.assertTrue(symbols.diameter)
                self.assertTrue(symbols.diameter_ocr_explicit)

    def test_gdt_position_symbol_is_not_diameter(self) -> None:
        symbols = detect_prefix_from_ocr_text("⊕20")
        self.assertFalse(symbols.diameter)
        result = compose_engineering_dimension("⊕20", None, symbols)
        self.assertFalse(result.text.startswith("Ø"))
        self.assertNotEqual(result.kind, "diameter")

    def test_o_like_ocr_is_ambiguous_not_diameter(self) -> None:
        for text in ("O20", "020", "Q20", "C20", "O", "0"):
            with self.subTest(text=text):
                symbols = detect_prefix_from_ocr_text(text)
                self.assertFalse(symbols.diameter)
                self.assertTrue(symbols.diameter_ocr_ambiguous)

    def test_ordinary_decimal_is_not_an_ambiguous_prefix(self) -> None:
        symbols = detect_prefix_from_ocr_text("0.5")
        self.assertFalse(symbols.diameter)
        self.assertFalse(symbols.diameter_ocr_ambiguous)

    def test_score_alone_never_becomes_diameter_during_merge(self) -> None:
        visual = DetectedSymbols(
            diameter=False,
            diameter_score=0.99,
            diameter_visual_score=0.99,
            diameter_visual_candidate=False,
        )
        merged = merge_symbol_scores(
            visual,
            detect_prefix_from_ocr_text("O20"),
        )
        self.assertFalse(merged.diameter)
        self.assertEqual(merged.diameter_visual_score, 0.99)
        self.assertFalse(merged.diameter_visual_candidate)
        self.assertTrue(merged.diameter_ocr_ambiguous)

    def test_visual_confirmation_survives_merge(self) -> None:
        visual = DetectedSymbols(
            diameter=True,
            diameter_score=0.91,
            diameter_visual_score=0.91,
        )
        merged = merge_symbol_scores(
            visual,
            detect_prefix_from_ocr_text("20H10"),
        )
        self.assertTrue(merged.diameter)

    def test_closed_round_shape_without_diagonal_is_rejected(self) -> None:
        evidence = _fuse_phi_scores(0.70, 0.90, 0.85, 1.0, 0.30)
        self.assertFalse(evidence["detected"])

    def test_diagonal_without_supported_shape_is_rejected(self) -> None:
        evidence = _fuse_phi_scores(0.30, 0.90, 0.20, 0.95, 0.95)
        self.assertFalse(evidence["detected"])

    def test_closed_ring_diagonal_and_shape_confirm_phi(self) -> None:
        evidence = _fuse_phi_scores(
            0.40,
            0.85,
            0.80,
            0.95,
            0.90,
            component_score=0.94,
        )
        self.assertTrue(evidence["detected"])

    def test_partial_legacy_agreement_without_glyph_topology_is_rejected(self) -> None:
        evidence = _fuse_phi_scores(0.40, 0.85, 0.80, 0.95, 0.90)
        self.assertFalse(evidence["detected"])

    def test_phi_gray_has_dark_ink_on_white_paper(self) -> None:
        image = Image.new("RGB", (24, 24), "white")
        ImageDraw.Draw(image).line((4, 4, 19, 19), fill="black", width=3)
        gray = _prepare_phi_gray(image)
        self.assertLess(int(gray[10, 10]), int(gray[0, 0]))


class PhiCompositionTests(unittest.TestCase):
    def test_vertical_tolerance_does_not_invent_diameter(self) -> None:
        vertical_crop = Image.new("RGB", (30, 120), "white")
        result = compose_engineering_dimension(
            "215.37±0.05",
            vertical_crop,
            DetectedSymbols(),
        )
        self.assertEqual(result.text, "215.37±0.05")
        self.assertEqual(result.kind, "tolerance")
        self.assertNotIn("vertical_dia_tol", result.applied)

    def test_high_unconfirmed_score_does_not_invent_diameter(self) -> None:
        result = compose_engineering_dimension(
            "20H10",
            None,
            DetectedSymbols(
                diameter=False,
                diameter_score=0.99,
                diameter_visual_score=0.99,
            ),
        )
        self.assertEqual(result.text, "20H10")
        self.assertEqual(result.kind, "linear")

    def test_confirmed_visual_phi_recovers_leading_zero_misread(self) -> None:
        result = compose_engineering_dimension(
            "020H10",
            None,
            DetectedSymbols(
                diameter=True,
                diameter_score=0.91,
                diameter_visual_score=0.91,
            ),
        )
        self.assertEqual(result.text, "Ø20H10")
        self.assertEqual(result.kind, "diameter")

    def test_explicit_empty_set_ocr_is_normalized_to_diameter(self) -> None:
        result = compose_engineering_dimension(
            "∅20H10",
            None,
            DetectedSymbols(),
        )
        self.assertEqual(result.text, "Ø20H10")
        self.assertEqual(result.kind, "diameter")

    def test_radius_text_wins_over_visual_diameter(self) -> None:
        result = compose_engineering_dimension(
            "R20",
            None,
            DetectedSymbols(diameter=True, diameter_score=0.95),
        )
        self.assertEqual(result.text, "R20")
        self.assertEqual(result.kind, "radius")


class PageBatchPhiTests(unittest.TestCase):
    @staticmethod
    def _pipeline_for_text(raw_text: str) -> OcrPipeline:
        pipeline = OcrPipeline.__new__(OcrPipeline)
        pipeline._page_batch_recognition_available = True
        pipeline._predict_text_orientations = lambda images, batch_size: [
            (0, 1.0) for _ in images
        ]
        pipeline._predict_text_recognition = lambda images, batch_size: [
            (raw_text, 0.99) for _ in images
        ]
        return pipeline

    def test_page_batch_uses_visual_phi_evidence(self) -> None:
        pipeline = self._pipeline_for_text("20H10")
        confirmed = DetectedSymbols(
            diameter=True,
            diameter_score=0.92,
            diameter_visual_score=0.92,
        )
        with patch(
            "ocr_pipeline.detect_diameter_symbol",
            return_value=(confirmed, {}),
        ) as detector:
            result = pipeline._recognize_page_batch(
                [Image.new("RGB", (100, 40), "white")],
                batch_size=1,
            )[0]
        detector.assert_called_once()
        self.assertEqual(result["text"], "Ø20H10")
        self.assertTrue(result["symbols_detected"]["diameter"])

    def test_page_batch_rejects_ambiguous_ocr_and_score_only(self) -> None:
        pipeline = self._pipeline_for_text("O20H10")
        unconfirmed = DetectedSymbols(
            diameter=False,
            diameter_score=0.99,
            diameter_visual_score=0.99,
        )
        with patch(
            "ocr_pipeline.detect_diameter_symbol",
            return_value=(unconfirmed, {}),
        ):
            result = pipeline._recognize_page_batch(
                [Image.new("RGB", (100, 40), "white")],
                batch_size=1,
            )[0]
        self.assertEqual(result["text"], "020H10")
        self.assertFalse(result["symbols_detected"]["diameter"])


class RenderedPhiVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if phi_detector.cv2 is None:
            raise unittest.SkipTest("OpenCV is not installed")
        candidates = (
            "DejaVuSans.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        for candidate in candidates:
            try:
                cls.font = ImageFont.truetype(candidate, 42)
                break
            except OSError:
                continue
        else:
            raise unittest.SkipTest("No suitable test font is installed")

    @classmethod
    def _render(cls, text: str) -> Image.Image:
        image = Image.new("RGB", (130, 64), "white")
        ImageDraw.Draw(image).text((3, 5), text, font=cls.font, fill="black")
        return image

    def test_phi_survives_horizontal_and_vertical_orientation(self) -> None:
        horizontal = self._render("Ø20")
        variants = (
            (horizontal, False),
            (horizontal.rotate(90, expand=True, fillcolor="white"), True),
            (horizontal.rotate(-90, expand=True, fillcolor="white"), True),
        )
        for image, vertical in variants:
            with self.subTest(vertical=vertical, size=image.size):
                detected, _, _ = detect_phi_multi_strip(image, vertical)
                self.assertTrue(detected)

    def test_round_and_diagonal_confusables_do_not_create_phi(self) -> None:
        for text in ("020", "O20", "820", "R20", "20"):
            horizontal = self._render(text)
            variants = (
                (horizontal, False),
                (horizontal.rotate(90, expand=True, fillcolor="white"), True),
                (horizontal.rotate(-90, expand=True, fillcolor="white"), True),
            )
            for image, vertical in variants:
                with self.subTest(text=text, vertical=vertical, size=image.size):
                    detected, _, _ = detect_phi_multi_strip(image, vertical)
                    self.assertFalse(detected)


class FieldLikePhiVisionTests(unittest.TestCase):
    """Small CAD-style glyphs and linework matching the field failure shape."""

    @classmethod
    def setUpClass(cls) -> None:
        if phi_detector.cv2 is None:
            raise unittest.SkipTest("OpenCV is not installed")
        candidates = (
            "DejaVuSans.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        for candidate in candidates:
            try:
                cls.font = ImageFont.truetype(candidate, 16)
                break
            except OSError:
                continue
        else:
            raise unittest.SkipTest("No suitable test font is installed")

    @classmethod
    def _render(
        cls,
        text: str,
        *,
        dimension_line: bool = False,
        vertical: bool = False,
    ) -> Image.Image:
        image = Image.new("RGB", (150, 50), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 12), text, font=cls.font, fill="black")
        if dimension_line:
            draw.line((0, 42, 149, 42), fill=(0, 0, 255), width=1)
        if vertical:
            return image.rotate(90, expand=True, fillcolor="white")
        return image

    def _detect(self, image: Image.Image) -> tuple[bool, float]:
        detected, score, _details = detect_phi_multi_strip(
            image,
            image.height > image.width * 1.35,
        )
        return detected, score

    def test_four_field_dimension_shapes_keep_phi(self) -> None:
        for text in ("Ø33", "Ø30±0.2", "Ø23.5-0.05", "Ø30-0.2"):
            with self.subTest(text=text):
                detected, score = self._detect(self._render(text))
                self.assertTrue(detected)
                self.assertGreaterEqual(score, 0.82)

    def test_dimension_line_does_not_hide_small_phi(self) -> None:
        detected, _score = self._detect(
            self._render("Ø33-0.05", dimension_line=True)
        )
        self.assertTrue(detected)

    def test_field_shaped_non_diameters_do_not_gain_phi(self) -> None:
        cases = (
            ("R2±0.2", False),
            ("R10±2", False),
            ("70±0.2", True),
            ("114.5±0.05", True),
        )
        for text, vertical in cases:
            with self.subTest(text=text, vertical=vertical):
                detected, _score = self._detect(
                    self._render(text, vertical=vertical)
                )
                self.assertFalse(detected)

    def test_field_shaped_bottom_values_compose_with_phi(self) -> None:
        cases = (
            ("Ø23.5-0.05", "23.5-0.05", "Ø23.5-0.05"),
            ("Ø30-0.2", "30-0.2", "Ø30-0.2"),
        )
        for rendered, raw_text, expected in cases:
            with self.subTest(rendered=rendered):
                image = self._render(rendered)
                symbols, _debug = detect_diameter_symbol(image)
                result = compose_engineering_dimension(raw_text, image, symbols)
                self.assertEqual(result.text, expected)
                self.assertEqual(result.kind, "diameter")


if __name__ == "__main__":
    unittest.main()
