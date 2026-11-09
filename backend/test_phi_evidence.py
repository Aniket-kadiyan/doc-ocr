from __future__ import annotations

import unittest

from PIL import Image, ImageDraw, ImageFont

import phi_detector
from dimension_compose import compose_engineering_dimension
from phi_detector import detect_phi_multi_strip
from symbol_vision import DetectedSymbols, detect_prefix_from_ocr_text


class PhiEvidenceRulesTests(unittest.TestCase):
    def test_explicit_diameter_glyphs_are_decisive(self) -> None:
        for text in ("Ø20", "ø20", "φ20", "Φ20", "⌀20"):
            with self.subTest(text=text):
                symbols = detect_prefix_from_ocr_text(text)
                self.assertTrue(symbols.diameter)

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


class RenderedPhiVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if phi_detector.cv2 is None:
            raise unittest.SkipTest("OpenCV is not installed")
        candidates = (
            "DejaVuSans.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
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
            "/System/Library/Fonts/Supplemental/Arial.ttf",
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
