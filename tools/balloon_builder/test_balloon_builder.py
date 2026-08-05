from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from PIL import Image, ImageDraw

from tools.balloon_builder.balloon_builder import (
    _fit_preview_font,
    BalloonStyleError,
    build_style_payload,
    clear_region,
    load_style_file,
    open_artwork,
    remove_edge_connected_background,
    trim_transparent,
    validate_style_payload,
    write_style_file,
)


class BalloonBuilderTests(unittest.TestCase):
    def test_png_and_jpeg_artwork_open_as_rgba(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".png", ".jpg"):
                path = Path(directory) / f"source{suffix}"
                Image.new("RGB", (12, 8), "orange").save(path)

                opened = open_artwork(path)

                self.assertEqual(opened.mode, "RGBA")
                self.assertEqual(opened.size, (12, 8))

    def test_edge_background_removal_preserves_enclosed_white_area(self) -> None:
        image = Image.new("RGB", (20, 20), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 15, 15), outline="black", width=2)

        result = remove_edge_connected_background(image, tolerance=5)

        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertEqual(result.getpixel((10, 10)), (255, 255, 255, 255))

    def test_trim_and_sample_number_clear_are_deterministic(self) -> None:
        image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 4, 16, 17), fill="black")

        trimmed = trim_transparent(image)
        cleared = clear_region(trimmed, (4, 4, 6, 5), "#ffffff")

        self.assertEqual(trimmed.size, (14, 14))
        self.assertEqual(cleared.getpixel((5, 5)), (255, 255, 255, 255))

    def test_round_trip_style_contains_normalized_geometry(self) -> None:
        image = Image.new("RGBA", (40, 60), (0, 0, 0, 0))
        ImageDraw.Draw(image).ellipse((2, 2, 38, 38), fill="white", outline="black")
        payload = build_style_payload(
            image,
            name="Test marker",
            number_area=(4, 4, 32, 30),
            anchor=(20, 59),
        )
        self.assertEqual(payload["numberArea"]["x"], 0.1)
        self.assertAlmostEqual(payload["anchor"]["y"], 59 / 60, places=6)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "balloon-style.json"
            write_style_file(path, payload)
            loaded, artwork = load_style_file(path)
        self.assertEqual(loaded["name"], "Test marker")
        self.assertEqual(artwork.size, (40, 60))

    def test_validator_rejects_out_of_bounds_number_area(self) -> None:
        image = Image.new("RGBA", (40, 60), (255, 255, 255, 255))
        payload = build_style_payload(
            image,
            name="Test marker",
            number_area=(4, 4, 32, 30),
            anchor=(20, 59),
        )
        payload["numberArea"]["width"] = 0.95
        with self.assertRaises(BalloonStyleError):
            validate_style_payload(payload)

    def test_validator_rejects_corrupt_artwork_and_dimension_mismatch(self) -> None:
        image = Image.new("RGBA", (40, 60), (255, 255, 255, 255))
        payload = build_style_payload(
            image,
            name="Test marker",
            number_area=(4, 4, 32, 30),
            anchor=(20, 59),
        )

        corrupt = dict(payload)
        corrupt["artwork"] = dict(payload["artwork"])
        corrupt["artwork"]["dataUrl"] = "data:image/png;base64,not-valid!"
        with self.assertRaises(BalloonStyleError):
            validate_style_payload(corrupt)

        mismatched = dict(payload)
        mismatched["artwork"] = dict(payload["artwork"])
        mismatched["artwork"]["width"] = 41
        with self.assertRaises(BalloonStyleError):
            validate_style_payload(mismatched)

    def test_preview_font_fits_three_digit_numbers(self) -> None:
        one_digit = _fit_preview_font("1", 22, 24, 7, 13, 2)
        three_digits = _fit_preview_font("128", 22, 24, 7, 13, 2)

        self.assertLess(three_digits, one_digit)
        self.assertGreaterEqual(three_digits, 7)


if __name__ == "__main__":
    unittest.main()
