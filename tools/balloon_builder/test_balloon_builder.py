from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from PIL import Image, ImageDraw

from tools.balloon_builder.balloon_builder import (
    BalloonStyleError,
    build_style_payload,
    load_style_file,
    remove_edge_connected_background,
    validate_style_payload,
    write_style_file,
)


class BalloonBuilderTests(unittest.TestCase):
    def test_edge_background_removal_preserves_enclosed_white_area(self) -> None:
        image = Image.new("RGB", (20, 20), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 15, 15), outline="black", width=2)

        result = remove_edge_connected_background(image, tolerance=5)

        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertEqual(result.getpixel((10, 10)), (255, 255, 255, 255))

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


if __name__ == "__main__":
    unittest.main()
