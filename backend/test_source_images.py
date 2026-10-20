from __future__ import annotations

import io
import unittest

from PIL import Image

from source_images import SourceImageError, detect_raster_format, render_source_page


def encoded_image(format_name: str, *, color: str = "white") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(output, format=format_name)
    return output.getvalue()


class SourceImageTests(unittest.TestCase):
    def test_detects_supported_signatures_when_mime_is_blank(self) -> None:
        self.assertEqual(detect_raster_format(encoded_image("PNG"), "part.png", ""), "png")
        self.assertEqual(detect_raster_format(encoded_image("JPEG"), "part.jpg", ""), "jpeg")
        self.assertEqual(detect_raster_format(encoded_image("TIFF"), "part.tif", ""), "tiff")

    def test_rejects_a_misleading_extension(self) -> None:
        with self.assertRaisesRegex(SourceImageError, "contents are PNG"):
            detect_raster_format(encoded_image("PNG"), "part.jpg", "image/jpeg")

    def test_renders_each_multipage_tiff_frame(self) -> None:
        output = io.BytesIO()
        first = Image.new("RGB", (12, 8), "red")
        second = Image.new("RGB", (7, 5), "blue")
        first.save(output, format="TIFF", save_all=True, append_images=[second])

        rendered = render_source_page(
            output.getvalue(), page=2, filename="drawing.tiff", content_type="image/tiff"
        )

        self.assertEqual(rendered.page_count, 2)
        self.assertEqual((rendered.width, rendered.height), (7, 5))
        self.assertEqual(rendered.format, "tiff")
        decoded = Image.open(io.BytesIO(rendered.png))
        self.assertEqual(decoded.format, "PNG")
        self.assertEqual(decoded.getpixel((0, 0)), (0, 0, 255))

    def test_rejects_a_page_outside_the_document(self) -> None:
        with self.assertRaisesRegex(SourceImageError, "has 1 page"):
            render_source_page(encoded_image("TIFF"), page=2, filename="one.tif")

    def test_flattens_transparency_to_white(self) -> None:
        output = io.BytesIO()
        image = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
        image.putpixel((1, 0), (0, 0, 0, 255))
        image.save(output, format="PNG")

        rendered = render_source_page(output.getvalue(), filename="drawing.png")
        decoded = Image.open(io.BytesIO(rendered.png))
        self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(decoded.getpixel((1, 0)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
