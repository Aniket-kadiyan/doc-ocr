"""DWG/DXF drawings are converted to a PDF the viewer opens like any other."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ezdxf
import pypdfium2 as pdfium

from cad_documents import CadDocumentError, _dwg_converter, convert_cad_to_pdf, detect_cad_format

SAMPLE_DWG = b"AC1032" + b"\0" * 4096


def sample_dxf() -> bytes:
    doc = ezdxf.new("R2018", setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 100), (0, 100)], close=True)
    # Yellow on white would vanish from the page without the black colour policy.
    msp.add_circle((60, 50), 20, dxfattribs={"color": 2})
    msp.add_text("%%c40.00", height=6).set_placement((40, 80))
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode()


def page_count(pdf: bytes) -> int:
    return len(pdfium.PdfDocument(pdf))


class DetectCadFormatTest(unittest.TestCase):
    def test_identifies_dwg_and_dxf_by_content(self):
        self.assertEqual(detect_cad_format(b"AC1032" + b"\0" * 32, "a.dwg"), "dwg")
        self.assertEqual(detect_cad_format(sample_dxf(), "a.dxf"), "dxf")

    def test_enforces_the_configured_size_limit(self):
        with self.assertRaisesRegex(CadDocumentError, "1 MB limit"):
            detect_cad_format(
                b"AC1015" + b"\0" * (1024 * 1024),
                "large.dwg",
                max_bytes=1024 * 1024,
            )

    def test_rejects_mislabelled_and_unknown_files(self):
        with self.assertRaisesRegex(CadDocumentError, "contents are DWG"):
            detect_cad_format(b"AC1015" + b"\0" * 32, "a.dxf")
        with self.assertRaisesRegex(CadDocumentError, "Only DWG and DXF"):
            detect_cad_format(b"%PDF-1.7", "a.dwg")
        with self.assertRaisesRegex(CadDocumentError, "empty"):
            detect_cad_format(b"", "a.dwg")


class ConvertCadToPdfTest(unittest.TestCase):
    def test_dxf_model_space_becomes_one_pdf_page(self):
        pdf, pages = convert_cad_to_pdf(sample_dxf(), "part.dxf")
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertEqual((pages, page_count(pdf)), (1, 1))

        page = pdfium.PdfDocument(pdf)[0]
        width, height = page.get_size()
        # Fitted to A1 at the drawing's 2:1 aspect.
        self.assertAlmostEqual(width / height, 2.0, delta=0.05)
        self.assertGreater(width, 2000)

        # The yellow circle is drawn dark, not faint.
        image = page.render(scale=0.5).to_pil().convert("L")
        self.assertLess(image.getextrema()[0], 80)

    def test_sheet_of_viewports_onto_nothing_falls_back_to_model_space(self):
        doc = ezdxf.read(io.StringIO(sample_dxf().decode()))
        sheet = doc.paperspace(doc.layout_names_in_taborder()[1])
        sheet.page_setup(size=(420, 297), margins=(0, 0, 0, 0), units="mm")
        # Looks far away from the part, so the sheet would print blank.
        sheet.add_viewport(center=(210, 150), size=(300, 200), view_center_point=(5000, 5000), view_height=100)
        stream = io.StringIO()
        doc.write(stream)

        pdf, pages = convert_cad_to_pdf(stream.getvalue().encode(), "part.dxf")
        width, height = pdfium.PdfDocument(pdf)[0].get_size()
        self.assertEqual(pages, 1)
        # The 2:1 model page, not the A3 sheet.
        self.assertAlmostEqual(width / height, 2.0, delta=0.05)

    def test_sheet_with_a_viewport_onto_the_part_stays_the_page(self):
        doc = ezdxf.read(io.StringIO(sample_dxf().decode()))
        sheet = doc.paperspace(doc.layout_names_in_taborder()[1])
        sheet.page_setup(size=(420, 297), margins=(0, 0, 0, 0), units="mm")
        sheet.add_viewport(center=(210, 150), size=(300, 200), view_center_point=(100, 50), view_height=120)
        stream = io.StringIO()
        doc.write(stream)

        pdf, pages = convert_cad_to_pdf(stream.getvalue().encode(), "part.dxf")
        width, height = pdfium.PdfDocument(pdf)[0].get_size()
        self.assertEqual(pages, 1)
        self.assertAlmostEqual(width, 420 / 25.4 * 72, delta=1)
        self.assertAlmostEqual(height, 297 / 25.4 * 72, delta=1)

    def test_dxf_needs_no_dwg_converter(self):
        with mock.patch("cad_documents._dwg_converter", return_value=None):
            _pdf, pages = convert_cad_to_pdf(sample_dxf(), "part.dxf")
        self.assertEqual(pages, 1)

    def test_dwg_without_converter_explains_what_is_missing(self):
        with mock.patch("cad_documents._dwg_converter", return_value=None):
            with self.assertRaisesRegex(CadDocumentError, "dwg2dxf-acadsharp"):
                convert_cad_to_pdf(SAMPLE_DWG, "sheet.dwg")


class DwgConverterDiscoveryTest(unittest.TestCase):
    def test_configured_path_wins_and_a_wrong_one_is_reported(self):
        missing = str(Path(__file__).parent / "no-such-converter")
        with mock.patch.dict(os.environ, {"DWG2DXF_ACADSHARP": missing}):
            self.assertEqual(_dwg_converter(), missing)
            with self.assertRaisesRegex(CadDocumentError, "DWG2DXF_ACADSHARP .* does not exist"):
                convert_cad_to_pdf(SAMPLE_DWG, "sheet.dwg")

    def test_converter_that_cannot_start_is_a_cad_error(self):
        # Not executable here, like a binary for another OS or CPU.
        with tempfile.NamedTemporaryFile(suffix="-dwg2dxf") as stub:
            with mock.patch.dict(os.environ, {"DWG2DXF_ACADSHARP": stub.name}):
                with self.assertRaisesRegex(CadDocumentError, "could not be started"):
                    convert_cad_to_pdf(SAMPLE_DWG, "sheet.dwg")

    def test_no_bundled_build_is_guessed_for_an_unknown_platform(self):
        environ = {key: value for key, value in os.environ.items() if key != "DWG2DXF_ACADSHARP"}
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch("shutil.which", return_value=None),
            mock.patch("platform.system", return_value="FreeBSD"),
        ):
            self.assertIsNone(_dwg_converter())


class ConvertDwgTest(unittest.TestCase):
    def test_converter_output_is_rendered_without_a_binary_fixture(self):
        def fake_run(command, *, timeout_seconds):
            Path(command[2]).write_bytes(sample_dxf())
            return mock.Mock(returncode=0, stdout="", stderr="")

        with (
            mock.patch("cad_documents._dwg_converter", return_value="/converter"),
            mock.patch("cad_documents._run", side_effect=fake_run),
        ):
            pdf, pages = convert_cad_to_pdf(SAMPLE_DWG, "sheet.dwg")

        self.assertEqual((pages, page_count(pdf)), (1, 1))

    def test_converter_failure_is_reported_without_private_drawing_data(self):
        failed = mock.Mock(
            returncode=1,
            stdout="",
            stderr="unsupported DWG object",
        )
        with (
            mock.patch("cad_documents._dwg_converter", return_value="/converter"),
            mock.patch("cad_documents._run", return_value=failed),
            self.assertRaisesRegex(CadDocumentError, "unsupported DWG object"),
        ):
            convert_cad_to_pdf(SAMPLE_DWG, "broken.dwg")


if __name__ == "__main__":
    unittest.main()
