"""DWG/DXF drawings are converted to a PDF the viewer opens like any other."""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path

import ezdxf
import pypdfium2 as pdfium

from cad_documents import CadDocumentError, convert_cad_to_pdf, detect_cad_format


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

    @unittest.skipUnless(shutil.which("dxf2dwg") and shutil.which("dwg2dxf"), "LibreDWG not installed")
    def test_dwg_converts_through_dxf(self):
        import subprocess

        with tempfile.TemporaryDirectory() as work:
            dxf = Path(work, "part.dxf")
            dwg = Path(work, "part.dwg")
            dxf.write_bytes(sample_dxf())
            subprocess.run(["dxf2dwg", "-y", "-o", str(dwg), str(dxf)], capture_output=True, check=False)
            if not dwg.exists():
                self.skipTest("dxf2dwg could not write a sample DWG")
            pdf, pages = convert_cad_to_pdf(dwg.read_bytes(), "part.dwg")
        self.assertEqual(page_count(pdf), pages)
        self.assertGreaterEqual(pages, 1)


if __name__ == "__main__":
    unittest.main()
