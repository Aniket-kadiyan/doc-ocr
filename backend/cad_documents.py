"""Convert DWG and DXF drawings to PDF so the viewer can open them.

Browsers cannot read CAD files, and the rest of the app (viewer, scans,
checksheets, exports) is built around PDF pages. Rather than teach every
stage a new format, an uploaded CAD file is converted once into a vector PDF
and then handled exactly like any other PDF drawing.

DWG is a closed binary format, so it is first turned into DXF by an external
converter: the ODA File Converter when it is installed (best fidelity, every
DWG version), otherwise LibreDWG's ``dwg2dxf``. DXF is then drawn with ezdxf:
one PDF page per paper-space layout that has content, or a single page of
model space when there are none. Everything is drawn black on white, because
CAD colours such as yellow or cyan are close to invisible on paper and to OCR.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SUPPORTED_CAD_FORMATS = frozenset({"dwg", "dxf"})
MAX_CAD_BYTES = int(os.getenv("CAD_MAX_BYTES", str(128 * 1024 * 1024)))
CONVERT_TIMEOUT_SECONDS = int(os.getenv("CAD_CONVERT_TIMEOUT", "180"))

# Model space has no paper size, so it is fitted inside an A1 sheet: large
# enough that small callouts stay legible at the viewer's render scale.
MODEL_SPACE_LONG_SIDE_MM = 841.0
MODEL_SPACE_SHORT_SIDE_MM = 594.0
MM_PER_INCH = 25.4

BINARY_DXF_SENTINEL = b"AutoCAD Binary DXF\r\n\x1a\x00"


class CadDocumentError(ValueError):
    """A CAD file cannot be identified or converted."""


def detect_cad_format(data: bytes, filename: str = "") -> str:
    """Identify a DWG or DXF file from its contents."""

    if not data:
        raise CadDocumentError("The CAD file is empty")
    if len(data) > MAX_CAD_BYTES:
        raise CadDocumentError(
            f"The CAD file is larger than the {MAX_CAD_BYTES // (1024 * 1024)} MB limit"
        )

    suffix = Path(filename or "").suffix.casefold()
    # Every DWG starts with its version tag, "AC1015" for AutoCAD 2000 etc.
    if data[:4] == b"AC10" or data[:4] == b"AC2." or data[:6] == b"AC1.40":
        actual = "dwg"
    elif data.startswith(BINARY_DXF_SENTINEL):
        actual = "dxf"
    elif b"SECTION" in data[:4096] and (
        b"HEADER" in data[:4096] or b"ENTITIES" in data[:65536]
    ):
        actual = "dxf"
    else:
        raise CadDocumentError("Only DWG and DXF CAD files are supported")

    if suffix in {".dwg", ".dxf"} and suffix[1:] != actual:
        raise CadDocumentError(
            f"The file contents are {actual.upper()}, but its name says {suffix[1:].upper()}"
        )
    return actual


def _oda_converter() -> str | None:
    configured = os.getenv("ODA_FILE_CONVERTER")
    if configured and Path(configured).exists():
        return configured
    candidates = [
        "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
        shutil.which("ODAFileConverter"),
    ]
    for root in (os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)")):
        if root:
            candidates.extend(
                str(path)
                for path in sorted(Path(root, "ODA").glob("ODAFileConverter*/ODAFileConverter.exe"))
            )
    return next((path for path in candidates if path and Path(path).exists()), None)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CONVERT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CadDocumentError(
            f"DWG conversion took longer than {CONVERT_TIMEOUT_SECONDS} s"
        ) from error


def _dwg_to_dxf(dwg_path: Path, work_dir: Path) -> Path:
    oda = _oda_converter()
    if oda:
        out_dir = work_dir / "oda-out"
        out_dir.mkdir()
        # in-dir, out-dir, version, type, recurse, audit, filter
        _run([oda, str(dwg_path.parent), str(out_dir), "ACAD2018", "DXF", "0", "1", dwg_path.name])
        produced = out_dir / f"{dwg_path.stem}.dxf"
        if produced.exists():
            return produced

    dwg2dxf = os.getenv("DWG2DXF") or shutil.which("dwg2dxf")
    if not dwg2dxf:
        for candidate in ("/opt/homebrew/bin/dwg2dxf", "/usr/local/bin/dwg2dxf"):
            if Path(candidate).exists():
                dwg2dxf = candidate
                break
    if not dwg2dxf:
        raise CadDocumentError(
            "DWG support needs a converter on the server: install the ODA File "
            "Converter or LibreDWG (brew install libredwg), or save the drawing as DXF or PDF"
        )

    produced = work_dir / f"{dwg_path.stem}.dxf"
    result = _run([dwg2dxf, "-y", "-o", str(produced), str(dwg_path)])
    # dwg2dxf reports recoverable problems with a non-zero exit but still
    # writes a usable file, so only a missing output counts as failure.
    if not produced.exists() or produced.stat().st_size == 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise CadDocumentError(f"Could not convert the DWG file: {detail[0]}")
    return produced


def _read_dxf(path: Path):
    from ezdxf import recover
    from ezdxf.lldxf.const import DXFStructureError

    try:
        # recover repairs what it can; DXF written by converters is often untidy.
        doc, _auditor = recover.readfile(str(path))
    except (IOError, DXFStructureError) as error:
        raise CadDocumentError(f"Could not read the DXF drawing: {error}") from error
    return doc


def _drawable_layouts(doc) -> list:
    """Paper-space sheets that have content, else model space."""

    sheets = [
        layout
        for name in doc.layout_names_in_taborder()
        if name.casefold() != "model"
        for layout in [doc.paperspace(name)]
        # Every paper layout owns its main viewport; anything beyond that is content.
        if len(layout) > 1
    ]
    return sheets or [doc.modelspace()]


def _page_size_inches(layout) -> tuple[float, float]:
    if not layout.is_modelspace:
        width, height = layout.dxf.paper_width, layout.dxf.paper_height
        if layout.dxf.plot_rotation in (1, 3):
            width, height = height, width
        if width and height and width > 1 and height > 1:
            return width / MM_PER_INCH, height / MM_PER_INCH

    from ezdxf import bbox

    extents = bbox.extents(layout, fast=True)
    if extents.has_data and extents.size.x > 0 and extents.size.y > 0:
        aspect = extents.size.x / extents.size.y
    else:
        aspect = MODEL_SPACE_LONG_SIDE_MM / MODEL_SPACE_SHORT_SIDE_MM
    if aspect >= 1:
        width = MODEL_SPACE_LONG_SIDE_MM
        height = min(MODEL_SPACE_SHORT_SIDE_MM, width / aspect)
        width = height * aspect
    else:
        height = MODEL_SPACE_LONG_SIDE_MM
        width = min(MODEL_SPACE_SHORT_SIDE_MM, height * aspect)
        height = width / aspect
    return width / MM_PER_INCH, height / MM_PER_INCH


def _render_pdf(doc) -> tuple[bytes, int]:
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.config import (
        BackgroundPolicy,
        ColorPolicy,
        Configuration,
        LineweightPolicy,
    )
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    config = Configuration(
        background_policy=BackgroundPolicy.WHITE,
        color_policy=ColorPolicy.BLACK,
        lineweight_policy=LineweightPolicy.RELATIVE,
    )
    layouts = _drawable_layouts(doc)
    output = io.BytesIO()
    with PdfPages(output) as pdf:
        for layout in layouts:
            page_size = _page_size_inches(layout)
            figure = Figure(figsize=page_size)
            axes = figure.add_axes((0, 0, 1, 1))
            context = RenderContext(doc)
            context.set_current_layout(layout)
            Frontend(context, MatplotlibBackend(axes), config=config).draw_layout(
                layout, finalize=True
            )
            # finalize() resizes the figure to its own default; restore the sheet.
            figure.set_size_inches(page_size)
            axes.set_position((0, 0, 1, 1))
            pdf.savefig(figure, facecolor="white")
    return output.getvalue(), len(layouts)


def convert_cad_to_pdf(data: bytes, filename: str = "") -> tuple[bytes, int]:
    """Return a PDF of the drawing and its page count."""

    source_format = detect_cad_format(data, filename)
    with tempfile.TemporaryDirectory(prefix="cad-") as work:
        work_dir = Path(work)
        source = work_dir / f"drawing.{source_format}"
        source.write_bytes(data)
        dxf_path = _dwg_to_dxf(source, work_dir) if source_format == "dwg" else source
        doc = _read_dxf(dxf_path)
        try:
            return _render_pdf(doc)
        except CadDocumentError:
            raise
        except Exception as error:  # ezdxf/matplotlib raise many unrelated types
            raise CadDocumentError(f"Could not draw the CAD file: {error}") from error
