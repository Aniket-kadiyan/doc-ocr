"""Convert DWG and DXF drawings to PDF so the viewer can open them.

Browsers cannot read CAD files, and the rest of the app (viewer, scans,
checksheets, exports) is built around PDF pages. Rather than teach every
stage a new format, an uploaded CAD file is converted once into a vector PDF
and then handled exactly like any other PDF drawing.

DWG is a closed binary format, so it is first turned into DXF by
``tools/dwg2dxf``, a small converter built on ACadSharp (MIT). Converters
with copyleft or proprietary licences (LibreDWG, the ODA File Converter) are
deliberately not used. DXF is then drawn with ezdxf: one PDF page per
paper-space layout that has content, or a single page of model space when
there are none. Everything is drawn black on white, because
CAD colours such as yellow or cyan are close to invisible on paper and to OCR.
"""

from __future__ import annotations

import io
import os
import platform
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


def _dwg_converter() -> str | None:
    # An explicit setting wins outright, so a wrong path is reported instead
    # of being quietly replaced by some other converter.
    configured = os.getenv("DWG2DXF_ACADSHARP")
    if configured:
        return configured
    executable = "dwg2dxf-acadsharp.exe" if os.name == "nt" else "dwg2dxf-acadsharp"
    candidates = [shutil.which(executable)]
    # Self-contained builds land in tools/dwg2dxf/dist/<runtime id>/ (see its
    # README); only this machine's runtime can run.
    system = {"Darwin": "osx", "Linux": "linux", "Windows": "win"}.get(platform.system(), "")
    machine = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        platform.machine().casefold(), ""
    )
    if system and machine:
        candidates.append(
            str(
                Path(__file__).resolve().parent.parent
                / "tools" / "dwg2dxf" / "dist" / f"{system}-{machine}" / executable
            )
        )
    return next((path for path in candidates if path and Path(path).is_file()), None)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=CONVERT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CadDocumentError(
            f"DWG conversion took longer than {CONVERT_TIMEOUT_SECONDS} s"
        ) from error
    except OSError as error:
        # Not executable, or built for another operating system or CPU.
        raise CadDocumentError(
            f"The DWG converter on the server could not be started: {error.strerror or error}"
        ) from error


def _dwg_to_dxf(dwg_path: Path, work_dir: Path) -> Path:
    converter = _dwg_converter()
    if not converter:
        raise CadDocumentError(
            "DWG support needs the dwg2dxf-acadsharp converter on the server "
            "(build tools/dwg2dxf), or save the drawing as DXF or PDF"
        )
    if not Path(converter).is_file():
        raise CadDocumentError(
            "The DWG converter set in DWG2DXF_ACADSHARP on the server does not exist"
        )

    produced = work_dir / f"{dwg_path.stem}.dxf"
    result = _run([converter, str(dwg_path), str(produced)])
    if result.returncode != 0 or not produced.exists() or produced.stat().st_size == 0:
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


def _viewport_shows_model(viewport, model_extents) -> bool:
    """Whether a viewport looks at any model-space geometry."""

    # id 1 is the sheet's own viewport and status 0 means switched off.
    if viewport.dxf.id == 1 or viewport.dxf.status == 0:
        return False
    if not model_extents.has_data:
        return False
    plan_view = (
        tuple(viewport.dxf.view_direction_vector) == (0, 0, 1)
        and tuple(viewport.dxf.view_target_point) == (0, 0, 0)
        and not viewport.dxf.view_twist_angle
    )
    if not plan_view:
        return True  # 3D or twisted views are not cheap to bound; trust them.
    view_height = viewport.dxf.view_height
    view_width = view_height * viewport.dxf.width / max(viewport.dxf.height, 1e-9)
    center = viewport.dxf.view_center_point
    low, high = model_extents.extmin, model_extents.extmax
    return (
        center.x - view_width / 2 <= high.x
        and center.x + view_width / 2 >= low.x
        and center.y - view_height / 2 <= high.y
        and center.y + view_height / 2 >= low.y
    )


def _drawable_layouts(doc) -> list:
    """Paper-space sheets that have content, else model space."""

    from ezdxf import bbox

    model_extents = bbox.extents(doc.modelspace(), fast=True)
    sheets = [
        layout
        for name in doc.layout_names_in_taborder()
        if name.casefold() != "model"
        for layout in [doc.paperspace(name)]
        # A sheet holding only viewports onto empty model space would be a
        # blank page; the model itself is the better page then.
        if any(
            entity.dxftype() != "VIEWPORT" or _viewport_shows_model(entity, model_extents)
            for entity in layout
        )
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
