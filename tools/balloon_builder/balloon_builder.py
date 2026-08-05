#!/usr/bin/env python3
r"""Create or validate the developer-owned ``public/balloon-style.json``.

Run without arguments to open the graphical builder. The GUI can remove an
edge-connected background, clear a sample number, mark the dynamic number area
and pointer-tip anchor, preview multiple numbers, and export the single JSON
file consumed by the application.

Pillow is already part of ``backend/requirements.txt``. On Windows, for example:

    backend\.venv\Scripts\python.exe tools\balloon_builder\balloon_builder.py

The non-GUI ``--build`` and ``--validate`` modes are useful for automation and
do not import Tkinter.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from PIL import Image, ImageColor, ImageDraw


SCHEMA_VERSION = 1
PNG_DATA_URL_PREFIX = "data:image/png;base64,"
MAX_ARTWORK_BYTES = 2_200_000


class BalloonStyleError(ValueError):
    """Raised when a style file is unsafe or incompatible with the renderer."""


def _as_rgba(image: Image.Image) -> Image.Image:
    return image.convert("RGBA")


def open_artwork(path: str | Path) -> Image.Image:
    """Load source artwork without retaining an open file handle."""

    with Image.open(path) as source:
        return _as_rgba(source.copy())


def _border_pixels(image: Image.Image) -> Iterable[tuple[int, int, int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    assert pixels is not None
    for x in range(width):
        yield pixels[x, 0]
        if height > 1:
            yield pixels[x, height - 1]
    for y in range(1, max(1, height - 1)):
        yield pixels[0, y]
        if width > 1:
            yield pixels[width - 1, y]


def _median_background(image: Image.Image) -> tuple[int, int, int]:
    border = list(_border_pixels(image))
    if not border:
        return (255, 255, 255)
    channels = zip(*border)
    medians: list[int] = []
    for values in channels:
        ordered = sorted(values)
        medians.append(ordered[len(ordered) // 2])
    return tuple(medians)  # type: ignore[return-value]


def remove_edge_connected_background(
    image: Image.Image,
    tolerance: int = 28,
) -> Image.Image:
    """Make only edge-connected pixels near the border colour transparent.

    Unlike global colour removal, this preserves a white number area enclosed by
    a dark circle even when the source image also has a white external backdrop.
    ``tolerance`` is the maximum per-pixel RGB Euclidean distance from the
    median border colour.
    """

    if not 0 <= tolerance <= 442:
        raise BalloonStyleError("Background tolerance must be between 0 and 442.")

    result = _as_rgba(image)
    width, height = result.size
    if width == 0 or height == 0:
        raise BalloonStyleError("Artwork is empty.")

    pixels = result.load()
    assert pixels is not None
    background = _median_background(result)
    limit = tolerance * tolerance
    visited = bytearray(width * height)
    queue: deque[int] = deque()

    def near_background(x: int, y: int) -> bool:
        red, green, blue, alpha = pixels[x, y]
        if alpha == 0:
            return True
        return (
            (red - background[0]) ** 2
            + (green - background[1]) ** 2
            + (blue - background[2]) ** 2
            <= limit
        )

    def enqueue(x: int, y: int) -> None:
        index = y * width + x
        if visited[index] or not near_background(x, y):
            return
        visited[index] = 1
        queue.append(index)

    for x in range(width):
        enqueue(x, 0)
        enqueue(x, height - 1)
    for y in range(1, height - 1):
        enqueue(0, y)
        enqueue(width - 1, y)

    while queue:
        index = queue.popleft()
        x = index % width
        y = index // width
        red, green, blue, _ = pixels[x, y]
        pixels[x, y] = (red, green, blue, 0)
        if x > 0:
            enqueue(x - 1, y)
        if x + 1 < width:
            enqueue(x + 1, y)
        if y > 0:
            enqueue(x, y - 1)
        if y + 1 < height:
            enqueue(x, y + 1)

    return result


def transparent_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    bounds = _as_rgba(image).getchannel("A").getbbox()
    if bounds is None:
        raise BalloonStyleError("Artwork is fully transparent.")
    return bounds


def trim_transparent(image: Image.Image) -> Image.Image:
    return _as_rgba(image).crop(transparent_bounds(image))


def clear_region(
    image: Image.Image,
    rectangle: Sequence[int],
    fill: str = "#ffffff",
) -> Image.Image:
    """Clear a printed sample number using the number area's background colour."""

    if len(rectangle) != 4:
        raise BalloonStyleError("Clear rectangle must be x,y,width,height.")
    x, y, width, height = (int(value) for value in rectangle)
    if width <= 0 or height <= 0:
        raise BalloonStyleError("Clear rectangle must have positive dimensions.")
    red, green, blue = ImageColor.getrgb(fill)[:3]
    result = _as_rgba(image)
    ImageDraw.Draw(result).rectangle(
        (x, y, x + width - 1, y + height - 1),
        fill=(red, green, blue, 255),
    )
    return result


def encode_png_data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    _as_rgba(image).save(output, format="PNG", optimize=True)
    payload = output.getvalue()
    if len(payload) > MAX_ARTWORK_BYTES:
        raise BalloonStyleError(
            "Encoded artwork is too large; resize it below 2.2 MB before export."
        )
    return PNG_DATA_URL_PREFIX + base64.b64encode(payload).decode("ascii")


def decode_png_data_url(data_url: str) -> Image.Image:
    if not data_url.startswith(PNG_DATA_URL_PREFIX):
        raise BalloonStyleError("Artwork must be an embedded PNG data URL.")
    try:
        raw = base64.b64decode(
            data_url[len(PNG_DATA_URL_PREFIX) :], validate=True
        )
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG":
                raise BalloonStyleError("Embedded artwork is not a PNG.")
            return _as_rgba(source.copy())
    except BalloonStyleError:
        raise
    except Exception as exc:
        raise BalloonStyleError("Embedded PNG data is invalid.") from exc


def _normal_rect(
    rectangle: Sequence[float], width: int, height: int
) -> dict[str, float]:
    if len(rectangle) != 4:
        raise BalloonStyleError("Number area must be x,y,width,height.")
    x, y, rect_width, rect_height = (float(value) for value in rectangle)
    if rect_width <= 0 or rect_height <= 0:
        raise BalloonStyleError("Number area must have positive dimensions.")
    if x < 0 or y < 0 or x + rect_width > width or y + rect_height > height:
        raise BalloonStyleError("Number area must stay within the artwork.")
    return {
        "x": round(x / width, 6),
        "y": round(y / height, 6),
        "width": round(rect_width / width, 6),
        "height": round(rect_height / height, 6),
    }


def build_style_payload(
    image: Image.Image,
    *,
    name: str,
    number_area: Sequence[float],
    anchor: Sequence[float],
    drawing_width: float = 32,
    sidebar_width: float = 22,
    text_color: str = "#111111",
    font_family: str = "Arial, sans-serif",
    font_weight: str = "bold",
    min_font_size: float = 7,
    max_font_size: float = 13,
    text_padding: float = 2,
) -> dict[str, Any]:
    artwork = _as_rgba(image)
    width, height = artwork.size
    if len(anchor) != 2:
        raise BalloonStyleError("Anchor must be x,y.")
    anchor_x, anchor_y = (float(value) for value in anchor)
    if not (0 <= anchor_x <= width and 0 <= anchor_y <= height):
        raise BalloonStyleError("Anchor must stay within the artwork.")

    payload: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "name": name.strip(),
        "artwork": {
            "mimeType": "image/png",
            "dataUrl": encode_png_data_url(artwork),
            "width": width,
            "height": height,
        },
        "numberArea": _normal_rect(number_area, width, height),
        "anchor": {
            "x": round(anchor_x / width, 6),
            "y": round(anchor_y / height, 6),
        },
        "display": {
            "drawingWidth": float(drawing_width),
            "sidebarWidth": float(sidebar_width),
        },
        "text": {
            "color": text_color.strip(),
            "fontFamily": font_family.strip(),
            "fontWeight": font_weight,
            "minFontSize": float(min_font_size),
            "maxFontSize": float(max_font_size),
            "padding": float(text_padding),
        },
    }
    validate_style_payload(payload)
    return payload


def _number(value: Any, path: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BalloonStyleError(f"{path} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise BalloonStyleError(
            f"{path} must be between {minimum:g} and {maximum:g}."
        )
    return parsed


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BalloonStyleError(f"{path} must be an object.")
    return value


def validate_style_payload(payload: Any) -> None:
    root = _object(payload, "Balloon style")
    if root.get("schemaVersion") != SCHEMA_VERSION:
        raise BalloonStyleError(f"schemaVersion must be {SCHEMA_VERSION}.")
    if not isinstance(root.get("name"), str) or not root["name"].strip():
        raise BalloonStyleError("name must be a non-empty string.")

    artwork = _object(root.get("artwork"), "artwork")
    if artwork.get("mimeType") != "image/png":
        raise BalloonStyleError("artwork.mimeType must be image/png.")
    width = _number(artwork.get("width"), "artwork.width", 1, 4096)
    height = _number(artwork.get("height"), "artwork.height", 1, 4096)
    if not width.is_integer() or not height.is_integer():
        raise BalloonStyleError("Artwork dimensions must be whole pixels.")
    image = decode_png_data_url(artwork.get("dataUrl", ""))
    if image.size != (int(width), int(height)):
        raise BalloonStyleError(
            "Declared artwork dimensions do not match the embedded PNG."
        )

    area = _object(root.get("numberArea"), "numberArea")
    area_x = _number(area.get("x"), "numberArea.x", 0, 1)
    area_y = _number(area.get("y"), "numberArea.y", 0, 1)
    area_width = _number(area.get("width"), "numberArea.width", 0.001, 1)
    area_height = _number(area.get("height"), "numberArea.height", 0.001, 1)
    if area_x + area_width > 1.000001 or area_y + area_height > 1.000001:
        raise BalloonStyleError("numberArea must stay within the artwork.")

    anchor = _object(root.get("anchor"), "anchor")
    _number(anchor.get("x"), "anchor.x", 0, 1)
    _number(anchor.get("y"), "anchor.y", 0, 1)

    display = _object(root.get("display"), "display")
    _number(display.get("drawingWidth"), "display.drawingWidth", 12, 160)
    _number(display.get("sidebarWidth"), "display.sidebarWidth", 12, 80)

    text = _object(root.get("text"), "text")
    color = text.get("color")
    if not isinstance(color, str):
        raise BalloonStyleError("text.color must be a hexadecimal CSS colour.")
    if re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", color) is None:
        raise BalloonStyleError("text.color must be a hexadecimal CSS colour.")
    family = text.get("fontFamily")
    if not isinstance(family, str) or not family.strip():
        raise BalloonStyleError("text.fontFamily must be a non-empty string.")
    if text.get("fontWeight") not in {"normal", "bold"}:
        raise BalloonStyleError('text.fontWeight must be "normal" or "bold".')
    minimum = _number(text.get("minFontSize"), "text.minFontSize", 4, 48)
    maximum = _number(text.get("maxFontSize"), "text.maxFontSize", 4, 72)
    if maximum < minimum:
        raise BalloonStyleError(
            "text.maxFontSize must be at least text.minFontSize."
        )
    _number(text.get("padding"), "text.padding", 0, 24)


def load_style_file(path: str | Path) -> tuple[dict[str, Any], Image.Image]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BalloonStyleError(f"Could not read style file: {exc}") from exc
    validate_style_payload(payload)
    return payload, decode_png_data_url(payload["artwork"]["dataUrl"])


def write_style_file(path: str | Path, payload: dict[str, Any]) -> None:
    validate_style_payload(payload)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_values(value: str, expected: int, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label} must contain comma-separated numbers."
        ) from exc
    if len(values) != expected:
        raise argparse.ArgumentTypeError(
            f"{label} must contain exactly {expected} numbers."
        )
    return values


def _fit_preview_font(
    value: str,
    width: float,
    height: float,
    minimum: float,
    maximum: float,
    padding: float,
) -> float:
    units = sum(0.5 if char == "1" else 0.4 if char in "-." else 0.68 for char in value)
    fitted = min(
        maximum,
        max(1, height - padding * 2) * 0.68,
        max(1, width - padding * 2) / max(units, 0.68),
    )
    return max(minimum, fitted)


def run_gui() -> None:
    """Launch the developer-only graphical conversion utility."""

    try:
        import tkinter as tk
        from tkinter import colorchooser, filedialog, messagebox, ttk
    except ImportError as exc:
        raise BalloonStyleError(
            "The graphical builder requires Tkinter. Use a standard Windows "
            "Python installation with Tcl/Tk enabled, or use --build/--validate."
        ) from exc
    from PIL import ImageTk

    class BalloonBuilderApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("Balloon Style Builder (developer tool)")
            self.root.minsize(1000, 700)

            self.image: Image.Image | None = None
            self.photo: ImageTk.PhotoImage | None = None
            self.number_area: tuple[float, float, float, float] | None = None
            self.anchor: tuple[float, float] | None = None
            self.mode: str | None = None
            self.drag_start: tuple[float, float] | None = None
            self.view_scale = 1.0
            self.view_offset = (0.0, 0.0)

            self.name = tk.StringVar(value="Client balloon")
            self.tolerance = tk.StringVar(value="28")
            self.clear_color = tk.StringVar(value="#ffffff")
            self.drawing_width = tk.StringVar(value="32")
            self.sidebar_width = tk.StringVar(value="22")
            self.text_color = tk.StringVar(value="#111111")
            self.font_family = tk.StringVar(value="Arial, sans-serif")
            self.font_weight = tk.StringVar(value="bold")
            self.min_font = tk.StringVar(value="7")
            self.max_font = tk.StringVar(value="13")
            self.padding = tk.StringVar(value="2")
            self.preview_number = tk.StringVar(value="35")
            self.status = tk.StringVar(
                value="Open a PNG/JPEG or an existing balloon-style.json."
            )

            self._build_ui(ttk)

        def _build_ui(self, ttk_module: Any) -> None:
            toolbar = ttk_module.Frame(self.root, padding=8)
            toolbar.pack(fill="x")
            for label, command in (
                ("Open image", self.open_image),
                ("Open style", self.open_style),
                ("Remove edge background", self.remove_background),
                ("Crop transparent", self.crop_image),
                ("Clear sample number", lambda: self.set_mode("clear")),
                ("Mark number area", lambda: self.set_mode("number")),
                ("Mark pointer tip", lambda: self.set_mode("anchor")),
                ("Export JSON", self.export_style),
            ):
                ttk_module.Button(toolbar, text=label, command=command).pack(
                    side="left", padx=3
                )

            body = ttk_module.Frame(self.root, padding=(8, 0, 8, 8))
            body.pack(fill="both", expand=True)
            body.columnconfigure(0, weight=1)
            body.rowconfigure(0, weight=1)

            self.canvas = tk.Canvas(
                body,
                background="#e2e8f0",
                highlightthickness=1,
                highlightbackground="#94a3b8",
            )
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.canvas.bind("<Configure>", lambda _event: self.redraw())
            self.canvas.bind("<ButtonPress-1>", self.canvas_down)
            self.canvas.bind("<B1-Motion>", self.canvas_drag)
            self.canvas.bind("<ButtonRelease-1>", self.canvas_up)

            panel = ttk_module.Frame(body, padding=(12, 0, 0, 0))
            panel.grid(row=0, column=1, sticky="ns")

            fields = (
                ("Style name", self.name),
                ("Background tolerance", self.tolerance),
                ("Clear colour", self.clear_color),
                ("Drawing width (px)", self.drawing_width),
                ("Sidebar width (px)", self.sidebar_width),
                ("Text colour", self.text_color),
                ("Font family", self.font_family),
                ("Minimum font (px)", self.min_font),
                ("Maximum font (px)", self.max_font),
                ("Text padding (px)", self.padding),
            )
            for row, (label, variable) in enumerate(fields):
                ttk_module.Label(panel, text=label).grid(
                    row=row, column=0, sticky="w", pady=(0, 2)
                )
                ttk_module.Entry(panel, textvariable=variable, width=24).grid(
                    row=row, column=1, sticky="ew", pady=(0, 7)
                )

            weight_row = len(fields)
            ttk_module.Label(panel, text="Font weight").grid(
                row=weight_row, column=0, sticky="w"
            )
            ttk_module.Combobox(
                panel,
                textvariable=self.font_weight,
                values=("normal", "bold"),
                state="readonly",
                width=21,
            ).grid(row=weight_row, column=1, sticky="ew", pady=(0, 7))

            preview_row = weight_row + 1
            ttk_module.Label(panel, text="Preview number").grid(
                row=preview_row, column=0, sticky="w"
            )
            preview = ttk_module.Combobox(
                panel,
                textvariable=self.preview_number,
                values=("1", "35", "128"),
                width=21,
            )
            preview.grid(row=preview_row, column=1, sticky="ew", pady=(0, 7))
            preview.bind("<<ComboboxSelected>>", lambda _event: self.redraw())
            preview.bind("<KeyRelease>", lambda _event: self.redraw())

            ttk_module.Button(
                panel,
                text="Choose clear colour",
                command=self.choose_clear_color,
            ).grid(row=preview_row + 1, column=0, columnspan=2, sticky="ew")

            help_text = (
                "Workflow\n"
                "1. Open artwork\n"
                "2. Remove the edge background\n"
                "3. Crop transparent margins\n"
                "4. Clear the printed sample number\n"
                "5. Mark the number rectangle\n"
                "6. Click the pointer tip\n"
                "7. Preview 1, 35 and 128\n"
                "8. Export balloon-style.json"
            )
            ttk_module.Label(panel, text=help_text, justify="left").grid(
                row=preview_row + 2,
                column=0,
                columnspan=2,
                sticky="nw",
                pady=(18, 0),
            )

            ttk_module.Label(
                self.root,
                textvariable=self.status,
                relief="sunken",
                anchor="w",
                padding=5,
            ).pack(fill="x", side="bottom")

        def set_mode(self, mode: str) -> None:
            if self.image is None:
                self.status.set("Open artwork before marking it.")
                return
            self.mode = mode
            prompts = {
                "clear": "Drag tightly over the printed sample number.",
                "number": "Drag the rectangle available for the dynamic number.",
                "anchor": "Click the exact pointer tip.",
            }
            self.status.set(prompts[mode])

        def choose_clear_color(self) -> None:
            chosen = colorchooser.askcolor(color=self.clear_color.get())[1]
            if chosen:
                self.clear_color.set(chosen)

        def open_image(self) -> None:
            path = filedialog.askopenfilename(
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ]
            )
            if not path:
                return
            try:
                self.image = open_artwork(path)
                self.number_area = None
                self.anchor = None
                self.status.set(f"Opened {Path(path).name} ({self.image.width}×{self.image.height}).")
                self.redraw()
            except Exception as exc:
                messagebox.showerror("Open failed", str(exc))

        def open_style(self) -> None:
            path = filedialog.askopenfilename(
                filetypes=[("Balloon style", "*.json"), ("All files", "*.*")]
            )
            if not path:
                return
            try:
                payload, self.image = load_style_file(path)
                width, height = self.image.size
                area = payload["numberArea"]
                self.number_area = (
                    area["x"] * width,
                    area["y"] * height,
                    area["width"] * width,
                    area["height"] * height,
                )
                anchor = payload["anchor"]
                self.anchor = (anchor["x"] * width, anchor["y"] * height)
                self.name.set(payload["name"])
                self.drawing_width.set(str(payload["display"]["drawingWidth"]))
                self.sidebar_width.set(str(payload["display"]["sidebarWidth"]))
                self.text_color.set(payload["text"]["color"])
                self.font_family.set(payload["text"]["fontFamily"])
                self.font_weight.set(payload["text"]["fontWeight"])
                self.min_font.set(str(payload["text"]["minFontSize"]))
                self.max_font.set(str(payload["text"]["maxFontSize"]))
                self.padding.set(str(payload["text"]["padding"]))
                self.status.set(f"Opened and validated {Path(path).name}.")
                self.redraw()
            except Exception as exc:
                messagebox.showerror("Style is invalid", str(exc))

        def remove_background(self) -> None:
            if self.image is None:
                return
            try:
                self.image = remove_edge_connected_background(
                    self.image, int(self.tolerance.get())
                )
                self.status.set(
                    "Removed only the background connected to the image edges."
                )
                self.redraw()
            except Exception as exc:
                messagebox.showerror("Background removal failed", str(exc))

        def crop_image(self) -> None:
            if self.image is None:
                return
            try:
                left, top, right, bottom = transparent_bounds(self.image)
                self.image = self.image.crop((left, top, right, bottom))
                if self.number_area is not None:
                    x, y, width, height = self.number_area
                    self.number_area = (x - left, y - top, width, height)
                if self.anchor is not None:
                    self.anchor = (self.anchor[0] - left, self.anchor[1] - top)
                self.status.set(
                    f"Cropped to transparent bounds ({self.image.width}×{self.image.height})."
                )
                self.redraw()
            except Exception as exc:
                messagebox.showerror("Crop failed", str(exc))

        def _image_point(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
            if self.image is None:
                return (0, 0)
            x = (canvas_x - self.view_offset[0]) / self.view_scale
            y = (canvas_y - self.view_offset[1]) / self.view_scale
            return (
                min(max(x, 0), self.image.width),
                min(max(y, 0), self.image.height),
            )

        def canvas_down(self, event: Any) -> None:
            if self.image is None or self.mode is None:
                return
            point = self._image_point(event.x, event.y)
            if self.mode == "anchor":
                self.anchor = point
                self.mode = None
                self.status.set("Pointer-tip anchor marked.")
                self.redraw()
                return
            self.drag_start = point

        def canvas_drag(self, event: Any) -> None:
            if self.drag_start is None or self.mode not in {"clear", "number"}:
                return
            end = self._image_point(event.x, event.y)
            self._draw_drag_rectangle(self.drag_start, end)

        def canvas_up(self, event: Any) -> None:
            if self.drag_start is None or self.mode not in {"clear", "number"}:
                return
            end = self._image_point(event.x, event.y)
            left = min(self.drag_start[0], end[0])
            top = min(self.drag_start[1], end[1])
            width = abs(end[0] - self.drag_start[0])
            height = abs(end[1] - self.drag_start[1])
            current_mode = self.mode
            self.drag_start = None
            self.mode = None
            if width < 1 or height < 1:
                self.status.set("Selection was too small; try again.")
                self.redraw()
                return
            if current_mode == "number":
                self.number_area = (left, top, width, height)
                self.status.set("Dynamic number area marked.")
            else:
                assert self.image is not None
                self.image = clear_region(
                    self.image,
                    (round(left), round(top), round(width), round(height)),
                    self.clear_color.get(),
                )
                self.status.set("Sample number cleared. Mark the full number area next.")
            self.redraw()

        def _draw_drag_rectangle(
            self, start: tuple[float, float], end: tuple[float, float]
        ) -> None:
            self.canvas.delete("drag")
            x1 = self.view_offset[0] + start[0] * self.view_scale
            y1 = self.view_offset[1] + start[1] * self.view_scale
            x2 = self.view_offset[0] + end[0] * self.view_scale
            y2 = self.view_offset[1] + end[1] * self.view_scale
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="#2563eb", width=2, tags="drag"
            )

        def redraw(self) -> None:
            self.canvas.delete("all")
            if self.image is None:
                self.canvas.create_text(
                    max(100, self.canvas.winfo_width() / 2),
                    max(100, self.canvas.winfo_height() / 2),
                    text="Open balloon artwork to begin",
                    fill="#475569",
                    font=("Arial", 16),
                )
                return

            canvas_width = max(200, self.canvas.winfo_width())
            canvas_height = max(200, self.canvas.winfo_height())
            self.view_scale = min(
                (canvas_width - 60) / self.image.width,
                (canvas_height - 60) / self.image.height,
                4,
            )
            display_size = (
                max(1, round(self.image.width * self.view_scale)),
                max(1, round(self.image.height * self.view_scale)),
            )
            preview = self.image.resize(display_size, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(preview)
            offset_x = (canvas_width - display_size[0]) / 2
            offset_y = (canvas_height - display_size[1]) / 2
            self.view_offset = (offset_x, offset_y)
            self.canvas.create_image(offset_x, offset_y, image=self.photo, anchor="nw")

            if self.number_area is not None:
                x, y, width, height = self.number_area
                view_x = offset_x + x * self.view_scale
                view_y = offset_y + y * self.view_scale
                view_width = width * self.view_scale
                view_height = height * self.view_scale
                self.canvas.create_rectangle(
                    view_x,
                    view_y,
                    view_x + view_width,
                    view_y + view_height,
                    outline="#2563eb",
                    width=2,
                    dash=(5, 3),
                )
                try:
                    drawing_width = float(self.drawing_width.get())
                    artwork_to_screen = drawing_width / self.image.width
                    font_size = _fit_preview_font(
                        self.preview_number.get(),
                        width * artwork_to_screen,
                        height * artwork_to_screen,
                        float(self.min_font.get()),
                        float(self.max_font.get()),
                        float(self.padding.get()),
                    )
                    self.canvas.create_text(
                        view_x + view_width / 2,
                        view_y + view_height / 2,
                        text=self.preview_number.get(),
                        fill=self.text_color.get(),
                        font=(
                            self.font_family.get().split(",")[0],
                            max(
                                6,
                                round(
                                    font_size
                                    / artwork_to_screen
                                    * self.view_scale
                                ),
                            ),
                            self.font_weight.get(),
                        ),
                    )
                except (ValueError, tk.TclError):
                    pass

            if self.anchor is not None:
                anchor_x = offset_x + self.anchor[0] * self.view_scale
                anchor_y = offset_y + self.anchor[1] * self.view_scale
                radius = 6
                self.canvas.create_line(
                    anchor_x - radius,
                    anchor_y,
                    anchor_x + radius,
                    anchor_y,
                    fill="#dc2626",
                    width=2,
                )
                self.canvas.create_line(
                    anchor_x,
                    anchor_y - radius,
                    anchor_x,
                    anchor_y + radius,
                    fill="#dc2626",
                    width=2,
                )

        def export_style(self) -> None:
            if self.image is None:
                messagebox.showwarning("Nothing to export", "Open artwork first.")
                return
            if self.number_area is None or self.anchor is None:
                messagebox.showwarning(
                    "Missing geometry",
                    "Mark both the number area and pointer-tip anchor before export.",
                )
                return
            path = filedialog.asksaveasfilename(
                initialfile="balloon-style.json",
                defaultextension=".json",
                filetypes=[("Balloon style", "*.json")],
            )
            if not path:
                return
            try:
                payload = build_style_payload(
                    self.image,
                    name=self.name.get(),
                    number_area=self.number_area,
                    anchor=self.anchor,
                    drawing_width=float(self.drawing_width.get()),
                    sidebar_width=float(self.sidebar_width.get()),
                    text_color=self.text_color.get(),
                    font_family=self.font_family.get(),
                    font_weight=self.font_weight.get(),
                    min_font_size=float(self.min_font.get()),
                    max_font_size=float(self.max_font.get()),
                    text_padding=float(self.padding.get()),
                )
                write_style_file(path, payload)
                self.status.set(f"Exported {path}. Replace public/balloon-style.json with it.")
                messagebox.showinfo(
                    "Style exported",
                    "The compatible one-file balloon style was created successfully.",
                )
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    root = tk.Tk()
    BalloonBuilderApp(root)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", metavar="STYLE_JSON", help="Validate a style file.")
    mode.add_argument("--build", metavar="IMAGE", help="Build a style from prepared artwork.")
    parser.add_argument("--output", help="Destination JSON for --build.")
    parser.add_argument("--name", default="Client balloon")
    parser.add_argument(
        "--number-area",
        type=lambda value: _parse_values(value, 4, "number area"),
        help="Pixel x,y,width,height in the prepared artwork.",
    )
    parser.add_argument(
        "--anchor",
        type=lambda value: _parse_values(value, 2, "anchor"),
        help="Pixel x,y position of the pointer tip.",
    )
    parser.add_argument("--drawing-width", type=float, default=32)
    parser.add_argument("--sidebar-width", type=float, default=22)
    parser.add_argument("--text-color", default="#111111")
    parser.add_argument("--font-family", default="Arial, sans-serif")
    parser.add_argument("--font-weight", choices=("normal", "bold"), default="bold")
    parser.add_argument("--min-font-size", type=float, default=7)
    parser.add_argument("--max-font-size", type=float, default=13)
    parser.add_argument("--text-padding", type=float, default=2)
    parser.add_argument(
        "--remove-background",
        action="store_true",
        help="Remove the edge-connected background before building.",
    )
    parser.add_argument("--background-tolerance", type=int, default=28)
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Crop transparent margins before interpreting geometry arguments.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.validate:
        payload, image = load_style_file(args.validate)
        print(
            f"OK: {payload['name']} — {image.width}x{image.height}, "
            f"drawing width {payload['display']['drawingWidth']:g}px"
        )
        return 0

    if args.build:
        if not args.output or args.number_area is None or args.anchor is None:
            parser.error("--build requires --output, --number-area and --anchor.")
        image = open_artwork(args.build)
        if args.remove_background:
            image = remove_edge_connected_background(
                image, args.background_tolerance
            )
        if args.trim:
            image = trim_transparent(image)
        payload = build_style_payload(
            image,
            name=args.name,
            number_area=args.number_area,
            anchor=args.anchor,
            drawing_width=args.drawing_width,
            sidebar_width=args.sidebar_width,
            text_color=args.text_color,
            font_family=args.font_family,
            font_weight=args.font_weight,
            min_font_size=args.min_font_size,
            max_font_size=args.max_font_size,
            text_padding=args.text_padding,
        )
        write_style_file(args.output, payload)
        print(f"Wrote {args.output}")
        return 0

    try:
        run_gui()
    except BalloonStyleError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
