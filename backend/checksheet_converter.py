"""Convert OCR inspection JSON into a Digital Checksheet template.

The conversion rules come from ``convert_ts2_to_checksheet.py`` supplied for
the POC.  This module removes the standalone script's editable test settings
and exposes two backend-friendly operations:

``convert_ts2_json_to_checksheet``
    Convert an already-loaded inspection payload in memory.

``save_checksheet_template``
    Convert that payload and atomically create/replace ``<TEMPLATE_ID>.json``
    inside a server-configured directory.

The browser never supplies a filesystem path.  Keeping the destination on the
server prevents accidental writes outside the Digital Checksheet template
directory and lets deployment change only ``CHECKSHEET_TEMPLATE_DIR``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CREATED_BY = "PPTPL"
VERSION = 1
TEMPLATE_DIRECTORY_ENV = "CHECKSHEET_TEMPLATE_DIR"

# Layout settings copied from the supplied converter.  Part input columns
# intentionally receive about 47% of the total row width.
WIDTH_SNO = 3.0
WIDTH_LABEL = 18.0
WIDTH_VALUE = 9.0
WIDTH_TOLERANCE = 7.0
WIDTH_METHOD = 6.0
WIDTH_TOOL = 10.0
WIDTH_PARTS_TOTAL = 47.0

DEFAULT_FONT_SIZE = 12
INPUT_FONT_SIZE = 11
DEFAULT_TOLERANCE = 0.0

# Unsigned decimal used by tolerance expressions. Leading-decimal OCR forms
# such as .1 are accepted as well as 0.1 and whole numbers.
_TOLERANCE_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"


class ChecksheetConfigurationError(RuntimeError):
    """Raised when the server has no template destination configured."""


@dataclass(frozen=True)
class SavedChecksheetTemplate:
    """Details of one template file written by the converter service."""

    path: Path
    template: dict[str, Any]
    replaced: bool


# ---------------------------------------------------------------------------
# File and naming helpers
# ---------------------------------------------------------------------------


def get_configured_template_dir() -> Path:
    """Return the deployment-specific Digital Checksheet template directory."""
    raw = os.environ.get(TEMPLATE_DIRECTORY_ENV, "").strip()
    if not raw:
        raise ChecksheetConfigurationError(
            f"{TEMPLATE_DIRECTORY_ENV} is not configured. Set it to the "
            "Digital Checksheet data/templates directory."
        )

    # expandvars supports normal environment-variable based deployment paths;
    # expanduser also permits a portable user-relative path where appropriate.
    return Path(os.path.expandvars(raw)).expanduser()


def load_json(file_path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON object from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source JSON not found: {path}")

    with path.open("r", encoding="utf-8") as source_file:
        result = json.load(source_file)

    if not isinstance(result, dict):
        raise ValueError("Invalid source JSON: expected a JSON object.")
    return result


def save_json(
    file_path: str | Path,
    data: dict[str, Any],
    *,
    overwrite: bool = True,
) -> bool:
    """Atomically save JSON and return whether an existing file was replaced.

    The temporary file is created beside the target, flushed, and then moved
    into place with ``os.replace``.  Readers therefore see either the previous
    complete template or the new complete template, never a partially-written
    JSON document.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()

    if existed and not overwrite:
        raise FileExistsError(
            f"Target JSON already exists and overwrite=False: {path}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(data, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return existed


def _portable_basename(file_name: str | Path) -> str:
    """Extract a basename consistently from Windows or POSIX-style input."""
    return re.split(r"[\\/]", str(file_name))[-1]


def make_template_id_from_filename(source_file_path: str | Path) -> str:
    """Convert an inspection filename to a safe uppercase template id."""
    file_name = _portable_basename(source_file_path)
    stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", stem)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Source filename cannot produce a valid template id.")
    return cleaned.upper()


def make_template_name_from_id(template_id: str) -> str:
    """Convert ``TS2_EXAMPLE_TEMPLATE`` into a human-readable name."""
    title_parts: list[str] = []
    for part in template_id.split("_"):
        if part.upper() == "TS2":
            title_parts.append("TS2")
        elif part.isdigit():
            title_parts.append(part)
        else:
            title_parts.append(part.capitalize())
    return " ".join(title_parts)


def make_safe_id_part(value: Any) -> str:
    """Create a safe field-id fragment from a row number or column name."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "x"


# ---------------------------------------------------------------------------
# Numeric/range parsing helpers
# ---------------------------------------------------------------------------


def round_for_json(value: float, digits: int = 6) -> int | float:
    """Round floating-point noise and emit whole numbers as JSON integers."""
    rounded = round(value, digits)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def contains_angle_symbols(text: str) -> bool:
    """Return whether text contains degree, minute, or second markers."""
    return any(symbol in text for symbol in ["°", "'", '"', "′", "″"])


def first_number(text: str) -> float | None:
    """Return the first signed integer or decimal found in a string."""
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def nominal_value_for_range(value_text: str) -> float | None:
    """Return a decimal-degree nominal for angles, or the first normal number.

    Valid degree/minute/second forms preserve readable minutes and seconds.
    If anything after the degree marker is malformed, only the degree value is
    used. The original OCR/display text is never changed.
    """
    text = str(value_text or "").strip()
    if not text:
        return None

    angle_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*°", text)
    if angle_match is None:
        return first_number(text)

    degrees = float(angle_match.group(1))
    remainder = text[angle_match.end():].strip()
    if not remainder:
        return degrees

    dms_match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*['′]"
        r'(?:\s*(\d+(?:\.\d+)?)\s*["″])?',
        remainder,
    )
    if dms_match is None:
        return degrees

    minutes = float(dms_match.group(1))
    seconds = float(dms_match.group(2) or 0)
    if not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return degrees

    fraction = minutes / 60 + seconds / 3600
    return degrees - fraction if degrees < 0 else degrees + fraction


def parse_tolerance_text(tolerance_text: str) -> tuple[float, float] | None:
    """Return (lower, upper), rejecting non-empty malformed expressions."""
    text = str(tolerance_text or "").strip()
    if not text:
        return None

    plain_match = re.fullmatch(rf"({_TOLERANCE_NUMBER})\s*°?", text)
    if plain_match:
        tolerance = float(plain_match.group(1))
        return tolerance, tolerance

    plus_minus_match = re.fullmatch(
        rf"±\s*({_TOLERANCE_NUMBER})\s*°?",
        text,
    )
    if plus_minus_match:
        tolerance = float(plus_minus_match.group(1))
        return tolerance, tolerance

    asymmetric_match = re.fullmatch(
        rf"\+\s*({_TOLERANCE_NUMBER})\s*°?\s*,?\s*"
        rf"-\s*({_TOLERANCE_NUMBER})\s*°?",
        text,
    )
    if asymmetric_match:
        upper_tolerance = float(asymmetric_match.group(1))
        lower_tolerance = float(asymmetric_match.group(2))
        return lower_tolerance, upper_tolerance

    raise ValueError(
        f"Malformed tolerance expression: {text!r}. "
        "Use a number, ±x, or +x -y."
    )


def _has_embedded_tolerance_intent(text: str) -> bool:
    """Return whether text after its nominal number contains tolerance signs."""
    nominal_match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if nominal_match is None:
        return any(symbol in text for symbol in ("±", "+", "-"))
    return any(
        symbol in text[nominal_match.end():]
        for symbol in ("±", "+", "-")
    )


def parse_range_from_value_text(value_text: str) -> dict[str, int | float] | None:
    """Parse ranges embedded directly in a value/specification string."""
    text = str(value_text or "").strip()
    if not text or contains_angle_symbols(text):
        return None

    range_match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*~\s*([-+]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        return {
            "min": round_for_json(float(range_match.group(1))),
            "max": round_for_json(float(range_match.group(2))),
        }

    plus_minus_match = re.fullmatch(
        rf"\s*([-+]?\d+(?:\.\d+)?)\s*±\s*"
        rf"({_TOLERANCE_NUMBER})\s*",
        text,
        flags=re.IGNORECASE,
    )
    if plus_minus_match:
        nominal = float(plus_minus_match.group(1))
        tolerance = float(plus_minus_match.group(2))
        return {
            "min": round_for_json(nominal - tolerance),
            "max": round_for_json(nominal + tolerance),
        }

    asymmetric_match = re.fullmatch(
        rf"\s*([-+]?\d+(?:\.\d+)?)\s*"
        rf"\+\s*({_TOLERANCE_NUMBER})\s*,?\s*"
        rf"-\s*({_TOLERANCE_NUMBER})\s*",
        text,
        flags=re.IGNORECASE,
    )
    if asymmetric_match:
        nominal = float(asymmetric_match.group(1))
        upper_tolerance = float(asymmetric_match.group(2))
        lower_tolerance = float(asymmetric_match.group(3))
        return {
            "min": round_for_json(nominal - lower_tolerance),
            "max": round_for_json(nominal + upper_tolerance),
        }

    maximum_match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:max|maximum)\b",
        text,
        flags=re.IGNORECASE,
    )
    if maximum_match:
        return {
            "min": 0,
            "max": round_for_json(float(maximum_match.group(1))),
        }

    minimum_match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:min|minimum)\b",
        text,
        flags=re.IGNORECASE,
    )
    if minimum_match:
        return {"min": round_for_json(float(minimum_match.group(1)))}

    # Do not silently apply the zero default when OCR detected tolerance-like
    # punctuation but the complete expression is not one of the valid forms.
    if _has_embedded_tolerance_intent(text):
        raise ValueError(
            f"Malformed value/tolerance expression: {text!r}. "
            "Use z ±x or z +x -y."
        )

    return None


def parse_range(
    value_text: str,
    tolerance_text: str,
) -> dict[str, int | float] | None:
    """Parse a range and use zero only when no tolerance was supplied."""
    value_text = str(value_text or "").strip()
    tolerance_text = str(tolerance_text or "").strip()

    nominal = nominal_value_for_range(value_text)
    parsed_tolerance = parse_tolerance_text(tolerance_text)
    if nominal is not None and parsed_tolerance is not None:
        lower_tolerance, upper_tolerance = parsed_tolerance
        return {
            "min": round_for_json(nominal - lower_tolerance),
            "max": round_for_json(nominal + upper_tolerance),
        }

    embedded_range = parse_range_from_value_text(value_text)
    if embedded_range is not None:
        return embedded_range

    if nominal is not None:
        return {
            "min": round_for_json(nominal - DEFAULT_TOLERANCE),
            "max": round_for_json(nominal + DEFAULT_TOLERANCE),
        }

    return None


# ---------------------------------------------------------------------------
# Digital Checksheet node builders
# ---------------------------------------------------------------------------


def lang_label(text: Any) -> dict[str, str]:
    """Build the bilingual label shape expected by Digital Checksheet."""
    safe_text = "" if text is None else str(text)
    return {"en": safe_text, "hi": safe_text}


def label_node(
    text: Any,
    *,
    bold: bool = False,
    font_size: int = DEFAULT_FONT_SIZE,
    horizontal_alignment: str = "left",
    vertical_alignment: str = "center",
) -> dict[str, Any]:
    """Build a static label node."""
    style: dict[str, Any] = {
        "font_size": font_size,
        "vertical_alignment": vertical_alignment,
        "horizontal_alignment": horizontal_alignment,
    }
    if bold:
        style["bold"] = True

    return {"type": "Label", "label": lang_label(text), "style": style}


def input_node(
    field_id: str,
    *,
    range_obj: dict[str, int | float] | None = None,
    required: bool = False,
) -> dict[str, Any]:
    """Build a numeric range input or an unrestricted text input."""
    if range_obj:
        return {
            "id": field_id,
            "type": "number",
            "input_type": "text",
            "range": range_obj,
            "required": required,
            "style": {"font_size": INPUT_FONT_SIZE},
        }

    return {
        "id": field_id,
        "type": "text",
        "input_type": "text",
        "required": required,
        "style": {"font_size": INPUT_FONT_SIZE},
    }


def section_cell(
    *,
    width: float,
    items: list[dict[str, Any]],
    orientation: str = "Vertical",
    fill: str = "uniform",
) -> dict[str, Any]:
    """Wrap one or more nodes inside a Section cell."""
    return {
        "repeat_per_shift": "true/false",
        "depends_on_sections": [],
        "type": "Section",
        "Orientation": orientation,
        "items": items,
        "fill": fill,
        "style": {"width": f"{round_for_json(width)}%"},
    }


def label_cell(
    text: Any,
    *,
    width: float,
    bold: bool = False,
    horizontal_alignment: str = "left",
    font_size: int = DEFAULT_FONT_SIZE,
) -> dict[str, Any]:
    """Build a static label cell."""
    return section_cell(
        width=width,
        items=[
            label_node(
                text,
                bold=bold,
                font_size=font_size,
                horizontal_alignment=horizontal_alignment,
            )
        ],
    )


def input_cell(
    field_id: str,
    *,
    width: float,
    range_obj: dict[str, int | float] | None,
) -> dict[str, Any]:
    """Build a part input cell."""
    return section_cell(
        width=width,
        items=[input_node(field_id, range_obj=range_obj, required=False)],
    )


def horizontal_row(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one horizontal table row."""
    return {
        "repeat_per_shift": "true/false",
        "depends_on_sections": [],
        "type": "Section",
        "Orientation": "Horizontal",
        "items": items,
        "fill": "uniform",
    }


# ---------------------------------------------------------------------------
# Template row builders
# ---------------------------------------------------------------------------


def get_extra_columns(source_json: dict[str, Any]) -> list[str]:
    """Read explicit part columns or infer keys named ``part1``, ``part2``..."""
    extra_columns = source_json.get("extra_columns")
    if isinstance(extra_columns, list) and extra_columns:
        return [str(column) for column in extra_columns]

    rows = source_json.get("data", [])
    discovered: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            if re.fullmatch(r"part\d+", str(key), flags=re.IGNORECASE):
                discovered.add(str(key))

    def part_sort_key(value: str) -> int:
        match = re.search(r"\d+", value)
        return int(match.group(0)) if match else 999999

    return sorted(discovered, key=part_sort_key)


def build_table_header_row(extra_columns: list[str]) -> dict[str, Any]:
    """Build the checksheet table header row."""
    part_width = WIDTH_PARTS_TOTAL / max(len(extra_columns), 1)
    cells: list[dict[str, Any]] = [
        label_cell(
            "S.No",
            width=WIDTH_SNO,
            bold=True,
            horizontal_alignment="center",
        ),
        label_cell(
            "Label",
            width=WIDTH_LABEL,
            bold=True,
            horizontal_alignment="center",
        ),
        label_cell(
            "Value",
            width=WIDTH_VALUE,
            bold=True,
            horizontal_alignment="center",
        ),
        label_cell(
            "Tolerance",
            width=WIDTH_TOLERANCE,
            bold=True,
            horizontal_alignment="center",
        ),
        label_cell(
            "Method",
            width=WIDTH_METHOD,
            bold=True,
            horizontal_alignment="center",
        ),
        label_cell(
            "Tool",
            width=WIDTH_TOOL,
            bold=True,
            horizontal_alignment="center",
        ),
    ]

    for column in extra_columns:
        cells.append(
            label_cell(
                column,
                width=part_width,
                bold=True,
                horizontal_alignment="center",
            )
        )
    return horizontal_row(cells)


def build_data_row(
    row: dict[str, Any],
    extra_columns: list[str],
) -> dict[str, Any]:
    """Build one Digital Checksheet row from one inspection JSON row."""
    serial_number = row.get("S.no", row.get("S.No", row.get("sno", "")))
    label = row.get("Label", "")
    value = row.get("Value", "")
    tolerance = row.get("Tolerance", "")
    method = row.get("Method", "")
    tool = row.get("Tool", "")

    row_id = make_safe_id_part(serial_number)
    part_width = WIDTH_PARTS_TOTAL / max(len(extra_columns), 1)
    range_obj = parse_range(
        value_text=str(value or ""),
        tolerance_text=str(tolerance or ""),
    )

    cells: list[dict[str, Any]] = [
        label_cell(
            serial_number,
            width=WIDTH_SNO,
            horizontal_alignment="center",
        ),
        label_cell(label, width=WIDTH_LABEL),
        label_cell(value, width=WIDTH_VALUE),
        label_cell(tolerance, width=WIDTH_TOLERANCE),
        label_cell(method, width=WIDTH_METHOD),
        label_cell(tool, width=WIDTH_TOOL),
    ]

    for column in extra_columns:
        column_id = make_safe_id_part(column)
        cells.append(
            input_cell(
                f"row_{row_id}_{column_id}",
                width=part_width,
                range_obj=range_obj,
            )
        )
    return horizontal_row(cells)


# ---------------------------------------------------------------------------
# Public conversion and save operations
# ---------------------------------------------------------------------------


def convert_ts2_json_to_checksheet(
    source_json: dict[str, Any],
    *,
    template_id: str,
    template_name: str,
    created_by: str = CREATED_BY,
    version: int = VERSION,
) -> dict[str, Any]:
    """Convert an already-loaded inspection JSON object into a template."""
    rows = source_json.get("data")
    if not isinstance(rows, list):
        raise ValueError("Invalid source JSON: expected key 'data' to be a list.")

    extra_columns = get_extra_columns(source_json)
    if not extra_columns:
        raise ValueError(
            "Invalid source JSON: no extra_columns found and no partN columns "
            "could be inferred."
        )

    items: list[dict[str, Any]] = [build_table_header_row(extra_columns)]
    for row in rows:
        if isinstance(row, dict):
            items.append(build_data_row(row, extra_columns))

    return {
        "template_id": template_id,
        "template_name": {"en": template_name, "hi": template_name},
        "description": (
            "Auto-generated checksheet template converted from TS2 source JSON. "
            "Static inspection data with range-limited part input columns where "
            "possible."
        ),
        "version": version,
        "created_by": created_by,
        "items": items,
    }


def save_checksheet_template(
    *,
    source_json: dict[str, Any],
    source_file_name: str,
    target_dir: str | Path | None = None,
    overwrite: bool = True,
    created_by: str = CREATED_BY,
    version: int = VERSION,
) -> SavedChecksheetTemplate:
    """Convert one in-memory export and save it in the template directory."""
    template_id = make_template_id_from_filename(source_file_name)
    template_name = make_template_name_from_id(template_id)
    template = convert_ts2_json_to_checksheet(
        source_json,
        template_id=template_id,
        template_name=template_name,
        created_by=created_by,
        version=version,
    )

    destination = (
        Path(target_dir)
        if target_dir is not None
        else get_configured_template_dir()
    )
    target_path = destination / f"{template_id}.json"
    replaced = save_json(target_path, template, overwrite=overwrite)
    return SavedChecksheetTemplate(
        path=target_path,
        template=template,
        replaced=replaced,
    )


def convert_ts2_file_to_checksheet(
    *,
    source_file_path: str | Path,
    target_dir: str | Path,
    target_file_name: str | None = None,
    overwrite: bool = True,
    created_by: str = CREATED_BY,
    version: int = VERSION,
) -> Path:
    """Compatibility wrapper for the supplied converter's file-based API."""
    source_path = Path(source_file_path)
    source_json = load_json(source_path)
    template_id = make_template_id_from_filename(source_path.name)
    template = convert_ts2_json_to_checksheet(
        source_json,
        template_id=template_id,
        template_name=make_template_name_from_id(template_id),
        created_by=created_by,
        version=version,
    )
    final_file_name = (
        _portable_basename(target_file_name)
        if target_file_name
        else f"{template_id}.json"
    )
    target_path = Path(target_dir) / final_file_name
    save_json(target_path, template, overwrite=overwrite)
    return target_path
