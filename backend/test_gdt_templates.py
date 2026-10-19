"""Foundation tests for procedural GD&T symbol templates.

These tests cover only the registry/template contract and rendering safety.
They do not enable or test page-level glyph detection.
"""

from __future__ import annotations

import numpy as np
import pytest

import gdt_registry as reg
import gdt_templates as tpl


requires_cv2 = pytest.mark.skipif(
    tpl.cv2 is None,
    reason="OpenCV is required to render procedural GD&T templates",
)


TEMPLATED = sorted(
    {
        symbol.template
        for symbol in reg.ALL_SYMBOLS
        if symbol.template and tpl.has_template(symbol.template)
    }
)


def test_every_registry_template_has_a_builder() -> None:
    missing = {
        symbol.template
        for symbol in reg.ALL_SYMBOLS
        if symbol.template and not tpl.has_template(symbol.template)
    }
    assert not missing, f"registry names templates with no builder: {missing}"


@pytest.mark.parametrize("name", TEMPLATED)
@requires_cv2
def test_each_template_draws_a_plausible_amount_of_ink(name: str) -> None:
    rendered = tpl.render(name, 48)
    ink_fraction = float((rendered > 127).mean())
    assert 0.02 < ink_fraction < 0.6, (
        f"{name} ink fraction {ink_fraction:.3f}"
    )


@requires_cv2
def test_templates_use_expected_shape_type_and_polarity() -> None:
    rendered = tpl.render("circularity", 48)
    assert rendered.shape == (48, 48)
    assert rendered.dtype == np.uint8
    assert rendered[0, 0] == tpl.BG == 0
    assert rendered.max() == tpl.INK == 255


@requires_cv2
def test_cached_render_returns_independent_arrays() -> None:
    first = tpl.render_cached("position", 48)
    second = tpl.render_cached("position", 48)
    assert np.array_equal(first, second)

    first[:] = tpl.BG
    third = tpl.render_cached("position", 48)
    assert third.max() == tpl.INK


@requires_cv2
def test_unknown_template_is_rejected() -> None:
    with pytest.raises(KeyError):
        tpl.render("not_a_gdt_template", 48)


def test_scale_selection_is_deduplicated_and_has_a_safe_floor() -> None:
    assert tpl.scales_for(1) == (10,)
    scales = tpl.scales_for(40, (1.0, 1.0, 1.25))
    assert scales == (40, 50)


@requires_cv2
def test_contact_sheet_contains_every_template_tile() -> None:
    size = 32
    columns = 7
    pad = 4
    sheet = tpl.contact_sheet(size=size, columns=columns, pad=pad)
    expected_rows = (len(tpl.template_names()) + columns - 1) // columns
    cell = size + 2 * pad
    assert sheet.shape == (expected_rows * cell, columns * cell)
    assert sheet.dtype == np.uint8
    assert sheet.max() == tpl.INK
