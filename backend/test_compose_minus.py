"""
Tests for the leading-minus (misread Ø/R prefix) repair in dimension_compose.

Run: PYTHONPATH=. python test_compose_minus.py
"""

from __future__ import annotations

from dimension_compose import compose_engineering_dimension
from symbol_vision import DetectedSymbols


def _check(name: str, cond: bool, detail: str = "") -> bool:
    print(("OK   " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    return cond


def main() -> int:
    fails = 0

    # Ø read as a leading minus, vision corroborates -> Ø restored (both dashes).
    for raw in ["-0.69±0.03", "−0.69±0.03"]:
        s = DetectedSymbols(diameter=True, diameter_score=0.4, plus_minus=True)
        c = compose_engineering_dimension(raw, None, s)
        fails += not _check(
            f"phi restored {raw!r}",
            c.text == "Ø0.69±0.03" and c.kind == "diameter",
            f"got {c.text!r}/{c.kind}",
        )

    # No symbol evidence: the spurious minus is still removed (value correct).
    c = compose_engineering_dimension("−0.69±0.03", None, DetectedSymbols(plus_minus=True))
    fails += not _check("minus dropped w/o Ø", c.text == "0.69±0.03",
                        f"got {c.text!r}")

    # Radius glyph misread as a dash -> R restored.
    c = compose_engineering_dimension("-2.70", None, DetectedSymbols(radius=True))
    fails += not _check("radius restored", c.text == "R2.70",
                        f"got {c.text!r}")

    # A plain value that isn't negative — leading dash still dropped.
    c = compose_engineering_dimension("-5.0", None, DetectedSymbols())
    fails += not _check("plain dash dropped", c.text == "5.0", f"got {c.text!r}")

    # Real Ø text must be untouched (no false strip / double prefix).
    c = compose_engineering_dimension("Ø174.07±0.05", None,
                                      DetectedSymbols(diameter=True, diameter_score=1.0))
    fails += not _check("real Ø untouched", c.text == "Ø174.07±0.05",
                        f"got {c.text!r}")

    print("\nPASS" if not fails else f"\n{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_no_phi_from_vertical_layout_alone():
    """A vertical ±two-decimal value is NOT promoted to Ø without ring evidence."""
    from PIL import Image

    tall = Image.new("RGB", (40, 160), "white")
    s = DetectedSymbols(plus_minus=True, diameter_score=0.45)
    c = compose_engineering_dimension("3.81±0.20", tall, s)
    assert c.text == "3.81±0.20" and c.kind == "tolerance"


def test_phi_from_topology_evidence():
    from PIL import Image

    tall = Image.new("RGB", (40, 160), "white")
    s = DetectedSymbols(diameter=True, diameter_score=0.92, plus_minus=True)
    c = compose_engineering_dimension("215.37±0.05", tall, s)
    assert c.text == "Ø215.37±0.05" and c.kind == "diameter"


def test_junk_letter_prefix_becomes_phi_with_topology():
    s = DetectedSymbols(diameter=True, diameter_score=0.92, plus_minus=True)
    c = compose_engineering_dimension("W215.37±0.05", None, s)
    assert c.text == "Ø215.37±0.05"
    # Without strong topology evidence the letter is left alone.
    c = compose_engineering_dimension("W215.37±0.05", None, DetectedSymbols(diameter_score=0.45))
    assert c.text == "W215.37±0.05"


def test_chamfer_degree_rule():
    c = compose_engineering_dimension("0.5×45", None, DetectedSymbols())
    assert c.text == "0.5×45°" and c.kind == "angle"
    c = compose_engineering_dimension("0.2-0.3 X 45", None, DetectedSymbols())
    assert c.text == "0.2-0.3×45°"
    # Spline / non-standard products are not chamfers.
    c = compose_engineering_dimension("20×19×1", None, DetectedSymbols())
    assert "°" not in c.text
    c = compose_engineering_dimension("2×7", None, DetectedSymbols())
    assert "°" not in c.text


def test_angle_tail_stripped():
    c = compose_engineering_dimension("12°±3°1", None, DetectedSymbols(plus_minus=True))
    assert c.text == "12°±3°"


def test_leading_stroke_is_not_phi_evidence():
    """A clipped leader stroke is removed, but it does not imply a Ø."""
    c = compose_engineering_dimension("/0.5×45°", None, DetectedSymbols())
    assert c.text == "0.5×45°", c.text
    c = compose_engineering_dimension("\\12.55", None, DetectedSymbols())
    assert c.text == "12.55", c.text
    # A slash between deviations is untouched.
    s = DetectedSymbols(diameter=True, diameter_score=0.92)
    c = compose_engineering_dimension("20H10 +0.084/0", None, s)
    assert c.text == "Ø20H10+0.084/0", c.text
