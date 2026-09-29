"""
Tests for dual_unit: inch [mm] cross-check + digit repair.

Run: PYTHONPATH=. python test_dual_unit.py
"""

from __future__ import annotations

from dual_unit import repair_dual


def _check(name: str, cond: bool) -> bool:
    print(("OK   " if cond else "FAIL ") + name)
    return cond


def main() -> int:
    fails = 0

    # --- Already-consistent dual pairs pass untouched (from the CUP drawing). ---
    for text in [
        ".30 [7.62]",
        "1.08 [27.43]",
        "6.13 [155.70]",
        "2.03 [51.56]",
        ".970 [24.64]",
        "R2.70 [68.58]",
        "Ø.81 [20.57]",
        "3.84 [97.53]",
    ]:
        r = repair_dual(text)
        fails += not _check(f"consistent {text!r} -> {r.status}",
                            r.status == "consistent")

    # --- No bracket -> absent (single-unit sheets untouched). ---
    for text in ["215.37", "Ø174.07±0.05", "32°20'40\"", "FULL RADIUS TYP"]:
        r = repair_dual(text)
        fails += not _check(f"absent {text!r} -> {r.status}",
                            r.status == "absent")

    # --- Repair a '1' misread as '7' on the primary using the mm value. ---
    r = repair_dual("7.08 [27.43]")
    fails += not _check(f"repair 7.08->1.08 got {r.text!r} ({r.status})",
                        r.status == "repaired" and r.text == "1.08 [27.43]")

    # --- Repair a '4' misread as '1' inside the bracket using the inch value. ---
    r = repair_dual("1.08 [27.13]")
    fails += not _check(f"repair 27.13->27.43 got {r.text!r} ({r.status})",
                        r.status == "repaired" and "27.43" in r.text)

    # --- Repair a spurious '1' inserted into the integer part (deletion). ---
    r = repair_dual("11.08 [27.43]")
    fails += not _check(f"repair 11.08->1.08 got {r.text!r} ({r.status})",
                        r.status == "repaired" and r.text == "1.08 [27.43]")

    # --- Preserve prefix/format on repair. ---
    r = repair_dual("R2.10 [68.58]")
    fails += not _check(f"repair R2.10->R2.70 got {r.text!r} ({r.status})",
                        r.status == "repaired" and r.text == "R2.70 [68.58]")

    # --- Genuinely inconsistent, unrepairable -> flagged, not mangled. ---
    r = repair_dual("5.00 [999.99]")
    fails += not _check(f"inconsistent 5.00 [999.99] -> {r.status}",
                        r.status == "inconsistent" and r.needs_review)

    # --- mm-first sheets auto-detect direction. ---
    r = repair_dual("27.43 [1.08]")
    fails += not _check(f"mm-first 27.43 [1.08] -> {r.status}/{r.direction}",
                        r.status == "consistent" and r.direction == "mm_inch")

    print("\nPASS" if not fails else f"\n{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
