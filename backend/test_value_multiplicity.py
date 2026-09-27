"""
Tests for segment_quality.count_dimension_values (content-aware split trigger).

Run: PYTHONPATH=. python test_value_multiplicity.py
"""

from __future__ import annotations

from segment_quality import count_dimension_values


def _check(name: str, cond: bool) -> bool:
    print(("OK   " if cond else "FAIL ") + name)
    return cond


def main() -> int:
    fails = 0

    # Two stacked dual-unit callouts fused into one read -> split trigger.
    fails += not _check(
        "two dual pairs -> 2",
        count_dimension_values("1.28 [32.51] 1.10 [27.94]") == 2,
    )
    fails += not _check(
        "glued dual pairs -> 2",
        count_dimension_values("2.32[58.93]2.23[56.64]") == 2,
    )

    # Two Ø feature callouts fused into one read -> split trigger.
    fails += not _check(
        "two Ø callouts -> 2",
        count_dimension_values("Ø20H10 +0.084 0 Ø18H10 +0.070 0") == 2,
    )
    # One Ø value with a fit + tolerance is a single value.
    fails += not _check("single Ø callout -> 1",
                        count_dimension_values("Ø20H10 +0.084 0") == 1)

    # Single values / tolerances / angles are one value -> no split.
    fails += not _check("single dual -> 1",
                        count_dimension_values("1.08 [27.43]") == 1)
    fails += not _check("tolerance -> 1",
                        count_dimension_values("Ø215.37±0.05") == 1)
    fails += not _check("angle -> 1",
                        count_dimension_values("32°20'40\"") == 1)
    fails += not _check("plain value -> 1",
                        count_dimension_values("6.13") == 1)

    print("\nPASS" if not fails else f"\n{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
