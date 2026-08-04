"""Light OCR cleanup — normalize symbols without inventing Ø/±."""

from __future__ import annotations

import re

from angle_utils import normalize_primes


def count_engineering_symbols(text: str) -> int:
    return len(re.findall(r"[±°Ø'\"]", text))


def engineering_quality_score(text: str, confidence: float) -> float:
    if not text.strip():
        return 0.0
    score = confidence * 10.0
    score += count_engineering_symbols(text) * 3.0
    score += len(re.findall(r"\d", text)) * 0.5
    return score


def fix_engineering_symbols_light(text: str) -> str:
    """Spacing and explicit symbols; phi→Ø only at line start."""
    if not text:
        return ""

    t = normalize_primes(text.strip())
    t = t.replace("+/−", "±").replace("+/-", "±").replace("＋", "+")
    # Leading diameter / radius markers only (not φ inside numbers)
    t = re.sub(r"^[\s]*(?:⊕|∅|⌀|[φΦø])", "Ø", t)
    t = re.sub(r"^[\s]*([Rr])(?=\d)", r"R", t)
    t = re.sub(r"\s*([±°Ø'\"])\s*", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fix_engineering_symbols(text: str) -> str:
    return fix_engineering_symbols_light(text)


def merge_ocr_candidates(candidates: list[tuple[str, float]]) -> tuple[str, float]:
    if not candidates:
        return "", 0.0

    fixed = [(fix_engineering_symbols_light(t), c) for t, c in candidates if t.strip()]
    if not fixed:
        return "", 0.0

    best_text, best_conf = max(
        fixed,
        key=lambda x: engineering_quality_score(x[0], x[1]),
    )
    return best_text, best_conf
