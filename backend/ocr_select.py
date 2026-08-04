"""Pick best PaddleOCR candidate — prioritize digit accuracy over symbol heuristics."""

from __future__ import annotations

import re


def digit_quality_score(text: str, confidence: float) -> float:
    """Higher = more trustworthy numeric read."""
    if not text or not text.strip():
        return 0.0

    t = text.strip()
    score = float(confidence) * 15.0

    digits = re.findall(r"\d", t)
    score += len(digits) * 2.5

    # Reward decimal structure (dimensions)
    if re.search(r"\d+\.\d+", t):
        score += 8.0
    if re.search(r"±|\+/-", t):
        score += 4.0
    if re.search(r"[ØøφΦ]", t):
        score += 3.0
    if re.search(r"±", t):
        score += 6.0

    # Comma often = misread ± (e.g. 215,370.05)
    if re.search(r"\d,\d{3}", t):
        score -= 12.0

    # Penalize garbage letters (mis-read noise)
    letters = re.sub(r"[ØøRr±°'\".\d\s]", "", t)
    score -= len(letters) * 4.0

    # Penalize very short or very long nonsense
    if len(digits) < 2:
        score -= 10.0
    if len(t) > 80:
        score -= 5.0

    return score


def pick_best_candidate(
    candidates: list[tuple[str, float]],
) -> tuple[str, float]:
    if not candidates:
        return "", 0.0

    best_text, best_conf = candidates[0]
    best_score = digit_quality_score(best_text, best_conf)

    for text, conf in candidates[1:]:
        s = digit_quality_score(text, conf)
        if s > best_score:
            best_text, best_conf, best_score = text, conf, s

    return best_text, best_conf
