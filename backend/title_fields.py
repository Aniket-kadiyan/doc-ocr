"""
Read configured title-block fields ("DWG NO.", "REV", "DATE"…) off a drawing.

The title block is the label/value table in the bottom-right corner. None of it
is a dimension, so the segmenter drops every box in it. The user configures the
labels worth keeping ("Keywords to look for") and this module pairs each one
with its value.

Three layouts appear in practice, all seen on real sheets:

  "DATE: 1/3/14"      one box, label and value split by the colon
  "DWG NO." "GEP5A34" two boxes, the value to the RIGHT of the label
  "REV" / "2"         two boxes, the value BELOW the label

Matching is deliberately conservative: a keyword row is only produced from a box
that actually reads as that label, so a value is never invented. A keyword that
is not found at all is reported with an empty value, which keeps a failed read
visible on the sheet instead of silently dropping the field.
"""

from __future__ import annotations

import re
from typing import Any


# Label chrome that is not part of the field name: "DWG NO." -> "DWG",
# "CHECKED BY:" -> "CHECKED". Keeps matching stable across sheets that write
# the same field slightly differently.
_LABEL_CHROME = re.compile(r"\b(NO|NUM|NUMBER|BY)\b\.?", re.I)
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")

# A value cell that is really just another label. Without this, "CHECKED BY:"
# sitting under "DRAWN: DX" would be read as the value of DRAWN.
_LOOKS_LIKE_LABEL = re.compile(r"^[A-Z][A-Z .\'/]*:\s*$")


def _normalize(text: str) -> str:
    """Upper-case, strip label chrome and punctuation, for keyword matching."""
    return _NON_ALNUM.sub(" ", _LABEL_CHROME.sub(" ", (text or "").upper())).strip()


def _contains_words(haystack: str, needle: str) -> bool:
    """
    True when `needle`'s words appear as a contiguous run of whole words.

    Substring matching is wrong here: a "REV" keyword would otherwise match the
    word REVISION inside a general note and pull a value out of the middle of a
    paragraph. Both sides are already normalized to space-separated tokens.
    """
    h = haystack.split()
    n = needle.split()
    if not n or len(n) > len(h):
        return False
    return any(h[i : i + len(n)] == n for i in range(len(h) - len(n) + 1))


def _is_value_like(text: str) -> bool:
    """True when a box could be somebody's answer rather than another label."""
    t = (text or "").strip()
    if not t:
        return False
    # The label's own chrome is not its value. "DWG NO." splits into separate
    # boxes at some scales, and without this the "NO." box sitting beside "DWG"
    # was read as the drawing number.
    if not _LABEL_CHROME.sub("", t).strip(" .:-"):
        return False
    return not _LOOKS_LIKE_LABEL.match(t)


def _split_inline(text: str) -> tuple[str, str] | None:
    """
    Split "DATE: 1/3/14" into ("DATE", "1/3/14").

    A colon is decisive: it says the value belongs on this line, so "SCALE:"
    with nothing after it is an EMPTY field, not an invitation to read the next
    row of the table. Returning ("SCALE", "") stops the neighbour search, which
    is what keeps "SCALE:" from swallowing the "SHEET 1 OF 4" printed under it.
    """
    if ":" not in text:
        return None
    label, _, value = text.partition(":")
    label = label.strip()
    if not label:
        return None
    return label, value.strip()


def _split_by_keyword(text: str, keyword: str) -> tuple[str, str] | None:
    """
    Split a colon-less "SHEET 1 OF 4" into ("SHEET", "1 OF 4").

    Label chrome is removed first so "DWG NO." leaves no remainder and falls
    through to the neighbour search, where its value actually is.
    """
    cleaned = re.sub(r"\s+", " ", _LABEL_CHROME.sub(" ", text)).strip()
    needle = keyword.strip().upper()
    if not needle or not cleaned.upper().startswith(needle):
        return None
    rest = cleaned[len(needle):].strip(" :.-")
    if not rest:
        return None
    return cleaned[: len(needle)].strip(), rest


def _vertical_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Shared height of two boxes as a fraction of the shorter one."""
    top = max(a["y"], b["y"])
    bottom = min(a["y"] + a["h"], b["y"] + b["h"])
    shorter = max(min(a["h"], b["h"]), 1.0)
    return max(0.0, bottom - top) / shorter


def _horizontal_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Shared width of two boxes as a fraction of the narrower one."""
    left = max(a["x"], b["x"])
    right = min(a["x"] + a["w"], b["x"] + b["w"])
    narrower = max(min(a["w"], b["w"]), 1.0)
    return max(0.0, right - left) / narrower


def _value_to_the_right(
    label: dict[str, Any], boxes: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Nearest value box on the label's own line, to its right."""
    best, best_gap = None, None
    for b in boxes:
        if b is label or not _is_value_like(b.get("text", "")):
            continue
        gap = b["x"] - (label["x"] + label["w"])
        # A value sits close to its label; anything past a few label widths is
        # in the next cell of the table.
        if gap < -0.25 * label["w"] or gap > 4.0 * label["h"]:
            continue
        if _vertical_overlap(label, b) < 0.4:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = b, gap
    return best


def _value_below(
    label: dict[str, Any], boxes: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """
    The value directly under the label, within the same cell.

    When several values are stacked in the column the label is not a cell label
    but a TABLE HEADING — a revisions table's "REV" column is the case that
    matters — and the answer is the bottom entry, because a revisions table is
    appended downward and the drawing's current revision is its last row.
    """
    column: list[tuple[float, dict[str, Any]]] = []
    for b in boxes:
        if b is label or not _is_value_like(b.get("text", "")):
            continue
        gap = b["y"] - (label["y"] + label["h"])
        if gap < -0.25 * label["h"] or gap > 2.5 * label["h"]:
            continue
        if _horizontal_overlap(label, b) < 0.3:
            continue
        column.append((gap, b))

    if not column:
        return None

    column.sort(key=lambda pair: pair[0])
    # Follow the column down only while the entries look like more rows of the
    # SAME column: stacked under the first value, and of comparable width. A
    # revision table's "1" over "2" qualifies; the "SCALE: N.T.S." cell sitting
    # under a title block's REV cell does not, and following it there returned
    # the scale as the revision.
    first = column[0][1]
    run = [column[0]]
    for gap, b in column[1:]:
        previous = run[-1][1]
        step = b["y"] - (previous["y"] + previous["h"])
        if step < -0.25 * label["h"] or step > 2.5 * label["h"]:
            break
        if _horizontal_overlap(first, b) < 0.6:
            break
        wider = max(b["w"], first["w"])
        narrower = max(min(b["w"], first["w"]), 1.0)
        if wider / narrower > 2.0:
            break
        run.append((gap, b))

    return run[-1][1] if len(run) > 1 else run[0][1]


def _bottom_right(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The boxes in the bottom-right quarter of whatever was passed in.

    The title block lives in that corner of a drawing sheet, and searching it
    first keeps a keyword from matching the same word somewhere in the middle of
    the sheet — a "DATE" in a revision history, or "REV" inside a general note.
    """
    if not boxes:
        return []
    right = max(b["x"] + b["w"] for b in boxes)
    bottom = max(b["y"] + b["h"] for b in boxes)
    left = min(b["x"] for b in boxes)
    top = min(b["y"] for b in boxes)
    x_cut = left + 0.5 * (right - left)
    y_cut = top + 0.5 * (bottom - top)
    return [
        b
        for b in boxes
        if (b["x"] + b["w"]) > x_cut and (b["y"] + b["h"]) > y_cut
    ]


def extract_title_fields(
    boxes: list[dict[str, Any]],
    keywords: list[str],
) -> list[dict[str, Any]]:
    """
    Pair each configured keyword with its value from the title block.

    Returns one entry per keyword, in the order the user configured them:
    ``{"keyword", "label", "value", "confidence", "bbox"}``. ``value`` is ""
    and ``bbox`` is None when the keyword was not found, so the caller can
    still show the field as an empty row to be filled in by hand.

    When a keyword matches more than one box the bottom-right-most match wins:
    that is where the title block lives, and a stray "DATE" in a revision table
    higher up the sheet should not displace it.
    """
    # Any box that reads as one of the OTHER configured labels is off-limits as
    # a value: "SCALE:" sitting above "SHEET 1 OF 4" must not claim it.
    other_labels = [_normalize(k) for k in keywords if _normalize(k)]

    # Two passes: the bottom-right table, then the whole image. The fallback is
    # what keeps this working when the user has cropped the title block itself,
    # where every box is already "the table".
    corner = _bottom_right(boxes)

    results: list[dict[str, Any]] = []
    for keyword in keywords:
        needle = _normalize(keyword)
        if not needle:
            continue

        # A field label STARTS with its own name ("DWG NO.", "REV"). Text that
        # merely mentions the word is not a label, and matching it produces
        # nonsense: "REV" matched note 4's "B633-LATEST REV. TYPE II" and read
        # the line beneath it as the revision. The rule is absolute — with no
        # leading match the field is reported empty rather than guessed at.
        def _opens_with_needle(b: dict[str, Any]) -> bool:
            return _normalize(b.get("text", "")).startswith(needle)

        # The title block's corner first, then the whole image — the fallback is
        # what keeps this working on a crop of the title block itself.
        candidates = [b for b in corner if _opens_with_needle(b)]
        if not candidates:
            candidates = [b for b in boxes if _opens_with_needle(b)]

        # Bottom-right-most first: the title block's corner of the sheet.
        candidates.sort(key=lambda b: (b["y"] + b["h"], b["x"] + b["w"]), reverse=True)

        others = [o for o in other_labels if o != needle]
        neighbours = [
            b
            for b in boxes
            if not any(
                _contains_words(_normalize(b.get("text", "")), o) for o in others
            )
        ]

        found: dict[str, Any] | None = None
        for label_box in candidates:
            text = (label_box.get("text") or "").strip()

            inline = _split_inline(text)
            split = inline or _split_by_keyword(text, keyword)
            if split and split[1]:
                found = {
                    "label": split[0],
                    "value": split[1],
                    "confidence": float(label_box.get("conf") or 0.0),
                    "bbox": label_box,
                }
                break

            # A colon says the value belongs on this LINE. The recogniser may
            # still have cut it into its own box ("DATE:" + "1/3/14"), so the
            # search to the right continues — but the search below does not:
            # an empty "SCALE:" must not claim the row printed beneath it.
            value_box = _value_to_the_right(label_box, neighbours)
            if value_box is None and inline is None:
                value_box = _value_below(label_box, neighbours)

            if value_box is None and inline is not None:
                # A colon with nothing beside it: the field is genuinely blank.
                found = {
                    "label": inline[0],
                    "value": "",
                    "confidence": float(label_box.get("conf") or 0.0),
                    "bbox": label_box,
                }
                break
            if value_box is not None:
                found = {
                    "label": text.rstrip(":").strip(),
                    "value": (value_box.get("text") or "").strip(),
                    "confidence": min(
                        float(label_box.get("conf") or 0.0),
                        float(value_box.get("conf") or 0.0),
                    ),
                    "bbox": {
                        "x": min(label_box["x"], value_box["x"]),
                        "y": min(label_box["y"], value_box["y"]),
                        "w": max(
                            label_box["x"] + label_box["w"],
                            value_box["x"] + value_box["w"],
                        )
                        - min(label_box["x"], value_box["x"]),
                        "h": max(
                            label_box["y"] + label_box["h"],
                            value_box["y"] + value_box["h"],
                        )
                        - min(label_box["y"], value_box["y"]),
                    },
                }
                break

        results.append(
            {
                "keyword": keyword,
                "label": found["label"] if found else keyword,
                "value": found["value"] if found else "",
                "confidence": found["confidence"] if found else 0.0,
                "bbox": found["bbox"] if found else None,
            }
        )
    return results
