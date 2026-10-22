"""PDF-first scans keep accepted, review, and Other candidates accounted."""

from text_layer_scan import scan_result_from_text_layer


def _item(text: str, x: float, *, review: bool = False) -> dict:
    return {
        "text": text,
        "bbox": {"x": x, "y": 10, "width": 20, "height": 10},
        "confidence": 1.0,
        "needs_review": review,
    }


def test_pdf_objects_share_the_scan_candidate_ledger() -> None:
    result = scan_result_from_text_layer(
        {
            "regions": [_item("Ø10", 10), _item("12", 40, review=True)],
            "excluded": [
                {
                    **_item("SECTION A-A", 70),
                    "rule": "pdf_text_non_callout",
                    "reason": "Not an inspection callout",
                }
            ],
            "text_blocks": [
                {
                    **_item("DRAWN BY", 100),
                    "category": "title",
                    "reason": "Title-block text",
                }
            ],
        },
        scope_kind="page",
        scope={"x": 0, "y": 0, "width": 200, "height": 100},
    )

    assert [region["text"] for region in result["regions"]] == ["Ø10"]
    assert [region["text"] for region in result["review_candidates"]] == ["12"]
    assert [item["state"] for item in result["candidate_outcomes"]] == [
        "eligible",
        "review",
        "excluded",
        "excluded",
    ]
    assert result["detected_count"] == 4
    assert result["eligible_count"] == 1
    assert result["review_count"] == 1
    assert result["excluded_count"] == 2


def test_section_scope_filters_every_candidate_kind() -> None:
    result = scan_result_from_text_layer(
        {
            "regions": [_item("Ø10", 10), _item("Ø20", 110)],
            "excluded": [_item("VIEW", 20), _item("TITLE", 120)],
            "text_blocks": [],
        },
        scope_kind="section",
        scope={"x": 0, "y": 0, "width": 80, "height": 100},
    )

    assert [region["text"] for region in result["regions"]] == ["Ø10"]
    assert [item["text"] for item in result["candidate_outcomes"]] == [
        "Ø10",
        "VIEW",
    ]
    assert result["detected_count"] == 2
