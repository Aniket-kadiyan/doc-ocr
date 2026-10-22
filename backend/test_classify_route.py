"""/ocr/classify applies the same labelling rules the scan routes use."""

from __future__ import annotations

import asyncio

from feature_classifier import classify_feature
from main import ClassifyRequest, classify_values


def test_each_text_gets_the_rule_engines_label_in_order():
    texts = ["Ø12", "4X Ø6.6 THRU", "0", "R5", ""]
    result = asyncio.run(classify_values(ClassifyRequest(texts=texts)))["results"]
    assert [r["text"] for r in result] == texts
    for row, text in zip(result, texts):
        feature = classify_feature(text)
        assert (row["category"], row["subtype"], row["label"]) == (
            feature.category,
            feature.subtype,
            feature.label,
        )
    # A falsy value is still a value and still gets its label; only empty
    # text has none.
    assert result[2]["label"] == "Linear Dimension"
    assert result[4]["label"] == ""


def test_an_empty_request_is_an_empty_answer():
    assert asyncio.run(classify_values(ClassifyRequest(texts=[]))) == {"results": []}


def test_every_value_leaving_the_api_carries_its_rule_label():
    # The whole-page route published regions with no category and no label;
    # the API check labels them from the same rules and counts them.
    from main import _serialize_segment_result

    seg = {
        "coordinate_space": "page",
        "regions": [
            {"bbox": {}, "text": "15°±3°", "type": "angle"},
            {"bbox": {}, "text": "Ø12", "type": "diameter", "category": "Diameter", "label": "Diameter"},
            # A label a pass set on purpose is kept.
            {"bbox": {}, "text": "⌖ Ø0.3 A", "type": "gdt", "category": "GD&T", "label": "GD&T Position"},
            {"bbox": {}, "text": "1. FREE FROM BURRS", "type": "General Note", "category": "General Note", "label": "General Notes"},
        ],
        "review_candidates": [{"bbox": {}, "text": "0", "type": "linear"}, {"bbox": {}, "text": ""}],
        "candidate_outcomes": [{"bbox": {}, "text": "R5", "state": "review"}],
    }
    out = _serialize_segment_result(seg)
    labels = [r["label"] for r in out["regions"]]
    assert labels == ["Angle", "Diameter", "GD&T Position", "General Notes"]
    assert out["regions"][0]["category"] == "Angle"
    # A falsy value is labelled; an empty one legitimately is not.
    assert out["review_candidates"][0]["label"] == "Linear Dimension"
    assert "label" not in out["review_candidates"][1]
    assert out["candidate_outcomes"][0]["label"] == "Radius"
    assert out["label_completeness"] == {"values": 6, "labelled": 6, "filled_by_check": 3}
    # The input was not modified in place.
    assert "label" not in seg["regions"][0]
