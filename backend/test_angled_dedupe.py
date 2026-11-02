"""Slanted reads against the upright reads of the same ink (129E01-13300-ID)."""

from __future__ import annotations

from ocr_pipeline import OcrPipeline, _levelled_min_aspect


def _angled(text, confidence, bbox, oriented):
    x, y, w, h = bbox
    ox, oy, ow, oh, rotation = oriented
    return {
        "text": text,
        "confidence": confidence,
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "oriented_box": {
            "x": ox, "y": oy, "width": ow, "height": oh, "rotation": rotation,
        },
        "rotation": rotation,
    }


def _upright(text, confidence, bbox):
    x, y, w, h = bbox
    return {
        "text": text,
        "confidence": confidence,
        "bbox": {"x": x, "y": y, "width": w, "height": h},
    }


def test_short_values_need_less_elongation_than_long_ones() -> None:
    # "R19" and "R4.5" levelled with their dimension line measured 1.47–1.59.
    assert _levelled_min_aspect("R19") < 1.47
    assert _levelled_min_aspect("R4.5") < 1.47
    assert _levelled_min_aspect("C3") == 1.3
    assert _levelled_min_aspect("Ø20H10+0.084") == 1.8


def test_levelled_read_completes_an_upright_read_cut_short() -> None:
    base = [_upright("R4.", 0.977, (2587.4, 2978.9, 97.9, 105.0))]
    angled = [
        _angled("R4.5", 1.0, (2578.1, 2942.5, 141.0, 154.1),
                (2578.1, 3055.3, 273.0, 128.0, -52.0)),
        # The same ink levelled 7° off runs on into the leader's ticks.
        _angled("R4.551", 0.973, (2568.9, 2940.2, 158.3, 158.3),
                (2568.9, 3042.2, 275.0, 152.0, -45.0)),
    ]

    kept = OcrPipeline._dedupe_angled(angled, base)

    assert kept == []
    assert base[0]["text"] == "R4.5"
    assert base[0]["enriched_from"] == "angled"


def test_a_speck_of_the_callout_does_not_block_it() -> None:
    base = [_upright("20", 0.91, (2385.4, 3189.0, 28.6, 27.5))]
    angled = [
        _angled("R4.5", 1.0, (2338.9, 3123.7, 163.2, 130.8),
                (2338.9, 3187.3, 233.0, 120.0, -26.0)),
    ]

    kept = OcrPipeline._dedupe_angled(angled, base)

    assert [r["text"] for r in kept] == ["R4.5"]
    assert base[0].get("superseded") is True


def test_a_leader_tick_read_as_one_does_not_outrank_the_levelled_read() -> None:
    base = [_upright("41.5", 0.991, (2882.1, 3019.7, 91.8, 76.5))]
    angled = [
        _angled("R4.5", 0.992, (2824.5, 3005.7, 151.8, 131.3),
                (2824.5, 3057.5, 168.0, 114.7, -24.0)),
    ]

    kept = OcrPipeline._dedupe_angled(angled, base)

    assert [r["text"] for r in kept] == ["R4.5"]
    assert base[0].get("superseded") is True


def test_a_value_written_beside_the_callout_still_blocks_it() -> None:
    # Same box as the "41.5" above, but its digits are not the levelled
    # read's plus ticks: it is another value, and the upright read stands.
    base = [_upright("47.4", 0.991, (2882.1, 3019.7, 91.8, 76.5))]
    angled = [
        _angled("R4.5", 0.992, (2824.5, 3005.7, 151.8, 131.3),
                (2824.5, 3057.5, 168.0, 114.7, -24.0)),
    ]

    kept = OcrPipeline._dedupe_angled(angled, base)

    assert kept == []
    assert not base[0].get("superseded")


def test_chained_dimensions_are_cut_at_their_shared_extension_line() -> None:
    from PIL import Image, ImageDraw

    from note_blocks import InkMap

    # "16.8 | 10.7 |": two values either side of an extension line, and the
    # next extension line closing the right-hand one.
    image = Image.new("RGB", (400, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 55, 185, 85), fill="black")
    draw.rectangle((215, 55, 300, 85), fill="black")
    draw.line((200, 0, 200, 139), fill="black", width=3)
    draw.line((318, 0, 318, 139), fill="black", width=3)
    read: list[tuple[int, int]] = []

    def recognize(crop, **_kwargs):
        read.append(crop.size)
        return {"text": "16.8" if len(read) == 1 else "10.7", "confidence": 1.0}

    pipeline = object.__new__(OcrPipeline)
    pipeline.recognize = recognize

    pieces = pipeline._split_at_interior_rules(
        image, InkMap(image), {"x": 40, "y": 50, "width": 285, "height": 40}
    )

    assert [p["text"] for p in pieces] == ["16.8", "10.7"]
    left, right = (p["bbox"] for p in pieces)
    assert left["x"] + left["width"] < 200 < right["x"]
    # Neither side carries a rule along its edge into the read.
    assert right["x"] + right["width"] < 317


def test_a_line_with_no_rule_through_it_is_not_cut() -> None:
    from PIL import Image, ImageDraw

    from note_blocks import InkMap

    image = Image.new("RGB", (400, 140), "white")
    ImageDraw.Draw(image).rectangle((45, 55, 300, 85), fill="black")
    pipeline = object.__new__(OcrPipeline)

    assert pipeline._split_at_interior_rules(
        image, InkMap(image), {"x": 40, "y": 50, "width": 285, "height": 40}
    ) is None
