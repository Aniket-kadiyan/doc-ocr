"""Custom symbol library: storage, detection accuracy, scan merge and routes."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import custom_symbol_routes
import custom_symbols as cs
import gdt_templates as gt
from auth_repository import UserRecord
from auth_routes import require_protected_request
from auth_service import AuthenticatedSession


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _glyph(name: str, size: int) -> np.ndarray:
    """A GD&T glyph drawn dark on white, the way a drawing shows it."""
    return 255 - gt.render(name, size)


def _upload(name: str, size: int) -> bytes:
    """A crop of ``name`` as a user would cut it from a drawing: some margin."""
    return _png(_glyph(name, size))


def _page(width: int = 900, height: int = 600) -> np.ndarray:
    return np.full((height, width), 255, np.uint8)


def _stamp(page: np.ndarray, glyph: np.ndarray, x: int, y: int) -> tuple[int, int, int, int]:
    h, w = glyph.shape
    page[y : y + h, x : x + w] = np.minimum(page[y : y + h, x : x + w], glyph)
    ys, xs = np.nonzero(glyph < 128)
    return x + xs.min(), y + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


def _rgb(page: np.ndarray) -> Image.Image:
    return Image.fromarray(page).convert("RGB")


def _near(match: cs.SymbolMatch, box: tuple[int, int, int, int], tol: int = 4) -> bool:
    return abs(match.x - box[0]) <= tol and abs(match.y - box[1]) <= tol


@pytest.fixture()
def store(tmp_path: Path) -> cs.CustomSymbolStore:
    return cs.CustomSymbolStore(tmp_path / "symbols")


# ── Storage ──────────────────────────────────────────────────────────────────


def test_symbols_persist_until_deleted(tmp_path: Path) -> None:
    root = tmp_path / "symbols"
    first = cs.CustomSymbolStore(root)
    created = first.create(
        name="Position mark",
        image_bytes=_upload("position", 48),
        label="True position",
        category="GD&T",
    )

    reopened = cs.CustomSymbolStore(root)  # a backend restart
    [loaded] = reopened.list()
    assert loaded.id == created.id
    assert loaded.label == "True position"
    assert loaded.enabled and loaded.threshold == cs.DEFAULT_THRESHOLD
    assert reopened.image_png(created.id).startswith(b"\x89PNG")

    reopened.delete(created.id)
    assert cs.CustomSymbolStore(root).list() == []
    assert not (root / "images" / f"{created.id}.png").exists()


def test_names_are_unique_case_insensitively(store: cs.CustomSymbolStore) -> None:
    store.create(name="Stamp", image_bytes=_upload("position", 40))
    with pytest.raises(cs.CustomSymbolConflictError):
        store.create(name="  stamp ", image_bytes=_upload("flatness", 40))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "", "image_bytes": _upload("position", 40)}, "name"),
        ({"name": "x", "image_bytes": b"not an image"}, "readable image"),
        ({"name": "x", "image_bytes": _png(_page(60, 60))}, "No symbol"),
        ({"name": "x", "image_bytes": _upload("position", 40), "category": "Banana"}, "category"),
        ({"name": "x", "image_bytes": _upload("position", 40), "threshold": 0.2}, "strictness"),
    ],
)
def test_bad_symbols_are_refused(store: cs.CustomSymbolStore, kwargs: dict, message: str) -> None:
    with pytest.raises(cs.CustomSymbolValidationError, match=message):
        store.create(**kwargs)
    assert store.list() == []


def test_update_changes_fields_and_disables(store: cs.CustomSymbolStore) -> None:
    symbol = store.create(name="Mark", image_bytes=_upload("position", 40))
    updated = store.update(symbol.id, {"label": "Pos", "enabled": False, "threshold": 0.9})
    assert (updated.label, updated.enabled, updated.threshold) == ("Pos", False, 0.9)
    assert store.enabled_templates() == []


def test_template_is_cleaned_light_on_dark_and_cropped() -> None:
    dark = gt.render("position", 60)  # white ink on black
    mask, sides = cs.prepare_template_with_sides(Image.fromarray(dark).convert("RGB"))
    assert mask.shape[0] < 60 and mask.shape[1] < 60  # cropped to the ink
    assert sides == []
    assert 0.05 < (mask > 0).mean() < 0.85


def test_crop_through_ink_records_open_sides() -> None:
    picture = _page(80, 40)
    cv2.circle(picture, (30, 20), 12, 0, 2)
    cv2.line(picture, (30, 20), (79, 20), 0, 2)  # centre line cut by the crop
    _, sides = cs.prepare_template_with_sides(_rgb(picture))
    assert sides == ["right"]


# ── Detection ────────────────────────────────────────────────────────────────


def test_finds_every_instance_and_nothing_else(store: cs.CustomSymbolStore) -> None:
    page = _page()
    truth = [_stamp(page, _glyph("position", 40), 100, 100), _stamp(page, _glyph("position", 48), 600, 380)]
    _stamp(page, _glyph("circularity", 40), 300, 100)
    _stamp(page, _glyph("diameter", 40), 450, 100)
    _stamp(page, _glyph("concentricity", 40), 300, 380)
    cv2.putText(page, "10.05 A 0.8", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    symbol = store.create(name="Position", image_bytes=_upload("position", 44))

    matches = cs.detect_custom_symbols(_rgb(page), store.enabled_templates())

    assert len(matches) == 2
    assert all(m.symbol.id == symbol.id for m in matches)
    assert all(any(_near(m, box) for m in matches) for box in truth)


def test_each_symbol_claims_only_its_own_glyphs(store: cs.CustomSymbolStore) -> None:
    page = _page()
    pos = _stamp(page, _glyph("position", 40), 100, 100)
    flat = _stamp(page, _glyph("flatness", 40), 400, 100)
    store.create(name="Position", image_bytes=_upload("position", 40))
    store.create(name="Flatness", image_bytes=_upload("flatness", 40))

    found = {(m.symbol.name, _near(m, pos), _near(m, flat)) for m in cs.detect_custom_symbols(_rgb(page), store.enabled_templates())}

    assert found == {("Position", True, False), ("Flatness", False, True)}


def test_line_junction_is_not_a_perpendicularity_symbol(store: cs.CustomSymbolStore) -> None:
    page = _page()
    real = _stamp(page, _glyph("perpendicularity", 40), 100, 100)
    # The same two strokes as part of a T-junction of construction lines.
    cv2.line(page, (300, 300), (700, 300), 0, 3)
    cv2.line(page, (500, 150), (500, 300), 0, 3)
    store.create(name="Perpendicular", image_bytes=_upload("perpendicularity", 40))

    matches = cs.detect_custom_symbols(_rgb(page), store.enabled_templates())

    assert len(matches) == 1 and _near(matches[0], real)


def test_a_letter_o_is_not_a_circle_symbol(store: cs.CustomSymbolStore) -> None:
    page = _page()
    real = _stamp(page, _glyph("circularity", 40), 100, 100)
    cv2.ellipse(page, (400, 120), (12, 20), 0, 0, 360, 0, 3)  # an O / 0
    store.create(name="Circle", image_bytes=_upload("circularity", 40))

    matches = cs.detect_custom_symbols(_rgb(page), store.enabled_templates())

    assert len(matches) == 1 and _near(matches[0], real)


def test_digit_loops_are_not_a_circle_symbol_at_any_size(store: cs.CustomSymbolStore) -> None:
    page = _page()
    real = _stamp(page, _glyph("circularity", 40), 100, 100)
    for row in range(3):
        cv2.putText(page, "69.96 6.9 96.6", (250, 300 + row * 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    # A clean, large picture searched at every size: the loop of a 6 or 9
    # comes into range, and only its tail running out of the box gives it away.
    store.create(name="Circle", image_bytes=_upload("circularity", 150), any_size=True)

    matches = cs.detect_custom_symbols(_rgb(page), store.enabled_templates())

    assert len(matches) == 1 and _near(matches[0], real)


def test_crop_through_a_centre_line_still_matches(store: cs.CustomSymbolStore) -> None:
    def projection(page: np.ndarray, cx: int, cy: int) -> None:
        cv2.circle(page, (cx, cy), 14, 0, 2)
        cv2.circle(page, (cx, cy), 6, 0, 2)
        cv2.line(page, (cx - 22, cy), (cx + 60, cy), 0, 2)
        cv2.line(page, (cx, cy - 22), (cx, cy + 22), 0, 2)

    source = _page(120, 80)
    projection(source, 40, 40)
    store.create(name="Projection", image_bytes=_png(source[10:70, 10:70]))  # cuts the line
    page = _page()
    projection(page, 300, 300)

    matches = cs.detect_custom_symbols(_rgb(page), store.enabled_templates())

    assert len(matches) == 1
    assert abs(matches[0].x - (300 - 22)) <= 4


def test_rotated_instances_need_rotation_matching(store: cs.CustomSymbolStore) -> None:
    page = _page()
    turned = np.ascontiguousarray(np.rot90(_glyph("perpendicularity", 40), -1))
    real = _stamp(page, turned, 200, 200)
    symbol = store.create(name="Perp", image_bytes=_upload("perpendicularity", 40))
    assert cs.detect_custom_symbols(_rgb(page), store.enabled_templates()) == []

    store.update(symbol.id, {"match_rotations": True})
    [match] = cs.detect_custom_symbols(_rgb(page), store.enabled_templates())
    assert _near(match, real) and match.rotation == 90


def test_sizes_far_from_the_upload_need_any_size(store: cs.CustomSymbolStore) -> None:
    page = _page()
    real = _stamp(page, _glyph("position", 32), 200, 200)
    symbol = store.create(name="Pos", image_bytes=_upload("position", 160))  # a clean, big picture
    assert cs.detect_custom_symbols(_rgb(page), store.enabled_templates()) == []

    store.update(symbol.id, {"any_size": True})
    [match] = cs.detect_custom_symbols(_rgb(page), store.enabled_templates())
    assert _near(match, real)


def test_scope_and_existing_balloons_are_respected(store: cs.CustomSymbolStore) -> None:
    page = _page()
    left = _stamp(page, _glyph("position", 40), 100, 100)
    right = _stamp(page, _glyph("position", 40), 600, 100)
    store.create(name="Pos", image_bytes=_upload("position", 40))
    symbols = store.enabled_templates()

    scoped = cs.detect_custom_symbols(_rgb(page), symbols, scope_bbox={"x": 0, "y": 0, "width": 400, "height": 600})
    assert len(scoped) == 1 and _near(scoped[0], left)

    existing = {"x": right[0] - 2, "y": right[1] - 2, "width": right[2] + 4, "height": right[3] + 4}
    skipped = cs.detect_custom_symbols(_rgb(page), symbols, skip_boxes=[existing])
    assert len(skipped) == 1 and _near(skipped[0], left)


# ── Scan result merge ────────────────────────────────────────────────────────


def _symbol(**overrides: object) -> cs.CustomSymbol:
    now = datetime.now(timezone.utc).isoformat()
    values = dict(
        id="a" * 32,
        name="Inspection stamp",
        label="",
        category="GD&T",
        threshold=0.82,
        match_rotations=False,
        enabled=True,
        width=20,
        height=20,
        created_at=now,
        updated_at=now,
    )
    values.update(overrides)
    return cs.CustomSymbol(**values)  # type: ignore[arg-type]


def test_merge_adds_linked_balloons_and_drops_glyph_readings() -> None:
    seg = {
        "regions": [
            {"candidate_id": "R1", "text": "O", "bbox": {"x": 102, "y": 102, "width": 10, "height": 12}},
            {"candidate_id": "R2", "text": "25.4", "bbox": {"x": 300, "y": 100, "width": 40, "height": 14}},
        ],
        "review_candidates": [],
        "candidate_outcomes": [
            {"candidate_id": "R1", "state": "eligible"},
            {"candidate_id": "R2", "state": "eligible"},
        ],
        "detected_count": 2,
        "recognized_count": 2,
        "eligible_count": 2,
        "excluded_count": 0,
        "review_count": 0,
        "filter_rule_counts": {},
    }
    match = cs.SymbolMatch(symbol=_symbol(), x=100, y=100, width=20, height=20, score=0.93, recall=1.0, precision=1.0)

    assert cs.merge_custom_symbol_matches(seg, [match]) == 1

    texts = [r["text"] for r in seg["regions"]]
    assert texts == ["25.4", "Inspection stamp"]
    added = seg["regions"][-1]
    assert added["custom_symbol_id"] == "a" * 32
    assert added["label"] == "Inspection stamp"  # no label set: the name
    assert added["category"] == "GD&T" and added["recognized"] and not added["needs_review"]
    assert seg["eligible_count"] == 2 and seg["excluded_count"] == 1
    assert seg["filter_rule_counts"] == {"custom_symbol": 1, "custom_symbol_overlap": 1}
    states = {o["candidate_id"]: o["state"] for o in seg["candidate_outcomes"]}
    assert states == {"R1": "excluded", "R2": "eligible", "S0001": "eligible"}


def test_label_field_overrides_the_name() -> None:
    match = cs.SymbolMatch(symbol=_symbol(label="Flatness check"), x=0, y=0, width=10, height=10, score=0.9, recall=1, precision=1)
    assert cs.match_region(match, 1)["label"] == "Flatness check"


# ── Routes ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    test_store = cs.CustomSymbolStore(tmp_path / "symbols")
    monkeypatch.setattr(custom_symbol_routes, "get_custom_symbol_store", lambda: test_store)
    now = datetime.now(timezone.utc)
    session = AuthenticatedSession(
        session_id="s",
        token_hash="t",
        csrf_token_hash="c",
        user=UserRecord(
            id="user-1",
            username="inspector",
            display_name="Inspector",
            is_active=True,
            roles=(),
            created_at=now,
            updated_at=now,
        ),
    )
    app = FastAPI()
    app.include_router(custom_symbol_routes.router)
    app.dependency_overrides[require_protected_request] = lambda: session
    return TestClient(app)


def test_routes_create_list_update_delete(client: TestClient) -> None:
    created = client.post(
        "/custom-symbols",
        data={"name": "Pos", "label": "Position", "category": "GD&T", "threshold": "0.85"},
        files={"file": ("pos.png", _upload("position", 40), "image/png")},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["created_by"] == "user-1"
    assert body["image_data_url"].startswith("data:image/png;base64,")

    listed = client.get("/custom-symbols").json()
    assert [s["name"] for s in listed["symbols"]] == ["Pos"]
    assert "GD&T" in listed["categories"]

    patched = client.patch(f"/custom-symbols/{body['id']}", json={"enabled": False})
    assert patched.status_code == 200 and patched.json()["enabled"] is False

    duplicate = client.post(
        "/custom-symbols",
        data={"name": "pos"},
        files={"file": ("pos.png", _upload("position", 40), "image/png")},
    )
    assert duplicate.status_code == 409

    assert client.delete(f"/custom-symbols/{body['id']}").status_code == 204
    assert client.delete(f"/custom-symbols/{body['id']}").status_code == 404
    assert client.get("/custom-symbols").json()["symbols"] == []
