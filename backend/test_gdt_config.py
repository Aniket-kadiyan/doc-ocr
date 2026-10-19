"""Configuration contract for the standalone GD&T detector."""

from __future__ import annotations

import gdt_config


def test_default_thresholds_preserve_review_band(monkeypatch) -> None:
    for name in (
        "GDT_ACCEPT_CONFIDENCE",
        "GDT_AMBIGUOUS_CONFIDENCE",
        "GDT_RIVAL_MARGIN",
    ):
        monkeypatch.delenv(name, raising=False)

    config = gdt_config.load()
    assert 0.0 < config.ambiguous_confidence < config.accept_confidence < 1.0
    assert 0.0 < config.rival_margin < config.accept_confidence


def test_environment_overrides_are_loaded(monkeypatch) -> None:
    monkeypatch.setenv("GDT_ENABLED", "false")
    monkeypatch.setenv("GDT_PAGE_SCAN_ENABLED", "true")
    monkeypatch.setenv("GDT_ACCEPT_CONFIDENCE", "0.73")
    monkeypatch.setenv("GDT_AMBIGUOUS_CONFIDENCE", "0.51")
    monkeypatch.setenv("GDT_MIN_GLYPH_PX", "12")
    monkeypatch.setenv("GDT_TILE_PX", "1800")

    config = gdt_config.load()
    assert config.enabled is False
    assert config.page_scan_enabled is True
    assert config.accept_confidence == 0.73
    assert config.ambiguous_confidence == 0.51
    assert config.min_glyph_px == 12
    assert config.tile_px == 1800


def test_malformed_numeric_values_fall_back_to_defaults(monkeypatch) -> None:
    defaults = gdt_config.GdtConfig()
    monkeypatch.setenv("GDT_TEMPLATE_MIN_SCORE", "not-a-number")
    monkeypatch.setenv("GDT_TILE_OVERLAP_PX", "not-an-integer")

    config = gdt_config.load()
    assert config.template_min_score == defaults.template_min_score
    assert config.tile_overlap_px == defaults.tile_overlap_px


def test_config_cache_can_be_reset(monkeypatch) -> None:
    gdt_config.reset_config()
    monkeypatch.setenv("GDT_ACCEPT_CONFIDENCE", "0.70")
    first = gdt_config.get_config()

    monkeypatch.setenv("GDT_ACCEPT_CONFIDENCE", "0.80")
    assert gdt_config.get_config() is first
    assert gdt_config.get_config().accept_confidence == 0.70

    gdt_config.reset_config()
    assert gdt_config.get_config().accept_confidence == 0.80
    gdt_config.reset_config()


def test_config_serializes_as_plain_data() -> None:
    payload = gdt_config.GdtConfig().to_dict()
    assert payload["enabled"] is True
    assert payload["page_scan_enabled"] is False
    assert payload["accept_confidence"] == 0.62
    assert payload["tile_px"] == 1400
