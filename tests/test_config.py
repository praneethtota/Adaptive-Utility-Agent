"""
tests/test_config.py — config loading, validation, and tier tests.
"""

from pathlib import Path

import pytest

from aua.config import (
    AVAILABLE_TIERS,
    AUAConfig,
    load_config,
    load_tier,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── Happy path ────────────────────────────────────────────────────────────────


def test_load_minimal_config(minimal_config):
    """Minimal config loads without error and has correct structure."""
    cfg = minimal_config
    assert isinstance(cfg, AUAConfig)
    assert cfg.backend == "vllm"
    assert len(cfg.specialists) == 1
    assert cfg.specialist("swe").field == "software_engineering"
    assert cfg.router.port == 19000


def test_specialist_endpoint_url(minimal_config):
    """Specialist endpoint URL must be well-formed."""
    url = minimal_config.specialist("swe").endpoint
    assert url.startswith("http://")
    assert "19001" in url
    assert url.endswith("/v1/chat/completions")


def test_specialist_for_field(minimal_config):
    """specialist_for_field() must resolve correctly."""
    spec = minimal_config.specialist_for_field("software_engineering")
    assert spec is not None
    assert spec.name == "swe"

    missing = minimal_config.specialist_for_field("nonexistent_field")
    assert missing is None


def test_vllm_command(minimal_config):
    """vllm_command() must include required flags."""
    cmd = minimal_config.specialist("swe").vllm_command()
    assert cmd[0] == "python"
    assert "--model" in cmd
    assert "--port" in cmd
    assert "--enforce-eager" in cmd  # enforce_eager=True in fixture


def test_blue_green_for(minimal_config):
    """blue_green_for() returns config or default."""
    bg = minimal_config.blue_green_for("swe")
    assert bg.delta == 0.025
    assert bg.T_min == 10

    default_bg = minimal_config.blue_green_for("nonexistent")
    assert default_bg.delta == 0.025  # default values


def test_all_endpoints(minimal_config):
    """all_endpoints() returns dict including arbiter."""
    eps = minimal_config.all_endpoints()
    assert "swe" in eps
    assert "arbiter" in eps
    for url in eps.values():
        assert url.startswith("http://")


# ── Tier loading ──────────────────────────────────────────────────────────────


def test_available_tiers():
    assert set(AVAILABLE_TIERS) == {"macbook", "rtx4090", "a100"}


@pytest.mark.parametrize("tier", ["macbook", "rtx4090", "a100"])
def test_load_tier(tier):
    """Every built-in tier must load without error."""
    cfg = load_tier(tier)
    assert isinstance(cfg, AUAConfig)
    assert len(cfg.specialists) >= 1
    assert cfg.arbiter.port > 0
    assert cfg.router.port > 0


def test_macbook_tier_uses_ollama():
    cfg = load_tier("macbook")
    assert cfg.backend == "ollama"


def test_rtx4090_tier_uses_vllm():
    cfg = load_tier("rtx4090")
    assert cfg.backend == "vllm"
    assert cfg.specialist("swe").quantization == "awq"


def test_a100_tier_no_enforce_eager():
    cfg = load_tier("a100")
    for spec in cfg.specialists:
        assert spec.enforce_eager is False


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown tier"):
        load_tier("nonexistent_tier")


# ── Error cases ───────────────────────────────────────────────────────────────


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/aua_config.yaml")


def test_unknown_specialist_raises(minimal_config):
    with pytest.raises(KeyError, match="nonexistent"):
        minimal_config.specialist("nonexistent")


# NOTE: duplicate-port and unknown-key validation are roadmap P-06 items.
# The fixtures exist ready for when those validators are added to config.py.
