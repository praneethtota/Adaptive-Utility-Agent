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


# ── P-05: host / scheme / endpoint URL ───────────────────────────────────────


def test_specialist_endpoint_uses_host(minimal_config):
    """Endpoint must use the configured host (127.0.0.1), never 'localhost'."""
    url = minimal_config.specialist("swe").endpoint
    assert "localhost" not in url
    assert "127.0.0.1" in url
    assert "19001" in url
    assert url.endswith("/v1/chat/completions")


def test_specialist_models_url_uses_host(minimal_config):
    """models_url must use 127.0.0.1, not localhost."""
    url = minimal_config.specialist("swe").models_url
    assert "localhost" not in url
    assert "127.0.0.1" in url
    assert url.endswith("/v1/models")


def test_arbiter_endpoint_uses_host(minimal_config):
    """Arbiter endpoint must use 127.0.0.1."""
    url = minimal_config.arbiter.endpoint
    assert "localhost" not in url
    assert "127.0.0.1" in url


def test_endpoint_override(fixtures_dir):
    """endpoint_override replaces auto-built URL entirely."""
    from aua.config import _parse_config

    raw = {
        "aua": {"version": "0.5", "mode": "local", "backend": "vllm"},
        "specialists": [
            {
                "name": "swe",
                "model": "fake/swe",
                "port": 9001,
                "field": "software_engineering",
                "endpoint_override": "http://custom-host:9999/v1/chat/completions",
            }
        ],
        "arbiter": {"model": "fake/arb", "port": 9003},
        "router": {"port": 8000},
        "blue_green": {},
    }
    cfg = _parse_config(raw, source="<test>")
    assert cfg.specialist("swe").endpoint == "http://custom-host:9999/v1/chat/completions"


def test_custom_scheme(fixtures_dir):
    """scheme=https must produce https:// endpoints."""
    from aua.config import _parse_config

    raw = {
        "aua": {"version": "0.5", "mode": "local", "backend": "vllm"},
        "specialists": [
            {
                "name": "swe",
                "model": "fake/swe",
                "port": 9001,
                "field": "software_engineering",
                "scheme": "https",
                "host": "swe.internal",
            }
        ],
        "arbiter": {"model": "fake/arb", "port": 9003},
        "router": {"port": 8000},
        "blue_green": {},
    }
    cfg = _parse_config(raw, source="<test>")
    assert cfg.specialist("swe").endpoint.startswith("https://")
    assert "swe.internal" in cfg.specialist("swe").endpoint


# ── P-05: RuntimeConfig ───────────────────────────────────────────────────────


def test_runtime_config_defaults(minimal_config):
    """RuntimeConfig exposes .logs .pids .state .checkpoints under .aua/."""
    rt = minimal_config.runtime
    assert rt.logs == rt.base / "logs"
    assert rt.pids == rt.base / "pids"
    assert rt.state == rt.base / "state"
    assert rt.checkpoints == rt.base / "checkpoints"


def test_runtime_ensure_creates_dirs(tmp_path):
    """RuntimeConfig.ensure() creates all four subdirectories."""
    from aua.config import RuntimeConfig

    rt = RuntimeConfig(base=tmp_path / ".aua")
    rt.ensure()
    assert rt.logs.is_dir()
    assert rt.pids.is_dir()
    assert rt.state.is_dir()
    assert rt.checkpoints.is_dir()


# ── P-05: cors_origins in RouterConfig ───────────────────────────────────────


def test_router_cors_defaults_to_wildcard(minimal_config):
    """cors_origins defaults to ['*'] when not specified in YAML."""
    assert minimal_config.router.cors_origins == ["*"]


# ── P-05: Validation — duplicate ports ───────────────────────────────────────


def test_duplicate_ports_raises(fixtures_dir):
    """Config with two services sharing a port must raise ValueError."""
    with pytest.raises(ValueError, match="[Dd]uplicate port"):
        load_config(fixtures_dir / "aua_config_invalid_duplicate_ports.yaml")


# ── P-05: Validation — unknown keys ──────────────────────────────────────────


def test_unknown_key_raises(fixtures_dir):
    """Config with an unrecognised key must raise ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown key"):
        load_config(fixtures_dir / "aua_config_invalid_unknown_key.yaml")


# ── P-05: Validation — threshold ranges ──────────────────────────────────────


def test_invalid_threshold_raises(fixtures_dir):
    """Config with out-of-range threshold must raise ValueError."""
    with pytest.raises(ValueError, match="must be in"):
        load_config(fixtures_dir / "aua_config_invalid_threshold.yaml")


def test_gpu_memory_utilization_zero_raises():
    """gpu_memory_utilization=0 must be rejected (exclusive lower bound)."""
    from aua.config import _parse_config

    raw = {
        "aua": {"version": "0.5", "mode": "local", "backend": "vllm"},
        "specialists": [
            {
                "name": "swe",
                "model": "fake/swe",
                "port": 9001,
                "field": "software_engineering",
                "gpu_memory_utilization": 0.0,
            }
        ],
        "arbiter": {"model": "fake/arb", "port": 9003},
        "router": {"port": 8000},
        "blue_green": {},
    }
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        _parse_config(raw, source="<test>")
