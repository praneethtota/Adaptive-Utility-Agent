"""
tests/test_cli_doctor.py — aua doctor command tests.

Doctor makes real hardware/port/dependency checks which vary by environment,
so we mock _detect_hardware and subprocess calls to get deterministic results.
"""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from aua.cli import main
from aua.doctor import _HWInfo

FIXTURES = Path(__file__).parent / "fixtures"

# Simulate an NVIDIA GPU environment (the happy-path for vllm config)
MOCK_NVIDIA = _HWInfo(
    kind="nvidia",
    devices=[{"index": 0, "name": "RTX 4090", "vram_mib": 24564}],
    system_ram_mib=65536,
)

# Simulate Apple Silicon (Ollama-compatible, vllm incompatible)
MOCK_APPLE = _HWInfo(
    kind="apple_silicon",
    devices=[{"index": 0, "name": "M3 Pro", "vram_mib": 18432}],
    system_ram_mib=18432,
)


def test_doctor_runs_without_crash(fixtures_dir):
    """aua doctor must not crash even if checks fail."""
    runner = CliRunner()
    with patch("aua.doctor._detect_hardware", return_value=MOCK_NVIDIA):
        result = runner.invoke(
            main, ["doctor", "--config", str(fixtures_dir / "aua_config_minimal.yaml")]
        )
    # Exit 0 (all pass) or 1 (some fail) — both are valid outcomes
    assert result.exit_code in (0, 1)
    # SystemExit is expected when exit_code=1; only fail on unexpected exceptions
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_doctor_config_check_passes(fixtures_dir):
    """Config check passes for valid minimal config."""
    from aua.doctor import _check_config

    checks = _check_config(str(fixtures_dir / "aua_config_minimal.yaml"))

    assert any(c.status == "pass" for c in checks)


def test_doctor_config_check_fails_missing_file():
    """Config check fails if file does not exist."""
    from aua.doctor import _check_config

    checks = _check_config("/nonexistent/aua_config.yaml")
    assert any(c.status == "fail" for c in checks)


def test_doctor_hardware_vllm_on_apple_fails(fixtures_dir):
    """vllm backend on Apple Silicon must produce a fail check."""
    from aua.config import load_config
    from aua.doctor import _check_hardware

    cfg = load_config(fixtures_dir / "aua_config_minimal.yaml")
    with patch("aua.doctor._detect_hardware", return_value=MOCK_APPLE):
        checks = _check_hardware(cfg)
    assert any(c.status == "fail" for c in checks)


def test_doctor_hardware_ollama_on_apple_passes():
    """ollama backend on Apple Silicon must not produce a fail check."""
    from aua.config import load_tier
    from aua.doctor import _check_hardware

    cfg = load_tier("macbook")
    with patch("aua.doctor._detect_hardware", return_value=MOCK_APPLE):
        checks = _check_hardware(cfg)
    assert not any(c.status == "fail" for c in checks)


def test_doctor_hardware_nvidia_vllm_passes(fixtures_dir):
    """NVIDIA GPU + vllm backend must not produce a fail check."""
    from aua.config import load_config
    from aua.doctor import _check_hardware

    cfg = load_config(fixtures_dir / "aua_config_minimal.yaml")
    with patch("aua.doctor._detect_hardware", return_value=MOCK_NVIDIA):
        checks = _check_hardware(cfg)
    assert not any(c.status == "fail" for c in checks)


def test_doctor_returns_integer(fixtures_dir):
    """run_doctor() returns an integer (number of failures)."""
    from aua.doctor import run_doctor

    with patch("aua.doctor._detect_hardware", return_value=MOCK_NVIDIA):
        result = run_doctor(str(fixtures_dir / "aua_config_minimal.yaml"))
    assert isinstance(result, int)
    assert result >= 0
