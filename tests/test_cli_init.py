"""
tests/test_cli_init.py — aua init command tests.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from aua.cli import main


def test_init_creates_directory():
    """aua init creates the project directory if it doesn't exist."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "my_project"
        result = runner.invoke(main, ["init", str(target)])
        assert result.exit_code == 0, result.output
        assert target.exists()


def test_init_creates_expected_files():
    """aua init scaffolds all required files and directories."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "proj"
        result = runner.invoke(main, ["init", str(p)])
        assert result.exit_code == 0, result.output

        assert (p / "aua_config.yaml").exists()
        assert (p / "models").exists()
        assert (p / "dpo_pairs").exists()
        assert (p / "results").exists()
        assert (p / "logs").exists()
        assert (p / ".gitignore").exists()


def test_init_gitignore_content():
    """Generated .gitignore must include models/ and logs/."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "proj"
        runner.invoke(main, ["init", str(p)])
        gi = (p / ".gitignore").read_text()
        assert "models/" in gi
        assert "logs/" in gi


def test_init_default_tier_is_rtx4090():
    """Default tier is rtx4090 when no --tier flag is given."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "proj"
        runner.invoke(main, ["init", str(p)])
        raw = yaml.safe_load((p / "aua_config.yaml").read_text())
        assert raw["aua"]["backend"] == "vllm"


def test_init_macbook_tier():
    """--tier macbook generates Ollama backend config."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "proj"
        result = runner.invoke(main, ["init", str(p), "--tier", "macbook"])
        assert result.exit_code == 0, result.output
        raw = yaml.safe_load((p / "aua_config.yaml").read_text())
        assert raw["aua"]["backend"] == "ollama"


def test_init_force_overwrites():
    """--force overwrites existing aua_config.yaml."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "proj"
        # First init
        runner.invoke(main, ["init", str(p), "--tier", "macbook"])
        # Force overwrite with different tier
        result = runner.invoke(main, ["init", str(p), "--tier", "rtx4090", "--force"])
        assert result.exit_code == 0, result.output
        raw = yaml.safe_load((p / "aua_config.yaml").read_text())
        assert raw["aua"]["backend"] == "vllm"


def test_init_refuses_overwrite_without_force():
    """Without --force, existing config is not overwritten."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "proj"
        runner.invoke(main, ["init", str(p), "--tier", "macbook"])
        # Try overwrite without --force — should warn but not fail
        result = runner.invoke(main, ["init", str(p), "--tier", "rtx4090"])
        assert result.exit_code == 0
        # Config unchanged — still ollama
        raw = yaml.safe_load((p / "aua_config.yaml").read_text())
        assert raw["aua"]["backend"] == "ollama"


def test_init_existing_dir_is_reused():
    """aua init works fine if the directory already exists."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)  # already exists
        result = runner.invoke(main, ["init", str(p)])
        assert result.exit_code == 0, result.output
        assert (p / "aua_config.yaml").exists()


@pytest.mark.parametrize("tier", ["macbook", "rtx4090", "a100"])
def test_init_all_tiers(tier):
    """aua init works for all supported tiers."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "proj"
        result = runner.invoke(main, ["init", str(p), "--tier", tier])
        assert result.exit_code == 0, result.output
        assert (p / "aua_config.yaml").exists()


@pytest.mark.parametrize("tier", ["macbook", "rtx4090", "a100"])
def test_init_all_tiers_generate_valid_config(tier):
    """Every tier generates a config that loads without error."""
    from aua.config import load_config

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "proj"
        runner.invoke(main, ["init", str(p), "--tier", tier])
        cfg = load_config(p / "aua_config.yaml")
        assert len(cfg.specialists) >= 1
