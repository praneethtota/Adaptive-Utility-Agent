"""
tests/test_rollback.py — aua rollback logic tests.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from aua.cli import main
from aua.rollback import load_promotions, record_promotion, run_rollback

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def project_dir():
    """Temporary project directory with a copy of the minimal config."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        shutil.copy(FIXTURES / "aua_config_minimal.yaml", p / "aua_config.yaml")
        (p / "results").mkdir()
        # P-09: create .aua/state/ for the new JSONL promotions log
        (p / ".aua" / "state").mkdir(parents=True)
        yield p


# ── record_promotion ──────────────────────────────────────────────────────────


def test_record_promotion_creates_log(project_dir):
    """record_promotion creates a promotions log file."""
    event = record_promotion(
        "swe",
        "fake/swe-v1",
        "fake/swe-v2",
        u_delta=0.045,
        project_dir=str(project_dir),
    )
    assert event.event == "promote"
    assert event.specialist == "swe"
    assert event.from_model == "fake/swe-v1"
    assert event.to_model == "fake/swe-v2"
    assert event.u_delta == 0.045
    assert not event.reverted


def test_load_promotions_empty(project_dir):
    """load_promotions returns empty list when no log exists."""
    events = load_promotions(str(project_dir))
    assert events == []


def test_load_promotions_after_record(project_dir):
    """load_promotions returns the recorded event."""
    record_promotion("swe", "from_model", "to_model", project_dir=str(project_dir))
    events = load_promotions(str(project_dir))
    assert len(events) == 1
    assert events[0].specialist == "swe"


# ── run_rollback ──────────────────────────────────────────────────────────────


def test_rollback_no_history_returns_1(project_dir):
    """run_rollback returns 1 if there is no promotion history."""
    result = run_rollback(
        config_path=str(project_dir / "aua_config.yaml"),
        specialist="swe",
        yes=True,
    )
    assert result == 1


def test_rollback_success(project_dir):
    """run_rollback reverts config and returns 0."""
    record_promotion(
        "swe",
        "fake/swe-blue",
        "fake/swe-green",
        u_delta=0.03,
        project_dir=str(project_dir),
    )
    result = run_rollback(
        config_path=str(project_dir / "aua_config.yaml"),
        specialist="swe",
        yes=True,
        restart=False,
    )
    assert result == 0


def test_rollback_updates_config(project_dir):
    """After rollback, aua_config.yaml model points to BLUE model."""
    blue_model = "fake/swe-blue"
    record_promotion(
        "swe",
        blue_model,
        "fake/swe-green",
        project_dir=str(project_dir),
    )
    run_rollback(
        config_path=str(project_dir / "aua_config.yaml"),
        specialist="swe",
        yes=True,
        restart=False,
    )
    raw = yaml.safe_load((project_dir / "aua_config.yaml").read_text())
    swe = next(s for s in raw["specialists"] if s["name"] == "swe")
    assert swe["model"] == blue_model


def test_rollback_marks_promotion_reverted(project_dir):
    """After rollback, the promotion log entry is marked reverted=True."""
    record_promotion("swe", "blue", "green", project_dir=str(project_dir))
    run_rollback(
        config_path=str(project_dir / "aua_config.yaml"),
        specialist="swe",
        yes=True,
        restart=False,
    )
    events = load_promotions(str(project_dir))
    promote_event = next(e for e in events if e.event == "promote")
    assert promote_event.reverted is True


def test_rollback_appends_rollback_event(project_dir):
    """After rollback, a 'rollback' event is appended to the log."""
    record_promotion("swe", "blue", "green", project_dir=str(project_dir))
    run_rollback(
        config_path=str(project_dir / "aua_config.yaml"),
        specialist="swe",
        yes=True,
        restart=False,
    )
    events = load_promotions(str(project_dir))
    rollback_events = [e for e in events if e.event == "rollback"]
    assert len(rollback_events) == 1


def test_double_rollback_returns_1(project_dir):
    """Second rollback on same specialist (already reverted) returns 1."""
    record_promotion("swe", "blue", "green", project_dir=str(project_dir))
    # First rollback — should succeed
    run_rollback(
        config_path=str(project_dir / "aua_config.yaml"),
        specialist="swe",
        yes=True,
        restart=False,
    )
    # Second rollback — nothing left to revert
    result = run_rollback(
        config_path=str(project_dir / "aua_config.yaml"),
        specialist="swe",
        yes=True,
        restart=False,
    )
    assert result == 1


def test_rollback_all_skips_specialists_with_no_history(project_dir):
    """--all skips specialists with no un-reverted promotions without error."""
    # Only record for swe — math has nothing
    record_promotion("swe", "blue", "green", project_dir=str(project_dir))
    result = run_rollback(
        config_path=str(project_dir / "aua_config.yaml"),
        all_specialists=True,
        yes=True,
        restart=False,
    )
    # swe reverted successfully; math skipped silently → overall 0
    assert result == 0


def test_rollback_cli_no_restart(project_dir):
    """aua rollback --no-restart exits 0 and updates config."""
    record_promotion("swe", "blue-model", "green-model", project_dir=str(project_dir))
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rollback",
            "--config",
            str(project_dir / "aua_config.yaml"),
            "--specialist",
            "swe",
            "--yes",
            "--no-restart",
        ],
    )
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load((project_dir / "aua_config.yaml").read_text())
    swe = next(s for s in raw["specialists"] if s["name"] == "swe")
    assert swe["model"] == "blue-model"
