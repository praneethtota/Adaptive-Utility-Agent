"""
tests/conftest.py — shared pytest fixtures for the AUA test suite.

Fixtures:
    minimal_config      load the minimal test config (no server needed)
    two_spec_config     load two-specialist config (no server needed)
    fake_swe_server     start a fake SWE specialist on a random port
    fake_two_servers    start fake SWE + arbiter on random ports
    fixtures_dir        Path to the test fixtures directory
"""

from pathlib import Path

import pytest

from aua.config import AUAConfig, load_config
from tests.fakes.openai_server import make_fake_specialist, start_fake_server

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def minimal_config() -> AUAConfig:
    """Load the minimal test config (no live servers required)."""
    return load_config(FIXTURES / "aua_config_minimal.yaml")


@pytest.fixture
def two_spec_config() -> AUAConfig:
    """Load the two-specialist test config (no live servers required)."""
    return load_config(FIXTURES / "aua_config_two_specialists.yaml")


@pytest.fixture
def fake_swe_server(minimal_config):
    """
    Start a fake SWE specialist + arbiter on the ports from minimal_config.
    Yields (swe_port, arbiter_port). Tears down after the test.
    """
    swe_app = make_fake_specialist(
        model_name="swe",
        response="Binary search runs in O(log n) time.",
        stream_tokens=["Binary", "search", "runs", "in", "O(log", "n)", "time."],
    )
    arb_app = make_fake_specialist(
        model_name="arbiter",
        response="VERDICT: A\nREASON: A is correct.\nCORRECTION: none",
    )

    swe_port = minimal_config.specialist("swe").port
    arb_port = minimal_config.arbiter.port

    _, stop_swe = start_fake_server(swe_app, port=swe_port)
    _, stop_arb = start_fake_server(arb_app, port=arb_port)

    yield swe_port, arb_port

    stop_swe()
    stop_arb()


@pytest.fixture
def fake_two_servers(two_spec_config):
    """
    Start fake SWE + math specialists + arbiter for fanout tests.
    Yields (swe_port, math_port, arb_port).
    """
    swe_app = make_fake_specialist(model_name="swe", response="SWE answer.")
    math_app = make_fake_specialist(model_name="math", response="Math answer.")
    arb_app = make_fake_specialist(
        model_name="arbiter",
        response="VERDICT: A\nREASON: A is correct.\nCORRECTION: none",
    )

    swe_port = two_spec_config.specialist("swe").port
    math_port = two_spec_config.specialist("math").port
    arb_port = two_spec_config.arbiter.port

    _, stop_swe = start_fake_server(swe_app, port=swe_port)
    _, stop_math = start_fake_server(math_app, port=math_port)
    _, stop_arb = start_fake_server(arb_app, port=arb_port)

    yield swe_port, math_port, arb_port

    stop_swe()
    stop_math()
    stop_arb()
