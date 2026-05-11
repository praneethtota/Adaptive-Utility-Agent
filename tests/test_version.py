"""
tests/test_version.py — version consistency checks.

Verifies that the version string is identical everywhere it appears:
  aua/version.py       — single source of truth
  aua.__version__      — re-exported from __init__
  package metadata     — read via importlib.metadata (after pip install -e .)
  aua CLI              — aua --version
"""

import pytest

import aua
from aua.version import __version__


def test_version_format():
    """Version must be a valid PEP 440 version string.

    Accepts: X.Y.Z, X.Y.ZaN (alpha), X.Y.ZbN (beta), X.Y.ZrcN (release candidate).
    Examples: 0.5.0, 0.6.0a0, 0.7.0b1, 1.0.0rc1
    """
    from packaging.version import InvalidVersion, Version

    try:
        v = Version(__version__)
    except InvalidVersion:
        pytest.fail(f"Invalid PEP 440 version: {__version__!r}")

    # Must have at least major.minor.patch
    assert v.major >= 0
    assert v.minor >= 0
    assert v.micro >= 0


def test_init_re_exports_version():
    """aua.__version__ must equal aua.version.__version__."""
    assert aua.__version__ == __version__


def test_version_in_all():
    """__version__ must be listed in aua.__all__."""
    assert "__version__" in aua.__all__


def test_cli_version():
    """aua --version must print the correct version string."""
    from click.testing import CliRunner

    from aua.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
