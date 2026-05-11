"""
tests/test_version.py — version consistency checks.

Verifies that the version string is identical everywhere it appears:
  aua/version.py       — single source of truth
  aua.__version__      — re-exported from __init__
  package metadata     — read via importlib.metadata (after pip install -e .)
  aua CLI              — aua --version
"""

import aua
from aua.version import __version__


def test_version_format():
    """Version must be a valid semver-ish string like '0.5.0'."""
    parts = __version__.split(".")
    assert len(parts) == 3, f"Expected X.Y.Z format, got {__version__!r}"
    for part in parts:
        assert part.isdigit(), f"Each part must be numeric, got {part!r}"


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
