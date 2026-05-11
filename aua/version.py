"""
aua/version.py — single source of truth for the AUA Framework version.

All other version references import from here:
    from aua.version import __version__

To release a new version, update ONLY this file.
pyproject.toml reads this file via hatchling dynamic versioning.
"""

__version__ = "0.6.0a0"
