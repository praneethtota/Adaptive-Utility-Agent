"""
aua/templates/registry.py — Prompt template registry.

Templates are versioned text files. Config can override which version
to use, or supply a custom template path.

Built-in templates:
    classifier_v1       Field classification prompt
    arbiter_balanced_v1 Balanced arbitration (default)
    abstention_v1       Abstention response when confidence is too low

YAML configuration:
    prompts:
      classifier_template: classifier_v1
      arbiter_template: arbiter_balanced_v1
      abstention_template: abstention_v1
      # Or custom:
      arbiter_template_path: ./prompts/my_arbiter.txt
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "prompts"


def get_template(name: str, override_path: str | None = None) -> str:
    """
    Load a prompt template by name or custom path.

    Args:
        name:          built-in template name (e.g. "classifier_v1")
        override_path: path to a custom template file (overrides name)

    Returns:
        Template string with {variable} placeholders.
    """
    if override_path:
        path = Path(override_path)
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {override_path}")
        return path.read_text()

    path = _TEMPLATE_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Built-in template {name!r} not found. " f"Available: {list_templates()}"
        )
    return path.read_text()


def render_template(name: str, override_path: str | None = None, **kwargs: object) -> str:
    """Load and render a template with provided variables."""
    template = get_template(name, override_path)
    return template.format(**kwargs)


def list_templates() -> list[str]:
    """Return names of all available built-in templates."""
    return sorted(p.stem for p in _TEMPLATE_DIR.glob("*.txt"))
