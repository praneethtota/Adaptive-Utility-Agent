"""
aua/cli.py — Command-line interface stub.

Full commands (#04–#08) are implemented in the next build round.
This stub makes `aua --help` and `aua --version` work immediately after install.
"""

import click


@click.group()
@click.version_option(version="0.5.0", prog_name="aua")
def main():
    """Adaptive Utility Agents — deployable specialist framework."""


@main.command()
@click.option("--config", default="aua_config.yaml", show_default=True,
              help="Path to aua_config.yaml")
def serve(config):
    """Start all specialists + router from config.  [#04 — coming next]"""
    click.echo(f"[aua serve] will start specialists defined in {config}")
    click.echo("Full implementation in #04. Run 'aua doctor' to check readiness.")


@main.command()
def doctor():
    """Check endpoints, VRAM, and dependencies.  [#07 — coming next]"""
    click.echo("[aua doctor] diagnostics — full implementation in #07.")


@main.command()
def status():
    """Live terminal dashboard.  [#06 — coming next]"""
    click.echo("[aua status] dashboard — full implementation in #06.")


@main.command()
@click.argument("project_dir", default=".")
def init(project_dir):
    """Scaffold a new AUA project.  [#05 — coming next]"""
    click.echo(f"[aua init] will scaffold a project in {project_dir} — coming in #05.")


@main.command()
@click.option("--config", default="aua_config.yaml", show_default=True)
def rollback(config):
    """Revert to the previous BLUE model.  [#08 — coming next]"""
    click.echo("[aua rollback] — full implementation in #08.")
