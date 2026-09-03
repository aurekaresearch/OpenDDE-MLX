# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Console entry point: ``opendde {pred,json,msa,mt,prep,convert,doctor}``."""

from __future__ import annotations

import difflib
import importlib

import click

from opendde.version import __version__

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "show_default": True}
_RUNTIME_COMMANDS = {
    "json": ("tojson", "Convert PDB or CIF files to OpenDDE inference JSON."),
    "msa": ("msa", "Run protein MSA search."),
    "mt": ("msatemplate", "Run protein MSA and template search."),
    "pred": ("predict", "Run OpenDDE structure prediction."),
    "prep": ("inputprep", "Prepare MSA, template, and RNA MSA input features."),
}
_COMMAND_HELP = {
    "doctor": "Print environment diagnostics.",
    "convert": "Convert a PyTorch .pt checkpoint to .safetensors.",
    **{name: help_text for name, (_, help_text) in _RUNTIME_COMMANDS.items()},
}


class LazyGroup(click.Group):
    """Import the heavy runtime module only for the commands that need it."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(_COMMAND_HELP)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None or cmd_name not in _RUNTIME_COMMANDS:
            return command
        module = importlib.import_module("runner.batch_inference")
        return getattr(module, _RUNTIME_COMMANDS[cmd_name][0])

    def resolve_command(self, ctx: click.Context, args: list[str]):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as exc:
            matches = difflib.get_close_matches(args[0], self.list_commands(ctx)) if args else []
            if matches:
                raise click.UsageError(
                    f"{exc.message}\n\nDid you mean one of these?\n    " + ", ".join(matches),
                    ctx=exc.ctx,
                ) from exc
            raise

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        with formatter.section("Commands"):
            formatter.write_dl([(name, _COMMAND_HELP[name]) for name in self.list_commands(ctx)])


@click.command(context_settings=CONTEXT_SETTINGS)
def doctor() -> None:
    """Print environment diagnostics."""
    from opendde.utils.environment import format_doctor_report

    click.echo(format_doctor_report())


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("pt_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=str, default=None, help="Output .safetensors path.")
def convert(pt_path: str, output: str | None) -> None:
    """Convert a PyTorch .pt checkpoint to .safetensors (requires torch)."""
    import os

    from opendde.model.checkpoint import convert_torch_checkpoint

    output = output or os.path.splitext(pt_path)[0] + ".safetensors"
    convert_torch_checkpoint(pt_path, output)
    click.echo(f"Saved {output}")


@click.group(name="opendde", cls=LazyGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
def opendde_cli() -> None:
    """OpenDDE-MLX: biomolecular co-folding on Apple Silicon."""


opendde_cli.add_command(doctor)
opendde_cli.add_command(convert)
