"""The CLI help text must be plain ASCII.

A non-ASCII glyph in an argparse help / description string crashes
``pybmodes <cmd> --help`` on a legacy Windows console (cp1252 / cp437)
with a ``UnicodeEncodeError`` when argparse writes the formatted help.
A rightwards arrow in the ``windio`` subcommand help did exactly that
and was caught by conda-forge's Windows build. These tests assert every
parser's formatted help is ASCII-encodable so it prints on any code
page, and guard against the regression returning.
"""

from __future__ import annotations

import argparse

import pytest

from pybmodes.cli import _build_parser


def _iter_parsers():
    """Yield (label, parser) for the main parser and every subparser."""
    parser = _build_parser()
    yield "pybmodes", parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield f"pybmodes {name}", sub


def test_all_cli_help_is_ascii() -> None:
    """Every parser's formatted help encodes as ASCII, so it never trips
    the console-encoding crash on a non-UTF-8 Windows terminal."""
    offenders: list[str] = []
    for label, parser in _iter_parsers():
        try:
            parser.format_help().encode("ascii")
        except UnicodeEncodeError as exc:
            offenders.append(f"{label}: {exc}")
    assert not offenders, "non-ASCII CLI help:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("encoding", ["cp1252", "cp437"])
def test_cli_help_encodes_on_legacy_windows_codepages(encoding: str) -> None:
    """Belt and braces: the help also encodes on the two common legacy
    Windows console code pages, the environments where the original
    crash happened."""
    for _label, parser in _iter_parsers():
        parser.format_help().encode(encoding)  # must not raise
