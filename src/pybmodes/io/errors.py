# Copyright 2024-2026 Jae Hoon Seo
# Marine Structural Mechanics and Integrity Lab (SMI Lab), Inha University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified parse-error base class for every ``pybmodes.io.*`` reader.

Every input-format parser ``pybmodes`` ships (BModes ``.bmi`` /
section-properties ``.dat`` / OpenFAST ElastoDyn / SubDyn / HydroDyn /
MoorDyn ``.dat`` / WAMIT ``.1`` ``.hst`` / BModes ``.out`` reference
output / WISDEM / WindIO ontology ``.yaml``) previously raised either a
bare :class:`ValueError` with a format-specific message or, in the
case of :mod:`pybmodes.io.out_parser`, the legacy
:class:`BModeOutParseError`. That made downstream
``try / except ValueError`` callers correct but indiscriminate, and
the per-error file / line context was unstructured prose.

:class:`ParseError` is the unified base. It inherits :class:`ValueError`
so existing ``except ValueError`` callers continue to catch every
parse error untouched — the inheritance addition is non-breaking.

Each parser will (incrementally) start raising :class:`ParseError`
subclasses with structured ``file`` / ``line`` / ``column`` /
``context`` fields. Callers that want file / line context can switch
to ``except ParseError`` and read the typed fields; callers that
only need "the file is broken" stay on ``except ValueError``.

Examples
--------

Catch a parse error and pull out the file / line / column context::

    from pybmodes.io.errors import ParseError
    from pybmodes.io.bmi import read_bmi

    try:
        bmi = read_bmi("malformed.bmi")
    except ParseError as err:
        print(f"parse failed in {err.file}:{err.line}: {err}")
        print(err.format_diagnostic())   # one-line diagnostic string

Keep an older ``except ValueError`` pattern working — :class:`ParseError`
is a subclass so the catch still triggers::

    try:
        bmi = read_bmi("malformed.bmi")
    except ValueError as err:
        print(f"oops: {err}")   # also catches every ParseError
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(eq=False)
class ParseError(ValueError):
    """Base class for every ``pybmodes.io.*`` parser exception.

    Inherits :class:`ValueError` so ``except ValueError`` catches it
    unchanged. Inheriting parsers add structured context fields below.

    Attributes
    ----------
    message : str
        The human-readable description of the failure. Same as
        ``str(err)``.
    file : pathlib.Path-compatible str | None
        The source file the parser was reading from, when known.
        ``None`` for in-memory parses (e.g. yaml from a string).
    line : int | None
        1-based line number where the error was detected, when
        known. The parser is encouraged to populate this even for
        token-level errors — fall back to the start of the
        containing block.
    column : int | None
        1-based column number where the error was detected, when
        known. Optional; many parsers don't track column.
    context : str | None
        A short snippet — typically the offending line or token —
        that gives the reader visual confirmation of *what* the
        parser tripped on. Up to one or two lines; the parser
        should truncate large blobs.
    """

    message: str
    file: str | None = None
    line: int | None = None
    column: int | None = None
    context: str | None = None

    def __post_init__(self) -> None:
        # Cooperate with ValueError's positional-argument convention:
        # ``str(err)`` should return ``message`` directly so existing
        # f-string error formatting keeps working.
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def format_diagnostic(self) -> str:
        """Render a one-line ``file:line:column — message`` diagnostic.

        Useful as a fallback when callers want a uniform error format
        regardless of which parser raised. Falls back to just the
        message when no file / line context is available.
        """
        parts: list[str] = []
        if self.file is not None:
            parts.append(str(self.file))
            if self.line is not None:
                parts[-1] += f":{self.line}"
                if self.column is not None:
                    parts[-1] += f":{self.column}"
        parts.append(self.message)
        head = " — ".join(parts)
        if self.context:
            return f"{head}\n    {self.context}"
        return head


@dataclass(eq=False)
class BMIParseError(ParseError):
    """Raised by :func:`pybmodes.io.bmi.read_bmi` and the companion
    section-properties parser when the input deck is malformed.

    The parser is line-oriented; ``line`` is the 1-based row in the
    source file. ``context`` is the offending line (truncated).
    """


@dataclass(eq=False)
class ElastoDynParseError(ParseError):
    """Raised by the ElastoDyn deck reader
    (:mod:`pybmodes.io.elastodyn_reader` / the private
    ``pybmodes.io._elastodyn`` sub-package) on malformed input.
    """


@dataclass(eq=False)
class SubDynParseError(ParseError):
    """Raised by :mod:`pybmodes.io.subdyn_reader` when the SubDyn
    joints / members / reaction-joint block can't be parsed."""


@dataclass(eq=False)
class WAMITParseError(ParseError):
    """Raised by :class:`pybmodes.io.wamit_reader.HydroDynReader` and
    the underlying ``.1`` / ``.hst`` readers on malformed WAMIT
    output (bad re-dimensionalisation, missing files behind
    ``PotFile``, asymmetric matrices that can't be mirrored)."""


@dataclass(eq=False)
class MoorDynParseError(ParseError):
    """Raised by :meth:`pybmodes.mooring.MooringSystem.from_moordyn`
    when a MoorDyn ``.dat`` carries an unrecognised layout or a
    point-ID column ordering the parser can't auto-detect."""


@dataclass(eq=False)
class WindIOParseError(ParseError):
    """Raised by the WindIO ontology readers
    (:mod:`pybmodes.io.windio` / :mod:`pybmodes.io.windio_blade` /
    :mod:`pybmodes.io.windio_floating`) on schema-drift or
    malformed published blocks.

    Includes both ``components.<component>`` lookup failures and the
    "elastic_properties block is present but unparseable" path that
    ``elastic="auto"`` / ``"file"`` distinguish."""


__all__ = [
    "BMIParseError",
    "ElastoDynParseError",
    "MoorDynParseError",
    "ParseError",
    "SubDynParseError",
    "WAMITParseError",
    "WindIOParseError",
]
