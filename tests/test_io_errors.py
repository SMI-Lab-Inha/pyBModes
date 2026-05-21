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

"""Tests for the unified :mod:`pybmodes.io.errors` parse-error hierarchy.

Locks in three properties of the typed-exception base added in
Phase 1 PR A3:

1. Inheritance chain — every per-format subclass inherits the new
   ``ParseError`` base, which in turn inherits :class:`ValueError`.
   This is the backward-compatibility contract: any code that
   already does ``except ValueError`` keeps working.
2. **Hashability** — exception instances stay hashable. Standard
   ``@dataclass`` generates ``__eq__`` and sets ``__hash__ = None``,
   which would break callers storing caught exceptions in sets /
   dict keys. The class uses ``@dataclass(eq=False)`` to preserve
   identity-based equality + hashability (the Python-built-in
   ``Exception`` semantics). Regression test for this.
3. ``format_diagnostic()`` renders the structured fields into a
   uniform ``file:line:column — message`` header.
"""
from __future__ import annotations

import pytest

from pybmodes.io.errors import (
    BMIParseError,
    ElastoDynParseError,
    MoorDynParseError,
    ParseError,
    SubDynParseError,
    WAMITParseError,
    WindIOParseError,
)
from pybmodes.io.out_parser import BModeOutParseError

_ALL_SUBCLASSES = [
    BMIParseError,
    ElastoDynParseError,
    SubDynParseError,
    WAMITParseError,
    MoorDynParseError,
    WindIOParseError,
    BModeOutParseError,   # legacy, re-rooted under ParseError
]


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

def test_parse_error_is_subclass_of_value_error() -> None:
    """``ParseError`` must inherit ``ValueError`` so existing
    ``except ValueError`` callers stay working."""
    assert issubclass(ParseError, ValueError)


@pytest.mark.parametrize("cls", _ALL_SUBCLASSES)
def test_every_subclass_inherits_parse_error_and_value_error(cls) -> None:
    assert issubclass(cls, ParseError)
    assert issubclass(cls, ValueError)


# ---------------------------------------------------------------------------
# Hashability (PR #68 static-review regression)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [ParseError, *_ALL_SUBCLASSES])
def test_instances_are_hashable(cls) -> None:
    """Built-in :class:`Exception` is hashable (identity-based).
    Plain ``@dataclass`` would inherit ``__eq__`` from the dataclass
    decorator and set ``__hash__ = None``, making instances
    unhashable — that breaks callers that put caught exceptions in
    ``set()`` or use them as dict keys. ``@dataclass(eq=False)``
    keeps the identity-based equality + hashability path.

    Static review flagged this as a P2 backward-compatibility
    regression in PR #68; this test pins the fix."""
    err = cls("test message", file="x.bmi", line=1, column=2)
    # ``hash()`` must not raise. The actual integer value is
    # identity-derived, not value-derived — two distinct instances
    # constructed with the same fields will hash differently.
    h1 = hash(err)
    assert isinstance(h1, int)

    # Two distinct instances with identical fields have distinct
    # identity, therefore distinct hashes (almost always; collisions
    # are vanishingly rare for ``object.__hash__``). Use them as
    # set members to confirm the standard pattern works.
    err2 = cls("test message", file="x.bmi", line=1, column=2)
    s = {err, err2}
    assert len(s) == 2


@pytest.mark.parametrize("cls", [ParseError, *_ALL_SUBCLASSES])
def test_equality_is_identity_based(cls) -> None:
    """Confirms that two ParseErrors constructed with identical
    fields are NOT equal (identity-based equality, matching
    built-in :class:`Exception`)."""
    err1 = cls("test", file="x.bmi", line=1)
    err2 = cls("test", file="x.bmi", line=1)
    assert err1 is not err2
    assert err1 != err2          # identity-based, not value-based
    assert err1 == err1          # but identical to itself


# ---------------------------------------------------------------------------
# Diagnostic formatting
# ---------------------------------------------------------------------------

def test_format_diagnostic_full_context() -> None:
    err = BMIParseError(
        "bad token",
        file="test.bmi",
        line=42,
        column=3,
        context="   abc = xyz",
    )
    out = err.format_diagnostic()
    # Header line has file:line:column — message
    assert "test.bmi:42:3" in out
    assert "bad token" in out
    # Context is indented below
    assert "    abc = xyz" in out


def test_format_diagnostic_no_file() -> None:
    err = ParseError("just a message")
    assert err.format_diagnostic() == "just a message"


def test_format_diagnostic_file_only() -> None:
    err = ParseError("oops", file="x.dat")
    out = err.format_diagnostic()
    assert "x.dat" in out
    assert "oops" in out
    # No ``:`` line/column suffix when those are unset
    assert ":" not in out.split("x.dat", 1)[1].split(" — ", 1)[0]


def test_format_diagnostic_file_and_line_only() -> None:
    err = ParseError("oops", file="x.dat", line=7)
    out = err.format_diagnostic()
    assert "x.dat:7 — oops" in out


# ---------------------------------------------------------------------------
# Str / message
# ---------------------------------------------------------------------------

def test_str_returns_message() -> None:
    err = ParseError("the message", file="x.dat", line=7)
    assert str(err) == "the message"
