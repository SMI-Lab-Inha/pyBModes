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

"""Low-level scanning helpers for the ElastoDyn parser.

Pure functions only — no dataclass imports here. The parser builds
``ElastoDynMain`` / ``ElastoDynTower`` / ``ElastoDynBlade`` records
above this layer; here we just turn raw lines into (value-string,
label) tuples and parse FORTRAN-style scalars.

Why label-based rather than position-based: ElastoDyn ``.dat`` files
have shifted columns and inserted/removed scalars across FAST v8 →
OpenFAST v3+. A label-driven parser remains robust to that drift; a
position-driven one would need a per-version dispatch table.
"""

from __future__ import annotations

import re
from typing import Optional

# Capture: <values...>  <label>  [- comment]
# label = identifier with optional (n) suffix (e.g. "BldFile(1)") or a digit
# tail with no parens (e.g. "BldFile1"). The lookahead requires whitespace +
# ``-`` afterward, but ``-`` may be missing on some legacy lines, so we make
# that branch tolerant.
_RE_LINE = re.compile(
    r"""
    ^\s*
    (?P<value>.+?)              # value(s), non-greedy
    \s+
    (?P<label>[A-Za-z][A-Za-z0-9_]*(?:\(\s*\d+\s*\))?)  # label
    (?:\s+-.*)?                 # optional " - comment" tail
    \s*$
    """,
    re.VERBOSE,
)


# Labels for which the IEA RWT files use the bare-digit form ``BldFile1``
# instead of the parenthesised ``BldFile(1)``. Extend this tuple if a new
# upstream pipeline ships another bare-digit indexed scalar.
_BARE_INDEXED_LABELS = ("BldFile",)


def _strip_quotes(tok: str) -> str:
    """Strip a single layer of straight quotes (single or double) from
    a token, leaving any internal whitespace and content intact."""
    t = tok.strip()
    if len(t) >= 2 and (
        (t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")
    ):
        return t[1:-1]
    return t


def _is_section_divider(line: str) -> bool:
    """``True`` if ``line`` looks like a section divider in an
    ElastoDyn file — three or more leading ``-`` or ``=``."""
    s = line.strip()
    return s.startswith("---") or s.startswith("===")


def _is_file_header(line: str) -> bool:
    """``True`` if ``line`` is the first-line header of an ElastoDyn
    main file (the one containing the literal ``ELASTODYN`` token)."""
    s = line.strip()
    return s.startswith("---") and "ELASTODYN" in s.upper()


def _parse_float(tok: str) -> float:
    """Parse a FORTRAN-or-Python-style float, normalising the FORTRAN
    ``D`` exponent marker to Python's ``e`` (e.g. ``1.0D+03 → 1.0e+03``).

    Rejects ``nan`` / ``inf`` — physical ElastoDyn quantities
    (NumBl, TipRad, EI columns, mode-shape coefficients, …) must be
    finite. The pass-4 review tightened the BMI and section-
    properties parsers; the pass-5 negative-paths audit caught that
    this ElastoDyn-specific copy was left permissive.
    """
    import math
    value = float(tok.strip().replace("d", "e").replace("D", "E"))
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite float in ElastoDyn token: {tok!r} parses to "
            f"{value!r}. Physical quantities must be finite."
        )
    return value


def _split_label_index(label: str) -> tuple[str, Optional[int]]:
    """Strip an array index from ``label``, returning ``(canon, idx)``.

    Two indexed-label forms are recognised:

    * Parenthesised — ``Foo(N)`` (the ElastoDyn convention for most arrays).
      Stripping the suffix preserves embedded digits in the base name, so
      ``Twr2Shft`` stays ``Twr2Shft`` and ``TwFAM1Sh(2)`` becomes
      ``TwFAM1Sh, idx=1``.
    * Bare-digit — ``FooN``, allowed only for labels listed in
      :data:`_BARE_INDEXED_LABELS` (currently only ``BldFile``, where the
      IEA RWT files use ``BldFile1`` while the 5MW deck uses ``BldFile(1)``).
    """
    m = re.match(r"^([A-Za-z][A-Za-z0-9_]*?)\s*\((\d+)\)\s*$", label)
    if m:
        return m.group(1), int(m.group(2)) - 1
    for base in _BARE_INDEXED_LABELS:
        m = re.match(rf"^{base}(\d+)$", label)
        if m:
            return base, int(m.group(1)) - 1
    return label, None


def _canon_label(label: str) -> str:
    """Return ``label`` with any array-index suffix stripped."""
    return _split_label_index(label)[0]


def _split_value_label(line: str) -> Optional[tuple[str, str]]:
    """Split a data line into ``(value-string, label)``; ``None`` if no
    label was matched."""
    if not line.strip() or _is_section_divider(line):
        return None
    m = _RE_LINE.match(line)
    if not m:
        return None
    return m.group("value").strip(), m.group("label")
