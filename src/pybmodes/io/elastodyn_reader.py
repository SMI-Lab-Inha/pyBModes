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

"""Parser, writer, and pyBmodes adapter for OpenFAST ElastoDyn ``.dat``
files.

ElastoDyn ``.dat`` files come in three flavours, all of which share a
common line-ordered convention: each significant line carries a value
(or short list) followed by a label keyword and an optional ``-
comment`` tail. The parsers below are **label-based** rather than
position-based, which keeps them robust across the FAST v8 → OpenFAST
v3+ format drift (renamed/added/removed scalars between versions;
``BldFile(1)`` vs ``BldFile1``).

Three entry points cover the three file flavours:

* :func:`read_elastodyn_main`  — the top-level ElastoDyn input file.
* :func:`read_elastodyn_tower` — the tower-properties file referenced
  via ``TwrFile``.
* :func:`read_elastodyn_blade` — the blade-properties file referenced
  via ``BldFile(1..3)``.

Each returns a dataclass holding the parsed values plus enough raw-
line metadata to re-emit a semantically identical file via the
matching ``write_*`` function.

Two adapter helpers turn a parsed ElastoDyn bundle into pyBmodes
``BMIFile`` + ``SectionProperties`` records ready for the FEM core:

* :func:`to_pybmodes_tower(main, tower, blade=None)`
* :func:`to_pybmodes_blade(main, blade)`

The implementation lives in the private sub-package
``pybmodes.io._elastodyn``:

* ``types``   — dataclasses
* ``lex``     — line/token scanning helpers
* ``parser``  — line-driven flavour parsers
* ``writer``  — canonical re-emitters
* ``adapter`` — pyBmodes BMI/SectionProperties synthesisers

This module re-exports the public names from those sub-modules so
existing ``from pybmodes.io.elastodyn_reader import …`` imports keep
working unchanged. Field-set discrepancies vs. the user-specified
spec are documented under each dataclass.

Round-trip contract
-------------------

``write_*`` functions emit a canonically formatted file that **parses
back to an equal dataclass** but is not byte-identical to the
original. Whitespace, label column position, and comment text are
normalised. The test suite compares the parse-emit-reparse fixed
point (see ``tests/test_elastodyn_reader.py``).
"""

from __future__ import annotations

# Re-export the public surface. Private helpers (leading-underscore
# names) are re-exported too — internal pyBmodes callers (currently
# ``pybmodes.io.subdyn_reader``) depend on them — and are listed in
# ``__all__`` below so ruff treats the imports as intentional rather
# than F401 "unused".
from pybmodes.io._elastodyn.adapter import (
    _build_bmi_skeleton,
    _resolve_relative,
    _rotary_inertia_floor,
    _stack_blade_section_props,
    _stack_tower_section_props,
    _tower_top_assembly_mass,
    to_pybmodes_blade,
    to_pybmodes_tower,
)
from pybmodes.io._elastodyn.parser import (
    read_elastodyn_blade,
    read_elastodyn_main,
    read_elastodyn_tower,
)
from pybmodes.io._elastodyn.types import (
    ElastoDynBlade,
    ElastoDynMain,
    ElastoDynTower,
)
from pybmodes.io._elastodyn.writer import (
    write_elastodyn_blade,
    write_elastodyn_main,
    write_elastodyn_tower,
)

# Public surface plus internal-re-export surface. Names starting with
# an underscore are *not* part of the stable public API; they're
# listed here only to (a) signal to ruff that the imports above are
# intentional, and (b) keep ``pybmodes.io.subdyn_reader``'s existing
# import path working. ``from pybmodes.io.elastodyn_reader import *``
# will pick them up too; callers writing wildcard imports get what
# they ask for.
__all__ = [
    "ElastoDynMain",
    "ElastoDynTower",
    "ElastoDynBlade",
    "read_elastodyn_main",
    "read_elastodyn_tower",
    "read_elastodyn_blade",
    "write_elastodyn_main",
    "write_elastodyn_tower",
    "write_elastodyn_blade",
    "to_pybmodes_tower",
    "to_pybmodes_blade",
    # Internal re-exports for pybmodes.io.subdyn_reader; not part of
    # the stable public API and may be reorganised without notice.
    "_rotary_inertia_floor",
    "_stack_blade_section_props",
    "_stack_tower_section_props",
    "_tower_top_assembly_mass",
    "_build_bmi_skeleton",
    "_resolve_relative",
]
