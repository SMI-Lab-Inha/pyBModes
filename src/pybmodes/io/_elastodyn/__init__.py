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

"""Internal ElastoDyn IO sub-package.

The public surface for ElastoDyn ``.dat`` files lives at
``pybmodes.io.elastodyn_reader`` and re-exports the names below. The
modules inside this package are implementation detail and may be
reorganised across pyBmodes releases without notice.

Module map:

* :mod:`pybmodes.io._elastodyn.types`   — dataclasses for the three
  file flavours (``ElastoDynMain``, ``ElastoDynTower``,
  ``ElastoDynBlade``).
* :mod:`pybmodes.io._elastodyn.lex`     — low-level scanning helpers
  (line splitting, label-index parsing, float coercion).
* :mod:`pybmodes.io._elastodyn.parser`  — line-driven parsers for the
  three file flavours.
* :mod:`pybmodes.io._elastodyn.writer`  — canonical re-emitters that
  parse-write-parse round-trip to an equal dataclass.
* :mod:`pybmodes.io._elastodyn.adapter` — converters that build
  pyBmodes ``BMIFile`` + ``SectionProperties`` from a parsed
  ElastoDyn bundle.

The re-exports below preserve the historical
``from pybmodes.io.elastodyn_reader import …`` surface.
"""

from __future__ import annotations

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
    # Private helpers re-exported because internal pyBmodes callers
    # (currently ``pybmodes.io.subdyn_reader``) depend on them. Keep
    # the leading-underscore signalling so external code knows these
    # are not part of the stable public surface.
    "_rotary_inertia_floor",
    "_stack_blade_section_props",
    "_stack_tower_section_props",
    "_tower_top_assembly_mass",
    "_build_bmi_skeleton",
    "_resolve_relative",
]
