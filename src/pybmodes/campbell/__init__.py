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

"""Campbell-diagram support: rotor-speed sweep with MAC-tracked blade
modes and constant-frequency tower modes overlaid on the same plot.

A Campbell diagram plots a turbine's natural frequencies against rotor
speed and overlays the per-revolution excitation lines (1P, 2P, 3P,
…); crossings between excitation lines and structural-mode lines flag
resonance risks. For a wind-turbine blade the centrifugal-stiffening
contribution to the FEM stiffness matrix raises flap-dominated
frequencies markedly with rotor speed while edgewise (lag-dominated)
modes barely move. The tower lives in an Earth-fixed frame, so its
fore-aft / side-to-side bending frequencies don't depend on rotor
speed at all and show up as horizontal lines on the diagram. The
NREL 5MW turbine's canonical resonance call-out — 3P crossing the
1st tower fore-aft mode near ~6.4 rpm — sits right where the cut-in
operating envelope begins, which is exactly the kind of constraint
this diagram is designed to surface.

Module layout
-------------

Phase 3 PR C1 of the v1.x architecture refactor split this from a
single 1301-line module into a sub-package. The public API is
unchanged; internal helpers live in private sub-modules so each file
covers one concern and stays under a few hundred lines:

- :mod:`pybmodes.campbell.result` — :class:`CampbellResult` dataclass
  plus its NPZ / CSV round-trip.
- ``_models`` — input dispatcher: path-vs-loaded model, ``.dat``
  vs. ``.bmi``, optional ``tower_input`` keyword.
- ``_classify`` — mode-naming heuristics (``1st flap`` / ``1st tower FA``
  / platform DOFs).
- ``_mac`` — MAC matrix helpers + Hungarian assignment used by the
  blade-sweep tracker.
- ``_sweep`` — rotor-speed sweep drivers and the public
  :func:`campbell_sweep` entry point.
- ``_plot`` — :func:`plot_campbell`.

Public API
----------

- :func:`campbell_sweep` — given an OpenFAST ElastoDyn main ``.dat``,
  loads the blade and tower from the same deck, sweeps the blade
  across ``omega_rpm`` (with MAC-based mode tracking), solves the
  tower once, and packs both into a single :class:`CampbellResult`.
  ``.bmi`` inputs are also accepted and route to blade-only or
  tower-only sweeps based on ``beam_type``; an explicit
  ``tower_input=...`` keyword adds a tower file alongside a blade
  ``.bmi``.
- :func:`plot_campbell` — renders the result with blade modes as
  solid coloured lines, tower modes as horizontal dashed dark-grey
  lines, and the per-rev excitation family as light grey rays from
  the origin. Optional vertical marker at the rated rotor speed.

Defaults are deliberately spare (``n_blade_modes=4``, ``n_tower_modes=4``)
so the diagram shows the modes that actually drive resonance design —
1st/2nd flap, 1st/2nd edge, 1st/2nd tower FA, 1st/2nd tower SS —
without crowding the plot with high-order modes that the per-rev
family doesn't reach inside any realistic operating envelope.
"""
from __future__ import annotations

# Public API — kept at the package top-level for back-compat with the
# original monolith. ``from pybmodes.campbell import CampbellResult,
# campbell_sweep, plot_campbell`` works unchanged.
#
# Underscore-private helpers are re-exported too because existing
# tests (``test_campbell.py``, ``test_campbell_helpers.py``) import
# them from ``pybmodes.campbell`` by name. The ``noqa: F401`` markers
# silence ruff's "unused import" lint — these are intentional
# re-exports, not dead imports.
from ._classify import (
    _label_blade_modes,  # noqa: F401
    _label_tower_modes,  # noqa: F401
    _label_tower_modes_with_overrides,  # noqa: F401
    _ordinal,  # noqa: F401
    _participation,  # noqa: F401
)
from ._mac import (
    _greedy_assignment,  # noqa: F401
    _hungarian_assignment,  # noqa: F401
    _mac_matrix,  # noqa: F401
    _shape_vector,  # noqa: F401
)
from ._models import _load_models, _model_pair  # noqa: F401
from ._plot import plot_campbell
from ._sweep import (
    _solve_blade_sweep,  # noqa: F401
    _solve_tower_once,  # noqa: F401
    campbell_sweep,
)
from .result import CampbellResult

__all__ = [
    "CampbellResult",
    "campbell_sweep",
    "plot_campbell",
]
