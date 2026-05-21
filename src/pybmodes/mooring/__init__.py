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

"""Quasi-static mooring linearisation for pyBmodes.

Solves the extensible elastic catenary per line, sums fairlead tensions
into a platform 6-DOF restoring force, and returns a finite-difference
6×6 mooring stiffness matrix ready for the ``PlatformSupport.mooring_K``
block.

References
==========

- **Jonkman, J. M. (2007).** *Dynamics Modeling and Loads Analysis of an
  Offshore Floating Wind Turbine*, NREL/TP-500-41958. Appendix B,
  equations B-1 / B-2 are the extensible-catenary boundary-condition
  equations implemented in :func:`pybmodes.mooring._catenary._catenary_residual`.
  B-7 / B-8 are the seabed-contact variants (no friction; ``CB = 0``).
- **Irvine, H. M. (1981).** *Cable Structures*, MIT Press, §2.4 the
  extensible elastic catenary. Equivalent derivation; the EA-correction
  terms ``H · L / EA`` and ``L · (V − ½WL) / EA`` come from §2.4 eqn
  (2.49).

Module layout
=============

Phase 3 PR C2 of the v1.x architecture refactor split this from a
single 1202-line module into a sub-package. The public API is
unchanged; internal helpers live in private sub-modules:

- :mod:`pybmodes.mooring.types` — :class:`LineType`, :class:`Point`,
  :class:`Line` dataclasses. :meth:`Line.solve_static` is here.
- :mod:`pybmodes.mooring.system` — :class:`MooringSystem`: multi-line
  force assembly, equilibrium Newton, 6×6 stiffness, plus the
  :meth:`MooringSystem.from_moordyn` and
  :meth:`MooringSystem.from_windio_mooring` classmethod parsers.
- ``_catenary`` — extensible-elastic-catenary residual + analytical
  2×2 Jacobian (Jonkman 2007 B-1 / B-2 + B-7 / B-8).
- ``_rotation`` — 3-2-1 intrinsic Euler rotation primitive (ElastoDyn
  attitude convention).
- ``_moordyn_parser`` — MoorDyn ``.dat`` section + row tokenisers.

Scope
=====

Implemented:

- Extensible elastic catenary per line (Newton on ``(H, V)``;
  analytical 2 × 2 Jacobian; ``tol = 1e-6`` m, ``MaxIter = 100``).
- Fully-suspended (``V_F ≥ W · L``) and anchor-on-seabed
  (``V_F < W · L``, zero friction) profiles, branched inside the
  residual function.
- Multi-line platform restoring force from a 6-DOF body offset.
- Central-difference linearisation around an arbitrary or zero offset
  producing ``K_mooring`` (6, 6).
- MoorDyn v1 (``CONNECTION``) and v2 (``POINT``) ``.dat`` parsing.

Known limitations:

- Seabed friction (``CB > 0``) is parsed from ``LineType`` but not
  consumed by the catenary solver.
- Sloped seabed, U-shape lines (one line touching the seabed mid-
  span), and the vertical-line degenerate case (``H → 0``) are not
  handled.
- Time-domain dynamics, hydrodynamic drag, and added mass on the
  lines themselves are out of scope — this is a quasi-static
  linearised model only.
- :meth:`MooringSystem.solve_equilibrium` defaults to the input
  offset; pure mooring has no z equilibrium without buoyancy / weight,
  so the Newton iteration is only meaningful for the in-plane DOFs
  (surge, sway, yaw) of a 3-fold-symmetric layout. Callers wanting
  platform equilibrium under a full force model should call
  :meth:`MooringSystem.restoring_force` and assemble the rest of the
  forces themselves.

Coordinate / unit conventions
=============================

SI throughout: m, N, kg, kg/m, N/m; radians for rotations. Origin at MSL;
z positive upward; anchors at negative z (below MSL). Matches
OpenFAST / HydroDyn / ElastoDyn without any coordinate transform.

Body rotation uses the 3-2-1 (z-y-x intrinsic) Euler angle convention
``R = R_z(yaw) · R_y(pitch) · R_x(roll)`` — the same convention as
ElastoDyn's platform 6-DOF state.
"""
from __future__ import annotations

from .system import MooringSystem
from .types import Line, LineType, Point

__all__ = ["LineType", "Point", "Line", "MooringSystem"]
