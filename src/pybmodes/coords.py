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

"""Coordinate / DOF conventions used throughout pyBmodes.

This module is **documentation, not implementation** — it carries no
runtime logic and zero coupling to the rest of the package. Its job
is to centralise the answer to "what order is index 0 in this 6×6
matrix?", so when a future contributor wonders whether a particular
constructor returns OpenFAST DOF order or BModes-historical order,
they read this module instead of reverse-engineering nondim_platform.

Single 6-DOF order used everywhere
==================================

pyBmodes uses **OpenFAST DOF order** for every 6 × 6 rigid-body
matrix exposed through the public API and stored in the
``PlatformSupport`` block:

==== ======== ========== ===================================
idx  symbol   DOF        physical sense
==== ======== ========== ===================================
0    x        surge      fore-aft translation (along wind)
1    y        sway       lateral translation
2    z        heave      vertical translation
3    φ        roll       rotation about the surge axis
4    θ        pitch      rotation about the sway axis
5    ψ        yaw        rotation about the heave axis
==== ======== ========== ===================================

This matches the convention in **Jonkman (2010) NREL/TP-500-47535**
Table 5-1 (and every subsequent OpenFAST publication), in
``OC3Hywind.bmi``'s mooring / hydro / inertia blocks, and in the
upstream WAMIT and MoorDyn outputs that
:class:`pybmodes.io.WamitReader` and
:class:`pybmodes.mooring.MooringSystem` consume. Across the entire
pipeline — from MoorDyn parsing through ``MooringSystem.stiffness_matrix``,
through ``PlatformSupport.mooring_K`` / ``hydro_K`` / ``hydro_M`` /
``i_matrix``, through ``pybmodes.fem.nondim.nondim_platform``'s T-
matrix — index 0 always means surge.

Why this matters
================

A latent DOF-order bug on an asymmetric mooring or hydro matrix would
silently swap surge↔sway / roll↔pitch coupling without raising; the
catenary cert tolerance ``test_certtest_oc3hywind`` (≤ 0.0003 % on
the first 9 modes against BModes JJ) cannot catch it because OC3 is
N-fold symmetric and the swapped entries are numerically equal. The
load-bearing invariant is checked by
``tests/test_mooring.py::test_oc3hywind_bmi_dof_order_matches_jonkman_2010``,
which asserts ``OC3Hywind.bmi.mooring_K[0, 4] = −2.821e6`` matches
Jonkman (2010) K_15 = surge → pitch coupling (negative sign), and
``mooring_K[1, 3] = +2.821e6`` matches K_24 = sway → roll (positive
sign). If anyone refactors the pipeline and the convention drifts,
that test fires.

Internal FEM tower-base DOFs
============================

The FEM solver internally uses a different per-node 6-DOF layout for
each tower / blade node:

==== ============== ==================================
idx  FEM DOF        physical sense
==== ============== ==================================
0    axial          along-beam translation
1    v_disp         lateral disp (aligned with surge)
2    v_slope        d(v_disp)/d(arclen)
3    w_disp         lateral disp (aligned with sway)
4    w_slope        d(w_disp)/d(arclen)
5    phi            torsional rotation about the beam
==== ============== ==================================

The conversion between platform 6-DOF (this module) and tower-base
FEM 6-DOF is encoded in :func:`pybmodes.fem.nondim.nondim_platform`'s
``T`` matrix. The mapping is deliberately *non-bijective* in
direction labels — for instance, ``v_disp`` corresponds to the
platform surge DOF (index 0) rather than the lexical "v = y" you
might expect from the FEM literature. The OC3 cert validates the
end-to-end pipeline, so this mismatch is naming-only.

Module surface
==============

Two constants are exported for downstream code that wants to refer
to the convention by name:

- :data:`DOF_NAMES` — six-element tuple of DOF labels in canonical
  index order. ``DOF_NAMES[0] == "surge"``.
- :data:`DOF_INDEX` — inverse mapping, ``DOF_INDEX["surge"] == 0``.

These are convenience aliases, not enums — they don't enforce
anything. The point is to give a single source of truth that
``pybmodes.mooring`` / ``pybmodes.io.wamit_reader`` / the
``PlatformSupport`` consumer can all reference instead of carrying
"# index 0 is surge" comments scattered across the codebase.
"""

from __future__ import annotations

DOF_NAMES: tuple[str, str, str, str, str, str] = (
    "surge", "sway", "heave", "roll", "pitch", "yaw",
)

DOF_INDEX: dict[str, int] = {name: i for i, name in enumerate(DOF_NAMES)}

__all__ = ["DOF_INDEX", "DOF_NAMES"]
