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

"""Connectivity vector construction and boundary-condition application.

Global DOF layout (0-based):

  Each element i (1=tip ... nselt=root) owns a 9-DOF "block" at global positions
  9*(i-1)+1 ... 9*(i-1)+9, plus a shared 6-DOF outboard-end node that overlaps
  with block i-1 (or the tip node for i=1).

  Within block for outboard-end node of element i (global DOFs 9*(i-1)+1..6):
    9*(i-1)+1 : axial  u
    9*(i-1)+2 : v (lag/edge) displacement
    9*(i-1)+3 : v (lag/edge) slope
    9*(i-1)+4 : w (flap) displacement
    9*(i-1)+5 : w (flap) slope
    9*(i-1)+6 : phi (twist)

  Internal DOFs for element i (global DOFs 9*(i-1)+7..9):
    9*(i-1)+7 : axial at 2/3 of element
    9*(i-1)+8 : twist at midpoint
    9*(i-1)+9 : axial at 1/3 of element

  Root node DOFs (shared inboard end of element nselt):
    9*nselt+1 .. 9*nselt+6  (= ndt-5 .. ndt)
    These are zeroed out for cantilevered BC.

  ndt  = 9*nselt + 6
  ngd  = ndt - ncon  where ncon=6 (cantilever) -> ngd = 9*nselt
"""

from __future__ import annotations

import numpy as np

# Hardwired 15-DOF element connectivity vector (1-based global offsets).
# Maps local DOF j (0-based) -> offset to add to (i-1)*9+1 to get global DOF.
_IVECBE = np.array([10, 9, 7, 1, 11, 12, 2, 3, 13, 14, 4, 5, 15, 8, 6], dtype=int)
NEDOF = 15
NNDOF = 6
NINTN = 3
NESH = 9


# ---------------------------------------------------------------------------
# Shared hub_conn dispatch
# ---------------------------------------------------------------------------
#
# Each ``hub_conn`` BC is defined once, here, by its set of constrained
# local-DOF indices on the root element (using the element's 15-DOF
# layout: 0=axial@ib, 4=v@ib, 5=v'@ib, 8=w@ib, 9=w'@ib, 12=phi@ib).
#
# Every consumer below (``build_connectivity`` / ``n_free_dof`` /
# ``active_dof_indices``) routes through ``_validate_hub_conn`` and
# ``_root_local_constrained`` so an unsupported ``hub_conn`` raises in
# all three rather than silently degenerating to free-free in some of
# them. The previous arrangement had ``n_free_dof`` and
# ``active_dof_indices`` silently treat unknown values as free-free,
# which made typos in BMI ``hub_conn`` fields slip through whenever
# the model bypassed ``build_connectivity``.

_HUB_CONN_ROOT_LOCAL: dict[int, tuple[int, ...]] = {
    1: (0, 4, 5, 8, 9, 12),  # cantilever — every base DOF locked
    2: (),                   # free-free floating — every base DOF released;
                             # reactions supplied externally by PlatformSupport
    3: (0, 12),              # soft monopile — axial + torsion locked, lateral
                             # + rocking free
    4: (0, 4, 8, 12),        # pinned-free (Bir 2009 cable BC) — translations
                             # + twist locked, bending slopes FREE
}

# Map from local element-DOF index (within the 15-DOF element layout) to
# the offset of the same physical DOF on the root *node* (within the
# 6-DOF root block). The four "@ib" local DOFs map to the bottom of the
# root block; the two "@ib slope" local DOFs map to slope positions
# inside it.
_LOCAL_TO_ROOT_NODE_OFFSET: dict[int, int] = {
    0: 0,   # axial    @ root node
    4: 1,   # v_disp   @ root node
    5: 2,   # v_slope  @ root node
    8: 3,   # w_disp   @ root node
    9: 4,   # w_slope  @ root node
    12: 5,  # phi      @ root node
}


def _validate_hub_conn(hub_conn: int) -> None:
    """Raise on any unsupported ``hub_conn``.

    Shared by ``build_connectivity`` / ``n_free_dof`` /
    ``active_dof_indices`` so a typo in a BMI ``hub_conn`` field can
    never silently degenerate to free-free.
    """
    if hub_conn not in _HUB_CONN_ROOT_LOCAL:
        raise ValueError(
            f"Unsupported hub_conn = {hub_conn!r}; expected one of "
            f"1 (cantilever), 2 (free-free floating), 3 (soft "
            f"monopile: axial + torsion locked), 4 (pinned-free / "
            f"Bir 2009 cable BC). A typo in the BMI deck silently "
            f"becoming a free-free solve was caught here so this can't "
            f"happen."
        )


def build_connectivity(nselt: int, hub_conn: int = 1) -> np.ndarray:
    """Build element connectivity array indeg[j, i] (1-based global DOF, or 0 if constrained)."""
    _validate_hub_conn(hub_conn)
    indeg = np.zeros((NEDOF, nselt), dtype=int)
    for i in range(nselt):
        nsh = i * NESH
        for j in range(NEDOF):
            indeg[j, i] = _IVECBE[j] + nsh

    last = nselt - 1
    for j in _HUB_CONN_ROOT_LOCAL[hub_conn]:
        indeg[j, last] = 0

    return indeg


def n_total_dof(nselt: int) -> int:
    """Total (unreduced) DOFs: ndt = 9*nselt + 6."""
    return NESH * nselt + NNDOF


def n_free_dof(nselt: int, hub_conn: int = 1) -> int:
    """Free (reduced) DOFs after BC application."""
    _validate_hub_conn(hub_conn)
    ndt = NESH * nselt + NNDOF
    return ndt - len(_HUB_CONN_ROOT_LOCAL[hub_conn])


def active_dof_indices(nselt: int, hub_conn: int = 1) -> np.ndarray:
    """Return sorted 0-based indices of unconstrained (active) global DOFs.

    Root-node DOF map (offset from ``root_base = NESH * nselt``):
      +0 axial · +1 v_disp · +2 v_slope · +3 w_disp · +4 w_slope · +5 phi

    Constrained sets:
      hub_conn=1 (cantilever):    all six       (+0 +1 +2 +3 +4 +5)
      hub_conn=2 (free-free):     none
      hub_conn=3 (axial+torsion): two           (+0 +5)
      hub_conn=4 (pinned-free):   four          (+0 +1 +3 +5) — slopes FREE
    """
    _validate_hub_conn(hub_conn)
    ndt = NESH * nselt + NNDOF
    root_base = NESH * nselt
    constrained = {
        root_base + _LOCAL_TO_ROOT_NODE_OFFSET[j]
        for j in _HUB_CONN_ROOT_LOCAL[hub_conn]
    }
    return np.array([i for i in range(ndt) if i not in constrained], dtype=int)
