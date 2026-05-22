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

"""Eigenvector post-processing: extract nodal mode shapes from global DOF vector.

The global DOF layout (0-based Python indices) for a system with nselt elements:

  For each element i (0=tip … nselt-1=root), the outboard-end node occupies
  global DOFs 9*i+0 … 9*i+5:
    9*i+0 : axial u
    9*i+1 : v (lag/edge) displacement
    9*i+2 : v (lag/edge) slope
    9*i+3 : w (flap) displacement
    9*i+4 : w (flap) slope
    9*i+5 : phi (twist)

  Internal DOFs for element i: 9*i+6, 9*i+7, 9*i+8 (axial 2/3, phi mid, axial 1/3)

  Root node (constrained, zero): global DOFs 9*nselt+0 … 9*nselt+5.

Output convention for each mode (matching the .out file column order):
  span_loc | w(flap)_disp | w(flap)_slope | v(lag)_disp | v(lag)_slope | twist
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NodeModeShape:
    """Nodal mode shape data for one mode."""
    mode_number: int
    freq_hz: float
    span_loc: np.ndarray    # normalised span, 0=root … 1=tip (root-to-tip order)
    flap_disp: np.ndarray   # w(flap) displacement
    flap_slope: np.ndarray  # w(flap) slope
    lag_disp: np.ndarray    # v(lag/edge) displacement
    lag_slope: np.ndarray   # v(lag/edge) slope
    twist: np.ndarray       # phi (torsion)


def extract_mode_shapes(
    eigvecs: np.ndarray,
    eigvals_hz: np.ndarray,
    nselt: int,
    el: np.ndarray,
    xb: np.ndarray,
    radius: float,
    hub_rad: float,
    bl_len: float,
    hub_conn: int = 1,
    active_dofs: np.ndarray | None = None,
) -> list[NodeModeShape]:
    """Extract per-node mode shape data from eigenvectors.

    The output includes nselt+1 stations: root + one per element outboard end.

    Parameters
    ----------
    eigvecs    : (ngd, n_modes) compact eigenvectors (L2-normalised)
    eigvals_hz : (n_modes,) frequencies in Hz
    nselt      : number of elements
    el         : element lengths (non-dim, tip-to-root)
    xb         : inboard end positions (non-dim, tip-to-root)
    radius     : total radius/height (m)
    hub_rad    : hub radius or tower base height (m)
    bl_len     : flexible length = radius - hub_rad (m)
    hub_conn   : root BC (1=cantilever, 2=free-free, 3=axial+torsion only)
    active_dofs: (ngd,) sorted 0-based indices of active DOFs used to expand
                 compact eigvecs to full ndt size.  If None, assumes hub_conn=1.

    Returns
    -------
    List of NodeModeShape, one per mode.
    """
    ndt     = 9 * nselt + 6
    n_modes = eigvecs.shape[1]

    # Output span locations: root + outboard ends of elements in root-to-tip order.
    # In tip-to-root element ordering, station k (0..nselt) corresponds to the
    # inboard end of element nselt-1-k for k>0, equivalently the outboard end of
    # element nselt-k.  ``xb[ie] + el[ie]`` is the outboard end of element ie.
    x_nodes_nd = np.empty(nselt + 1)
    x_nodes_nd[0] = xb[nselt - 1]                      # root = inboard end of root element
    x_nodes_nd[1:] = (xb + el)[::-1]                    # outboard ends, root-to-tip
    span_loc = (x_nodes_nd * radius - hub_rad) / bl_len

    # Expand compact eigenvectors to full (ndt, n_modes) once, scattering active DOFs.
    ev_full = np.zeros((ndt, n_modes))
    if active_dofs is not None:
        ev_full[active_dofs, :] = eigvecs
    else:
        # hub_conn=1 default: compact eigvec covers the first 9*nselt DOFs
        ev_full[: eigvecs.shape[0], :] = eigvecs

    # Per-station global DOF base index, root-to-tip:
    #   station 0 (root) -> 9*nselt          (root node block)
    #   station k>0      -> 9*(nselt-k)      (outboard end of element nselt-k)
    station_base = 9 * np.arange(nselt, -1, -1)        # shape (nselt+1,)

    # Local-DOF offsets for each component (see docstring).
    AXIAL_U_OFFSET = 0   # noqa: F841 (kept for documentation)
    V_DISP, V_SLOPE = 1, 2     # lag (edge / side-side)
    W_DISP, W_SLOPE = 3, 4     # flap (fore-aft)
    PHI            = 5

    flap_disp_all  = ev_full[station_base + W_DISP,  :]   # (nstations, n_modes)
    flap_slope_all = ev_full[station_base + W_SLOPE, :]
    lag_disp_all   = ev_full[station_base + V_DISP,  :]
    lag_slope_all  = ev_full[station_base + V_SLOPE, :]
    twist_all      = ev_full[station_base + PHI,     :]

    return [
        NodeModeShape(
            mode_number=m + 1,
            freq_hz=eigvals_hz[m],
            span_loc=span_loc.copy(),
            flap_disp=flap_disp_all[:, m].copy(),
            flap_slope=flap_slope_all[:, m].copy(),
            lag_disp=lag_disp_all[:, m].copy(),
            lag_slope=lag_slope_all[:, m].copy(),
            twist=twist_all[:, m].copy(),
        )
        for m in range(n_modes)
    ]
