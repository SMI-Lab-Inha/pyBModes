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

"""Global mass and stiffness matrix assembly."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .boundary import NESH, active_dof_indices, build_connectivity, n_total_dof
from .element import _element_matrices_batch
from .nondim import PlatformND, TipMassND


def assemble(
    nselt: int,
    el: np.ndarray,
    xb: np.ndarray,
    cfe: np.ndarray,
    eiy: np.ndarray,
    eiz: np.ndarray,
    gj: np.ndarray,
    eac: np.ndarray,
    rmas: np.ndarray,
    skm1: np.ndarray,
    skm2: np.ndarray,
    eg: np.ndarray,
    ea: np.ndarray,
    omega2: float,
    sec_loc: np.ndarray,
    str_tw: np.ndarray,
    tip_mass: Optional[TipMassND] = None,
    wire_k_nd: Optional[np.ndarray] = None,
    wire_node_attach: Optional[list] = None,
    hub_conn: int = 1,
    platform_nd: Optional[PlatformND] = None,
    elm_distr_k: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble global stiffness gk and mass gm matrices.

    All inputs are non-dimensional (already processed by models layer).

    Parameters
    ----------
    nselt      : number of elements
    el         : element lengths (nselt,), tip-to-root ordering
    xb         : inboard end positions (nselt,)
    cfe        : centrifugal tension at outboard end (nselt,)
    eiy        : flap bending stiffness per element (nselt,)
    eiz        : edge bending stiffness per element (nselt,)
    gj         : torsion stiffness per element (nselt,)
    eac        : axial stiffness per element (nselt,)
    rmas       : mass/length per element (nselt,)
    skm1       : flap mass moment of inertia/length (nselt,)
    skm2       : lag mass moment of inertia/length (nselt,)
    eg         : CG offset per element (= cg_offst, non-dim) (nselt,)
    ea         : tension-centre offset per element (= tc_offst, non-dim) (nselt,)
    omega2     : (omega/romg)²  non-dimensional rotational speed squared
    sec_loc    : station span locations for twist interpolation (non-dim)
    str_tw     : structural twist at each station (radians)
    hub_conn   : root BC (1=cantilever, 2=free-free, 3=axial+torsion only)
    platform_nd: optional non-dim platform 6×6 matrices added at root DOFs

    Returns
    -------
    gk    : (ngd, ngd) global stiffness matrix  (ngd = n_free_dof(nselt, hub_conn))
    gm    : (ngd, ngd) global mass matrix
    indeg : (NEDOF, nselt) connectivity array (used by post-processor)
    """
    ndt   = n_total_dof(nselt)
    indeg = build_connectivity(nselt, hub_conn)
    gk    = np.zeros((ndt, ndt))
    gm    = np.zeros((ndt, ndt))

    # Compute every element's stiffness and mass matrices in one batched call.
    distr_k = np.asarray(elm_distr_k, dtype=float) if elm_distr_k is not None else np.zeros(nselt)
    ek_batch, em_batch = _element_matrices_batch(
        eli=el, xbi=xb, eiy=eiy, eiz=eiz, gj=gj, eac=eac, rmas=rmas,
        skm1=skm1, skm2=skm2, eg=eg, ea=ea, axfi=cfe,
        omega2=omega2, sec_loc=sec_loc, str_tw=str_tw, distr_k=distr_k,
    )

    # Scatter each element block into the global matrices.  Per-element
    # connectivity has 15 distinct global DOFs so a fancy-index block-add
    # is safe; the cost of a Python-level element loop here is ~20 numpy
    # ops per element which is far below the eigh solve.
    for i in range(nselt):
        _scatter(gm, gk, em_batch[i], ek_batch[i], indeg[:, i])

    if tip_mass is not None:
        _add_tip_mass(gm, tip_mass)

    if wire_k_nd is not None and wire_node_attach is not None:
        _add_wire_stiffness(gk, nselt, wire_k_nd, wire_node_attach)

    if platform_nd is not None:
        _add_platform_support(gk, gm, nselt, platform_nd)

    # Compact to active DOFs only (removes constrained rows/cols)
    active = active_dof_indices(nselt, hub_conn)
    gk_c = gk[np.ix_(active, active)]
    gm_c = gm[np.ix_(active, active)]

    return gk_c, gm_c, indeg


def _scatter(
    gm: np.ndarray,
    gk: np.ndarray,
    em: np.ndarray,
    ek: np.ndarray,
    ind: np.ndarray,
) -> None:
    """Add element matrices em, ek into global gm, gk using connectivity ind.

    ind is 1-based (0 marks a constrained DOF that must be skipped).  Within
    a single element each non-zero entry of ind is unique, so a fancy-index
    block-add is safe and equivalent to the original Fortran scatter.
    """
    valid = np.flatnonzero(ind > 0)
    if valid.size == 0:
        return
    g_idx = ind[valid] - 1                 # 0-based global DOFs
    rows, cols = np.ix_(g_idx, g_idx)
    sub = np.ix_(valid, valid)
    gm[rows, cols] += em[sub]
    gk[rows, cols] += ek[sub]


def _add_tip_mass(gm: np.ndarray, tm: TipMassND) -> None:
    """Add concentrated tip/tower-top mass to the global mass matrix.

    Global DOFs 0-5 are the tip-node DOFs: [axial, v_disp, v_slope, w_disp, w_slope, phi].
    """
    tcm_loc    = tm.mass * tm.cm_loc
    tcm_axial  = tm.mass * tm.cm_axial
    tcd        = tm.mass * tm.cm_loc * tm.cm_axial
    tcm_loc2   = tm.mass * tm.cm_loc   ** 2
    tcm_axial2 = tm.mass * tm.cm_axial ** 2

    # Diagonal mass terms
    gm[0, 0] += tm.mass       # axial
    gm[1, 1] += tm.mass       # v_disp
    gm[3, 3] += tm.mass       # w_disp

    # Axial – v_slope coupling
    gm[0, 2] += -tcm_loc
    gm[2, 0]  = gm[0, 2]

    # v_disp – v_slope coupling
    gm[1, 2] += tcm_axial
    gm[2, 1]  = gm[1, 2]

    # v_slope – v_slope
    gm[2, 2] += tcm_loc2 + tcm_axial2 + tm.izz

    # v_slope – w_slope coupling
    gm[2, 4] += -tm.iyz
    gm[4, 2]  = gm[2, 4]

    # v_slope – phi coupling
    gm[2, 5] += tm.izx
    gm[5, 2]  = gm[2, 5]

    # w_disp – w_slope
    gm[3, 4] += tcm_axial
    gm[4, 3]  = gm[3, 4]

    # w_disp – phi
    gm[3, 5] += tcm_loc
    gm[5, 3]  = gm[3, 5]

    # w_slope – w_slope
    gm[4, 4] += tm.iyy + tcm_axial2

    # w_slope – phi
    gm[4, 5] += -tm.ixy + tcd
    gm[5, 4]  = gm[4, 5]

    # phi – phi
    gm[5, 5] += tm.ixx + tcm_loc2


def _add_wire_stiffness(
    gk: np.ndarray,
    nselt: int,
    wire_k_nd: np.ndarray,
    wire_node_attach: list,
) -> None:
    """Add tension-wire lateral stiffness to the global stiffness matrix.

    Each wire set adds k_nd to both the v (lag/SS) and w (flap/FA) displacement
    DOFs of its attachment node.

    Parameters
    ----------
    gk              : global stiffness matrix (modified in place)
    nselt           : number of elements
    wire_k_nd       : non-dimensional stiffness per attachment set
    wire_node_attach: root-first 1-based node numbers (from .bmi file)

    Node mapping (elements ordered tip-to-root, nodes 1-based from root):
      glob_node (0-based tip-first) = nselt + 1 - node_attach
      v_dof = 9 * glob_node + 1  (lag/SS displacement)
      w_dof = 9 * glob_node + 3  (flap/FA displacement)
    """
    for node_attach, k_nd in zip(wire_node_attach, wire_k_nd):
        glob_node_py = nselt + 1 - node_attach   # 0-based, tip-first
        v_dof = 9 * glob_node_py + 1
        w_dof = 9 * glob_node_py + 3
        gk[v_dof, v_dof] += k_nd
        gk[w_dof, w_dof] += k_nd


def _add_platform_support(
    gk: np.ndarray,
    gm: np.ndarray,
    nselt: int,
    platform_nd: PlatformND,
) -> None:
    """Add non-dim platform 6×6 stiffness and mass matrices at the root node.

    ``platform_nd`` is already in FEM DOF ordering (produced by
    ``nondim_platform``): ``[axial(0), v_disp(1), v_slope(2), w_disp(3),
    w_slope(4), phi(5)]``.  These map directly to root-node global DOF
    offsets 0..5.
    """
    root = slice(NESH * nselt, NESH * nselt + 6)
    gk[root, root] += platform_nd.stiffness
    gm[root, root] += platform_nd.mass


def compute_element_props(
    nselt: int,
    el_loc: np.ndarray,
    sp: Any,
    hub_r: float = 0.0,
    bl_frac: float | None = None,
) -> tuple[np.ndarray, ...]:
    """Interpolate section properties at element midpoints and compute geometry.

    Parameters
    ----------
    nselt   : number of elements
    el_loc  : element boundary locations (normalised to beam length [0..1]),
              root-to-tip ordering as stored in BMIFile.el_loc
    sp      : SectionProperties object with .span_loc in radius-normalised coords
    hub_r   : hub_rad / radius  (0 for towers)
    bl_frac : total_beam / radius — scaling factor from el_loc [0..1] to
              radius-normalised coordinates.  Defaults to (1 - hub_r) for the
              onshore/blade case, but must be set to (radius + draft) / radius
              for offshore towers.

    Returns
    -------
    el, xb, cfe : element lengths, inboard positions, centrifugal tensions
    eiy, eiz, gj, eac, rmas, skm1, skm2, eg, ea : per-element section props (non-dim)

    Note: All element arrays follow TIP-TO-ROOT ordering (element 0 = tip element).
          All positions (xb, xmid) are in radius-normalised units.
    """
    if bl_frac is None:
        bl_frac = 1.0 - hub_r

    # Convert el_loc from beam-length [0..1] to radius-normalised coords
    el_loc_r = el_loc * bl_frac + hub_r

    # Reverse to tip-to-root order
    el_loc_f = el_loc_r[::-1]   # el_loc_f[0]=1 (tip), el_loc_f[-1]=hub_r (root)

    el  = np.abs(np.diff(el_loc_f))   # element lengths in radius-normalised units
    xb  = el_loc_f[1:]                # inboard end of each element (radius-normalised)
    xmid = xb + 0.5 * el             # midpoint (radius-normalised)

    # Interpolate section properties at element midpoints
    eiy_  = np.interp(xmid, sp.span_loc, sp.flp_stff)
    eiz_  = np.interp(xmid, sp.span_loc, sp.edge_stff)
    gj_   = np.interp(xmid, sp.span_loc, sp.tor_stff)
    eac_  = np.interp(xmid, sp.span_loc, sp.axial_stff)
    rmas_ = np.interp(xmid, sp.span_loc, sp.mass_den)
    eg_   = np.interp(xmid, sp.span_loc, sp.cg_offst)
    ea_   = np.interp(xmid, sp.span_loc, sp.tc_offst)

    flp_iner  = np.interp(xmid, sp.span_loc, sp.flp_iner)
    edge_iner = np.interp(xmid, sp.span_loc, sp.edge_iner)
    mass_safe = np.where(rmas_ > 0.0, rmas_, 1.0)
    skm1_ = flp_iner  / mass_safe
    skm2_ = edge_iner / mass_safe

    # Centrifugal tension accumulated tip-to-root.
    # Per-element contribution is 0.5*m*(x_outboard^2 - x_inboard^2); cfe[i] is
    # the tension at the *outboard* face of element i (i.e. the sum of all
    # tip-side contributions strictly above element i, omitting the omega^2
    # factor which is applied later in element_matrices).
    contrib = 0.5 * rmas_ * ((xb + el) ** 2 - xb ** 2)
    cfe = np.empty(nselt)
    cfe[0] = 0.0
    cfe[1:] = np.cumsum(contrib[:-1])

    return el, xb, cfe, eiy_, eiz_, gj_, eac_, rmas_, skm1_, skm2_, eg_, ea_
