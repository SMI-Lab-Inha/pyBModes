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

"""15-DOF Bernoulli–Euler beam element mass and stiffness matrices.

Free-vibration formulation about the undeformed frame; straight elements
only (no sweep, droop, or elastic coupling terms ``eb1/eb2/ec1/ec2``).

Local DOF layout (0-based):

* 0–3   axial *u*       — cubic Lagrange  ``hu``
* 4–7   *v* (lag/edge)  — Hermite cubic   ``h`` (uses ``eiz``, edge stiffness)
* 8–11  *w* (flap)      — Hermite cubic   ``h`` (uses ``eiy``, flap stiffness)
* 12–14 torsion *phi*   — quadratic       ``hf``

The two public entry points are:

* :func:`element_matrices` — single-element scalar interface used by tests
  and downstream consumers.
* :func:`_element_matrices_batch` — vectorised core that operates on all
  ``nselt`` elements at once via :func:`numpy.einsum`.  The single-element
  function is implemented as a thin shim around the batch core.

Note on axis labelling: the ``v`` DOFs (4–7) couple to ``eiz`` (edge / lag
stiffness) and the ``w`` DOFs (8–11) couple to ``eiy`` (flap stiffness).
This is correct for the coordinate system adopted here, even though the
names look transposed.
"""

from __future__ import annotations

import numpy as np

from .gauss import gauss_6pt

# 6-point Gauss quadrature on [0, 1] (cached at module load).
_GQP, _GQW = gauss_6pt()
_NGAUSS = 6


# ---------------------------------------------------------------------------
# Shape-function constants — depend only on Gauss-point coordinates, not on
# element length, so we evaluate them once at import time.
# ---------------------------------------------------------------------------

_CI  = _GQP                                   # (NG,)
_CIS = _CI ** 2
_CIC = _CI ** 3
_ZERO = np.zeros_like(_CI)

# Cubic Lagrange shape functions for the axial DOF (4 nodes at 0, 1/3, 2/3, 1).
_HU = np.stack([
    -4.5  * _CIC +  9.0 * _CIS - 5.5 * _CI + 1.0,
     13.5 * _CIC - 22.5 * _CIS + 9.0 * _CI,
    -13.5 * _CIC + 18.0 * _CIS - 4.5 * _CI,
     4.5  * _CIC -  4.5 * _CIS + _CI,
], axis=-1)                                   # (NG, 4)

# Derivative of the axial Lagrange shape functions (multiplied later by 1/eli).
_HUP_FACTOR = np.stack([
    -13.5 * _CIS + 18.0 * _CI - 5.5,
     40.5 * _CIS - 45.0 * _CI + 9.0,
    -40.5 * _CIS + 36.0 * _CI - 4.5,
     13.5 * _CIS -  9.0 * _CI + 1.0,
], axis=-1)                                   # (NG, 4)

# Hermite cubic shape functions for transverse displacement.
# Components 0 and 2 are eli-independent; 1 and 3 carry an extra factor of eli.
_H_CONST = np.stack([
     2.0 * _CIC - 3.0 * _CIS + 1.0,           # disp at inboard
     _ZERO,
    -2.0 * _CIC + 3.0 * _CIS,                  # disp at outboard
     _ZERO,
], axis=-1)
_H_LIN = np.stack([
     _ZERO,
     _CIC - 2.0 * _CIS + _CI,                  # slope at inboard (× eli)
     _ZERO,
     _CIC - _CIS,                              # slope at outboard (× eli)
], axis=-1)

# First derivative of the Hermite cubic.
_HP_CONST = np.stack([
     _ZERO,
     3.0 * _CIS - 4.0 * _CI + 1.0,
     _ZERO,
     3.0 * _CIS - 2.0 * _CI,
], axis=-1)
_HP_INV = np.stack([                          # multiply by 1/eli
     6.0 * (_CIS - _CI),
     _ZERO,
    -6.0 * (_CIS - _CI),
     _ZERO,
], axis=-1)

# Second derivative of the Hermite cubic.
_HS_INV2 = np.stack([                         # multiply by 1/eli^2
     12.0 * _CI - 6.0,
     _ZERO,
    -12.0 * _CI + 6.0,
     _ZERO,
], axis=-1)
_HS_INV1 = np.stack([                         # multiply by 1/eli
     _ZERO,
     6.0 * _CI - 4.0,
     _ZERO,
     6.0 * _CI - 2.0,
], axis=-1)

# Quadratic shape functions for torsion (3 nodes at 0, 1/2, 1).
_HF = np.stack([
     2.0 * _CIS - 3.0 * _CI + 1.0,
    -4.0 * _CIS + 4.0 * _CI,
     2.0 * _CIS - _CI,
], axis=-1)                                   # (NG, 3)

# Derivative of the torsion shape functions (multiplied later by 1/eli).
_HFP_FACTOR = np.stack([
     4.0 * _CI - 3.0,
    -8.0 * _CI + 4.0,
     4.0 * _CI - 1.0,
], axis=-1)                                   # (NG, 3)


# ---------------------------------------------------------------------------
# Public single-element entry point
# ---------------------------------------------------------------------------

def element_matrices(
    eli: float,
    xbi: float,
    eiy: float,
    eiz: float,
    gj: float,
    eac: float,
    rmas: float,
    skm1: float,
    skm2: float,
    eg: float,
    ea: float,
    axfi: float,
    omega2: float,
    sec_loc: np.ndarray,
    str_tw: np.ndarray,
    distr_k: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 15×15 element stiffness and mass matrices.

    Parameters
    ----------
    eli   : element length (non-dimensional)
    xbi   : inboard (root-side) end position (non-dimensional)
    eiy   : flap bending stiffness :math:`EI_y`  (non-dimensional)
    eiz   : edge bending stiffness :math:`EI_z`  (non-dimensional)
    gj    : torsion stiffness :math:`GJ` (non-dimensional)
    eac   : axial stiffness :math:`EA` (non-dimensional)
    rmas  : mass per unit length (non-dimensional)
    skm1  : flap mass moment of inertia per unit length (= :math:`I_\\text{flap}/m`)
    skm2  : lag  mass moment of inertia per unit length (= :math:`I_\\text{lag}/m`)
    eg    : CG offset × mass (= ``cg_offst × rmas``, non-dimensional)
    ea    : tension-centre offset (``tc_offst``, non-dimensional)
    axfi  : centrifugal tension at outboard end (non-dimensional)
    omega2: :math:`(\\omega/\\omega_\\text{ref})^2` non-dim rotational speed squared
    sec_loc: spanwise station locations (non-dimensional, for twist interpolation)
    str_tw : structural twist at each station (radians)
    distr_k: optional distributed foundation stiffness (non-dimensional)

    Returns
    -------
    ek, em : (15, 15) element stiffness and mass matrices.
    """
    ek_b, em_b = _element_matrices_batch(
        eli  = np.atleast_1d(np.asarray(eli,  dtype=float)),
        xbi  = np.atleast_1d(np.asarray(xbi,  dtype=float)),
        eiy  = np.atleast_1d(np.asarray(eiy,  dtype=float)),
        eiz  = np.atleast_1d(np.asarray(eiz,  dtype=float)),
        gj   = np.atleast_1d(np.asarray(gj,   dtype=float)),
        eac  = np.atleast_1d(np.asarray(eac,  dtype=float)),
        rmas = np.atleast_1d(np.asarray(rmas, dtype=float)),
        skm1 = np.atleast_1d(np.asarray(skm1, dtype=float)),
        skm2 = np.atleast_1d(np.asarray(skm2, dtype=float)),
        eg   = np.atleast_1d(np.asarray(eg,   dtype=float)),
        ea   = np.atleast_1d(np.asarray(ea,   dtype=float)),
        axfi = np.atleast_1d(np.asarray(axfi, dtype=float)),
        omega2  = float(omega2),
        sec_loc = np.asarray(sec_loc, dtype=float),
        str_tw  = np.asarray(str_tw,  dtype=float),
        distr_k = np.atleast_1d(np.asarray(distr_k, dtype=float)),
    )
    return ek_b[0], em_b[0]


# ---------------------------------------------------------------------------
# Vectorised batched core
# ---------------------------------------------------------------------------

def _element_matrices_batch(
    eli:     np.ndarray,
    xbi:     np.ndarray,
    eiy:     np.ndarray,
    eiz:     np.ndarray,
    gj:      np.ndarray,
    eac:     np.ndarray,
    rmas:    np.ndarray,
    skm1:    np.ndarray,
    skm2:    np.ndarray,
    eg:      np.ndarray,
    ea:      np.ndarray,
    axfi:    np.ndarray,
    omega2:  float,
    sec_loc: np.ndarray,
    str_tw:  np.ndarray,
    distr_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Element matrices vectorised over Gauss points *and* elements.

    All per-element inputs (``eli``, ``xbi``, …, ``axfi``, ``distr_k``) are
    1-D arrays of shape ``(nselt,)`` in tip-to-root order.  Returns ``ek``
    and ``em`` of shape ``(nselt, 15, 15)``.

    The inner loops over Gauss points and local DOF pairs become
    :func:`numpy.einsum` tensor contractions; the global element loop is
    eliminated by carrying a leading element axis on every per-element
    quantity.
    """
    nselt = eli.shape[0]

    # Broadcast distr_k to (nselt,) if a scalar slipped in.
    if distr_k.shape != (nselt,):
        distr_k = np.broadcast_to(distr_k, (nselt,))

    inv_eli  = 1.0 / eli                            # (nselt,)
    inv_eli2 = inv_eli ** 2

    # --- Gauss-point absolute positions: x[e, n] = xbi[e] + ci[n] * eli[e] -
    x = xbi[:, None] + _CI[None, :] * eli[:, None]   # (nselt, NG)

    # --- Quadrature weights ------------------------------------------------
    gqwl  = _GQW[None, :] * eli[:, None]             # (nselt, NG)
    gqwml = gqwl * rmas[:, None]                      # (nselt, NG)

    # --- Shape functions ---------------------------------------------------
    # (NG, ·) — independent of eli (constants):
    hu = _HU
    hf = _HF

    # (nselt, NG, ·) — eli-dependent:
    hup = _HUP_FACTOR[None, :, :] * inv_eli[:, None, None]
    hfp = _HFP_FACTOR[None, :, :] * inv_eli[:, None, None]
    h   = _H_CONST[None, :, :] + _H_LIN[None, :, :] * eli[:, None, None]
    hp  = _HP_CONST[None, :, :] + _HP_INV[None, :, :] * inv_eli[:, None, None]
    hs  = (_HS_INV2[None, :, :] * inv_eli2[:, None, None]
           + _HS_INV1[None, :, :] * inv_eli[:, None, None])

    # --- Twist at each Gauss point per element -----------------------------
    twx = np.interp(x.ravel(), sec_loc, str_tw).reshape(nselt, _NGAUSS)
    ct  = np.cos(twx)
    st  = np.sin(twx)
    cst = ct * st
    cts = ct ** 2
    sts = st ** 2
    c2t = cts - sts

    # --- Centrifugal tension at each Gauss point ---------------------------
    xbils = (xbi + eli) ** 2
    fi = (axfi[:, None] + 0.5 * rmas[:, None]
          * (xbils[:, None] - x * x)) * omega2          # (nselt, NG)

    # --- Bending coupling --------------------------------------------------
    eiy_b   = eiy[:, None]
    eiz_b   = eiz[:, None]
    dei_b   = (eiz - eiy)[:, None]
    ceicss  = eiz_b * cts + eiy_b * sts                   # flap-direction
    ceiscs  = eiz_b * sts + eiy_b * cts                   # lag-direction
    deicst  = dei_b * cst                                  # cross-bending

    # CG-offset coupling (flap/lag — torsion mass and stiffness).
    eg_b    = eg[:, None]
    egct    = eg_b * ct
    egst    = eg_b * st
    egxct   = egct * x
    egxst   = egst * x

    # Tension-centre offset coupling (axial — flap/lag stiffness).
    eacea_b = (eac * ea)[:, None]
    eaeact  = eacea_b * ct
    eaeast  = eacea_b * st

    # Torsion mass / centrifugal-softening coupling.
    dsk_b   = (skm2 - skm1)[:, None]
    skm_b   = (skm1 + skm2)[:, None]
    dskc2t  = dsk_b * c2t

    # --- Allocate -----------------------------------------------------------
    ek = np.zeros((nselt, 15, 15))
    em = np.zeros((nselt, 15, 15))

    # ``H_outer`` = ∫ h_i h_j w_l  (used by both flap and lag bending blocks
    # for distributed-foundation stiffness, and again — weighted by gqwml —
    # for the centrifugal mass softening on flap and the transverse mass
    # entries on flap and lag).
    H_outer = np.einsum('en,eni,enj->eij', gqwl,  h, h)
    HH_mass = np.einsum('en,eni,enj->eij', gqwml, h, h)

    # ----- Stiffness blocks ------------------------------------------------

    # axial-axial: ek[0:4, 0:4] += eac * ∫ hup_i hup_j w_l
    ek[:, 0:4, 0:4] = (eac[:, None, None]
                       * np.einsum('en,eni,enj->eij', gqwl, hup, hup))

    # axial — flap (v) coupling:  -∫ eaeact · hup_i hs_j  w_l   (and symm.)
    aef = -np.einsum('en,eni,enj->eij', gqwl * eaeact, hup, hs)
    ek[:, 0:4, 4:8] = aef
    ek[:, 4:8, 0:4] = np.swapaxes(aef, 1, 2)

    # axial — lag (w) coupling:  -∫ eaeast · hup_i hs_j  w_l   (and symm.)
    aew = -np.einsum('en,eni,enj->eij', gqwl * eaeast, hup, hs)
    ek[:, 0:4, 8:12] = aew
    ek[:, 8:12, 0:4] = np.swapaxes(aew, 1, 2)

    # flap (v) bending: centrifugal + bending + foundation − centrifugal softening.
    ek[:, 4:8, 4:8] = (
        np.einsum('en,eni,enj->eij', gqwl * fi,     hp, hp)
        + np.einsum('en,eni,enj->eij', gqwl * ceicss, hs, hs)
        + distr_k[:, None, None] * H_outer
        - omega2 * HH_mass
    )

    # flap–lag (v–w) cross-bending coupling.
    block_fl = np.einsum('en,eni,enj->eij', gqwl * deicst, hs, hs)
    ek[:, 4:8, 8:12] = block_fl
    ek[:, 8:12, 4:8] = np.swapaxes(block_fl, 1, 2)

    # lag (w) bending: centrifugal + bending + foundation (no softening).
    ek[:, 8:12, 8:12] = (
        np.einsum('en,eni,enj->eij', gqwl * fi,     hp, hp)
        + np.einsum('en,eni,enj->eij', gqwl * ceiscs, hs, hs)
        + distr_k[:, None, None] * H_outer
    )

    # flap (v) — torsion stiffness.
    block_ft = omega2 * (
        np.einsum('en,eni,nj->eij', gqwml * egst,  h,  hf)
        - np.einsum('en,eni,nj->eij', gqwml * egxst, hp, hf)
    )
    ek[:, 4:8, 12:15] = block_ft
    ek[:, 12:15, 4:8] = np.swapaxes(block_ft, 1, 2)

    # lag (w) — torsion stiffness.
    block_lt = omega2 * np.einsum('en,eni,nj->eij', gqwml * egxct, hp, hf)
    ek[:, 8:12, 12:15] = block_lt
    ek[:, 12:15, 8:12] = np.swapaxes(block_lt, 1, 2)

    # torsion–torsion: centrifugal softening + GJ bending.
    ek[:, 12:15, 12:15] = (
        omega2 * np.einsum('en,ni,nj->eij', gqwml * dskc2t, hf, hf)
        + gj[:, None, None] * np.einsum('en,eni,enj->eij', gqwl, hfp, hfp)
    )

    # ----- Mass blocks ------------------------------------------------------

    # axial mass.
    em[:, 0:4, 0:4] = np.einsum('en,ni,nj->eij', gqwml, hu, hu)

    # transverse (flap and lag) mass — same integral.
    em[:, 4:8, 4:8] = HH_mass
    em[:, 8:12, 8:12] = HH_mass

    # flap–torsion mass coupling.
    em_ft = -np.einsum('en,eni,nj->eij', gqwml * egst, h, hf)
    em[:, 4:8, 12:15] = em_ft
    em[:, 12:15, 4:8] = np.swapaxes(em_ft, 1, 2)

    # lag–torsion mass coupling.
    em_lt = np.einsum('en,eni,nj->eij', gqwml * egct, h, hf)
    em[:, 8:12, 12:15] = em_lt
    em[:, 12:15, 8:12] = np.swapaxes(em_lt, 1, 2)

    # torsion–torsion mass.
    em[:, 12:15, 12:15] = np.einsum('en,ni,nj->eij', gqwml * skm_b, hf, hf)

    return ek, em
