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

"""Decouple a 6×6 cross-sectional stiffness / inertia matrix into the
*decoupled Euler–Bernoulli* properties an ElastoDyn / BModes beam
model consumes — referenced at the **elastic (tension) centre** and
the **principal elastic axes**, not the raw reference-axis diagonal.

Why this exists (issue #50)
---------------------------
A WindIO ``elastic_properties`` / ``elastic_properties_mb`` (and the
companion BeamDyn) 6×6 sectional stiffness matrix is expressed about
the blade **reference axis** (typically the pitch axis), in the
section frame — *not* about the elastic centre and *not* aligned with
the principal elastic axes. For a real, offset / pre-twisted blade
the raw diagonal terms ``K44``/``K55`` therefore are **not** the
physical ``EI_flap``/``EI_edge``: they carry the axial–bending
coupling (``K34``/``K35``) and the bending–bending coupling
(``K45``) that a tension-centre translation + principal-axis rotation
remove. Reading the diagonal directly (the pre-1.5.1 behaviour) is
exact only if the reference point happens to coincide with the
elastic centre *and* the reference axes happen to be principal — it
biases the result for every realistic blade.

Method (standard, citable)
--------------------------
A sectional stiffness matrix transforms under a rigid in-plane offset
of the reference point by ``(d1, d2)`` (span axis = 3) as a congruence
``K' = Tᵀ K T``, where ``T`` maps generalized strains at the new
reference to the old (Bauchau, *Flexible Multibody Dynamics*, §16;
the same construction the DNV Bladed "WindIO to Bladed" pre-processor
documents under *Evaluation of axial and bending stiffnesses, elastic
centre, and principal elastic axes orientation*). With DOF order
``[γ1, γ2, ε3, κ1, κ2, κ3]`` the columns of ``T`` (old strain for a
unit new strain) are::

    γ1' → [ 1, 0, 0, 0, 0, 0]
    γ2' → [ 0, 1, 0, 0, 0, 0]
    ε3' → [ 0, 0, 1, 0, 0, 0]
    κ1' → [ 0, 0, d2,1, 0, 0]      (axial gets +d2·κ1)
    κ2' → [ 0, 0,-d1,0, 1, 0]      (axial gets −d1·κ2)
    κ3' → [-d2,d1, 0, 0, 0, 1]     (shear gets ∓·κ3 — shear centre)

1. **Elastic (tension) centre.** Choosing ``d2 = −K[2,3]/K[2,2]`` and
   ``d1 = K[2,4]/K[2,2]`` makes ``K'[2,3] = K'[2,4] = 0`` (a pure
   axial force produces no curvature). ``EA = K[2,2]`` is invariant.
2. **Principal elastic axes.** The 2×2 bending block of the
   tension-centre matrix, ``B = K'[3:5, 3:5]`` (now the parallel-axis
   ``Σ EI`` about the elastic centre), is symmetric; its
   eigenvalues are the principal ``EI`` and the eigenvector angle is
   the principal-axis orientation.
3. **Shear centre & torsion.** Solving the 2×2 system that nulls
   ``K[0,5]`` / ``K[1,5]`` gives the shear-centre offset; ``GJ`` is
   the torsion diagonal of the matrix translated there.

Sign convention
---------------
The reported ``x_tc/y_tc`` (and ``x_sc/y_sc``, ``x_cg/y_cg``) is the
**translation applied to the input matrix that decouples it** — i.e.
the offset ``(d1, d2)`` for which ``T(d1,d2)ᵀ K T(d1,d2)`` has no
axial↔bending coupling. Equivalently it is the position of the
elastic centre expressed in the input matrix's reference frame.
Consequently, if a decoupled section's matrix is *re-expressed* about
a reference displaced by ``r`` (``K = T(r)ᵀ K₀ T(r)``), the recovered
offset is ``−r`` (you translate back by ``−r`` to decouple again).
The principal values (``EA``, ``EI``, ``GJ``, ``mass``) are
sign-convention-independent.

The inertia 6×6 is decoupled the same way (mass centre + principal
mass moments of inertia) — the kinematic transform for a rigid offset
is structurally identical.

Pure ``numpy``; no third-party / vendored reference code (the project
independence stance) — the transform above is implemented directly
from the cited construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Section-frame DOF indices (WindIO / BeamDyn order
# ``[F1, F2, F3(axial), M1, M2, M3(torsion)]``; span axis = 3).
_AXIAL = 2
_BEND = (3, 4)
_TORSION = 5
_SHEAR = (0, 1)


def _offset_transform(d1: float, d2: float) -> np.ndarray:
    """6×6 congruence ``T`` for a rigid in-plane reference-point offset
    ``(d1, d2)`` (span axis 3). ``K_new = Tᵀ K_old T``."""
    T = np.eye(6)
    # κ1 (col 3) and κ2 (col 4) feed the axial strain (row 2).
    T[_AXIAL, 3] = d2
    T[_AXIAL, 4] = -d1
    # κ3 (col 5) feeds the transverse shears (rows 0, 1).
    T[_SHEAR[0], _TORSION] = -d2
    T[_SHEAR[1], _TORSION] = d1
    return T


@dataclass
class DecoupledStiffness:
    EA: float
    EI_flap: float          # principal bending, flap-aligned axis
    EI_edge: float          # principal bending, edge-aligned axis
    GJ: float
    x_tc: float             # tension-centre offset, section axis 1 (m)
    y_tc: float             # tension-centre offset, section axis 2 (m)
    x_sc: float             # shear-centre offset (m)
    y_sc: float
    principal_angle: float  # rad, reference → principal bending axes


@dataclass
class DecoupledInertia:
    mass: float             # per unit length (kg/m)
    i_flap: float           # principal mass moment, flap axis (kg·m)
    i_edge: float           # principal mass moment, edge axis (kg·m)
    x_cg: float             # mass-centre offset, section axis 1 (m)
    y_cg: float
    principal_angle: float  # rad


def _principal_2x2(B: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Eigen-decompose a symmetric 2×2 → (λ_a, λ_b, angle, eigenvecs),
    where ``angle`` rotates the reference axes onto the principal ones
    and the eigenvectors are returned column-wise, eigenvalue-sorted
    so column 0 ↔ λ_a."""
    Bs = 0.5 * (B + B.T)
    w, V = np.linalg.eigh(Bs)          # ascending, orthonormal
    angle = float(np.arctan2(V[1, 0], V[0, 0]))
    return float(w[0]), float(w[1]), angle, V


def _assign_flap_edge(
    lam0: float, lam1: float, V: np.ndarray
) -> tuple[float, float]:
    """Map the two principal bending/inertia values to (flap, edge).

    ``flap`` is the principal axis closest to section axis 1 (the
    flap-normal / out-of-plane direction in the WindIO blade frame),
    ``edge`` the one closest to axis 2 — matching how the rest of the
    pipeline and the BeamDyn oracle label the pair. Validation also
    compares the two as an unordered pair, so a near-isotropic section
    where the choice is ambiguous still passes."""
    # Column j of V is the eigenvector for λ_j. The principal axis more
    # aligned with reference axis-1 (|V[0, j]| largest) is "flap".
    if abs(V[0, 0]) >= abs(V[0, 1]):
        return lam0, lam1
    return lam1, lam0


def decouple_stiffness(K: np.ndarray) -> DecoupledStiffness:
    """Reduce a 6×6 sectional **stiffness** matrix to decoupled
    Euler–Bernoulli ``EA / EI_flap / EI_edge / GJ`` about the elastic
    and shear centres / principal elastic axes."""
    K = np.asarray(K, dtype=float)
    if K.shape != (6, 6):
        raise ValueError(f"stiffness must be 6×6; got {K.shape}")
    K = 0.5 * (K + K.T)                       # enforce symmetry

    EA = float(K[_AXIAL, _AXIAL])
    if not np.isfinite(EA) or EA <= 0.0:
        raise ValueError("non-positive axial stiffness K[2,2]")

    # 1. Tension centre — nulls the axial↔bending coupling.
    d2 = -K[_AXIAL, _BEND[0]] / EA
    d1 = K[_AXIAL, _BEND[1]] / EA
    Kt = _offset_transform(d1, d2).T @ K @ _offset_transform(d1, d2)

    # 2. Principal bending stiffnesses about the elastic centre.
    B = Kt[np.ix_(_BEND, _BEND)]
    la, lb, ang, V = _principal_2x2(B)
    EI_flap, EI_edge = _assign_flap_edge(la, lb, V)

    # 3. Shear centre — nulls shear↔torsion coupling — then GJ there.
    s_mat = np.array([[-K[_SHEAR[0], _SHEAR[0]], K[_SHEAR[0], _SHEAR[1]]],
                      [-K[_SHEAR[1], _SHEAR[0]], K[_SHEAR[1], _SHEAR[1]]]])
    s_rhs = np.array([-K[_SHEAR[0], _TORSION], -K[_SHEAR[1], _TORSION]])
    try:
        # solve [-K00 s2 + K01 s1 ; -K10 s2 + K11 s1] = [-K05; -K15]
        sol = np.linalg.solve(s_mat, s_rhs)        # [s2, s1]
        s2, s1 = float(sol[0]), float(sol[1])
        if not (np.isfinite(s1) and np.isfinite(s2)):
            raise np.linalg.LinAlgError
        Ks = _offset_transform(s1, s2).T @ K @ _offset_transform(s1, s2)
        GJ = float(Ks[_TORSION, _TORSION])
        if not np.isfinite(GJ) or GJ <= 0.0:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        # Negligible / ill-conditioned shear block (a near-pure-EB
        # input): torsion is already effectively at the shear centre.
        s1 = s2 = 0.0
        GJ = float(K[_TORSION, _TORSION])

    return DecoupledStiffness(
        EA=EA, EI_flap=EI_flap, EI_edge=EI_edge, GJ=GJ,
        x_tc=float(d1), y_tc=float(d2),
        x_sc=float(s1), y_sc=float(s2),
        principal_angle=float(ang),
    )


def decouple_inertia(M: np.ndarray) -> DecoupledInertia:
    """Reduce a 6×6 sectional **inertia** matrix to mass per length +
    principal flap/edge mass moments about the mass centre.

    The rigid-offset kinematic transform is structurally the same as
    for stiffness (``M' = Tᵀ M T``): the mass-centre offset nulls the
    translational↔rotational coupling, and the 2×2 rotary block there
    gives the principal mass moments of inertia."""
    M = np.asarray(M, dtype=float)
    if M.shape != (6, 6):
        raise ValueError(f"inertia must be 6×6; got {M.shape}")
    M = 0.5 * (M + M.T)

    mass = float(M[0, 0])
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("non-positive mass M[0,0]")

    # Mass centre: the translation that nulls the translational-inertia
    # ↔ bending-rotation coupling (same closed form as the tension
    # centre, axial row → translational row).
    d2 = -M[_AXIAL, _BEND[0]] / mass
    d1 = M[_AXIAL, _BEND[1]] / mass
    Mt = _offset_transform(d1, d2).T @ M @ _offset_transform(d1, d2)

    Irot = Mt[np.ix_(_BEND, _BEND)]
    la, lb, ang, V = _principal_2x2(Irot)
    i_flap, i_edge = _assign_flap_edge(la, lb, V)

    return DecoupledInertia(
        mass=mass, i_flap=i_flap, i_edge=i_edge,
        x_cg=float(d1), y_cg=float(d2), principal_angle=float(ang),
    )


__all__ = [
    "DecoupledInertia",
    "DecoupledStiffness",
    "decouple_inertia",
    "decouple_stiffness",
]
