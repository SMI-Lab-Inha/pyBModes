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

"""Classical-lamination-theory (CLT) primitives for the composite
blade cross-section reduction (issue #35).

This module is deliberately **orientation-agnostic and IO-free**: it
knows nothing about airfoils, span, or WindIO. It turns a stack of
plies — each a WindIO material plus a fibre angle and a thickness —
into the laminate ``A`` / ``B`` / ``D`` matrices and the
membrane-condensed stiffness ``Atilde = A - B D^-1 B`` that the
thin-wall beam reduction integrates around the section
perimeter. References: Jones, *Mechanics of Composite Materials*
(2nd ed.) §2.5–§4.3; Bir 2006, *User's Guide to PreComp*,
NREL/TP-500-38929, §3.

Convention: the laminate in-plane axes are ``(1, 2)`` where ``1`` is
the ply / wall **longitudinal** direction (the blade-span direction)
and ``2`` the transverse direction. ``Q`` is the plane-stress reduced
stiffness in the *ply principal* axes; ``Qbar``
is ``Q`` rotated by the fibre angle into the *wall* axes. ``z`` runs
through the wall thickness, the laminate mid-surface at ``z = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlyElastic:
    """In-plane elastic constants of one material (plane stress).

    ``E1`` / ``E2`` longitudinal / transverse Young's moduli (Pa),
    ``G12`` in-plane shear modulus (Pa), ``nu12`` major Poisson ratio,
    ``rho`` density (kg/m^3). For an isotropic material
    ``E1 == E2``, ``G12 = E / (2 (1 + nu))``.
    """

    E1: float
    E2: float
    G12: float
    nu12: float
    rho: float


def material_plane_stress(material: dict) -> PlyElastic:
    """Extract the in-plane elastic constants from a WindIO material.

    Handles both the isotropic form (``orth: 0`` or absent — scalar
    ``E``/``nu``/optional ``G``) and the orthotropic form (``orth: 1``
    — ``E``/``G``/``nu`` are 3-vectors ``[*, *12/*23/...]``; only the
    in-plane ``(E1, E2, G12, nu12)`` subset is used, since a thin-wall
    beam reduction is a plane-stress problem).

    ``rho`` is always scalar. ``density`` is accepted as a synonym for
    ``rho`` (some WindIO files use it), mirroring
    :func:`pybmodes.io.windio.read_windio_tubular`.
    """
    name = material.get("name", "<unnamed>")
    if "E" not in material:
        raise KeyError(f"WindIO material {name!r} has no 'E'")
    rho = material.get("rho", material.get("density"))
    if rho is None:
        raise KeyError(f"WindIO material {name!r} has neither 'rho' nor 'density'")
    rho = float(rho)

    E = material["E"]
    G = material.get("G")
    nu = material.get("nu", 0.3)

    if isinstance(E, (list, tuple)):
        # Orthotropic: E=[E1,E2,E3], G=[G12,G13,G23], nu=[nu12,nu13,nu23].
        E1, E2 = float(E[0]), float(E[1])
        if G is None or not isinstance(G, (list, tuple)):
            raise ValueError(
                f"WindIO material {name!r} is orthotropic (E is a vector) "
                f"but G is missing or scalar; an orthotropic ply needs "
                f"G=[G12,G13,G23]."
            )
        G12 = float(G[0])
        nu12 = float(nu[0]) if isinstance(nu, (list, tuple)) else float(nu)
        return PlyElastic(E1=E1, E2=E2, G12=G12, nu12=nu12, rho=rho)

    # Isotropic.
    E1 = float(E)
    nu12 = float(nu[0]) if isinstance(nu, (list, tuple)) else float(nu)
    G12 = float(G) if G is not None else E1 / (2.0 * (1.0 + nu12))
    return PlyElastic(E1=E1, E2=E1, G12=G12, nu12=nu12, rho=rho)


def reduced_stiffness(ply: PlyElastic) -> np.ndarray:
    """Plane-stress reduced stiffness ``Q`` (3×3) in ply principal axes.

    Order ``[11, 22, 12]`` engineering notation:
    ``Q = [[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]]`` with
    ``Q11 = E1/(1-ν12 ν21)``, ``Q22 = E2/(1-ν12 ν21)``,
    ``Q12 = ν12 E2/(1-ν12 ν21)``, ``Q66 = G12`` and
    ``ν21 = ν12 E2/E1`` (Jones eq. 2.61). Reduces to the isotropic
    ``E/(1-ν²)`` form when ``E1 == E2``.
    """
    nu21 = ply.nu12 * ply.E2 / ply.E1
    denom = 1.0 - ply.nu12 * nu21
    if denom <= 0.0:
        raise ValueError(
            f"non-physical Poisson coupling: 1 - nu12*nu21 = {denom:.3g} "
            f"(nu12={ply.nu12}, E1={ply.E1}, E2={ply.E2})"
        )
    q11 = ply.E1 / denom
    q22 = ply.E2 / denom
    q12 = ply.nu12 * ply.E2 / denom
    q66 = ply.G12
    return np.array(
        [[q11, q12, 0.0],
         [q12, q22, 0.0],
         [0.0, 0.0, q66]],
        dtype=float,
    )


def transform_reduced_stiffness(Q: np.ndarray, theta: float) -> np.ndarray:
    """Rotate ``Q`` by fibre angle ``theta`` (rad) into wall axes → ``Qbar``.

    Standard plane-stress transformation (Jones eqs. 2.80; the
    explicit closed form avoids forming the ``T`` matrix and its
    inverse). ``theta`` is measured from the wall longitudinal axis to
    the fibre direction.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    c2, s2 = c * c, s * s
    c4, s4 = c2 * c2, s2 * s2
    cs2 = c2 * s2

    q11, q12, q22, q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

    qb11 = q11 * c4 + 2.0 * (q12 + 2.0 * q66) * cs2 + q22 * s4
    qb22 = q11 * s4 + 2.0 * (q12 + 2.0 * q66) * cs2 + q22 * c4
    qb12 = (q11 + q22 - 4.0 * q66) * cs2 + q12 * (s4 + c4)
    qb66 = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * cs2 + q66 * (s4 + c4)
    qb16 = ((q11 - q12 - 2.0 * q66) * s * c2 * c
            + (q12 - q22 + 2.0 * q66) * s2 * s * c)
    qb26 = ((q11 - q12 - 2.0 * q66) * s2 * s * c
            + (q12 - q22 + 2.0 * q66) * s * c2 * c)
    return np.array(
        [[qb11, qb12, qb16],
         [qb12, qb22, qb26],
         [qb16, qb26, qb66]],
        dtype=float,
    )


def abd_matrices(
    plies: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the laminate ``A`` / ``B`` / ``D`` matrices.

    ``plies`` is the through-thickness stack, **first entry = innermost
    face**, each ``(Qbar, thickness)``. ``z`` is measured from the
    laminate mid-surface (Jones eqs. 4.22–4.24)::

        A_ij = Σ Qbar_ij (z_k - z_{k-1})
        B_ij = ½ Σ Qbar_ij (z_k² - z_{k-1}²)
        D_ij = ⅓ Σ Qbar_ij (z_k³ - z_{k-1}³)

    A symmetric stack gives ``B = 0`` (to round-off); a single ply
    gives ``B = 0`` exactly.
    """
    if not plies:
        raise ValueError("abd_matrices: empty ply stack")
    total_t = sum(t for _, t in plies)
    z = -0.5 * total_t
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for Qbar, t in plies:
        z_lo = z
        z_hi = z + t
        A += Qbar * (z_hi - z_lo)
        B += 0.5 * Qbar * (z_hi**2 - z_lo**2)
        D += (Qbar * (z_hi**3 - z_lo**3)) / 3.0
        z = z_hi
    return A, B, D


def membrane_condensed(
    A: np.ndarray, B: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """Membrane stiffness with wall-bending condensed out: ``A - B D⁻¹ B``.

    For a thin wall whose bending moments are unrestrained at the
    section level (the PreComp assumption), the effective in-plane
    membrane stiffness governing the beam ``EA`` / ``EI`` / ``GJ`` is
    the Schur complement of the laminate ``[[A,B],[B,D]]`` on the
    curvature block. A symmetric laminate (``B = 0``) returns ``A``
    unchanged.
    """
    if not np.any(B):
        return A.copy()
    return A - B @ np.linalg.solve(D, B)
