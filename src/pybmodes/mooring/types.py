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

"""Mooring-line data types: :class:`LineType`, :class:`Point`, :class:`Line`.

:class:`Line` owns its catenary solve via :meth:`Line.solve_static` (which
calls the residual function in :mod:`pybmodes.mooring._catenary`).
:class:`Point` knows how to project its body-frame coordinates into the
world frame via :func:`pybmodes.mooring._rotation._rotation_matrix`.

Multi-line force assembly + stiffness lives in
:class:`pybmodes.mooring.MooringSystem`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._catenary import _catenary_residual
from ._rotation import _rotation_matrix


@dataclass(frozen=True)
class LineType:
    """Material spec for a mooring line.

    Attributes
    ----------
    name : str
        Identifier referenced by ``Line.line_type`` and by MoorDyn LINES
        rows.
    diam : float
        Outer diameter (m).
    mass_per_length_air : float
        Mass density in air (kg/m).
    EA : float
        Axial stiffness (N). Inextensible limit is ``EA → ∞``.
    w : float
        Wet weight per unit length (N/m). For a homogeneous line of
        diameter ``d`` in water of density ``ρ_w``:
        ``w = (m - ρ_w · π/4 · d²) · g``.
    CB : float, default 0
        Seabed friction coefficient (sliding friction along the resting
        segment of a partly-grounded line). Parsed for round-trip
        identity with MoorDyn ``.dat`` inputs; the current catenary
        solver only handles the ``CB = 0`` (frictionless seabed) case.
    """

    name: str
    diam: float
    mass_per_length_air: float
    EA: float
    w: float
    CB: float = 0.0


@dataclass
class Point:
    """Endpoint of a mooring line.

    Attributes
    ----------
    id : int
        MoorDyn point ID (preserved for round-trip identification).
    attachment : str
        One of ``Fixed`` / ``Vessel`` / ``Free`` (case-insensitive on
        construction; stored title-cased).
    r_body : ndarray, shape (3,)
        Position in *body frame* for ``Vessel`` points; *world frame* for
        ``Fixed`` and ``Free`` points. ``Free`` points are essentially
        ``Fixed`` placeholders for this quasi-static solver — they don't
        participate in the body-equilibrium DOFs.
    """

    id: int
    attachment: str
    r_body: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.r_body, np.ndarray):
            self.r_body = np.asarray(self.r_body, dtype=float)
        if self.r_body.shape != (3,):
            raise ValueError(
                f"Point.r_body must be shape (3,); got {self.r_body.shape}"
            )
        # Normalise MoorDyn v1 abbreviations ('Fix' / 'Connect') to the
        # v2 canonical names ('Fixed' / 'Free'). MoorDyn v2 also accepts
        # 'Anchor' as a synonym for 'Fixed' and 'Body' / 'Coupled' for
        # vessel-attached; we map those too. Unknown strings raise so a
        # typo in user-authored input surfaces immediately.
        raw = self.attachment.strip().title()
        _aliases = {
            "Fixed": "Fixed", "Fix": "Fixed", "Anchor": "Fixed",
            "Vessel": "Vessel", "Body": "Vessel", "Coupled": "Vessel",
            "Free": "Free", "Connect": "Free", "Connection": "Free",
        }
        if raw not in _aliases:
            raise ValueError(
                f"Point.attachment must be one of "
                f"'Fixed' / 'Vessel' / 'Free' (or a known MoorDyn alias "
                f"such as 'Fix' / 'Connect' / 'Body' / 'Anchor'); "
                f"got {self.attachment!r}"
            )
        self.attachment = _aliases[raw]

    def r_world(self, body_r6: np.ndarray) -> np.ndarray:
        """World-frame position for this point at platform state ``body_r6``.

        For ``Fixed`` and ``Free`` points the platform state is ignored;
        for ``Vessel`` points the rotation
        ``R(roll, pitch, yaw) · r_body + r_body_origin`` is applied with
        the 3-2-1 intrinsic Euler convention.
        """
        if self.attachment == "Vessel":
            roll, pitch, yaw = body_r6[3], body_r6[4], body_r6[5]
            R = _rotation_matrix(roll, pitch, yaw)
            return R @ self.r_body + body_r6[:3]
        return self.r_body.copy()


@dataclass
class Line:
    """A single mooring line connecting two :class:`Point` endpoints.

    The catenary solve is owned by this class; the multi-line force
    assembly is delegated to :class:`MooringSystem`.

    Attributes
    ----------
    seabed_contact : bool, default True
        Whether the anchor sits on a seabed. When ``True``, a solver
        iterate with ``V_F < W·L`` triggers the seabed-contact branch
        of the catenary equations (Jonkman 2007 B-7 / B-8 with
        ``CB = 0``); the anchor-side portion of the line is treated as
        resting on the seabed. When ``False`` the fully-suspended
        equations (B-1 / B-2) are used unconditionally — appropriate
        for analytical tests where both endpoints are in free air and
        the line just sags between them. FOWT use cases default to
        ``True``.
    """

    line_type: LineType
    point_a: Point
    point_b: Point
    unstretched_length: float
    seabed_contact: bool = True

    def solve_static(
        self,
        r_a: np.ndarray,
        r_b: np.ndarray,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> tuple[float, float, np.ndarray]:
        """Solve the elastic catenary between ``r_a`` (anchor) and ``r_b``
        (fairlead).

        Implements Jonkman 2007 Appendix B equations B-1 and B-2 for the
        fully-suspended branch; B-7 / B-8 with ``CB = 0`` for the seabed-
        contact branch. The two unknowns are ``H`` (horizontal tension,
        constant along the line) and ``V_F`` (vertical tension at the
        fairlead, positive when the line pulls the fairlead downward).

        Returns
        -------
        H : float
            Horizontal tension (N).
        V_F : float
            Vertical fairlead tension (N).
        f_on_fairlead : ndarray, shape (3,)
            World-frame force the line exerts on ``r_b`` (the fairlead):
            horizontal component pulls toward ``r_a``, vertical component
            is ``-V_F`` (line pulls fairlead down).
        """
        dr = np.asarray(r_b) - np.asarray(r_a)
        dx_h = math.hypot(dr[0], dr[1])
        dz = float(dr[2])
        L = float(self.unstretched_length)
        W = float(self.line_type.w)
        EA = float(self.line_type.EA)

        if L <= 0.0:
            raise ValueError(f"Line.unstretched_length must be > 0; got {L}")
        if W <= 0.0:
            raise ValueError(
                f"LineType.w (wet weight per length) must be > 0; "
                f"got {W} — a neutrally-buoyant or floating line is not "
                f"supported by the standard catenary formulation"
            )
        if EA <= 0.0:
            raise ValueError(f"LineType.EA must be > 0; got {EA}")

        # Initial guess heuristic — converges in fewer than 10 Newton
        # iterations on all OC3-style FOWT lines from r6 = 0.
        H = max(0.25 * W * L, 1.0)
        V = max(0.5 * W * L, dz * W)

        for _ in range(max_iter):
            residual, J = _catenary_residual(
                H, V, dx_h, dz, L, W, EA,
                seabed_contact=self.seabed_contact,
            )
            if np.linalg.norm(residual) < tol:
                break
            try:
                step = -np.linalg.solve(J, residual)
            except np.linalg.LinAlgError as err:
                raise RuntimeError(
                    f"Line.solve_static: singular Jacobian (H={H}, V={V})"
                ) from err
            # Damped Newton: cap each component to ±50 % of current
            # magnitude so we don't overshoot into nonphysical territory.
            max_dH = 0.5 * abs(H)
            max_dV = 0.5 * max(abs(V), W * L * 0.01)
            step[0] = max(-max_dH, min(max_dH, step[0]))
            step[1] = max(-max_dV, min(max_dV, step[1]))
            H = max(H + step[0], 1e-6 * W * L)  # floor H to keep asinh well-defined
            V = V + step[1]
        else:
            raise RuntimeError(
                f"Line.solve_static: failed to converge after {max_iter} "
                f"iterations (final residual norm = "
                f"{float(np.linalg.norm(residual)):.3e} m)"
            )

        # Force on fairlead from line.
        if dx_h > 1e-12:
            unit_to_anchor = np.array(
                [-dr[0] / dx_h, -dr[1] / dx_h, 0.0]
            )
        else:
            unit_to_anchor = np.zeros(3)
        f_on_fairlead = H * unit_to_anchor + np.array([0.0, 0.0, -V])
        return H, V, f_on_fairlead
