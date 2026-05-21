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
  equations implemented in :func:`_catenary_residual`. B-7 / B-8 are
  the seabed-contact variants (no friction; ``CB = 0``).
- **Irvine, H. M. (1981).** *Cable Structures*, MIT Press, §2.4 the
  extensible elastic catenary. Equivalent derivation; the EA-correction
  terms ``H · L / EA`` and ``L · (V − ½WL) / EA`` come from §2.4 eqn
  (2.49).

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

import math
import pathlib
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["LineType", "Point", "Line", "MooringSystem"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Catenary residual + Jacobian
# ---------------------------------------------------------------------------

def _catenary_residual(
    H: float, V: float, dx_h: float, dz: float,
    L: float, W: float, EA: float,
    *, seabed_contact: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual ``[f1, f2]`` and analytical Jacobian for the catenary
    boundary-condition equations.

    Branches:

    - ``seabed_contact=True`` and ``V < W · L``: anchor portion rests on
      seabed at zero friction — Jonkman 2007 B-7 / B-8 with ``CB = 0``.
    - Otherwise: fully suspended — Jonkman 2007 B-1 / B-2. This is used
      for any line with both endpoints in free air (``seabed_contact=
      False``), plus for FOWT lines whose Newton iterate happens to land
      at ``V ≥ W · L``.

    The Jacobian is the analytical 2 × 2 derived in Irvine 1981 §2.4
    eqn (2.51) (fully-suspended branch); the seabed-contact-branch
    Jacobian is re-derived from B-7 / B-8 and matches Jonkman 2010 OC3
    reference computations.
    """
    WL = W * L
    if seabed_contact and V < WL:
        use_seabed = True
    else:
        use_seabed = False
    if not use_seabed:
        # Fully suspended — Jonkman 2007 eq. B-1, B-2.
        u = V / H
        v = (V - WL) / H
        su = math.sqrt(1.0 + u * u)
        sv = math.sqrt(1.0 + v * v)
        asu_diff = math.asinh(u) - math.asinh(v)
        ssu_diff = su - sv
        inv_diff = 1.0 / su - 1.0 / sv
        usu_diff = u / su - v / sv

        f1 = (H / W) * asu_diff + H * L / EA - dx_h
        f2 = (H / W) * ssu_diff + (L / EA) * (V - 0.5 * WL) - dz

        # ∂F1/∂H = (1/W)[asinh(u) − asinh(v) − (u/su − v/sv)] + L/EA
        # ∂F1/∂V = (1/W)[1/su − 1/sv]
        # ∂F2/∂H = ∂F1/∂V                            (Hessian symmetry)
        # ∂F2/∂V = (1/W)[u/su − v/sv] + L/EA
        J11 = (asu_diff - usu_diff) / W + L / EA
        J12 = inv_diff / W
        J21 = J12
        J22 = usu_diff / W + L / EA
    else:
        # Seabed contact, no friction — Jonkman 2007 eq. B-7, B-8 with
        # CB = 0. Suspended length L_S = V/W; resting length L_B = L − L_S.
        # The seabed portion contributes L_B to ΔX and zero to ΔZ; the
        # suspended portion's lower endpoint sees V = 0, so the standard
        # catenary holds with V_A → 0.
        u = V / H
        su = math.sqrt(1.0 + u * u)
        L_S = V / W
        L_B = L - L_S
        if L_B < 0.0:
            # Numerical edge; should be caught by branch condition.
            L_B = 0.0

        f1 = L_B + (H / W) * math.asinh(u) + H * L / EA - dx_h
        f2 = (H / W) * (su - 1.0) + V * V / (2.0 * W * EA) - dz

        J11 = (math.asinh(u) - u / su) / W + L / EA
        J12 = (1.0 / su - 1.0) / W
        J21 = J12
        J22 = (u / su) / W + V / (W * EA)

    return (
        np.array([f1, f2]),
        np.array([[J11, J12], [J21, J22]]),
    )


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3-2-1 intrinsic Euler rotation: ``R = R_z(yaw) · R_y(pitch) · R_x(roll)``.

    Matches ElastoDyn's platform attitude convention so a ``r_body``
    expressed in body coords maps to world coords via ``R · r_body``.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


# ---------------------------------------------------------------------------
# MooringSystem
# ---------------------------------------------------------------------------

class MooringSystem:
    """A collection of catenary lines connecting a platform to anchors.

    The system is fully assembled by the constructor or
    :meth:`from_moordyn`. The downstream API is:

    - :meth:`fairlead_positions` — world-frame fairlead positions at a
      given body offset.
    - :meth:`restoring_force` — 6-vector force / moment from all lines
      on the body, in world frame, about the body origin.
    - :meth:`solve_equilibrium` — Newton iteration over body DOFs to
      drive ``restoring_force`` to zero (may not converge for pure
      mooring without buoyancy — see module docstring).
    - :meth:`stiffness_matrix` — finite-difference 6 × 6 about a chosen
      offset (or zero by default; see note in the docstring below).
    """

    def __init__(
        self,
        depth: float,
        rho: float = 1025.0,
        g: float = 9.80665,
        line_types: Optional[dict[str, LineType]] = None,
        points: Optional[dict[int, Point]] = None,
        lines: Optional[list[Line]] = None,
    ) -> None:
        self.depth = float(depth)
        self.rho = float(rho)
        self.g = float(g)
        self.line_types: dict[str, LineType] = dict(line_types or {})
        self.points: dict[int, Point] = dict(points or {})
        self.lines: list[Line] = list(lines or [])

    # -----------------------------------------------------------------
    # Platform-state queries
    # -----------------------------------------------------------------

    def fairlead_positions(self, body_r6: np.ndarray) -> list[np.ndarray]:
        """World-frame positions of every ``Vessel``-attached point."""
        body_r6 = np.asarray(body_r6, dtype=float)
        return [
            p.r_world(body_r6) for p in self.points.values()
            if p.attachment == "Vessel"
        ]

    def restoring_force(self, body_r6: np.ndarray) -> np.ndarray:
        """6-vector force/moment from all lines on the platform body.

        ``F[:3]`` = sum of world-frame forces at every Vessel-attached
        endpoint; ``F[3:6]`` = sum of moments (``r_endpoint_world −
        r_body_origin``) × ``F_endpoint``, about the body origin.

        For each line, the endpoint forces are derived from the catenary
        solve in this order:

        - ``F_on_B`` (B = the "fairlead" passed as ``r_b`` to
          ``solve_static``) = ``H · ê_B→A + (-V_F) ẑ``.
        - ``F_on_A`` = ``H · ê_A→B + V_A ẑ`` where ``V_A`` = max(0,
          V_F − W·L). Fully suspended: ``V_A = V_F − W·L > 0`` (line
          pulls anchor up). Seabed contact (V_F < W·L, CB = 0): the
          anchor is on the seabed; horizontal tension at the anchor is
          still ``H`` (no friction decay), and ``V_A = 0``.

        Lines with neither endpoint attached to the body contribute
        nothing. Lines with both endpoints attached to the body
        contribute both endpoint reactions.
        """
        body_r6 = np.asarray(body_r6, dtype=float)
        F = np.zeros(6)
        body_origin = body_r6[:3]
        for line in self.lines:
            attach_a = line.point_a.attachment
            attach_b = line.point_b.attachment
            if attach_a != "Vessel" and attach_b != "Vessel":
                continue
            r_a = line.point_a.r_world(body_r6)
            r_b = line.point_b.r_world(body_r6)
            try:
                H, V_F, f_on_b = line.solve_static(r_a, r_b)
            except RuntimeError as err:
                raise RuntimeError(
                    f"Line {line.point_a.id}->{line.point_b.id} failed to "
                    f"converge at body_r6={body_r6}: {err}"
                ) from err
            if attach_b == "Vessel":
                F[:3] += f_on_b
                F[3:6] += np.cross(r_b - body_origin, f_on_b)
            if attach_a == "Vessel":
                WL = line.line_type.w * line.unstretched_length
                V_A = max(0.0, V_F - WL)
                dr = r_b - r_a
                dx_h = math.hypot(dr[0], dr[1])
                if dx_h > 1e-12:
                    unit_to_b = np.array(
                        [dr[0] / dx_h, dr[1] / dx_h, 0.0]
                    )
                else:
                    unit_to_b = np.zeros(3)
                f_on_a = H * unit_to_b + np.array([0.0, 0.0, V_A])
                F[:3] += f_on_a
                F[3:6] += np.cross(r_a - body_origin, f_on_a)
        return F

    # -----------------------------------------------------------------
    # Equilibrium + linearisation
    # -----------------------------------------------------------------

    def solve_equilibrium(
        self,
        body_r6_init: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        max_iter: int = 50,
        dx: float = 0.1,
        dtheta: float = 0.1,
    ) -> np.ndarray:
        """Newton iteration over body 6-DOF to drive ``restoring_force``
        to zero.

        Warning: pure mooring without buoyancy / weight has no z
        equilibrium (the lines always pull the platform down). For a
        3-fold-symmetric mooring at zero offset the in-plane DOFs
        (surge, sway, yaw) are already balanced; the heave DOF will
        not converge. Pass a ``body_r6_init`` close to your expected
        operating point and accept the result as a "best effort"
        local-minimum.
        """
        r6 = (
            np.zeros(6) if body_r6_init is None
            else np.asarray(body_r6_init, dtype=float).copy()
        )
        for _ in range(max_iter):
            F = self.restoring_force(r6)
            if np.linalg.norm(F) < tol:
                return r6
            J = self._restoring_jacobian(r6, dx=dx, dtheta=dtheta)
            try:
                step = -np.linalg.solve(J, F)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "MooringSystem.solve_equilibrium: singular Jacobian "
                    "(typical for mooring-only systems with no buoyancy "
                    "balance); returning current iterate.",
                    UserWarning,
                    stacklevel=2,
                )
                return r6
            # Cap step so we don't blow past sensible offsets.
            max_step = 10.0 * dx
            step_norm = float(np.linalg.norm(step))
            if step_norm > max_step:
                step *= max_step / step_norm
            r6 = r6 + step
        warnings.warn(
            f"MooringSystem.solve_equilibrium: did not converge in "
            f"{max_iter} iterations (final ||F|| = "
            f"{float(np.linalg.norm(F)):.3e}); returning last iterate.",
            UserWarning,
            stacklevel=2,
        )
        return r6

    def stiffness_matrix(
        self,
        body_r6: Optional[np.ndarray] = None,
        dx: float = 0.1,
        dtheta: float = 0.1,
    ) -> np.ndarray:
        """Linearised 6 × 6 mooring stiffness about ``body_r6``.

        Central finite differences with perturbation ``dx`` (m) on
        translational DOFs and ``dtheta`` (rad) on rotational DOFs.
        The trans-rot off-diagonal blocks are symmetrised after
        differencing — mooring linearised at static equilibrium is the
        Hessian of a potential and must therefore be symmetric;
        finite-difference noise gets averaged out.

        ``body_r6 = None`` is treated as ``np.zeros(6)`` (the typical
        FOWT linearisation point). Pure mooring has no z-direction
        equilibrium without buoyancy, so a solve-for-equilibrium
        default would diverge; pass an explicit ``body_r6`` if you
        want a different linearisation point.

        Returns
        -------
        K : ndarray, shape (6, 6)
            Stiffness in N/m / N / N·m/rad block structure (trans-trans:
            N/m, rot-trans / trans-rot: N (= N·m/m), rot-rot: N·m/rad).
        """
        if body_r6 is None:
            r6 = np.zeros(6)
        else:
            r6 = np.asarray(body_r6, dtype=float).copy()
        K = self._restoring_jacobian(r6, dx=dx, dtheta=dtheta)
        # Symmetrise trans-rot off-diagonal blocks (Hessian of potential).
        K[3:, :3] = 0.5 * (K[3:, :3] + K[:3, 3:].T)
        K[:3, 3:] = K[3:, :3].T
        # Symmetrise the full result for numerical hygiene.
        K = 0.5 * (K + K.T)
        # Sign: stiffness is dF/dr where F is the restoring force.
        # Restoring force opposes offset, so dF/dr is negative-definite
        # in the conservative sense. Conventional "K" returned to callers
        # is the *positive* stiffness = -dF/dr.
        return -K

    def _restoring_jacobian(
        self, r6: np.ndarray, dx: float, dtheta: float,
    ) -> np.ndarray:
        """Central-difference Jacobian of ``restoring_force`` w.r.t. ``r6``."""
        J = np.zeros((6, 6))
        for i in range(6):
            step = dx if i < 3 else dtheta
            r_plus = r6.copy()
            r_plus[i] += step
            r_minus = r6.copy()
            r_minus[i] -= step
            F_plus = self.restoring_force(r_plus)
            F_minus = self.restoring_force(r_minus)
            J[:, i] = (F_plus - F_minus) / (2.0 * step)
        return J

    # -----------------------------------------------------------------
    # MoorDyn parsing
    # -----------------------------------------------------------------

    @classmethod
    def from_moordyn(
        cls,
        dat_path: pathlib.Path | str,
        rho: float = 1025.0,
        g: float = 9.80665,
    ) -> MooringSystem:
        """Parse a MoorDyn v1 / v2 ``.dat`` and return a populated system.

        Sections recognised:

        - **LINE TYPES** (or **LINE DICTIONARY**): rows ``Name Diam
          MassDenInAir EA …``. Wet weight is derived as
          ``w = (MassDenInAir − ρ · π/4 · d²) · g``.
        - **POINTS** or **CONNECTION PROPERTIES**: rows
          ``ID Attachment X Y Z …``. ``Attachment`` accepted
          case-insensitively as ``Fixed``, ``Vessel``, or ``Free``.
        - **LINES** or **LINE PROPERTIES**: rows
          ``ID LineType AttachA AttachB UnstrLen …``.
        - **OPTIONS** (or **SOLVER OPTIONS**): ``WtrDpth`` / ``depth`` and
          ``rhoW`` / ``rho`` if present override the constructor defaults.

        Each section header is detected by ``startswith('---')`` plus a
        keyword (case-insensitive); the immediately-following 1-2 rows
        are skipped as column headers / units rows.
        """
        path = pathlib.Path(dat_path)
        if not path.is_file():
            raise FileNotFoundError(f"MoorDyn .dat not found at {path}")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines_raw = fh.readlines()

        sections = _split_sections(lines_raw)

        # Parse OPTIONS first so depth / rho overrides are available
        # when we derive wet weight from LINE TYPES. The three keys
        # we recognise (WtrDpth / rhoW / g) are load-bearing — rhoW
        # in particular feeds straight into wet-weight ``w = (m_air
        # - rho_w · A) · g_eff`` so a typo there silently shifts
        # every mooring stiffness. Once the key matches one of our
        # recognised forms we therefore strictly parse the value;
        # unknown keys are tolerated (informational rows are common
        # in MoorDyn OPTIONS blocks).
        depth_override: Optional[float] = None
        rho_override: Optional[float] = None
        g_override: Optional[float] = None
        if "OPTIONS" in sections:
            for raw in sections["OPTIONS"]:
                parts = raw.split()
                if len(parts) < 2:
                    continue
                value, key = parts[0], parts[1]
                key_lower = key.lower().rstrip(":")
                if key_lower in ("wtrdpth", "depth"):
                    depth_override = _parse_finite_option(
                        value, key, path,
                    )
                elif key_lower in ("rhow", "rho"):
                    rho_override = _parse_finite_option(
                        value, key, path,
                    )
                elif key_lower == "g":
                    g_override = _parse_finite_option(
                        value, key, path,
                    )

        depth = depth_override if depth_override is not None else 0.0
        rho_w = rho_override if rho_override is not None else rho
        g_eff = g_override if g_override is not None else g

        # LINE TYPES. Rows that ``_looks_like_header_row`` flags as
        # column-name / units lines are skipped silently (handles
        # MoorDyn variants with 1-row vs 2-row table headers); rows
        # that LOOK like data but fail strict parsing raise so a
        # transcription error in a real LineType row can't silently
        # drop the line from the model. Pass-2 review.
        line_types: dict[str, LineType] = {}
        if "LINE TYPES" in sections:
            for raw in sections["LINE TYPES"]:
                parts = raw.split()
                if _looks_like_header_row(parts):
                    continue
                if len(parts) < 4:
                    raise ValueError(
                        f"Malformed LINE TYPES row in {path}: expected "
                        f"≥ 4 columns (Name Diam MassPerLength EA), "
                        f"got {len(parts)}: {raw.strip()!r}"
                    )
                try:
                    name = parts[0]
                    diam = float(parts[1])
                    mass_air = float(parts[2])
                    ea = float(parts[3])
                    if not (math.isfinite(diam) and math.isfinite(mass_air)
                            and math.isfinite(ea)):
                        raise ValueError("non-finite numeric")
                except ValueError as err:
                    raise ValueError(
                        f"Malformed LINE TYPES row in {path} for type "
                        f"{parts[0]!r}: {raw.strip()!r}; one of "
                        f"Diam / MassPerLength / EA is not a finite "
                        f"number."
                    ) from err
                area = math.pi * 0.25 * diam * diam
                w = (mass_air - rho_w * area) * g_eff
                line_types[name] = LineType(
                    name=name,
                    diam=diam,
                    mass_per_length_air=mass_air,
                    EA=ea,
                    w=w,
                )

        # POINTS (or CONNECTION). Same strict-with-header-skip pattern
        # as LINE TYPES above.
        points: dict[int, Point] = {}
        if "POINTS" in sections:
            for raw in sections["POINTS"]:
                parts = raw.split()
                if _looks_like_header_row(parts):
                    continue
                if len(parts) < 5:
                    raise ValueError(
                        f"Malformed POINTS row in {path}: expected ≥ 5 "
                        f"columns (ID Attachment X Y Z), got "
                        f"{len(parts)}: {raw.strip()!r}"
                    )
                try:
                    pid = int(parts[0])
                    attachment = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    if not (math.isfinite(x) and math.isfinite(y)
                            and math.isfinite(z)):
                        raise ValueError("non-finite coordinate")
                except ValueError as err:
                    raise ValueError(
                        f"Malformed POINTS row in {path} (ID column "
                        f"{parts[0]!r}): {raw.strip()!r}; expected "
                        f"integer ID and finite X / Y / Z."
                    ) from err
                points[pid] = Point(
                    id=pid,
                    attachment=attachment,
                    r_body=np.array([x, y, z]),
                )

        # LINES — MoorDyn v2 column order is
        #   ``ID LineType AttachA AttachB UnstrLen NumSegs Outputs``
        # MoorDyn v1 (older ``LINE PROPERTIES`` sections) used
        #   ``ID LineType UnstrLen NumSegs NodeAnch NodeFair``
        # so the integer columns sit at different positions. We probe
        # v2 first and validate point IDs against the parsed ``points``
        # dict; if either doesn't resolve to a known point we fall back
        # to v1 column order. Pre-1.0 review surfaced the v1 gap.
        ln_list: list[Line] = []
        if "LINES" in sections:
            for raw in sections["LINES"]:
                parts = raw.split()
                if _looks_like_header_row(parts):
                    continue
                if len(parts) < 5:
                    raise ValueError(
                        f"Malformed LINES row in {path}: expected ≥ 5 "
                        f"columns (ID LineType plus v1/v2 attachment "
                        f"+ length triple), got {len(parts)}: "
                        f"{raw.strip()!r}"
                    )
                try:
                    _id = int(parts[0])
                    line_type_name = parts[1]
                except ValueError as err:
                    raise ValueError(
                        f"Malformed LINES row in {path}: expected "
                        f"integer ID then LineType name, got "
                        f"{raw.strip()!r}."
                    ) from err
                if line_type_name not in line_types:
                    raise ValueError(
                        f"Line {_id} references unknown LineType "
                        f"{line_type_name!r}; known types: "
                        f"{sorted(line_types.keys())}"
                    )
                spec = _parse_lines_row_v2(parts, points)
                if spec is None and len(parts) >= 6:
                    spec = _parse_lines_row_v1(parts, points)
                if spec is None:
                    raise ValueError(
                        f"Line {_id}: could not parse row under either "
                        f"MoorDyn v2 (AttachA AttachB UnstrLen) or v1 "
                        f"(UnstrLen NumSegs NodeAnch NodeFair) column "
                        f"order; row = {raw.strip()!r}; known points = "
                        f"{sorted(points.keys())}"
                    )
                attach_a, attach_b, unstr_len = spec
                ln_list.append(
                    Line(
                        line_type=line_types[line_type_name],
                        point_a=points[attach_a],
                        point_b=points[attach_b],
                        unstretched_length=unstr_len,
                    )
                )

        return cls(
            depth=depth,
            rho=rho_w,
            g=g_eff,
            line_types=line_types,
            points=points,
            lines=ln_list,
        )

    @classmethod
    def from_windio_mooring(
        cls,
        floating,
        *,
        depth: float,
        moordyn_fallback: "pathlib.Path | str | None" = None,
        rho: float = 1025.0,
        g: float = 9.80665,
    ) -> MooringSystem:
        """Build a system from a WindIO ``mooring`` block (issue #35).

        ``floating`` is a parsed
        :class:`pybmodes.io.windio_floating.WindIOFloating` — its
        ``joints`` table supplies every anchor / fairlead world
        position (fairleads are the axial joints resolved during parsing). Line
        **topology** (nodes / lines) comes from the WindIO mooring
        block; line **properties** (mass/length, EA, wet weight) are
        resolved in order of preference:

        1. explicit WindIO ``line_types`` fields — ``mass_density`` /
           ``linear_density`` and ``stiffness`` / ``EA`` /
           ``axial_stiffness`` — when present;
        2. a companion MoorDyn deck (``moordyn_fallback``): the
           accurate path, equivalent to how WISDEM/RAFT delegate
           chain sizing to MoorPy ``MoorProps`` (line types matched by
           name, or the sole entry);
        3. a documented studless-chain diameter regression (MoorPy
           ``MoorProps`` default coefficients ``m ≈ 19.9e3·d²``,
           ``EA ≈ 0.854e11·d²``) — a rough last resort that emits a
           ``UserWarning``; supply a deck or explicit props for
           quantitative work.

        Catenary engine + ``stiffness_matrix`` are unchanged, so the
        WindIO-topology system and the companion-MoorDyn system are
        consistent by construction (cross-path consistency anchor).
        """
        moor = getattr(floating, "mooring", None) or {}
        joints = floating.joints
        if not moor.get("lines"):
            raise KeyError(
                "WindIO floating component has no mooring.lines block"
            )

        deck_types: dict[str, LineType] = {}
        if moordyn_fallback is not None:
            deck_types = dict(
                cls.from_moordyn(moordyn_fallback, rho, g).line_types
            )

        line_types: dict[str, LineType] = {}
        for lt in moor.get("line_types", []):
            name = lt["name"]
            d = float(lt["diameter"])
            m = lt.get("mass_density", lt.get("linear_density"))
            ea = lt.get("EA", lt.get("stiffness",
                                     lt.get("axial_stiffness")))
            if m is not None and ea is not None:
                m, ea = float(m), float(ea)
                w = (m - rho * 0.25 * np.pi * d * d) * g
                cb = float(lt.get("CB", 0.0))
            elif name in deck_types:
                dt = deck_types[name]
                m, ea, w, cb = (dt.mass_per_length_air, dt.EA, dt.w,
                                dt.CB)
            elif len(deck_types) == 1:
                dt = next(iter(deck_types.values()))
                m, ea, w, cb = (dt.mass_per_length_air, dt.EA, dt.w,
                                dt.CB)
            else:
                import warnings
                m = 19.9e3 * d * d            # MoorPy MoorProps studless
                ea = 0.854e11 * d * d         # chain default regression
                w = (m - rho * 0.25 * np.pi * d * d) * g
                cb = 0.0
                warnings.warn(
                    f"WindIO mooring line_type {name!r} has no "
                    f"mass/EA and no MoorDyn fallback was supplied; "
                    f"using the rough MoorPy studless-chain "
                    f"diameter regression — pass moordyn_fallback or "
                    f"explicit line-type properties for quantitative "
                    f"results.",
                    UserWarning,
                    stacklevel=2,
                )
            line_types[name] = LineType(
                name=name, diam=d, mass_per_length_air=float(m),
                EA=float(ea), w=float(w), CB=float(cb),
            )

        points: dict[int, Point] = {}
        name_to_id: dict[str, int] = {}
        for i, nd in enumerate(moor.get("nodes", []), start=1):
            jn = nd["joint"]
            if jn not in joints:
                raise KeyError(
                    f"mooring node {nd['name']!r} references joint "
                    f"{jn!r} not in the floating joints "
                    f"{sorted(joints)}"
                )
            ntype = str(nd.get("node_type", "")).lower()
            attach = "Vessel" if ntype == "vessel" else "Fixed"
            points[i] = Point(id=i, attachment=attach,
                               r_body=np.asarray(joints[jn], float))
            name_to_id[nd["name"]] = i

        ln_list: list[Line] = []
        for ln in moor["lines"]:
            lt_name = ln["line_type"]
            if lt_name not in line_types:
                raise KeyError(
                    f"mooring line {ln['name']!r} references line_type "
                    f"{lt_name!r} not in {sorted(line_types)}"
                )
            ln_list.append(Line(
                line_type=line_types[lt_name],
                point_a=points[name_to_id[ln["node1"]]],
                point_b=points[name_to_id[ln["node2"]]],
                unstretched_length=float(ln["unstretched_length"]),
            ))

        return cls(depth=float(depth), rho=rho, g=g,
                   line_types=line_types, points=points, lines=ln_list)


# ---------------------------------------------------------------------------
# MoorDyn helpers
# ---------------------------------------------------------------------------


def _looks_like_header_row(parts: list[str]) -> bool:
    """Return True if a MoorDyn-section row looks like a column-name
    or units header rather than a data row.

    Two heuristics, taken together:

    1. **Units row** — every token is parenthesised (e.g. ``(m)``,
       ``(kg/m)``, ``(-)``). Common second header line.
    2. **Column-name row** — no token in the first four columns
       parses as a number. Every MoorDyn data row across LINE TYPES /
       POINTS / LINES carries at least one numeric column in its
       first four (diameter, ID, attachment id, X-coordinate, …),
       so this distinguishes the column-name header
       (``Name Diam MassPerLength EA Diff``) from a data row whose
       first column happens to be a string LineType name.

    Used by ``MooringSystem.from_moordyn`` so the section parsers can
    tolerate MoorDyn-deck variants that ship a 1-row header (only
    column names, no units row) or no header at all, without
    accidentally eating real data rows. The previous hardcoded ``pending_skip = 2`` in
    ``_split_sections`` assumed exactly two header rows, which is
    safe on the OC3 / IEA-15 reference decks the suite already
    covers but not on every valid deck in the wild.
    """
    if not parts:
        return True
    if all(p.startswith("(") and p.endswith(")") for p in parts):
        return True
    # MoorDyn-v1 dialects emit a leading "count" row inside each section
    # (e.g. ``1   NTypes  - number of LineTypes``). That row carries a
    # single integer in column 0 and an ``N*`` label in column 1 — every
    # MoorDyn count label follows the ``N<Capital>...`` convention
    # (NTypes, NConnects, NPoints, NLines, NRods, NBodies, ...). Treat
    # this as a header so the strict per-section parsers don't try to
    # interpret it as data. The IFE UPSCALE 25 MW deck (Sandua-Fernández
    # et al. 2023) is the in-tree example.
    if (
        len(parts) >= 2
        and _looks_like_integer(parts[0])
        and len(parts[1]) >= 2
        and parts[1][0] == "N"
        and parts[1][1].isupper()
    ):
        return True
    return not any(_looks_like_number(p) for p in parts[:4])


def _looks_like_number(token: str) -> bool:
    """Return True if ``token`` parses as a finite float. Used to
    distinguish data rows (which carry at least one numeric column
    among their first few) from column-name headers (which don't)."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _looks_like_integer(token: str) -> bool:
    """Return True if ``token`` parses as a base-10 integer (no leading
    sign, no fractional part). Used by :func:`_looks_like_header_row` to
    spot MoorDyn-v1 ``NTypes`` / ``NConnects`` / ``NLines`` count rows."""
    return token.isdigit()


def _parse_lines_row_v2(
    parts: list[str], points: dict[int, Point],
) -> Optional[tuple[int, int, float]]:
    """Try to parse a LINES row as MoorDyn v2 (``ID LineType AttachA
    AttachB UnstrLen ...``). Returns ``(attach_a, attach_b, unstr_len)``
    or ``None`` if the columns don't validate against ``points``.
    Non-finite ``unstr_len`` returns ``None`` so the v1 fallback gets
    a fair try; a downstream ``unstr_len <= 0`` check also rejects
    NaN (``nan <= 0`` is False but ``unstr_len > 0`` is also False)."""
    try:
        attach_a = int(parts[2])
        attach_b = int(parts[3])
        unstr_len = float(parts[4])
    except ValueError:
        return None
    if not math.isfinite(unstr_len):
        return None
    if attach_a not in points or attach_b not in points or unstr_len <= 0:
        return None
    return attach_a, attach_b, unstr_len


def _parse_lines_row_v1(
    parts: list[str], points: dict[int, Point],
) -> Optional[tuple[int, int, float]]:
    """Try to parse a LINES row as MoorDyn v1 (``ID LineType UnstrLen
    NumSegs NodeAnch NodeFair``). Returns ``(attach_a, attach_b,
    unstr_len)`` or ``None`` if the columns don't validate. Non-finite
    ``unstr_len`` returns ``None`` for the same reason as the v2
    helper above."""
    try:
        unstr_len = float(parts[2])
        attach_a = int(parts[4])
        attach_b = int(parts[5])
    except ValueError:
        return None
    if not math.isfinite(unstr_len):
        return None
    if attach_a not in points or attach_b not in points or unstr_len <= 0:
        return None
    return attach_a, attach_b, unstr_len


def _parse_finite_option(value: str, key: str, path: pathlib.Path) -> float:
    """Strict-parse a MoorDyn OPTIONS row value. Raises ``ValueError``
    with the key, path, and offending token on any parse failure or
    non-finite result. Used for the three load-bearing keys
    (``WtrDpth`` / ``rhoW`` / ``g``); unknown OPTIONS keys stay
    permissive (callers don't reach this helper). A bare
    ``try / except: pass`` would silently swallow typos in these
    keys, which directly shift mooring
    stiffness through the wet-weight calculation."""
    try:
        out = float(value)
    except ValueError as err:
        raise ValueError(
            f"Malformed OPTIONS row in {path}: value {value!r} for "
            f"key {key!r} is not a number."
        ) from err
    if not math.isfinite(out):
        raise ValueError(
            f"Malformed OPTIONS row in {path}: value {value!r} for "
            f"key {key!r} parses to {out!r}, which is not finite. "
            f"Physical quantities (depth, density, gravity) must be "
            f"finite."
        )
    return out



# Map of (lowercase keywords that appear in a section header) -> canonical
# section name. We pick the FIRST keyword that matches inside each header
# line so trailing decorations don't trip us up.
_SECTION_KEYWORDS = {
    "line types": "LINE TYPES",
    "line dictionary": "LINE TYPES",
    "rod types": "ROD TYPES",  # reserved (not parsed)
    "rod dictionary": "ROD TYPES",
    "bodies": "BODIES",         # reserved (not parsed)
    "points": "POINTS",
    "connection properties": "POINTS",   # v1 alias
    "connections": "POINTS",             # v1 alias
    "lines": "LINES",
    "line properties": "LINES",
    "options": "OPTIONS",
    "solver options": "OPTIONS",
    "outputs": "OUTPUTS",       # reserved (not parsed)
    "output list": "OUTPUTS",
}


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Group MoorDyn file lines into sections keyed by canonical name.

    Header detection: a line that starts with three dashes (after
    stripping whitespace) is a section divider; its lowercase content
    (with decoration stripped) is matched against
    ``_SECTION_KEYWORDS``. Inside a recognised section, up to two
    column-name / units rows immediately after the divider are skipped
    if they match :func:`_looks_like_header_row`; the moment a row that
    looks like data appears we stop eating header rows. This handles
    both the de-facto MoorDyn convention (column-names + units lines,
    two header rows) and hand-edited variants with one header row or
    none at all. The previous
    fixed ``pending_skip = 2`` ate the first data row on decks shipped
    without a units line.

    Comment lines (``!``, ``#``) and blank lines are dropped.
    """
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    pending_header_skip = 0  # rows of header to inspect-and-maybe-skip
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("!") or stripped.startswith("#"):
            continue
        if stripped.startswith("---"):
            # Section divider. Find a known keyword inside it.
            content = stripped.strip("- ").lower()
            section = None
            for kw, canon in _SECTION_KEYWORDS.items():
                if kw in content:
                    section = canon
                    break
            current = section
            if current is not None:
                sections.setdefault(current, [])
                # Allow up to two consecutive header rows after the
                # divider. OPTIONS has no header / units convention —
                # every row there is ``value label``.
                pending_header_skip = 0 if current == "OPTIONS" else 2
            continue
        # Detect "END" sentinel.
        if stripped.upper() == "END":
            current = None
            continue
        if current is None:
            continue
        if pending_header_skip > 0:
            parts = raw.split()
            if _looks_like_header_row(parts):
                pending_header_skip -= 1
                continue
            # First non-header row marks the start of real data; stop
            # the inspect-and-skip loop so any subsequent rows that
            # incidentally pattern-match the header heuristic still
            # land in the section list (where the strict-parse path
            # raises on them rather than silently dropping).
            pending_header_skip = 0
        sections[current].append(raw)
    return sections
