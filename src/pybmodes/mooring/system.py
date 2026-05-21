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

""":class:`MooringSystem` — multi-line restoring-force assembly + 6×6 stiffness.

Holds the system of catenary lines, computes the world-frame restoring
force on a 6-DOF floating body, and exposes a finite-difference 6×6
stiffness matrix ready for the ``PlatformSupport.mooring_K`` block.
Two parsers ship here as classmethods:

- :meth:`MooringSystem.from_moordyn` — MoorDyn v1 / v2 ``.dat`` ingest.
  The section-splitter + row-parser primitives live in
  :mod:`pybmodes.mooring._moordyn_parser`.
- :meth:`MooringSystem.from_windio_mooring` — WindIO ``floating_platform``
  mooring block (issue #35). Duck-typed on ``floating`` so this module
  doesn't import :mod:`pybmodes.io.windio_floating` (cycle break).
"""
from __future__ import annotations

import math
import pathlib
import warnings
from typing import Optional

import numpy as np

from ._moordyn_parser import (
    _looks_like_header_row,
    _parse_finite_option,
    _parse_lines_row_v1,
    _parse_lines_row_v2,
    _split_sections,
)
from .types import Line, LineType, Point


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
