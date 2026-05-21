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

"""Thin-wall composite cross-section reduction — multi-cell
(issue #35).

Given an airfoil perimeter (:class:`pybmodes.io._precomp.profile.Profile`),
the chord, the reference-axis chordwise location, the resolved shell
layers, and the resolved shear webs at one span station, reduce the
section to the *uncoupled diagonal* 1-D beam properties pyBmodes' FEM
consumes: axial ``EA``, chordline bending ``EI_flap`` / ``EI_edge`` (+
principal-axis angle), torsion ``GJ``, mass per length, flap/edge mass
moments, and the tension- / c.g. / shear-centre offsets.

Method (textbook thin-wall + classical lamination theory; Bir 2006,
*User's Guide to PreComp*, NREL/TP-500-38929, §3 — diagonal subset):

* Per wall segment (shell perimeter *and* every web line), stack the
  covering plies through the wall and form the laminate **membrane**
  stiffness ``A`` (:mod:`.laminate`). The wall's own bending-extension
  coupling ``B``/``D`` is NOT condensed in — the section bending is
  carried by the parallel-axis assembly and the thin wall's
  self-bending is second order (the PreComp thin-wall assumption).
  Effective longitudinal modulus·thickness with the laminate free to
  strain transversely (``N_y = 0``): ``Ex = A11 − A12²/A22``;
  shear ``Gxy = A66``.
* ``EA = Σ Ex·ds`` over shell + web segments; tension centre =
  ``Ex``-weighted centroid; ``EI`` = parallel-axis ``Σ Ex·d²`` about
  it (chordline frame), cross term → principal-axis angle.
* Torsion: **single-cell** Bredt–Batho ``GJ = 4 A² / ∮(ds/Gxy)`` when
  there are no webs; **multi-cell** when webs split the section into a
  chain of cells (LE cell · between-web cells · TE cell): solve the
  coupled cell shear-flow system ``(1/2Aᵢ)[qᵢ δᵢ − Σ qⱼ δ_shared] =
  θ'`` for unit twist, ``GJ = 2 Σ Aᵢ qᵢ``. A symmetric interior web
  carries zero shear flow under torsion, so it leaves ``GJ`` equal to
  the webless single-cell value — an exact self-check.
* Mass: ``Σ ρt·ds``; c.g. = mass-weighted centroid; flap/edge mass
  moments about it. Shear centre is approximated by the tension centre
  (exact SC needs the shear-flow warping solve; for the symmetric
  closed-form gates SC ≡ centroid ≡ TC, so the approximation is exact
  there).

Clean-room; the WISDEM PreComp port is the studied reference, not
vendored (independence stance; see ``VALIDATION.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pybmodes.io._precomp.laminate import (
    PlyElastic,
    abd_matrices,
    reduced_stiffness,
    transform_reduced_stiffness,
)
from pybmodes.io._precomp.profile import Profile


@dataclass
class LayerStation:
    """One WindIO shell layer at a single span station: material, ply
    thickness (m), fibre angle (rad, from the blade-span axis), and the
    ``nd_arc`` band ``[s0, s1]`` it covers."""

    ply: PlyElastic
    thickness: float
    theta: float
    s0: float
    s1: float


@dataclass
class WebStation:
    """One shear web at a single span station: its suction- and
    pressure-side ``nd_arc`` feet and its through-wall ply stack
    ``[(ply, thickness, theta), …]`` (the on-web layers)."""

    s_start: float                                        # suction foot
    s_end: float                                          # pressure foot
    plies: list = field(default_factory=list)


@dataclass
class SectionResult:
    """Uncoupled diagonal beam properties at one station (SI)."""

    EA: float              # N
    EI_flap: float         # N·m²  (out-of-plane / flap, chordline frame)
    EI_edge: float         # N·m²  (in-plane / edge)
    GJ: float              # N·m²
    mass: float            # kg/m
    flap_iner: float       # kg·m
    edge_iner: float       # kg·m
    x_tc: float            # chordwise tension-centre offset re ref axis (m)
    y_tc: float            # flap-normal tension-centre offset (m)
    x_cg: float            # chordwise c.g. offset (m)
    y_cg: float            # flap-normal c.g. offset (m)
    x_sc: float            # chordwise shear-centre offset (m) [≈ tc]
    y_sc: float            # flap-normal shear-centre offset (m) [≈ tc]
    principal_angle: float  # rad, chordline → principal bending axes
    n_cells: int = 1       # torsion cells (1 + number of webs)


def _stack_props(plies) -> tuple[float, float, float]:
    """``(Ex, Gxy, rho_t)`` for a through-wall ply stack: longitudinal
    modulus·thickness (N/m, ``N_y = 0``), shear modulus·thickness
    (N/m), mass per wall area (kg/m²). Empty → all zero."""
    if not plies:
        return 0.0, 0.0, 0.0
    stack = []
    rho_t = 0.0
    for ply, t, theta in plies:
        Q = reduced_stiffness(ply)
        stack.append((transform_reduced_stiffness(Q, theta), t))
        rho_t += ply.rho * t
    A, _B, _D = abd_matrices(stack)
    a11, a12, a22, a66 = A[0, 0], A[0, 1], A[1, 1], A[2, 2]
    ex = a11 - (a12 * a12 / a22 if a22 > 0.0 else 0.0)
    return float(ex), float(a66), float(rho_t)


def reduce_section(
    profile: Profile,
    chord: float,
    ref_axis_xc: float,
    shell_layers: list[LayerStation],
    webs: list[WebStation] | None = None,
    *,
    n_perim: int = 400,
    n_web: int = 50,
) -> SectionResult:
    """Reduce one station to its uncoupled diagonal beam properties.

    ``profile`` — airfoil perimeter (unit-chord ``nd_arc`` spine).
    ``chord`` — section chord (m). ``ref_axis_xc`` — chordwise
    reference (pitch) axis location as a chord fraction (offsets are
    reported relative to it, on the chord line). ``shell_layers`` —
    resolved shell layers, **outermost first**. ``webs`` — resolved
    shear webs (each a straight wall between its suction/pressure
    ``nd_arc`` feet, with its own ply stack); ``None``/empty →
    single-cell torsion.
    """
    webs = list(webs or [])

    def pt(s):
        x, y = profile.arc_to_xy(np.atleast_1d(np.asarray(s, float)))
        return (x - ref_axis_xc) * chord, y * chord

    seg = np.hypot(np.diff(profile.xc), np.diff(profile.yc))
    perim_phys = float(seg.sum()) * float(chord)
    if perim_phys <= 0.0:
        raise ValueError("degenerate section: zero physical perimeter")

    # --- shell wall segments -------------------------------------------
    s_edges = np.linspace(0.0, 1.0, n_perim + 1)
    s_mid = 0.5 * (s_edges[:-1] + s_edges[1:])
    ds = perim_phys / n_perim
    xs, ys = pt(s_mid)

    ex = np.zeros(n_perim)
    gxy = np.zeros(n_perim)
    rho_t = np.zeros(n_perim)
    for k in range(n_perim):
        sk = s_mid[k]
        plies = [
            (ly.ply, ly.thickness, ly.theta)
            for ly in shell_layers
            if ly.s0 - 1e-12 <= sk <= ly.s1 + 1e-12
        ]
        ex[k], gxy[k], rho_t[k] = _stack_props(plies)

    x_parts = [xs]
    y_parts = [ys]
    ea_parts = [ex * ds]
    m_parts = [rho_t * ds]

    # --- web wall segments (add their material to EA / EI / mass) ------
    web_geom: list = []   # (s_ss, s_se, ex_w, gxy_w, length) per web
    for w in webs:
        (x0,), (y0,) = pt(w.s_start)
        (x1,), (y1,) = pt(w.s_end)
        Lw = float(np.hypot(x1 - x0, y1 - y0))
        ex_w, gxy_w, rho_w = _stack_props(list(w.plies))
        if Lw > 0.0 and (ex_w > 0.0 or rho_w > 0.0):
            tt = (np.arange(n_web) + 0.5) / n_web
            xm = x0 + tt * (x1 - x0)
            ym = y0 + tt * (y1 - y0)
            dsw = Lw / n_web
            x_parts.append(xm)
            y_parts.append(ym)
            ea_parts.append(np.full(n_web, ex_w * dsw))
            m_parts.append(np.full(n_web, rho_w * dsw))
        web_geom.append((float(w.s_start), float(w.s_end), ex_w, gxy_w, Lw))

    X = np.concatenate(x_parts)
    Y = np.concatenate(y_parts)
    EA_seg = np.concatenate(ea_parts)
    M_seg = np.concatenate(m_parts)

    EA = float(EA_seg.sum())
    mass = float(M_seg.sum())
    if EA <= 0.0 or mass <= 0.0:
        raise ValueError(
            "section has no load-bearing material (EA or mass is zero) — "
            "check the resolved layer bands cover the perimeter"
        )

    x_tc = float((EA_seg * X).sum() / EA)
    y_tc = float((EA_seg * Y).sum() / EA)
    x_cg = float((M_seg * X).sum() / mass)
    y_cg = float((M_seg * Y).sum() / mass)

    dxt, dyt = X - x_tc, Y - y_tc
    ei_edge = float((EA_seg * dxt * dxt).sum())
    ei_flap = float((EA_seg * dyt * dyt).sum())
    ei_xy = float((EA_seg * dxt * dyt).sum())
    principal = 0.5 * float(np.arctan2(2.0 * ei_xy, ei_edge - ei_flap))

    dxc, dyc = X - x_cg, Y - y_cg
    edge_iner = float((M_seg * dxc * dxc).sum())
    flap_iner = float((M_seg * dyc * dyc).sum())

    GJ, n_cells = _torsion_constant(
        profile, chord, ref_axis_xc, s_mid, gxy, ds, web_geom
    )

    return SectionResult(
        EA=EA, EI_flap=ei_flap, EI_edge=ei_edge, GJ=GJ, mass=mass,
        flap_iner=flap_iner, edge_iner=edge_iner,
        x_tc=x_tc, y_tc=y_tc, x_cg=x_cg, y_cg=y_cg,
        x_sc=x_tc, y_sc=y_tc, principal_angle=principal, n_cells=n_cells,
    )


def _torsion_constant(
    profile: Profile, chord: float, ref_axis_xc: float,
    s_mid: np.ndarray, gxy: np.ndarray, ds: float, web_geom: list,
) -> tuple[float, int]:
    """Bredt–Batho torsion: single-cell (no webs) or the multi-cell
    chain (webs split the section LE→TE). Returns ``(GJ, n_cells)``."""

    def poly_area(xv, yv) -> float:
        return 0.5 * abs(float(np.sum(xv[:-1] * yv[1:] - xv[1:] * yv[:-1])))

    def cell_xy(s_a, s_b, n=240):
        ss = np.linspace(s_a, s_b, n)
        x, y = profile.arc_to_xy(ss)
        return (x - ref_axis_xc) * chord, y * chord

    def shell_delta(s_a, s_b) -> float:
        lo, hi = (s_a, s_b) if s_a <= s_b else (s_b, s_a)
        m = (s_mid >= lo) & (s_mid <= hi) & (gxy > 0.0)
        return float((ds / gxy[m]).sum()) if np.any(m) else 0.0

    # full closed perimeter
    sv = np.linspace(0.0, 1.0, len(s_mid) + 1)
    xv, yv = profile.arc_to_xy(sv)
    xv = (xv - ref_axis_xc) * chord
    yv = yv * chord
    a_full = poly_area(xv, yv)

    valid_webs = [
        wg for wg in web_geom
        if wg[3] > 0.0 and wg[4] > 0.0
        and wg[0] < profile.s_le < wg[1]          # suction foot, pressure foot
    ]
    if not valid_webs:
        m = gxy > 0.0
        if a_full <= 0.0 or not np.any(m):
            return 0.0, 1
        return 4.0 * a_full * a_full / float((ds / gxy[m]).sum()), 1

    # LE-most web first (largest suction-foot nd_arc is closest to LE).
    W = sorted(valid_webs, key=lambda wg: wg[0], reverse=True)
    M = len(W)
    n_cells = M + 1
    dw = [wg[4] / wg[3] for wg in W]              # web ∮ ds/Gt = L / (Gt)

    areas = np.zeros(n_cells)
    delta = np.zeros(n_cells)                     # full loop ∮ ds/Gt
    for i in range(n_cells):
        if i == 0:                                # LE cell
            ss0, se0 = W[0][0], W[0][1]
            x, y = cell_xy(ss0, se0)
            areas[i] = poly_area(np.append(x, x[0]), np.append(y, y[0]))
            delta[i] = shell_delta(ss0, se0) + dw[0]
        elif i == M:                              # TE cell (wraps the TE)
            ssm, sem = W[M - 1][0], W[M - 1][1]
            x1, y1 = cell_xy(ssm, 0.0)
            x2, y2 = cell_xy(1.0, sem)
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
            areas[i] = poly_area(np.append(x, x[0]), np.append(y, y[0]))
            delta[i] = (shell_delta(0.0, ssm) + shell_delta(sem, 1.0)
                        + dw[M - 1])
        else:                                     # between web i-1 and i
            x1, y1 = cell_xy(W[i][0], W[i - 1][0])      # suction arc
            x2, y2 = cell_xy(W[i - 1][1], W[i][1])      # pressure arc
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
            areas[i] = poly_area(np.append(x, x[0]), np.append(y, y[0]))
            delta[i] = (shell_delta(W[i][0], W[i - 1][0])
                        + shell_delta(W[i - 1][1], W[i][1])
                        + dw[i - 1] + dw[i])

    # Coupled cell shear-flow system: K q = 2 A, web j shared by
    # cells j and j+1.
    K = np.diag(delta)
    for j in range(M):
        K[j, j + 1] -= dw[j]
        K[j + 1, j] -= dw[j]
    q = np.linalg.solve(K, 2.0 * areas)
    GJ = 2.0 * float(np.dot(areas, q))
    return GJ, n_cells
