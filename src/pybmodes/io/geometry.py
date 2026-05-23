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

"""Build pyBmodes section properties from tubular *geometry* instead
of pre-computed structural properties.

Wind-turbine towers and monopiles are circular tubes: given the outer
diameter, wall thickness, and material, every structural property the
FEM needs is an exact closed-form expression — so the user supplies
only what they actually know (geometry) and pyBmodes derives mass /
EI / GJ / EA, eliminating the hand-computation error class
(issue #35).

For a circular tube of outer radius ``Ro`` and inner ``Ri`` made of an
isotropic material ``(E, rho, nu)``::

    A   = pi (Ro^2 - Ri^2)                      cross-section area
    I   = (pi/4) (Ro^4 - Ri^4)                  area 2nd moment (FA == SS)
    J   = 2 I                                   polar 2nd moment (tube)
    G   = E / (2 (1 + nu))                      shear modulus

    mass_den   = rho * A * outfitting_factor    kg/m   (outfitting lumps
                                                 internals / flanges /
                                                 paint into the mass)
    flp_stff   = edge_stff = E * I              N*m^2
    tor_stff   = G * J = E * I / (1 + nu)       N*m^2
    axial_stff = E * A                          N
    flp_iner   = edge_iner = rho * I            kg*m   (rotary inertia)

These are the same homogeneous-material identities the floating
section-property path uses (``axial = mass.E/rho``,
``tor = EI/(1+nu)``, ``rho.I = EI.rho/E``) — here derived forward from
geometry rather than back-solved from stiffness. ``outfitting_factor``
multiplies *only* the mass terms (it is non-structural mass), never
the stiffness — the same separation the AdjTwMa fix established.
"""

from __future__ import annotations

import warnings

import numpy as np

from pybmodes.io.sec_props import SectionProperties

# Construction-layer plausibility bands for domain-aware input validation
# (issue #102). These run on the user's RAW geometry + material — the only
# safe place for material checks, since derived ``SectionProperties`` carry
# convention-dependent placeholders (ElastoDyn towers are axially rigid, so
# their ``axial_stff`` is not the physical E·A).
#
# Bands are calibrated against EVERY WindIO reference turbine (fixed + floating)
# so they never false-positive on a validated model: across IEA-3.4 / 10 / 15 /
# 22, NREL 5MW and their monopiles/floaters, E is 200–210 GPa and ρ is
# 7800–8500 kg/m³, and D/t spans 56–1096 — the high end is the IEA-15
# VolturnUS-S *floating* tower, whose thin upper wall is legitimate (a floating
# tower carries far less bending than a fixed-bottom one). The D/t band is
# therefore a *gross* unit/geometry sanity (it catches a ×1000 D-vs-t unit
# mismatch or a near-solid / sub-mm wall), NOT a fixed-tower shell-buckling
# code check — a tight buckling band would need to know the support type
# (fixed vs floating), which this layer doesn't (tracked in #102).
_E_MIN_PA, _E_MAX_PA = 1.0e9, 1.0e12          # structural moduli (1 GPa–1 TPa)
_RHO_MIN, _RHO_MAX = 100.0, 25000.0           # kg/m³ (wood → dense ballast)
_DT_MIN, _DT_MAX = 5.0, 10000.0               # outer-diameter / wall-thickness


def _warn_implausible_tube(
    do: np.ndarray, t: np.ndarray, E: float, rho: float, title: str,
) -> None:
    """Emit ``UserWarning`` for physically implausible tube material /
    geometry — the domain-engineering guardrail a non-specialist needs
    (issue #102). Soft (warn, not raise): the values are usable, just
    almost certainly a unit error or a buckling-implausible section.
    """
    if not (_E_MIN_PA <= E <= _E_MAX_PA):
        warnings.warn(
            f"[{title}] Young's modulus E = {E:.3g} Pa is outside the "
            f"plausible structural range [{_E_MIN_PA:.0e}, {_E_MAX_PA:.0e}] Pa "
            f"(steel is ~2.0e11 Pa). Common cause: E supplied in GPa or MPa "
            f"instead of Pa.",
            UserWarning, stacklevel=3,
        )
    if not (_RHO_MIN <= rho <= _RHO_MAX):
        warnings.warn(
            f"[{title}] material density rho = {rho:.3g} kg/m^3 is outside the "
            f"plausible range [{_RHO_MIN:.0f}, {_RHO_MAX:.0f}] kg/m^3 (steel is "
            f"~7850). Common cause: density in t/m^3 or g/cm^3 instead of "
            f"kg/m^3.",
            UserWarning, stacklevel=3,
        )
    dt = do / t
    lo, hi = float(np.min(dt)), float(np.max(dt))
    if hi > _DT_MAX or lo < _DT_MIN:
        worst = hi if hi > _DT_MAX else lo
        warnings.warn(
            f"[{title}] tube diameter-to-thickness ratio D/t reaches "
            f"{worst:.0f}, outside the broad plausible range "
            f"[{_DT_MIN:.0f}, {_DT_MAX:.0f}] (real wind-turbine towers / "
            f"monopiles span ~56–1100). Likely a unit mismatch between D and "
            f"t (they must share the same length unit) or a non-physical wall "
            f"thickness.",
            UserWarning, stacklevel=3,
        )
    if do.size >= 2 and float(do[-1]) > float(do[0]) * 1.02:
        warnings.warn(
            f"[{title}] outer diameter increases from base ({float(do[0]):.2f} "
            f"m) to top ({float(do[-1]):.2f} m); a tower / monopile normally "
            f"tapers down or stays constant. Check the station ordering "
            f"(span_loc must run base → top).",
            UserWarning, stacklevel=3,
        )


def tubular_section_props(
    span_loc: np.ndarray,
    outer_diameter: np.ndarray,
    wall_thickness: np.ndarray,
    *,
    E: float,
    rho: float,
    nu: float = 0.3,
    outfitting_factor: float = 1.0,
    title: str = "geometry-derived tubular section properties",
) -> SectionProperties:
    """Exact circular-tube section properties for a steel/iso tower.

    Parameters
    ----------
    span_loc : (n,) normalised station locations in ``[0, 1]`` (root
        -> tip), strictly the same convention the solver expects.
    outer_diameter, wall_thickness : (n,) metres, per station.
    E, rho, nu : isotropic material — Young's modulus (Pa), density
        (kg/m^3), Poisson's ratio.
    outfitting_factor : non-structural mass multiplier (internals,
        flanges, paint, bolts). Multiplies the distributed mass
        density ONLY. Rotary inertia is treated as a structural
        section property and is *not* scaled (it stays ``rho *
        i_area``); stiffness is never scaled.

    Returns
    -------
    SectionProperties (FA == SS, no twist / offsets — an axisymmetric
    tube has none), ready for the FEM pipeline.
    """
    z = np.asarray(span_loc, dtype=float)
    do = np.asarray(outer_diameter, dtype=float)
    t = np.asarray(wall_thickness, dtype=float)
    if not (z.shape == do.shape == t.shape):
        raise ValueError(
            f"span_loc / outer_diameter / wall_thickness must have the "
            f"same shape; got {z.shape}, {do.shape}, {t.shape}"
        )
    if np.any(do <= 0.0) or np.any(t <= 0.0):
        raise ValueError("outer_diameter and wall_thickness must be > 0")
    if np.any(2.0 * t >= do):
        raise ValueError(
            "wall thickness must be < outer radius (2*t < outer_diameter) "
            "for every station; got a section with t >= Ro"
        )

    # Domain-aware plausibility warnings (material units, shell D/t, taper
    # direction) — caught at the construction layer on the raw inputs.
    _warn_implausible_tube(do, t, float(E), float(rho), title)

    ro = 0.5 * do
    ri = ro - t
    area = np.pi * (ro**2 - ri**2)
    i_area = 0.25 * np.pi * (ro**4 - ri**4)        # FA == SS for a tube
    g_mod = E / (2.0 * (1.0 + nu))

    mass_den = rho * area * outfitting_factor
    flp_stff = E * i_area
    tor_stff = g_mod * (2.0 * i_area)              # G*J, J = 2 I
    axial_stff = E * area
    # Rotary inertia is a physical (structural) section property -> it
    # does NOT carry the non-structural outfitting mass.
    rot_iner = rho * i_area

    zeros = np.zeros_like(z)
    return SectionProperties(
        title=title,
        n_secs=int(z.size),
        span_loc=z,
        str_tw=zeros.copy(),
        tw_iner=zeros.copy(),
        mass_den=mass_den,
        flp_iner=rot_iner,
        edge_iner=rot_iner.copy(),
        flp_stff=flp_stff,
        edge_stff=flp_stff.copy(),
        tor_stff=tor_stff,
        axial_stff=axial_stff,
        cg_offst=zeros.copy(),
        sc_offst=zeros.copy(),
        tc_offst=zeros.copy(),
    )
