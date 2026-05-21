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

"""Adapters that turn a parsed ElastoDyn bundle into pyBmodes
``BMIFile`` + ``SectionProperties`` records ready for the FEM core.

These helpers do **not** mutate the source dataclasses; they assemble
fresh ``BMIFile`` and ``SectionProperties`` objects, applying the
adjustment factors (``AdjBlMs``, ``AdjFlSt``, ``AdjEdSt``,
``AdjTwMa``, ``AdjFASt``, ``AdjSSSt``) so the downstream solver sees
the tuned stiffness/mass distribution rather than the raw deck
values.

Conventions worth keeping in mind when reading the code:

- **Rotary inertia per section** is zero in ElastoDyn input (only
  translational mass is carried). We set a tiny positive floor via
  :func:`_rotary_inertia_floor` so the global mass matrix stays
  positive-definite without fabricating physical inertia values.
- **Torsion and axial stiffness** are pinned far above bending via
  the ``_GJ_OVER_EI`` and ``_EA_OVER_EI_PER_LEN_SQ`` factors, taking
  twist and axial DOFs out of the bending-mode frequency range.
- **Tower-top RNA mass** is assembled via :func:`_tower_top_assembly_mass`,
  which sums nacelle + hub + blades via the parallel-axis theorem at
  the tower top.
"""

from __future__ import annotations

import math
import pathlib
from typing import Optional

import numpy as np

from pybmodes.io._elastodyn.types import (
    ElastoDynBlade,
    ElastoDynMain,
    ElastoDynTower,
)

# Stiffness multipliers used when synthesising the GJ / EA columns that
# ElastoDyn does not carry. These pin torsion and axial DOFs out of the
# bending mode range; raise them by another order of magnitude if a
# specific case shows torsion-bending coupling artefacts. They are safe
# ONLY for a clamped-base (cantilever / monopile) tower, where the
# base axial + torsion DOFs are locked and sit out of the bending
# band. The validated cert frequency targets (e.g.
# ``test_5mw_tower_frequency_target`` = 0.3324 Hz) depend on this path.
_GJ_OVER_EI = 100.0
_EA_OVER_EI_PER_LEN_SQ = 1.0e6  # times (mean EI), gives rigid-axial behaviour

# Physical material constants for a welded-steel tubular tower, used by
# the FREE-base floating path (``from_elastodyn_with_mooring``,
# ``hub_conn = 2``). There the proxies above (EA ≈ 1e6·EI, ~5e6× too
# stiff for a real tower) wreck the conditioning of the global matrices
# and — on an OC3-Hywind-style asymmetric platform that routes through
# the general (non-symmetric) eigensolver — collapse the soft
# rigid-body modes into an ``n_modes``-dependent degenerate cluster
# (see v1.1.1 / ``test_floating_samples_spectra``). These relations are
# exact homogeneous-section material identities, so no extra geometry
# input is needed:
#   axial_stff = E·A      = (ρ·A)·(E/ρ)      = mass_den · (E/ρ)
#   flp_iner   = ρ·I       = (E·I)·(ρ/E)      = flp_stff · (ρ/E)
#   tor_stff   = G·J       = (E·I)·(G/E)(J/I) = EI / (1+ν)
# The last uses J/I = 2 (thin-wall circular tube). ρ = 8500 kg/m³ is
# the effective-density convention (structural steel inflated to
# absorb bolts / flanges / paint) matching the OC3 Hywind reference
# tower; all three reproduce that deck's published section table to
# the printed digits. Mirrors ``build.py``'s floating emitter.
_STEEL_E = 210.0e9          # Young's modulus (Pa)
_STEEL_RHO = 8500.0         # effective density incl. secondary mass (kg/m³)
_POISSON = 0.3              # Poisson's ratio
_E_OVER_RHO = _STEEL_E / _STEEL_RHO
_RHO_OVER_E = _STEEL_RHO / _STEEL_E
_GJ_OVER_EI_TUBE = 1.0 / (1.0 + _POISSON)   # (G/E)·(J/I) = [1/2(1+ν)]·2


# Forward references resolved lazily inside function bodies to avoid
# import cycles with ``pybmodes.io.bmi`` and ``pybmodes.io.sec_props``.
if False:  # pragma: no cover
    from pybmodes.io.bmi import BMIFile, TipMassProps  # noqa: F401
    from pybmodes.io.sec_props import SectionProperties  # noqa: F401


def _resolve_relative(main: ElastoDynMain, ref: str) -> pathlib.Path:
    """Resolve a path possibly given relative to the main-file directory."""
    p = pathlib.Path(ref)
    if p.is_absolute():
        return p
    if main.source_file is not None:
        return (main.source_file.parent / p).resolve()
    return p.resolve()


def _rotary_inertia_floor(
    mass_den: np.ndarray,
    char_length: float,
) -> np.ndarray:
    """Strictly-positive regularisation for the rotary-inertia columns.

    ElastoDyn carries no per-section rotary mass moments of inertia. The
    physically correct value for an Euler-Bernoulli beam at the bending
    frequencies pyBmodes resolves is *zero* — rotary inertia is a
    higher-order Timoshenko correction that's negligible for slender
    structures. However, the global mass matrix needs every diagonal
    block strictly positive to stay positive-definite, so we set a tiny
    floor here. ``1e-6 · mass_den · L²`` keeps the floor at the parts-per-
    million level relative to translational mass while killing the
    singularity. ``L`` is a per-element characteristic length supplied
    by the caller (chord proxy for blades, mean radius for towers).
    """
    return np.full_like(mass_den, 1.0e-6 * char_length ** 2) * mass_den


def _stack_blade_section_props(
    blade: ElastoDynBlade,
    rot_rpm: float,  # noqa: ARG001 — kept for parity with the tower variant
    chord_estimate: float = 4.0,
) -> "SectionProperties":  # forward-ref — imported lazily to dodge cycles
    """Convert an ElastoDyn blade record to pyBmodes section properties.

    Rotary mass moments of inertia (``flp_iner``, ``edge_iner``) are
    physically zero for a thin-beam Euler-Bernoulli model; we set them
    to a tiny PD-safety floor only. See :func:`_rotary_inertia_floor`.
    """
    from pybmodes.io.sec_props import SectionProperties

    span = blade.bl_fract.astype(float)
    str_tw = blade.strc_twst.astype(float)
    mass_den = blade.b_mass_den.astype(float) * blade.adj_bl_ms
    flp_stff = blade.flp_stff.astype(float) * blade.adj_fl_st
    edg_stff = blade.edg_stff.astype(float) * blade.adj_ed_st

    ei_max = np.maximum(flp_stff, edg_stff)
    tor_stff = ei_max * _GJ_OVER_EI
    axial_stff = ei_max * _EA_OVER_EI_PER_LEN_SQ

    # Per-spec: rotary inertia is zero for thin beams; PD floor only.
    flp_iner = _rotary_inertia_floor(mass_den, chord_estimate)
    edge_iner = _rotary_inertia_floor(mass_den, chord_estimate)
    zeros = np.zeros_like(span)

    return SectionProperties(
        title="ElastoDyn-derived blade section properties",
        n_secs=int(span.size),
        span_loc=span,
        str_tw=str_tw,
        tw_iner=str_tw.copy(),  # ElastoDyn lacks an independent inertia twist
        mass_den=mass_den,
        flp_iner=flp_iner,
        edge_iner=edge_iner,
        flp_stff=flp_stff,
        edge_stff=edg_stff,
        tor_stff=tor_stff,
        axial_stff=axial_stff,
        cg_offst=zeros.copy(),
        sc_offst=zeros.copy(),
        tc_offst=zeros.copy(),
        source_file=blade.source_file,
    )


def _stack_tower_section_props(
    tower: ElastoDynTower,
    radius_estimate: float = 3.0,
    *,
    physical: bool = False,
) -> "SectionProperties":
    """Convert an ElastoDyn tower record to pyBmodes section properties.

    ``physical=False`` (default) synthesises torsion / axial stiffness
    from the large ``_GJ_OVER_EI`` / ``_EA_OVER_EI_PER_LEN_SQ`` proxies
    and a tiny rotary-inertia floor. Correct ONLY for a clamped-base
    cantilever / monopile (``from_elastodyn`` /
    ``from_elastodyn_with_subdyn``), where the base axial + torsion
    DOFs are locked and out of the bending band — the validated cert
    frequency targets depend on this path, so it is unchanged.

    ``physical=True`` derives torsion / axial / rotary inertia from the
    exact homogeneous-steel material identities (see the ``_STEEL_*``
    constants). Required for the FREE-base floating path
    (``from_elastodyn_with_mooring``, ``hub_conn = 2``): with the
    proxies the axial column is ~5e6× too stiff, which on an
    OC3-Hywind-style asymmetric platform collapses the soft rigid-body
    modes into an ``n_modes``-dependent degenerate cluster. The
    bundled-sample equivalent of this fix shipped in v1.1.1; this
    extends it to the in-memory ``from_elastodyn_with_mooring`` path so
    a user's own asymmetric spar/semi deck is solved consistently.
    """
    from pybmodes.io.sec_props import SectionProperties

    span = tower.ht_fract.astype(float)
    struct_mass_den = tower.t_mass_den.astype(float)
    mass_den = struct_mass_den * tower.adj_tw_ma
    flp_stff = tower.tw_fa_stif.astype(float) * tower.adj_fa_st
    edg_stff = tower.tw_ss_stif.astype(float) * tower.adj_ss_st

    ei_max = np.maximum(flp_stff, edg_stff)
    if physical:
        # Exact material identities for a homogeneous steel section
        # (thin-wall tube for the torsion J/I = 2). No geometry input
        # needed; reproduces the canonical OC3 Hywind section table to
        # the printed digits.
        #
        # axial_stff = E·A is derived from the *structural* mass
        # density (ρ·A), i.e. the UN-adjusted ``t_mass_den``. AdjTwMa
        # is purely a mass-density calibration knob — it does not
        # change the cross-sectional area, so it must not bleed into
        # the axial stiffness. (Pre-fix it used the AdjTwMa-scaled
        # mass_den, so a deck tuning tower mass via AdjTwMa would
        # silently soften/stiffen the axial DOF and could reintroduce
        # the bad conditioning this whole path exists to avoid.) The
        # adjusted ``mass_den`` still feeds the mass matrix; only the
        # E·A *stiffness* uses the structural value. Decks with
        # AdjTwMa = 1 (all current references) are unchanged.
        tor_stff = ei_max * _GJ_OVER_EI_TUBE
        axial_stff = struct_mass_den * _E_OVER_RHO
        flp_iner = flp_stff * _RHO_OVER_E
        edge_iner = edg_stff * _RHO_OVER_E
    else:
        tor_stff = ei_max * _GJ_OVER_EI
        axial_stff = ei_max * _EA_OVER_EI_PER_LEN_SQ
        flp_iner = _rotary_inertia_floor(mass_den, radius_estimate)
        edge_iner = _rotary_inertia_floor(mass_den, radius_estimate)
    zeros = np.zeros_like(span)

    return SectionProperties(
        title="ElastoDyn-derived tower section properties",
        n_secs=int(span.size),
        span_loc=span,
        str_tw=zeros.copy(),
        tw_iner=zeros.copy(),
        mass_den=mass_den,
        flp_iner=flp_iner,
        edge_iner=edge_iner,
        flp_stff=flp_stff,
        edge_stff=edg_stff,
        tor_stff=tor_stff,
        axial_stff=axial_stff,
        cg_offst=zeros.copy(),
        sc_offst=zeros.copy(),
        tc_offst=zeros.copy(),
        source_file=tower.source_file,
    )


def _build_bmi_skeleton(
    *,
    title: str,
    beam_type: int,
    radius: float,
    hub_rad: float,
    rot_rpm: float,
    precone: float,
    n_elements: int,
    el_loc: np.ndarray,
    tip_mass_props: "TipMassProps",
) -> "BMIFile":
    from pybmodes.io.bmi import BMIFile, ScalingFactors

    return BMIFile(
        title=title,
        echo=False,
        beam_type=beam_type,
        rot_rpm=rot_rpm,
        rpm_mult=1.0,
        radius=radius,
        hub_rad=hub_rad,
        precone=precone,
        bl_thp=0.0,
        hub_conn=1,
        n_modes_print=20,
        tab_delim=True,
        mid_node_tw=False,
        tip_mass=tip_mass_props,
        id_mat=1,
        sec_props_file="",  # in-memory; not resolved
        scaling=ScalingFactors(),
        n_elements=n_elements,
        el_loc=el_loc.astype(float),
        tow_support=0,
        support=None,
        source_file=None,
    )


def _tower_top_assembly_mass(
    main: ElastoDynMain,
    blade: Optional[ElastoDynBlade],
) -> "TipMassProps":
    """Lump the rotor-nacelle assembly (RNA) into a single ``TipMassProps``
    at the tower top via full rigid-body parallel-axis assembly.

    Bodies summed (each at its CM in the tower-top reference frame
    ``x = downwind, y = lateral, z = vertical``):

    1. **Nacelle** — mass ``NacMass`` at ``(NacCMxn, NacCMyn, NacCMzn)``.
       Inertia tensor: ``NacYIner`` about the yaw (z) axis; transverse
       components default to ``½·NacYIner`` as a slender-body
       approximation (ElastoDyn does not carry separate Ixx/Iyy for the
       nacelle).
    2. **Hub** — mass ``HubMass`` at the hub-mass location, which is the
       rotor apex translated by ``HubCM`` along the shaft axis. The shaft
       is tilted by ``ShftTilt`` from horizontal and the apex itself is
       at ``(OverHang·cos(ShftTilt), 0, Twr2Shft + OverHang·sin(ShftTilt))``
       relative to the tower top.
    3. **Blades** — total mass ``N_bl × ∫BMassDen ds`` from the blade
       file, treated as a point mass at the rotor apex. Distributed-blade
       rotational inertia is dropped — it's <1% of the parallel-axis
       contribution from the apex offset for a utility-scale RNA.

    Spec deviation: the prompt says ``m_total = NacMass + HubMass + 3·TipMass``,
    but in ElastoDyn ``TipMass(i)`` is the per-blade *tip-brake* mass
    (zero for the 5MW deck). Using that literally would drop the blade
    mass entirely. We use ``N_bl·∫BMassDen ds`` instead, matching what
    ElastoDyn itself computes as ``RotMass``.

    Inertia at the tower top is then ``∑_i [I_i + m_i·(|r_i|²·E - r_i⊗r_i)]``
    (parallel-axis theorem in tensor form). Cross-products are
    preserved. The diagonal and off-diagonal entries are passed through
    as ``ixx, iyy, izz, ixy, izx, iyz`` on the BMI tip-mass record;
    ``cm_offset`` and ``cm_axial`` are set to zero so the downstream
    nondimensionaliser does not re-apply parallel-axis terms (the
    tensor we hand it is already at the tower top).
    """
    from pybmodes.io.bmi import TipMassProps

    tilt = math.radians(main.shft_tilt)
    cos_t = math.cos(tilt)
    sin_t = math.sin(tilt)

    # --- Body 1: Nacelle ---
    m_nac = float(main.nac_mass)
    r_nac = np.array([main.nac_cm_xn, main.nac_cm_yn, main.nac_cm_zn], dtype=float)
    # Slender-body proxy: half of NacYIner on the transverse axes. For a
    # rectangular block this is ~Iy ≈ Iz/3 + small; ½ is a generous middle.
    I_nac = np.diag([0.5 * main.nac_y_iner, 0.5 * main.nac_y_iner, main.nac_y_iner])

    # --- Body 2: Hub ---
    m_hub = float(main.hub_mass)
    apex = np.array([main.overhang * cos_t, 0.0, main.twr2shft + main.overhang * sin_t])
    shaft_dir = np.array([cos_t, 0.0, sin_t])
    r_hub = apex + main.hub_cm * shaft_dir
    # HubIner is the inertia about the shaft (rotor) axis. Other axes
    # default to ½·HubIner (sphere-like proxy). For small ShftTilt the
    # shaft is nearly aligned with x, so we approximate the hub tensor as
    # diagonal in the tower-top frame; a proper rotation by ShftTilt is
    # left as a tightening target.
    I_hub = np.diag([main.hub_iner, 0.5 * main.hub_iner, 0.5 * main.hub_iner])

    # --- Body 3: Blades (lumped at the apex) ---
    # ElastoDyn does not carry per-section rotational inertia, and the
    # standard rigid-RNA approximation used to derive published reference
    # frequencies (e.g. Jonkman 2009 Tower FA = 0.3240 Hz) treats the
    # blades as a translational point mass at the rotor apex — the blade
    # rotational dynamics live in their own DOFs in OpenFAST and don't
    # rigidly couple to tower-mode rotation at low frequencies. Mirroring
    # that convention here keeps the tower modal frequencies close to the
    # published targets without fabricating distributed-blade inertia
    # data that ElastoDyn never had.
    if blade is not None and blade.bl_fract.size > 0:
        bl_len = main.tip_rad - main.hub_rad
        bmd = blade.b_mass_den * blade.adj_bl_ms
        s = blade.bl_fract * bl_len
        m_bl_each = float(np.trapezoid(bmd, s))
    else:
        m_bl_each = 0.0
    m_blades = main.num_bl * m_bl_each
    r_blades = apex.copy()
    I_blades = np.zeros((3, 3))

    bodies = [
        (m_nac,     r_nac,     I_nac),
        (m_hub,     r_hub,     I_hub),
        (m_blades,  r_blades,  I_blades),
    ]

    m_total = sum(m for m, _, _ in bodies)
    if m_total <= 0.0:
        return TipMassProps(
            mass=0.0, cm_offset=0.0, cm_axial=0.0,
            ixx=0.0, iyy=0.0, izz=0.0,
            ixy=0.0, izx=0.0, iyz=0.0,
        )

    # Assembly CM (relative to tower top, in tower-top frame). The
    # explicit ``start=np.zeros(3)`` keeps the static type at ndarray
    # even when ``bodies`` happens to be empty (handled above by the
    # m_total guard, but mypy doesn't see that path).
    cm: np.ndarray = sum(
        (m * r for m, r, _ in bodies),
        start=np.zeros(3),
    ) / m_total

    # Inertia tensor at the tower top via parallel-axis theorem.
    eye = np.eye(3)
    I_tt = np.zeros((3, 3))
    for m, r, I_body in bodies:
        rsq = float(r @ r)
        I_tt = I_tt + I_body + m * (rsq * eye - np.outer(r, r))

    return TipMassProps(
        mass=m_total,
        # cm_axial carries the BMI cm-axial lever arm used for kinematic
        # coupling terms (translation/rotation cross-blocks). cm_offset
        # is zeroed because the tower path drops the BMI horizontal
        # offset — the horizontal contribution is folded into ``ixx``,
        # ``iyy``, ``izz`` via the tensor parallel-axis above.
        cm_offset=0.0,
        cm_axial=float(cm[2]),
        ixx=float(I_tt[0, 0]),
        iyy=float(I_tt[1, 1]),
        izz=float(I_tt[2, 2]),
        ixy=float(I_tt[0, 1]),
        izx=float(I_tt[2, 0]),
        iyz=float(I_tt[1, 2]),
    )


_DUPLICATE_STATION_TOL = 1.0e-4
"""HtFract gap below which two adjacent stations are treated as a
property-step encoding (duplicate-pair trick used by some preprocessors,
e.g. the IFE UPSCALE 25 MW deck). Stations spaced this tightly would
produce mm-scale FEM elements with catastrophic conditioning."""


def _tower_element_boundaries(ht_fract: np.ndarray) -> np.ndarray:
    """Return FEM element boundaries for a tower station list.

    Most ElastoDyn tower decks list stations that are well-separated, so
    we use them directly as FEM element boundaries. Some preprocessors
    encode property-step discontinuities as pairs of near-coincident
    stations (HtFract gap ~ 1e-5). Using those as FEM nodes produces
    mm-scale elements that wreck the conditioning of the bending
    stiffness matrix — the resulting spectrum collapses to spurious
    zero eigenvalues plus a degenerate high-frequency pair. When this
    pattern is detected we switch to a uniform mesh; the full station
    list (with the step) stays in ``SectionProperties.span_loc`` and is
    sampled at element midpoints in :func:`compute_element_props`, so
    the step semantics survive.
    """
    ht = np.asarray(ht_fract, dtype=float)
    if ht.size < 2:
        return ht
    if float(np.diff(ht).min()) < _DUPLICATE_STATION_TOL:
        n_elements = max(int(ht.size) - 1, 1)
        return np.linspace(float(ht[0]), float(ht[-1]), n_elements + 1)
    return ht


def to_pybmodes_tower(
    main: ElastoDynMain,
    tower: ElastoDynTower,
    blade: Optional[ElastoDynBlade] = None,
    *,
    physical_sec_props: bool = False,
) -> tuple["BMIFile", "SectionProperties"]:
    """Build pyBmodes ``BMIFile`` and ``SectionProperties`` for tower modal
    analysis from a parsed ElastoDyn bundle. ``blade`` is optional; when
    omitted, the rotor mass is approximated as ``HubMass`` only.

    ``physical_sec_props`` selects the section-property synthesis (see
    :func:`_stack_tower_section_props`): ``False`` (default) for the
    clamped-base cantilever / monopile path; ``True`` for the free-base
    floating path, where the cantilever proxies wreck conditioning.
    """
    sp = _stack_tower_section_props(tower, physical=physical_sec_props)
    tip = _tower_top_assembly_mass(main, blade)

    el_loc = _tower_element_boundaries(tower.ht_fract)
    flexible_height = main.tower_ht - main.tower_bs_ht
    bmi = _build_bmi_skeleton(
        title=main.title or "ElastoDyn tower",
        beam_type=2,
        radius=flexible_height,
        hub_rad=0.0,
        rot_rpm=0.0,
        precone=0.0,
        n_elements=max(el_loc.size - 1, 1),
        el_loc=el_loc,
        tip_mass_props=tip,
    )
    return bmi, sp


def to_pybmodes_blade(
    main: ElastoDynMain,
    blade: ElastoDynBlade,
) -> tuple["BMIFile", "SectionProperties"]:
    """Build pyBmodes ``BMIFile`` and ``SectionProperties`` for blade modal
    analysis at the operating ``RotSpeed`` from the main file."""
    from pybmodes.io.bmi import TipMassProps

    sp = _stack_blade_section_props(blade, rot_rpm=main.rot_speed_rpm)

    tip = TipMassProps(
        mass=main.tip_mass[0],
        cm_offset=0.0, cm_axial=0.0,
        ixx=0.0, iyy=0.0, izz=0.0,
        ixy=0.0, izx=0.0, iyz=0.0,
    )

    bmi = _build_bmi_skeleton(
        title=main.title or "ElastoDyn blade",
        beam_type=1,
        radius=main.tip_rad,
        hub_rad=main.hub_rad,
        rot_rpm=main.rot_speed_rpm,
        precone=main.pre_cone[0],
        n_elements=max(blade.n_bl_inp_st - 1, 1),
        el_loc=blade.bl_fract,
        tip_mass_props=tip,
    )
    return bmi, sp
