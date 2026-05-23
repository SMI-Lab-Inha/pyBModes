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

"""Internal FEM pipeline shared by RotatingBlade and Tower."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pybmodes.fem.assembly import assemble, compute_element_props
from pybmodes.fem.boundary import active_dof_indices
from pybmodes.fem.nondim import make_params, nondim_platform, nondim_section_props, nondim_tip_mass
from pybmodes.fem.normalize import extract_mode_shapes
from pybmodes.fem.solver import eigvals_to_hz, solve_modes
from pybmodes.io.bmi import BMIFile, PlatformSupport, TensionWireSupport
from pybmodes.io.sec_props import SectionProperties, read_sec_props
from pybmodes.models.result import ModalResult


@dataclass
class _SectionPropsView:
    """Minimal struct of pre-nondimensionalised section-property
    columns ``compute_element_props`` reads via attribute access.

    Replaces an inline ``type('_SP', (), {...})()`` dynamic-class
    construction at the call site; the dataclass form is typed, easy
    to debug, and survives ``mypy`` without ``Any``-annotations.
    """

    span_loc:   np.ndarray
    flp_stff:   np.ndarray
    edge_stff:  np.ndarray
    tor_stff:   np.ndarray
    axial_stff: np.ndarray
    mass_den:   np.ndarray
    cg_offst:   np.ndarray
    tc_offst:   np.ndarray
    flp_iner:   np.ndarray
    edge_iner:  np.ndarray


def run_fem(
    bmi: BMIFile,
    n_modes: int = 20,
    sp: SectionProperties | None = None,
) -> ModalResult:
    """Execute the full FEM pipeline for a pre-parsed BMIFile.

    ``sp`` may be supplied directly when the section properties have been
    synthesised in memory (e.g. by an ElastoDyn adapter); otherwise they
    are read from ``bmi.resolve_sec_props_path()``.
    """
    if sp is None:
        sp = read_sec_props(bmi.resolve_sec_props_path())

    effective_rpm = bmi.rot_rpm * bmi.rpm_mult
    # For offshore towers with a submerged pile/platform, the structural beam
    # extends below MSL by draft metres; total beam = radius + draft.
    draft = bmi.support.draft if isinstance(bmi.support, PlatformSupport) else 0.0
    nd = make_params(bmi.radius, bmi.hub_rad, effective_rpm, draft=draft)

    props_nd = nondim_section_props(sp, nd, id_form=1, beam_type=bmi.beam_type)

    sc = bmi.scaling
    props_nd['mass_den']   *= sc.sec_mass
    props_nd['flp_stff']   *= sc.flp_stff
    props_nd['edge_stff']  *= sc.edge_stff
    props_nd['tor_stff']   *= sc.tor_stff
    props_nd['axial_stff'] *= sc.axial_stff
    props_nd['cg_offst']   *= sc.cg_offst
    props_nd['tc_offst']   *= sc.tc_offst
    props_nd['sq_km1']     *= sc.flp_iner
    props_nd['sq_km2']     *= sc.lag_iner

    hub_r = nd.hub_rad / nd.radius

    el, xb, cfe, eiy, eiz, gj, eac, rmas, skm1, skm2, eg, ea = compute_element_props(
        nselt   = bmi.n_elements,
        el_loc  = bmi.el_loc,
        sp      = _SectionPropsView(
            span_loc   = props_nd['sec_loc'],
            flp_stff   = props_nd['flp_stff'],
            edge_stff  = props_nd['edge_stff'],
            tor_stff   = props_nd['tor_stff'],
            axial_stff = props_nd['axial_stff'],
            mass_den   = props_nd['mass_den'],
            cg_offst   = props_nd['cg_offst'],
            tc_offst   = props_nd['tc_offst'],
            flp_iner   = sp.flp_iner,
            edge_iner  = sp.edge_iner,
        ),
        hub_r   = hub_r,
        bl_frac = nd.bl_len / nd.radius,
    )

    xmid = xb + 0.5 * el
    skm1 = np.interp(xmid, props_nd['sec_loc'], props_nd['sq_km1'])
    skm2 = np.interp(xmid, props_nd['sec_loc'], props_nd['sq_km2'])

    tip_mass_nd = None
    if bmi.tip_mass is not None and bmi.tip_mass.mass > 0.0:
        tip_mass_nd = nondim_tip_mass(bmi.tip_mass, nd,
                                      beam_type=bmi.beam_type, id_form=1,
                                      hub_conn=bmi.hub_conn)

    # Tension-wire stiffness (tow_support==1 with TensionWireSupport)
    wire_k_nd = None
    wire_node_attach = None
    if bmi.tow_support == 1 and isinstance(bmi.support, TensionWireSupport):
        sup = bmi.support
        wire_k_nd = np.array([
            sup.n_wires[i] * sup.wire_stiffness[i]
            * np.cos(np.radians(sup.th_wire[i])) ** 2
            / nd.ref1
            for i in range(sup.n_attachments)
        ])
        wire_node_attach = sup.node_attach

    # Offshore platform support.  The parser normalizes both offshore BMI
    # dialects to ``PlatformSupport``; key off the parsed support object so
    # hand-built BMIFile instances follow the same path.
    platform_nd = None
    if isinstance(bmi.support, PlatformSupport):
        platform_nd = nondim_platform(bmi.support, nd)
        # Embedded tension wires within the platform block
        plat_wires = bmi.support.wires
        if plat_wires is not None and plat_wires.n_attachments > 0:
            wire_k_nd = np.array([
                plat_wires.n_wires[i] * plat_wires.wire_stiffness[i]
                * np.cos(np.radians(plat_wires.th_wire[i])) ** 2
                / nd.ref1
                for i in range(plat_wires.n_attachments)
            ])
            wire_node_attach = plat_wires.node_attach

    hub_conn = bmi.hub_conn

    # Distributed soil/foundation stiffness (offshore only, when distr_k data present)
    elm_distr_k = None
    if isinstance(bmi.support, PlatformSupport) and len(bmi.support.distr_k) > 0:
        plat = bmi.support
        rmom2 = nd.rm * nd.romg ** 2
        # A hand-injected PlatformSupport can carry an inconsistent
        # distributed-soil array; a parsed BMI is safer but the API
        # accepts arbitrary input, so validate before it poisons the
        # FEM matrices.
        z_dk = np.asarray(plat.distr_k_z, dtype=float)
        k_dk = np.asarray(plat.distr_k, dtype=float)
        if z_dk.shape != k_dk.shape:
            raise ValueError(
                "PlatformSupport.distr_k_z and distr_k must have the "
                f"same shape; got {z_dk.shape} vs {k_dk.shape}"
            )
        if not (np.all(np.isfinite(z_dk)) and np.all(np.isfinite(k_dk))):
            raise ValueError(
                "PlatformSupport.distr_k_z / distr_k contain "
                "non-finite (NaN / inf) values"
            )
        if np.any(k_dk < 0.0):
            raise ValueError(
                "PlatformSupport.distr_k (distributed soil stiffness) "
                "must be non-negative"
            )
        # ``np.interp`` requires the sample coordinates to be
        # non-decreasing; an unsorted ``distr_k_z`` would otherwise
        # interpolate to silently wrong soil stiffness. Fail loud.
        if z_dk.size > 1 and np.any(np.diff(z_dk) < 0.0):
            raise ValueError(
                "PlatformSupport.distr_k_z must be sorted ascending "
                "for the distributed-soil-stiffness interpolation; "
                f"got {plat.distr_k_z!r}"
            )
        # z_distr_k is in metres from the flexible tower base; normalise to radius units
        z_dk_nd  = (z_dk + nd.hub_rad) / nd.radius
        # Use the validated float ndarray (``k_dk``), not the raw
        # ``plat.distr_k``: an injected Python list passes validation
        # via ``np.asarray`` but ``list / rmom2`` would raise.
        dk_nd    = k_dk / rmom2
        elm_distr_k = np.interp(xmid, z_dk_nd, dk_nd, left=0.0, right=0.0)

    gk, gm, _ = assemble(
        nselt            = bmi.n_elements,
        el               = el,
        xb               = xb,
        cfe              = cfe,
        eiy              = eiy,
        eiz              = eiz,
        gj               = gj,
        eac              = eac,
        rmas             = rmas,
        skm1             = skm1,
        skm2             = skm2,
        eg               = eg,
        ea               = ea,
        omega2           = nd.omega2,
        sec_loc          = props_nd['sec_loc'],
        str_tw           = props_nd['str_tw'],
        tip_mass         = tip_mass_nd,
        wire_k_nd        = wire_k_nd,
        wire_node_attach = wire_node_attach,
        hub_conn         = hub_conn,
        platform_nd      = platform_nd,
        elm_distr_k      = elm_distr_k,
    )

    eigvals, eigvecs, diagnostics = solve_modes(
        gk, gm, n_modes=n_modes, return_diagnostics=True,
    )
    freqs_hz = eigvals_to_hz(eigvals, nd.romg)

    active = active_dof_indices(bmi.n_elements, hub_conn)

    # Use the non-dim total radius (= bmi.radius + draft for offshore)
    # so that span_loc spans [0, 1] of the full flexible length —
    # matching the convention the polynomial fitter and Bir-style
    # plots both expect. For onshore decks (draft = 0) this equals
    # bmi.radius and the value is unchanged.
    shapes = extract_mode_shapes(
        eigvecs   = eigvecs,
        eigvals_hz= freqs_hz,
        nselt     = bmi.n_elements,
        el        = el,
        xb        = xb,
        radius    = nd.radius,
        hub_rad   = bmi.hub_rad,
        bl_len    = nd.bl_len,
        hub_conn  = hub_conn,
        active_dofs = active,
    )

    # Name the platform rigid-body modes (surge / sway / heave / roll
    # / pitch / yaw) for a free-free floating model. Cantilever /
    # monopile models have no rigid-body modes, so mode_labels stays
    # None (the classifier is never invoked).
    mode_labels = None
    if hub_conn == 2 and platform_nd is not None:
        from pybmodes.fem.platform_modes import classify_platform_modes

        mode_labels = classify_platform_modes(
            eigvecs=eigvecs,
            active_dofs=active,
            nselt=bmi.n_elements,
            platform_mass=platform_nd.mass,
            frequencies=freqs_hz,
        )

    # Record any parsed-but-not-modelled physics so the result is honest
    # about its fidelity rather than silently dropping it. Distributed
    # hydrodynamic added mass (``distr_m``) is read from the BMI but not
    # yet assembled into the mass matrix; ``distr_k`` (soil) IS assembled
    # (``elm_distr_k`` above), so it is not listed here.
    ignored: list[str] = []
    if isinstance(bmi.support, PlatformSupport) \
            and np.asarray(bmi.support.distr_m).size > 0:
        ignored.append("distributed added mass (distr_m)")

    return ModalResult(
        frequencies=freqs_hz, shapes=shapes, mode_labels=mode_labels,
        ignored_physics=tuple(ignored), diagnostics=diagnostics,
    )
