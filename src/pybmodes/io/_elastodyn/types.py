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

"""Dataclasses for the three ElastoDyn ``.dat`` file flavours.

Each dataclass holds the parsed typed fields *plus* enough raw-line
metadata (verbatim section dividers, captured-scalar map, distributed-
table header lines) for the writer to round-trip a parse → emit →
re-parse cycle to an equal dataclass. Field comments below identify
which extras serve the round-trip rather than the modal-analysis
pipeline.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ElastoDynMain:
    """Parsed top-level ElastoDyn input file."""

    # File-level metadata
    header: str
    title: str
    source_file: Optional[pathlib.Path] = None

    # Geometry / configuration
    num_bl: int = 3
    tip_rad: float = 0.0
    hub_rad: float = 0.0
    pre_cone: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hub_cm: float = 0.0
    overhang: float = 0.0
    shft_tilt: float = 0.0
    twr2shft: float = 0.0
    tower_ht: float = 0.0
    tower_bs_ht: float = 0.0

    # Nacelle CM offsets (tower-top frame, downwind/lateral/vertical)
    nac_cm_xn: float = 0.0
    nac_cm_yn: float = 0.0
    nac_cm_zn: float = 0.0

    # Initial conditions (only RotSpeed is needed for centrifugal stiffening)
    rot_speed_rpm: float = 0.0

    # Mass and inertia
    tip_mass: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hub_mass: float = 0.0
    hub_iner: float = 0.0
    gen_iner: float = 0.0
    nac_mass: float = 0.0
    nac_y_iner: float = 0.0
    yaw_br_mass: float = 0.0

    # File references
    bld_file: list[str] = field(default_factory=lambda: ["", "", ""])
    twr_file: str = ""

    # Captured raw scalar map: maps every ``LABEL`` (canonical, no parens) to
    # the verbatim value-token string from the source line. Used by the writer
    # to round-trip scalars we don't break out into typed fields.
    scalars: dict[str, str] = field(default_factory=dict)

    # Out-list (verbatim, including the END marker)
    out_list: list[str] = field(default_factory=list)
    nodal_out_list: list[str] = field(default_factory=list)

    # Section divider lines from the source, in order, for re-emit.
    section_dividers: list[str] = field(default_factory=list)

    @property
    def hub_ht(self) -> float:
        """Hub height above tower base, derived from geometry."""
        return self.tower_ht + self.twr2shft + self.overhang * math.sin(
            math.radians(self.shft_tilt)
        )

    def compute_rot_mass(self, blade: "ElastoDynBlade") -> float:
        """Total rotor mass = hub + N · AdjBlMs · ∫ BMassDen ds along
        the blade.

        Requires the blade file to integrate the distributed mass
        density. The ``AdjBlMs`` scalar from the blade file is an
        ElastoDyn-side multiplier on the entire mass distribution
        (the "blade-mass adjustment factor"); ignoring it under-
        / over-reports rotor mass on any deck where it deviates from
        1.0. The adapter at :func:`to_pybmodes_blade` already applies
        it; this method previously did not.
        """
        if blade.bl_fract.size == 0:
            return self.hub_mass
        # Trapezoidal integral of (AdjBlMs · BMassDen) over the blade
        # length.
        bl_len = self.tip_rad - self.hub_rad
        s = blade.bl_fract * bl_len
        bl_mass_per_blade = float(
            blade.adj_bl_ms * np.trapezoid(blade.b_mass_den, s)
        )
        return self.hub_mass + self.num_bl * bl_mass_per_blade


@dataclass
class ElastoDynTower:
    """Parsed ElastoDyn tower input file."""

    header: str
    title: str
    source_file: Optional[pathlib.Path] = None

    n_tw_inp_st: int = 0
    twr_fa_dmp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    twr_ss_dmp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    fa_st_tunr: list[float] = field(default_factory=lambda: [1.0, 1.0])
    ss_st_tunr: list[float] = field(default_factory=lambda: [1.0, 1.0])
    adj_tw_ma: float = 1.0
    adj_fa_st: float = 1.0
    adj_ss_st: float = 1.0

    # Distributed properties — always at least the 4 mandatory columns.
    ht_fract: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_mass_den: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_fa_stif: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_ss_stif: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Optional extra columns (not present in any of the bundled RWTs;
    # populated as zero-length arrays unless the source file carries them).
    tw_fa_iner: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_ss_iner: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_fa_cg_of: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_ss_cg_of: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Embedded mode-shape polynomial coefficients (degrees 2..6).
    tw_fa_m1_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    tw_fa_m2_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    tw_ss_m1_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    tw_ss_m2_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Verbatim column-header lines for re-emit.
    distr_header_lines: list[str] = field(default_factory=list)
    section_dividers: list[str] = field(default_factory=list)


@dataclass
class ElastoDynBlade:
    """Parsed ElastoDyn blade input file.

    ElastoDyn ships only translational mass density (``BMassDen``) and
    bending stiffnesses (``FlpStff``, ``EdgStff``) per spanwise station;
    it has no per-section *rotary* mass moments of inertia. Those live
    in BeamDyn or come from a cross-section pre-processor (VABS, PreComp).

    The :attr:`rotary_inertia_available` flag is therefore always
    ``False`` after parsing an ElastoDyn blade file. Downstream code
    (``to_pybmodes_blade``) treats the rotary inertia contributions as
    zero, which is the correct Euler-Bernoulli limit for slender blades
    and is sufficient for the bending modes (1–4 flap/edge) pyBmodes
    targets. A tiny regularisation floor is added in the section-property
    builder to keep the global mass matrix positive-definite without
    fabricating physically meaningful rotary terms.
    """

    header: str
    title: str
    source_file: Optional[pathlib.Path] = None

    n_bl_inp_st: int = 0
    bld_fl_dmp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    bld_ed_dmp: list[float] = field(default_factory=lambda: [0.0])
    fl_st_tunr: list[float] = field(default_factory=lambda: [1.0, 1.0])
    adj_bl_ms: float = 1.0
    adj_fl_st: float = 1.0
    adj_ed_st: float = 1.0

    # Distributed properties — mandatory columns plus any extras present.
    bl_fract: np.ndarray = field(default_factory=lambda: np.empty(0))
    pitch_axis: Optional[np.ndarray] = None  # present in 6-col format
    strc_twst: np.ndarray = field(default_factory=lambda: np.empty(0))
    b_mass_den: np.ndarray = field(default_factory=lambda: np.empty(0))
    flp_stff: np.ndarray = field(default_factory=lambda: np.empty(0))
    edg_stff: np.ndarray = field(default_factory=lambda: np.empty(0))

    # ElastoDyn does not carry per-section rotary inertia columns; this
    # flag stays False after parse and downstream synthesis treats the
    # rotary contributions as zero (negligible for the low bending modes
    # pyBmodes targets — see class docstring).
    rotary_inertia_available: bool = False

    # Mode-shape polynomial coefficients (degrees 2..6).
    bld_fl1_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    bld_fl2_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    bld_edg_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))

    distr_header_lines: list[str] = field(default_factory=list)
    section_dividers: list[str] = field(default_factory=list)
