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

"""Mudline coupled-spring foundation for soft monopile soil-pile interaction.

The coupled-spring (CS) model represents the soil-pile reaction at the
mudline as three springs: a lateral spring ``K_hh``, a rotational
spring ``K_rr``, and a cross-coupling ``K_hr``. The model is well
established for monopile-supported offshore wind turbines and is
endorsed by Yu and Amdahl (2023) for first three modes when the
spring stiffnesses are calculated properly.

This module wires closed-form formulas for K_hh, K_hr, K_rr from pile
geometry and soil properties into a 6x6 mooring_K block that drops
straight into :class:`pybmodes.io.bmi.PlatformSupport` of a
``hub_conn = 3`` soft-monopile BMI. The dispatch covers two formula
families and three soil profiles, classifying pile behaviour via
Randolph (1981).

Scope. ``MudlineFoundation`` produces the linearised mudline stiffness
used for coupled-frequency prediction. For ElastoDyn polynomial
coefficient generation the cantilever path
(:meth:`pybmodes.models.Tower.from_elastodyn` or
:meth:`pybmodes.models.Tower.from_geometry`) is still required
regardless of soil flexibility. The mudline stiffness affects the
coupled-system frequency but NOT the polynomial basis. ElastoDyn's
SHP ansatz requires clamped-base mode shapes
(``src/pybmodes/_examples/reference_decks/FLOATING_CASES.md`` records
the source-code citations and ``cases/ECOSYSTEM_FINDING.md`` the
audit trail).

References
----------
- Yu, Z. and Amdahl, J. (2023). A Rayleigh-Ritz solution for high
  order natural frequencies and eigenmodes of monopile supported
  offshore wind turbines considering tapered towers and soil-pile
  interactions. *Marine Structures* 92, 103482.
  https://doi.org/10.1016/j.marstruc.2023.103482
- Shadlou, M. and Bhattacharya, S. (2016). Dynamic stiffness of
  monopiles supporting offshore wind turbine generators.
  *Soil Dynamics and Earthquake Engineering* 88, 15-32.
- Psaroudakis, E. G., Mylonakis, G. and Antonopoulos, A. (2021).
  Analytical formulas for the lateral response of monopiles in
  homogeneous Winkler-type foundations. (As cited in Yu and Amdahl
  2023, Eq 25.)
- Randolph, M. F. (1981). The response of flexible piles to lateral
  loading. *Geotechnique* 31, 247-259.
"""

from __future__ import annotations

import math
import pathlib
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

SoilProfile = Literal["homogeneous", "parabolic", "linear"]
PileBehaviour = Literal["flexible", "rigid"]
FormulaFamily = Literal["shadlou", "psaroudakis"]


_SHADLOU_FLEXIBLE = {
    "homogeneous": {
        "K_hh": (2.9, 0.186),
        "K_hr": (1.2, 0.5),
        "K_rr": (1.5, 0.73),
    },
    "parabolic": {
        "K_hh": (2.03, 0.27),
        "K_hr": (1.17, 0.52),
        "K_rr": (1.42, 0.76),
    },
    "linear": {
        "K_hh": (1.58, 0.34),
        "K_hr": (1.07, 0.567),
        "K_rr": (1.38, 0.78),
    },
}

_SHADLOU_RIGID = {
    "homogeneous": {
        "K_hh": (6.4, 0.62),
        "K_hr": (7.1, 1.56),
        "K_rr": (13.2, 2.5),
    },
    "parabolic": {
        "K_hh": (5.33, 1.07),
        "K_hr": (7.2, 2.0),
        "K_rr": (13.0, 3.0),
    },
    "linear": {
        "K_hh": (4.7, 1.53),
        "K_hr": (7.1, 2.5),
        "K_rr": (12.7, 3.45),
    },
}


def _f_nu(nu_s: float) -> float:
    """Soil Poisson-ratio factor from Shadlou and Bhattacharya (2016)."""
    return 1.0 + 0.6 * abs(nu_s - 0.25)


def _equivalent_pile_E(pile_EI: float, pile_diameter: float) -> float:
    """``E_pe`` per Yu and Amdahl (2023) page 5.

    Treats the monopile as a solid cylinder for the equivalent
    Young's modulus regardless of wall thickness. This matches the
    convention in Shadlou and Bhattacharya (2016) on which the
    Table 1 coefficients are calibrated.
    """
    return pile_EI / (math.pi * pile_diameter**4 / 64.0)


def _classify_pile(
    pile_diameter: float,
    pile_length_embedded: float,
    pile_EI: float,
    soil_E: float,
    soil_nu: float,
) -> tuple[str, float, float]:
    """Apply Randolph (1981) pile-behaviour classification.

    Returns ``(label, ratio, threshold_flexible)`` where ``label`` is
    one of ``"flexible"``, ``"rigid"``, or ``"intermediate"`` and the
    two numerical values are the L/D ratio and the flexible threshold
    so a caller can report them in a warning.
    """
    G_s = soil_E / (2.0 * (1.0 + soil_nu))
    G_star = G_s * (1.0 + 3.0 * soil_nu / 4.0)
    E_pe = _equivalent_pile_E(pile_EI, pile_diameter)
    ratio_LD = pile_length_embedded / pile_diameter
    stiffness_ratio = E_pe / G_star
    threshold_flexible = stiffness_ratio ** (2.0 / 7.0)
    threshold_rigid = 0.05 * stiffness_ratio ** (1.0 / 2.0)
    if ratio_LD >= threshold_flexible:
        return "flexible", ratio_LD, threshold_flexible
    if ratio_LD <= threshold_rigid:
        return "rigid", ratio_LD, threshold_flexible
    return "intermediate", ratio_LD, threshold_flexible


def _shadlou(
    behaviour: PileBehaviour,
    soil_profile: SoilProfile,
    pile_diameter: float,
    pile_length_embedded: float,
    pile_EI: float,
    soil_E: float,
    soil_nu: float,
) -> tuple[float, float, float]:
    """Shadlou and Bhattacharya (2016) formulas via Yu Table 1."""
    f_nu_value = _f_nu(soil_nu)
    radius = pile_diameter / 2.0
    if behaviour == "flexible":
        coeffs = _SHADLOU_FLEXIBLE[soil_profile]
        E_pe = _equivalent_pile_E(pile_EI, pile_diameter)
        x = E_pe / soil_E
    else:
        coeffs = _SHADLOU_RIGID[soil_profile]
        x = pile_length_embedded / pile_diameter

    a_hh, b_hh = coeffs["K_hh"]
    a_hr, b_hr = coeffs["K_hr"]
    a_rr, b_rr = coeffs["K_rr"]

    K_hh = f_nu_value * soil_E * radius * a_hh * x**b_hh
    K_hr = -f_nu_value * soil_E * radius**2 * a_hr * x**b_hr
    K_rr = f_nu_value * soil_E * radius**3 * a_rr * x**b_rr
    return K_hh, K_hr, K_rr


def _psaroudakis(
    pile_diameter: float,
    pile_length_embedded: float,
    pile_EI: float,
    soil_E: float,
) -> tuple[float, float, float]:
    """Psaroudakis et al. (2021) closed form, Yu and Amdahl Eq 25.

    Homogeneous soil only. ``k_sub`` (subgrade reaction modulus) is
    taken equal to ``E_SO`` per the homogeneous Winkler-foundation
    assumption stated in Yu and Amdahl (2023) Eq 25.
    """
    k_sub = soil_E
    beta = (k_sub * pile_diameter / (4.0 * pile_EI)) ** 0.25
    bL = beta * pile_length_embedded
    two_bL = 2.0 * bL

    sin_2 = math.sin(two_bL)
    cos_2 = math.cos(two_bL)
    sinh_2 = math.sinh(two_bL)
    cosh_2 = math.cosh(two_bL)

    denom = 2.0 + cos_2 + cosh_2

    K_hh = 4.0 * pile_EI * beta**3 * (sin_2 + sinh_2) / denom
    K_hr = -2.0 * pile_EI * beta**2 * (-cos_2 + cosh_2) / denom
    K_rr = 2.0 * pile_EI * beta * (-sin_2 + sinh_2) / denom
    return K_hh, K_hr, K_rr


@dataclass
class MudlineFoundation:
    """Three coupled springs at the mudline for a monopile foundation.

    Spring convention matches Eq (3) of Yu and Amdahl (2023): the
    mudline force-moment vector is ``[F, M] = [[K_hh, K_hr],
    [K_hr, K_rr]] @ [rho, theta]`` where ``rho`` is the mudline
    lateral displacement and ``theta`` the mudline rotation in the
    same 2-D plane. By the right-hand convention used in OpenFAST
    (Jonkman 2010 NREL/TP-500-47535 Table 5-1), the K_hr term is
    negative for typical sands and clays.

    The :meth:`as_mooring_K` accessor maps the 2-D coupled-spring
    matrix to the OpenFAST 6-DOF order
    ``[surge, sway, heave, roll, pitch, yaw]`` and is symmetric.
    Heave and yaw are not modelled by this CS surrogate and stay at
    zero in the returned matrix. Wire it into the soft monopile via
    :class:`pybmodes.io.bmi.PlatformSupport.mooring_K` of a BMI built
    for ``hub_conn = 3``.
    """

    K_hh: float
    K_hr: float
    K_rr: float
    pile_behaviour: PileBehaviour
    soil_profile: SoilProfile
    formula: FormulaFamily

    @property
    def pile_behavior(self) -> PileBehaviour:
        """US-spelling alias preserved for prompt-style external code."""
        return self.pile_behaviour

    def as_mooring_K(self) -> np.ndarray:
        """Return the 6x6 mudline stiffness in OpenFAST DOF order.

        Mapping (Eq 3 of Yu and Amdahl 2023 lifted into 6 DOFs):

        - ``K[0, 0] = K[1, 1] = K_hh`` (lateral on surge and sway, the
          three-fold axisymmetric assumption that pins both diagonals
          to the same value).
        - ``K[3, 3] = K[4, 4] = K_rr`` (rotational on roll and pitch).
        - ``K[0, 4] = K[4, 0] = K_hr`` (surge-pitch coupling).
        - ``K[1, 3] = K[3, 1] = -K_hr`` (sway-roll coupling, opposite
          sign by right-hand rule).
        - ``K[2, 2] = 0`` (heave not modelled by the CS surrogate).
        - ``K[5, 5] = 0`` (yaw not modelled by the CS surrogate).

        The cross-coupling sign convention is pinned by
        ``tests/test_mooring.py::test_oc3hywind_bmi_dof_order_matches_jonkman_2010``
        against Jonkman (2010) OC3 Table 5-1, which reports
        ``K_15 = -2.821e6`` N for surge-pitch and ``K_24 = +2.816e6``
        N for sway-roll.
        """
        K = np.zeros((6, 6))
        K[0, 0] = self.K_hh
        K[1, 1] = self.K_hh
        K[3, 3] = self.K_rr
        K[4, 4] = self.K_rr
        K[0, 4] = self.K_hr
        K[4, 0] = self.K_hr
        K[1, 3] = -self.K_hr
        K[3, 1] = -self.K_hr
        return K

    @classmethod
    def from_soil_properties(
        cls,
        pile_diameter: float,
        pile_length_embedded: float,
        pile_EI: float,
        soil_E: float,
        soil_nu: float = 0.3,
        soil_profile: SoilProfile = "homogeneous",
        pile_behaviour: str = "auto",
        formula: FormulaFamily = "shadlou",
    ) -> MudlineFoundation:
        """Compute K_hh, K_hr, K_rr from pile geometry and soil properties.

        Parameters
        ----------
        pile_diameter
            Outer diameter of the monopile, ``D_P`` in m.
        pile_length_embedded
            Embedded pile length, ``L_P`` in m.
        pile_EI
            Pile bending stiffness ``E_P * I_P`` in N m^2. For a
            tubular pile with diameter D and wall thickness t,
            ``I_P = pi / 64 * (D^4 - (D - 2 t)^4)``.
        soil_E
            Soil Young's modulus ``E_SO`` in Pa. For an inhomogeneous
            profile this is the reference modulus at the depth used by
            the chosen formula family.
        soil_nu
            Soil Poisson's ratio. Default 0.3.
        soil_profile
            One of ``"homogeneous"`` (constant E_SO with depth),
            ``"parabolic"`` (E_SO proportional to sqrt of depth),
            ``"linear"`` (E_SO proportional to depth).
        pile_behaviour
            One of ``"flexible"``, ``"rigid"``, ``"auto"``. When set
            to ``"auto"``, the pile is classified per Randolph (1981).
            An intermediate L/D ratio falls back to the flexible
            formulas with a ``UserWarning``.
        formula
            ``"shadlou"`` (default) uses Shadlou and Bhattacharya
            (2016) per Yu Table 1 and covers all three soil profiles.
            ``"psaroudakis"`` uses Yu Eq 25 and is restricted to the
            homogeneous profile.

        Returns
        -------
        :class:`MudlineFoundation`
            Coupled-spring stiffness in SI units.

        Raises
        ------
        ValueError
            On non-positive geometry or soil parameters, an unknown
            ``soil_profile`` or ``pile_behaviour`` token, or
            ``formula="psaroudakis"`` paired with a non-homogeneous
            soil profile.
        """
        if pile_diameter <= 0.0:
            raise ValueError("pile_diameter must be positive")
        if pile_length_embedded <= 0.0:
            raise ValueError("pile_length_embedded must be positive")
        if pile_EI <= 0.0:
            raise ValueError("pile_EI must be positive")
        if soil_E <= 0.0:
            raise ValueError("soil_E must be positive")
        if soil_nu <= -1.0 or soil_nu >= 0.5:
            raise ValueError(
                "soil_nu must satisfy -1 < nu < 0.5 (Poisson-ratio bound)"
            )
        if soil_profile not in {"homogeneous", "parabolic", "linear"}:
            raise ValueError(
                f"soil_profile must be 'homogeneous', 'parabolic' or "
                f"'linear', got {soil_profile!r}"
            )
        if pile_behaviour not in {"flexible", "rigid", "auto"}:
            raise ValueError(
                f"pile_behaviour must be 'flexible', 'rigid' or 'auto', "
                f"got {pile_behaviour!r}"
            )
        if formula not in {"shadlou", "psaroudakis"}:
            raise ValueError(
                f"formula must be 'shadlou' or 'psaroudakis', got "
                f"{formula!r}"
            )
        if formula == "psaroudakis" and soil_profile != "homogeneous":
            raise ValueError(
                "Psaroudakis et al. (2021) formula is derived for "
                "homogeneous soil only. Pair it with "
                "soil_profile='homogeneous' or switch to the Shadlou "
                "formula which covers all three profiles."
            )

        if pile_behaviour == "auto":
            label, ratio_LD, threshold_flex = _classify_pile(
                pile_diameter, pile_length_embedded, pile_EI,
                soil_E, soil_nu,
            )
            if label == "intermediate":
                warnings.warn(
                    f"Randolph (1981) pile classification is "
                    f"intermediate (L/D = {ratio_LD:.2f}, flexible "
                    f"threshold = {threshold_flex:.2f}). Falling back "
                    f"to flexible formulas; consider verifying "
                    f"against a higher-fidelity P-y model.",
                    UserWarning,
                    stacklevel=2,
                )
                behaviour: PileBehaviour = "flexible"
            else:
                behaviour = label  # type: ignore[assignment]
        else:
            behaviour = pile_behaviour  # type: ignore[assignment]

        if formula == "shadlou":
            K_hh, K_hr, K_rr = _shadlou(
                behaviour, soil_profile, pile_diameter,
                pile_length_embedded, pile_EI, soil_E, soil_nu,
            )
        else:
            K_hh, K_hr, K_rr = _psaroudakis(
                pile_diameter, pile_length_embedded, pile_EI, soil_E,
            )

        return cls(
            K_hh=K_hh,
            K_hr=K_hr,
            K_rr=K_rr,
            pile_behaviour=behaviour,
            soil_profile=soil_profile,
            formula=formula,
        )

    @classmethod
    def from_windio(
        cls,
        yaml_path: str | pathlib.Path,
        *,
        soil_E: float,
        soil_nu: float = 0.3,
        soil_profile: SoilProfile = "homogeneous",
        pile_behaviour: str = "auto",
        formula: FormulaFamily = "shadlou",
        component_monopile: str = "monopile",
        water_depth: float | None = None,
    ) -> MudlineFoundation:
        """Build the mudline foundation from a WindIO monopile ontology plus
        soil properties, auto-extracting the pile geometry (issue #118).

        Only the soil is specified; the pile terms the coupled-spring model
        needs are read from the ontology's ``monopile`` component:

        - **pile diameter** — the monopile outer diameter at the mudline.
        - **embedded length** — the mudline down to the pile base
          (``-water_depth`` minus the monopile ``reference_axis.z`` base).
        - **pile EI** — ``E * pi/64 (D^4 - (D - 2 t)^4)`` at the mudline, with
          ``E`` the monopile material modulus from the ontology.

        then defers to :meth:`from_soil_properties`. Pair the result with
        :meth:`pybmodes.models.Tower.attach_mudline_foundation`, or let
        :meth:`~pybmodes.models.Tower.from_windio_with_monopile` build and
        attach it for you via its ``soil_E`` keyword.

        ``water_depth`` (m, positive) locates the mudline and defaults to the
        ontology's ``environment.water_depth``. Raises ``ValueError`` when the
        mudline is unknown or the monopile does not extend below it (no
        embedded length to model soil reaction over). Requires the optional
        ``[windio]`` extra (PyYAML).
        """
        from pybmodes.io.windio import _read_water_depth, read_windio_tubular

        mp = read_windio_tubular(yaml_path, component=component_monopile)
        wd = _read_water_depth(yaml_path, water_depth)
        if wd is None:
            raise ValueError(
                "water_depth is required to place the mudline for the soil "
                "springs; pass water_depth=... or set environment.water_depth "
                "in the ontology."
            )
        mudline = -wd
        if not (mp.z_base < mudline < mp.z_top):
            raise ValueError(
                f"the monopile (reference_axis.z {mp.z_base:g}..{mp.z_top:g} m) "
                f"does not extend below the mudline (z = {mudline:g} m), so "
                f"there is no embedded length to model soil reaction over. "
                f"Check water_depth and the monopile reference_axis.z."
            )
        z_phys = mp.z_base + mp.station_grid * (mp.z_top - mp.z_base)
        pile_diameter = float(np.interp(mudline, z_phys, mp.outer_diameter))
        wall = float(np.interp(mudline, z_phys, mp.wall_thickness))
        inner = pile_diameter - 2.0 * wall
        pile_EI = mp.E * math.pi / 64.0 * (pile_diameter**4 - inner**4)
        return cls.from_soil_properties(
            pile_diameter=pile_diameter,
            pile_length_embedded=mudline - mp.z_base,
            pile_EI=pile_EI,
            soil_E=soil_E,
            soil_nu=soil_nu,
            soil_profile=soil_profile,
            pile_behaviour=pile_behaviour,
            formula=formula,
        )
