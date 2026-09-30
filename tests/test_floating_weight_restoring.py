"""Weight restoring on floating models built from OpenFAST decks.

A WAMIT ``.hst`` holds only the hydrostatic restoring ``ρ g I_wp +
ρ g V z_B``; OpenFAST adds the weight of the platform, tower and RNA
through ElastoDyn's gravity. The classical restoring of a floating body
is ``C44 = C55 = ρ g V (z_B − z_G) + ρ g I_wp`` (Faltinsen 1990, *Sea
Loads on Ships and Offshore Structures*, ch. 2), so a model assembled
from the decks has to add ``−m g z_G`` back. Without it the OC3 Hywind
spar, which is stabilised by ballast rather than by its waterplane, read
``C44 = C55 = −5.0e9 N·m/rad`` and its roll and pitch modes silently
disappeared from the result.

The end-to-end reference is the OC3 Hywind full-system linearisation in
Jonkman (2010), *Definition of the Floating System for Phase IV of OC3*,
NREL/TP-500-47535, full-system natural frequencies from the FAST
linearisation: surge / sway 0.0080 Hz, heave 0.0324 Hz, roll 0.0342 Hz,
pitch 0.0343 Hz and yaw 0.1210 Hz — the last with the additional
9.834e7 N·m/rad yaw spring the report's mooring-system definition uses to
stand in for the delta-line crowfoot. The r-test MoorDyn deck models
plain catenaries without it.
"""

from __future__ import annotations

import copy
import pathlib
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pybmodes.io.bmi import PlatformSupport, TipMassProps
from pybmodes.models._platform import STANDARD_GRAVITY, _gravitational_restoring

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLES = (
    REPO_ROOT / "src" / "pybmodes" / "_examples" / "sample_inputs"
    / "reference_turbines"
)
OC3_SAMPLE = (
    SAMPLES / "07_nrel5mw_oc3hywind_spar"
    / "07_nrel5mw_oc3hywind_spar_tower.bmi"
)
OC3_DECKS = (
    REPO_ROOT / "external" / "OpenFAST_files" / "r-test" / "glue-codes"
    / "openfast" / "5MW_OC3Spar_DLL_WTurb_WavesIrr"
)
OC3_ELASTODYN = OC3_DECKS / "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn.dat"
OC3_MOORDYN = OC3_DECKS / "NRELOffshrBsline5MW_OC3Hywind_MoorDyn.dat"
OC3_HYDRODYN = OC3_DECKS / "NRELOffshrBsline5MW_OC3Hywind_HydroDyn.dat"

# Jonkman (2010) NREL/TP-500-47535, FAST full-system natural frequencies.
JONKMAN_2010_HZ = {
    "surge": 0.0080, "sway": 0.0080, "heave": 0.0324,
    "roll": 0.0342, "pitch": 0.0343, "yaw": 0.1210,
}
# Jonkman (2010) mooring-system definition: additional yaw spring that
# stands in for the delta-line crowfoot.
OC3_ADDITIONAL_YAW_STIFFNESS = 9.834e7

G = STANDARD_GRAVITY


def _support(**overrides) -> PlatformSupport:
    fields = dict(
        draft=-10.0, cm_pform=90.0, mass_pform=7.0e6,
        i_matrix=np.diag([7.0e6, 7.0e6, 7.0e6, 4.0e9, 4.0e9, 1.6e8]),
        ref_msl=0.0, hydro_M=np.zeros((6, 6)), hydro_K=np.zeros((6, 6)),
        mooring_K=np.zeros((6, 6)),
        distr_m_z=np.zeros(0), distr_m=np.zeros(0),
        distr_k_z=np.zeros(0), distr_k=np.zeros(0),
    )
    fields.update(overrides)
    return PlatformSupport(**fields)


def _model(support: PlatformSupport, *, mass_den: float = 3000.0,
           tip: float = 3.5e5, cm_axial: float = 2.0,
           sec_mass: float = 1.0, point_masses=()):
    """A hand-built BMI / section-property pair: uniform tower from z = 10
    to z = 90 above MSL, with the floating ``radius = tower-top z``
    convention the deck constructors use."""
    bmi = SimpleNamespace(
        support=support, radius=90.0,
        scaling=SimpleNamespace(sec_mass=sec_mass),
        tip_mass=TipMassProps(
            mass=tip, cm_offset=0.0, cm_axial=cm_axial,
            ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
        ),
        point_masses=tuple(point_masses),
    )
    sp = SimpleNamespace(
        span_loc=np.linspace(0.0, 1.0, 11),
        mass_den=np.full(11, mass_den),
    )
    return bmi, sp


class TestClosedForm:
    """``−g Σ m_i (z_i − z_ref)`` evaluated by hand."""

    def test_platform_tower_and_rna(self):
        bmi, sp = _model(_support())
        c = _gravitational_restoring(bmi, sp)
        # Platform CM 90 m below MSL; tower 80 m long, centroid at 50 m;
        # RNA 2 m above the 90 m tower top.
        moment = 7.0e6 * -90.0 + 3000.0 * 80.0 * 50.0 + 3.5e5 * 92.0
        assert c[3, 3] == pytest.approx(-G * moment, rel=1e-12)
        assert c[4, 4] == pytest.approx(-G * moment, rel=1e-12)
        mask = np.ones((6, 6), dtype=bool)
        mask[3, 3] = mask[4, 4] = False
        assert np.all(c[mask] == 0.0)

    def test_ballast_below_stabilises_and_topside_destabilises(self):
        deep, sp = _model(_support(), mass_den=0.0, tip=0.0)
        assert _gravitational_restoring(deep, sp)[3, 3] > 0.0
        top, sp = _model(_support(mass_pform=0.0))
        assert _gravitational_restoring(top, sp)[3, 3] < 0.0

    def test_lever_is_taken_about_the_reference_point(self):
        """The ``.hst`` is about ``ref_msl``; so must the weight be. Moving
        the reference down by ``d`` adds ``g · m_total · d``."""
        bmi0, sp = _model(_support())
        bmi5, _ = _model(_support(ref_msl=5.0))
        m_total = 7.0e6 + 3000.0 * 80.0 + 3.5e5
        delta = (_gravitational_restoring(bmi5, sp)[3, 3]
                 - _gravitational_restoring(bmi0, sp)[3, 3])
        assert delta == pytest.approx(-G * m_total * 5.0, rel=1e-9)

    def test_mass_scaling_and_point_masses_count(self):
        from pybmodes.io.bmi import PointMass

        base, sp = _model(_support())
        scaled, _ = _model(_support(), sec_mass=1.1,
                           point_masses=(PointMass(height=40.0, mass=2.0e4),))
        delta = (_gravitational_restoring(scaled, sp)[3, 3]
                 - _gravitational_restoring(base, sp)[3, 3])
        # 10 % more tower (centroid 50 m) and 20 t at 10 + 40 = 50 m.
        assert delta == pytest.approx(
            -G * (0.1 * 3000.0 * 80.0 * 50.0 + 2.0e4 * 50.0), rel=1e-9,
        )


def _oc3_sample():
    from pybmodes.models import Tower

    return Tower(OC3_SAMPLE)


class TestPointMassKeepsWeightInStep:
    """``add_point_mass`` after construction must reach the weight term too,
    or the FEM mass and the roll / pitch restoring disagree."""

    def test_lump_adds_its_own_weight(self):
        tower = _oc3_sample()
        tower._weight_g = G   # as the deck-built constructors set it
        ps = tower._bmi.support
        before = np.array(ps.hydro_K, copy=True)
        tower.add_point_mass(30.0, 5.0e4)
        z_rel = -ps.draft + 30.0 + ps.ref_msl
        delta = ps.hydro_K - before
        assert delta[3, 3] == pytest.approx(-G * 5.0e4 * z_rel, rel=1e-12)
        assert delta[4, 4] == delta[3, 3]
        mask = np.ones((6, 6), dtype=bool)
        mask[3, 3] = mask[4, 4] = False
        assert np.all(delta[mask] == 0.0)

    def test_model_without_a_weight_term_is_untouched(self):
        tower = _oc3_sample()
        before = np.array(tower._bmi.support.hydro_K, copy=True)
        tower.add_point_mass(30.0, 5.0e4)
        np.testing.assert_array_equal(tower._bmi.support.hydro_K, before)


class TestNegativeRestoringIsNeverSilent:
    """A negative-stiffness rigid-body mode must be reported, not dropped."""

    def test_stable_platform_is_quiet(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res = _oc3_sample().run(n_modes=9, check_model=False)
        assert set(res.mode_labels[:6]) == {
            "surge", "sway", "heave", "roll", "pitch", "yaw",
        }

    def test_unstable_roll_and_pitch_warn(self):
        tower = _oc3_sample()
        ps = copy.deepcopy(tower._bmi.support)
        # Strip the weight the BModes deck folds into mooring_K, leaving
        # hydrostatics + catenary only — the pre-fix deck-built state.
        ps.mooring_K[3, 3] -= 7.46633e6 * G * 89.9155
        ps.mooring_K[4, 4] -= 7.46633e6 * G * 89.9155
        tower._bmi.support = ps
        with pytest.warns(UserWarning, match=r"negative-stiffness.*roll, pitch"):
            res = tower.run(n_modes=9, check_model=False)
        assert "roll" not in res.mode_labels
        assert "pitch" not in res.mode_labels


_needs_oc3 = pytest.mark.skipif(
    not (OC3_ELASTODYN.is_file() and OC3_MOORDYN.is_file()
         and OC3_HYDRODYN.is_file()),
    reason=f"OC3 Hywind r-test decks not present under {OC3_DECKS}",
)


def _oc3_from_decks():
    from pybmodes.models import Tower

    return Tower.from_elastodyn_with_mooring(
        OC3_ELASTODYN, OC3_MOORDYN, OC3_HYDRODYN,
    )


def _by_label(res) -> dict[str, float]:
    return {
        lbl: float(f)
        for lbl, f in zip(res.mode_labels, res.frequencies)
        if lbl is not None
    }


@pytest.mark.integration
@_needs_oc3
class TestOC3HywindFromDecks:
    def test_all_six_rigid_body_modes_are_returned(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res = _oc3_from_decks().run(n_modes=9, check_model=False)
        assert set(res.mode_labels[:6]) == set(JONKMAN_2010_HZ)

    def test_weight_is_added_to_the_hst(self):
        from pybmodes.io.wamit_reader import HydroDynReader

        tower = _oc3_from_decks()
        c_hst = HydroDynReader(OC3_HYDRODYN).read_platform_matrices().C_hst
        c_w = tower._bmi.support.hydro_K - c_hst
        # Platform ballast (+6.58e9) outweighs tower + RNA (−4.1e8).
        assert c_hst[3, 3] == pytest.approx(-4.999e9, rel=1e-3)
        assert c_w[3, 3] == pytest.approx(6.171e9, rel=1e-3)
        assert c_w[4, 4] == c_w[3, 3]

    @pytest.mark.parametrize(
        "dof, rel",
        [("surge", 0.03), ("sway", 0.03), ("heave", 0.02),
         ("roll", 0.03), ("pitch", 0.03)],
    )
    def test_matches_jonkman_2010(self, dof, rel):
        f = _by_label(_oc3_from_decks().run(n_modes=9, check_model=False))
        assert f[dof] == pytest.approx(JONKMAN_2010_HZ[dof], rel=rel)

    def test_yaw_with_the_published_additional_spring(self):
        """The deck carries no crowfoot spring, so yaw is the catenary-only
        0.041 Hz. Adding the report's spring brings it to the published
        value; the remaining ~5 % is the RNA yaw inertia, which the
        ElastoDyn adapter assembles from a slender-body nacelle proxy."""
        tower = _oc3_from_decks()
        assert _by_label(tower.run(n_modes=9, check_model=False))["yaw"] == (
            pytest.approx(0.0412, rel=0.02)
        )
        tower._bmi.support.mooring_K[5, 5] += OC3_ADDITIONAL_YAW_STIFFNESS
        f = _by_label(tower.run(n_modes=9, check_model=False))
        assert f["yaw"] == pytest.approx(JONKMAN_2010_HZ["yaw"], rel=0.08)

    def test_independent_of_the_requested_mode_count(self):
        tower = _oc3_from_decks()
        f9 = tower.run(n_modes=9, check_model=False).frequencies
        f20 = tower.run(n_modes=20, check_model=False).frequencies
        np.testing.assert_allclose(f20[:9], f9, rtol=1e-6)

    def test_point_mass_matches_a_full_recompute(self):
        from pybmodes.io.wamit_reader import HydroDynReader

        tower = _oc3_from_decks().add_point_mass(40.0, 1.0e5)
        c_hst = HydroDynReader(OC3_HYDRODYN).read_platform_matrices().C_hst
        np.testing.assert_allclose(
            tower._bmi.support.hydro_K - c_hst,
            _gravitational_restoring(tower._bmi, tower._sp),
            rtol=1e-12, atol=1e-3,
        )

    def test_mooring_only_model_gets_no_weight_term(self):
        """Without HydroDyn there is no buoyancy to balance the weight, so
        adding it would not describe a floating body."""
        from pybmodes.models import Tower

        tower = Tower.from_elastodyn_with_mooring(OC3_ELASTODYN, OC3_MOORDYN)
        assert np.all(tower._bmi.support.hydro_K == 0.0)
