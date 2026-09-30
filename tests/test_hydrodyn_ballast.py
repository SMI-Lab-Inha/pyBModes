"""Filled-member ballast from HydroDyn decks.

HydroDyn treats a closed fill group as body-fixed ballast. The fill mass,
centroid and inertia are precomputed per element and the internal
pressure datum follows the ballast's high point, so the net load is the
weight of a rigid mass (``FloodedBallastPartSegmentCyl`` and the fill
terms in ``Morison.f90``). The self-contained tests check the reader
against the closed-form mass properties of solid cylinders and frustums.

The integration anchor is the OC4 DeepCwind semi, which keeps all of its
water ballast in two fill groups while ElastoDyn's ``PtfmMass`` holds only
the 3.852e6 kg of steel. Robertson et al. (2014), *Definition of the
Semisubmersible Floating System for Phase II of OC4*, NREL/TP-5000-60601,
give the ballasted platform as 1.3473e7 kg with its centre of mass
13.46 m below the still water line.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest

from pybmodes._numeric import trapezoid
from pybmodes.io._hydrodyn_ballast import read_filled_ballast
from pybmodes.models._platform import _add_filled_ballast

RHO = 1025.0

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OC4_DECKS = (
    REPO_ROOT / "external" / "OpenFAST_files" / "r-test" / "glue-codes"
    / "openfast" / "5MW_OC4Semi_WSt_WavesWN"
)
OC4_PREFIX = "NRELOffshrBsline5MW_OC4DeepCwindSemi_"
OC4_ELASTODYN = OC4_DECKS / f"{OC4_PREFIX}ElastoDyn.dat"
OC4_MOORDYN = OC4_DECKS / f"{OC4_PREFIX}MoorDyn.dat"
OC4_HYDRODYN = OC4_DECKS / f"{OC4_PREFIX}HydroDyn.dat"

# Robertson et al. (2014) NREL/TP-5000-60601, platform including ballast.
OC4_PLATFORM_MASS = 1.3473e7
OC4_PLATFORM_CM_Z = -13.46


def _deck(
    tmp_path: pathlib.Path,
    joints: list[tuple[int, float, float, float]],
    props: list[tuple[int, float, float]],
    members: list[tuple[int, int, int, int, int]],
    fills: list[str],
    *,
    legacy: bool = False,
    sec_geom: dict[int, int] | None = None,
    wtr_dens: str | None = None,
) -> pathlib.Path:
    """Write the HydroDyn sections the ballast reader looks at."""
    sec_geom = sec_geom or {}
    out = ["------- HydroDyn Input File (synthetic) -------"]
    if wtr_dens is not None:
        out.append(f"{wtr_dens}   WtrDens   - Water density (kg/m^3)")
    out.append("---------------------- MEMBER JOINTS ----------------------")
    out.append(f"{len(joints)}   NJoints   - Number of joints (-)")
    out.append("JointID Jointxi Jointyi Jointzi JointAxID JointOvrlp")
    out.append("  (-)     (m)     (m)     (m)     (-)      (switch)")
    out += [f"{j} {x} {y} {z} 1 0" for j, x, y, z in joints]
    if legacy:
        out.append("------- MEMBER CROSS-SECTION PROPERTIES -------")
        out.append(f"{len(props)}   NPropSets   - Number of member property sets")
    else:
        out.append("------- CYLINDRICAL MEMBER CROSS-SECTION PROPERTIES -------")
        out.append(f"{len(props)}   NPropSetsCyl   - Number of cylindrical sets")
    out.append("PropSetID PropD PropThck")
    out.append("  (-)      (m)    (m)")
    out += [f"{p} {d} {t}" for p, d, t in props]
    if not legacy:
        out.append("------- RECTANGULAR MEMBER CROSS-SECTION PROPERTIES -------")
        out.append("0   NPropSetsRec   - Number of rectangular sets")
        out.append("PropSetID PropA PropB PropThck")
        out.append("  (-)      (m)   (m)    (m)")
    out.append("-------------------- MEMBERS --------------------")
    out.append(f"{len(members)}   NMembers   - Number of members (-)")
    if legacy:
        out.append("MemberID MJointID1 MJointID2 MPropSetID1 MPropSetID2 "
                   "MDivSize MCoefMod PropPot")
        out.append("  (-) (-) (-) (-) (-) (m) (switch) (flag)")
        out += [f"{m} {a} {b} {p} {q} 1.0 1 TRUE"
                for m, a, b, p, q in members]
    else:
        out.append("MemberID MJointID1 MJointID2 MPropSetID1 MPropSetID2 "
                   "MSecGeom MSpinOrient MDivSize MCoefMod MHstLMod PropPot")
        out.append("  (-) (-) (-) (-) (-) (switch) (deg) (m) (switch) "
                   "(switch) (flag)")
        out += [f"{m} {a} {b} {p} {q} {sec_geom.get(m, 1)} 0 1.0 1 1 TRUE"
                for m, a, b, p, q in members]
    out.append("---------------------- FILLED MEMBERS ----------------------")
    out.append(f"{len(fills)}   NFillGroups   - Number of filled member groups")
    out.append("FillNumM FillMList FillFSLoc FillDens")
    out.append("  (-)      (-)       (m)     (kg/m^3)")
    out += fills
    out.append("---------------------- MARINE GROWTH ----------------------")
    out.append("0   NMGDepths   - Number of marine-growth depths specified")
    path = tmp_path / "hd.dat"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _vertical_column(tmp_path, *, fs: float, **kw) -> pathlib.Path:
    # Outer D = 10 m, wall 0.5 m, so inner radius 4.5 m; z from -20 to +10.
    return _deck(
        tmp_path,
        joints=[(1, 3.0, -2.0, -20.0), (2, 3.0, -2.0, 10.0)],
        props=[(1, 10.0, 0.5)],
        members=[(1, 1, 2, 1, 1)],
        fills=[f"1 1 {fs} {RHO}"],
        **kw,
    )


class TestClosedForms:
    def test_partially_filled_vertical_cylinder(self, tmp_path):
        b = read_filled_ballast(_vertical_column(tmp_path, fs=-8.0))
        r, h = 4.5, 12.0
        m = RHO * np.pi * r ** 2 * h
        assert b.mass == pytest.approx(m, rel=1e-12)
        np.testing.assert_allclose(b.cg, [3.0, -2.0, -14.0], atol=1e-9)
        # Solid cylinder about its own centroid (axis along z).
        i_perp = m * (3.0 * r ** 2 + h ** 2) / 12.0
        np.testing.assert_allclose(
            b.inertia_cg, np.diag([i_perp, i_perp, m * r ** 2 / 2.0]),
            rtol=1e-10, atol=1e-6 * i_perp,
        )

    def test_member_ends_are_order_independent(self, tmp_path):
        a = read_filled_ballast(_vertical_column(tmp_path, fs=-8.0))
        path = _deck(
            tmp_path,
            joints=[(1, 3.0, -2.0, -20.0), (2, 3.0, -2.0, 10.0)],
            props=[(1, 10.0, 0.5)],
            members=[(1, 2, 1, 1, 1)],
            fills=[f"1 1 -8.0 {RHO}"],
        )
        b = read_filled_ballast(path)
        assert b.mass == pytest.approx(a.mass, rel=1e-12)
        np.testing.assert_allclose(b.inertia_cg, a.inertia_cg, rtol=1e-10)

    def test_free_surface_above_member_fills_it_completely(self, tmp_path):
        b = read_filled_ballast(_vertical_column(tmp_path, fs=50.0))
        assert b.mass == pytest.approx(RHO * np.pi * 4.5 ** 2 * 30.0)

    def test_free_surface_below_member_holds_nothing(self, tmp_path):
        assert read_filled_ballast(_vertical_column(tmp_path, fs=-25.0)) is None

    def test_tapered_frustum_volume(self, tmp_path):
        # Inner radius tapers 5.0 -> 2.0 m over a fully filled 9 m length.
        path = _deck(
            tmp_path,
            joints=[(1, 0.0, 0.0, -30.0), (2, 0.0, 0.0, -21.0)],
            props=[(1, 10.2, 0.1), (2, 4.2, 0.1)],
            members=[(1, 1, 2, 1, 2)],
            fills=[f"1 1 0.0 {RHO}"],
        )
        b = read_filled_ballast(path)
        r1, r2, h = 5.0, 2.0, 9.0
        vol = np.pi * h / 3.0 * (r1 ** 2 + r1 * r2 + r2 ** 2)
        z_c = h * (r1 ** 2 + 2 * r1 * r2 + 3 * r2 ** 2) / (
            4 * (r1 ** 2 + r1 * r2 + r2 ** 2))
        assert b.mass == pytest.approx(RHO * vol, rel=1e-12)
        assert b.cg[2] == pytest.approx(-30.0 + z_c, rel=1e-12)
        # Axial moment of a frustum, (3/10) m (r1^5 - r2^5)/(r1^3 - r2^3).
        izz = 0.3 * RHO * vol * (r1 ** 5 - r2 ** 5) / (r1 ** 3 - r2 ** 3)
        assert b.inertia_cg[2, 2] == pytest.approx(izz, rel=1e-12)

    def test_horizontal_member_is_all_or_nothing(self, tmp_path):
        def deck(fs):
            return _deck(
                tmp_path,
                joints=[(1, -5.0, 0.0, -10.0), (2, 5.0, 0.0, -10.0)],
                props=[(1, 2.0, 0.0)],
                members=[(1, 1, 2, 1, 1)],
                fills=[f"1 1 {fs} {RHO}"],
            )
        full = read_filled_ballast(deck(-9.0))
        assert full.mass == pytest.approx(RHO * np.pi * 10.0)
        m, r, L = full.mass, 1.0, 10.0
        np.testing.assert_allclose(
            np.diag(full.inertia_cg),
            [m * r ** 2 / 2, m * (3 * r ** 2 + L ** 2) / 12,
             m * (3 * r ** 2 + L ** 2) / 12],
            rtol=1e-10,
        )
        # HydroDyn tests FillFSLoc >= Zb first, so a surface level with
        # the member still counts it as full.
        assert read_filled_ballast(deck(-10.0)).mass == pytest.approx(full.mass)
        assert read_filled_ballast(deck(-10.5)) is None

    def test_inclined_member_fills_along_its_axis(self, tmp_path):
        # 45 degree member from z = -10 to z = 0; free surface at -4 gives
        # an along-axis fill length of 6·sqrt(2).
        path = _deck(
            tmp_path,
            joints=[(1, 0.0, 0.0, -10.0), (2, 10.0, 0.0, 0.0)],
            props=[(1, 2.0, 0.0)],
            members=[(1, 1, 2, 1, 1)],
            fills=[f"1 1 -4.0 {RHO}"],
        )
        b = read_filled_ballast(path)
        assert b.mass == pytest.approx(RHO * np.pi * 6.0 * np.sqrt(2.0))
        np.testing.assert_allclose(b.cg, [3.0, 0.0, -7.0], atol=1e-12)

    def test_two_groups_combine_by_parallel_axis(self, tmp_path):
        path = _deck(
            tmp_path,
            joints=[(1, 10.0, 0.0, -20.0), (2, 10.0, 0.0, 0.0),
                    (3, -10.0, 0.0, -20.0), (4, -10.0, 0.0, 0.0)],
            props=[(1, 2.0, 0.0)],
            members=[(1, 1, 2, 1, 1), (2, 3, 4, 1, 1)],
            fills=[f"1 1 -10.0 {RHO}", f"1 2 -10.0 {RHO}"],
        )
        b = read_filled_ballast(path)
        m1 = RHO * np.pi * 10.0
        assert b.mass == pytest.approx(2 * m1)
        np.testing.assert_allclose(b.cg, [0.0, 0.0, -15.0], atol=1e-12)
        i_own_perp = m1 * (3.0 + 100.0) / 12.0
        assert b.inertia_cg[1, 1] == pytest.approx(
            2 * (i_own_perp + m1 * 100.0), rel=1e-12)
        assert b.inertia_cg[0, 0] == pytest.approx(2 * i_own_perp, rel=1e-12)
        assert b.inertia_cg[2, 2] == pytest.approx(
            2 * (m1 / 2.0 + m1 * 100.0), rel=1e-12)
        assert set(b.member_mass) == {1, 2}


class TestDeckHandling:
    def test_no_fill_groups_returns_none(self, tmp_path):
        path = _deck(
            tmp_path, joints=[(1, 0, 0, -1), (2, 0, 0, 1)],
            props=[(1, 1.0, 0.1)], members=[(1, 1, 2, 1, 1)], fills=[],
        )
        assert read_filled_ballast(path) is None

    def test_legacy_layout(self, tmp_path):
        new = read_filled_ballast(_vertical_column(tmp_path, fs=-8.0))
        old = read_filled_ballast(
            _vertical_column(tmp_path, fs=-8.0, legacy=True))
        assert old.mass == pytest.approx(new.mass, rel=1e-12)

    @pytest.mark.parametrize("token", ["DEFAULT", '"default"'])
    def test_default_fill_density_is_the_water_density(self, tmp_path, token):
        path = _deck(
            tmp_path, joints=[(1, 0, 0, -10), (2, 0, 0, 0)],
            props=[(1, 2.0, 0.0)], members=[(1, 1, 2, 1, 1)],
            fills=[f"1 1 0.0 {token}"], wtr_dens="1000",
        )
        b = read_filled_ballast(path)
        assert b.mass == pytest.approx(1000.0 * np.pi * 10.0)
        b2 = read_filled_ballast(path, water_density=1025.0)
        assert b2.mass == pytest.approx(1025.0 * np.pi * 10.0)

    def test_rectangular_member_is_rejected(self, tmp_path):
        path = _vertical_column(tmp_path, fs=-8.0, sec_geom={1: 2})
        with pytest.raises(NotImplementedError, match="rectangular"):
            read_filled_ballast(path)

    def test_unknown_member_is_rejected(self, tmp_path):
        path = _deck(
            tmp_path, joints=[(1, 0, 0, -10), (2, 0, 0, 0)],
            props=[(1, 2.0, 0.0)], members=[(1, 1, 2, 1, 1)],
            fills=[f"1 7 0.0 {RHO}"],
        )
        with pytest.raises(ValueError, match="member 7"):
            read_filled_ballast(path)

    def test_non_physical_section_is_rejected(self, tmp_path):
        path = _deck(
            tmp_path, joints=[(1, 0, 0, -10), (2, 0, 0, 0)],
            props=[(1, 2.0, 1.5)], members=[(1, 1, 2, 1, 1)],
            fills=[f"1 1 0.0 {RHO}"],
        )
        with pytest.raises(ValueError, match="non-physical"):
            read_filled_ballast(path)

    @pytest.mark.parametrize("token", ["nan", "NaN", "inf", "-Infinity"])
    @pytest.mark.parametrize("where", ["FillDens", "FillFSLoc", "joint", "PropD"])
    def test_non_finite_deck_scalars_are_rejected(self, tmp_path, token, where):
        joints = [(1, 0, 0, -10), (2, 0, 0, 0)]
        props = [(1, 2.0, 0.0)]
        fs, dens = "0.0", f"{RHO}"
        if where == "FillDens":
            dens = token
        elif where == "FillFSLoc":
            fs = token
        elif where == "joint":
            joints = [(1, 0, token, -10), (2, 0, 0, 0)]
        else:
            props = [(1, token, 0.0)]
        path = _deck(
            tmp_path, joints=joints, props=props,
            members=[(1, 1, 2, 1, 1)], fills=[f"1 1 {fs} {dens}"],
        )
        with pytest.raises(ValueError, match="(?i)non-finite|malformed"):
            read_filled_ballast(path)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_water_density_is_rejected(self, tmp_path, bad):
        path = _vertical_column(tmp_path, fs=-8.0)
        with pytest.raises(ValueError, match="water_density"):
            read_filled_ballast(path, water_density=bad)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_filled_ballast(tmp_path / "absent.dat")


class TestMergeIntoPlatform:
    def _ptfm(self):
        return {
            "PtfmMass": 4.0e6, "PtfmCMxt": 0.0, "PtfmCMyt": 0.0,
            "PtfmCMzt": -8.0, "PtfmRIner": 2.0e9, "PtfmPIner": 2.0e9,
            "PtfmYIner": 3.0e9,
        }

    def _i_mat(self, p):
        return np.diag([p["PtfmMass"]] * 3
                       + [p["PtfmRIner"], p["PtfmPIner"], p["PtfmYIner"]])

    def test_combined_mass_cm_and_parallel_axis(self, tmp_path):
        path = _vertical_column(tmp_path, fs=-8.0)
        b = read_filled_ballast(path)
        p = self._ptfm()
        out, i_mat = _add_filled_ballast(p, self._i_mat(p), path)

        m = p["PtfmMass"] + b.mass
        c_s = np.array([0.0, 0.0, -8.0])
        c = (p["PtfmMass"] * c_s + b.mass * b.cg) / m
        assert out["PtfmMass"] == pytest.approx(m)
        np.testing.assert_allclose(
            [out["PtfmCMxt"], out["PtfmCMyt"], out["PtfmCMzt"]], c)
        np.testing.assert_allclose(np.diag(i_mat)[:3], [m] * 3)

        def shift(mass, d):
            return mass * (d @ d * np.eye(3) - np.outer(d, d))
        expect = (np.diag([2.0e9, 2.0e9, 3.0e9]) + shift(p["PtfmMass"], c_s - c)
                  + b.inertia_cg + shift(b.mass, b.cg - c))
        np.testing.assert_allclose(i_mat[3:, 3:], expect, rtol=1e-12)
        # An off-axis column gives a genuine roll-yaw product of inertia.
        assert i_mat[3, 5] != 0.0
        np.testing.assert_allclose(i_mat, i_mat.T)
        # The input dict is not mutated.
        assert p["PtfmMass"] == 4.0e6

    def test_water_density_resolves_default_fill(self, tmp_path):
        """from_windio_floating passes its rho so the ballast and the
        hydrostatics agree on the water density."""
        path = _deck(
            tmp_path, joints=[(1, 0, 0, -10), (2, 0, 0, 0)],
            props=[(1, 2.0, 0.0)], members=[(1, 1, 2, 1, 1)],
            fills=["1 1 0.0 DEFAULT"],
        )
        p = self._ptfm()
        out, _ = _add_filled_ballast(
            p, self._i_mat(p), path, water_density=1000.0)
        assert out["PtfmMass"] == pytest.approx(4.0e6 + 1000.0 * np.pi * 10.0)

    def test_no_fill_leaves_platform_unchanged(self, tmp_path):
        path = _deck(
            tmp_path, joints=[(1, 0, 0, -1), (2, 0, 0, 1)],
            props=[(1, 1.0, 0.1)], members=[(1, 1, 2, 1, 1)], fills=[],
        )
        p = self._ptfm()
        i0 = self._i_mat(p)
        out, i_mat = _add_filled_ballast(p, i0, path)
        assert out is p and i_mat is i0


_needs_oc4 = pytest.mark.skipif(
    not (OC4_ELASTODYN.is_file() and OC4_MOORDYN.is_file()
         and OC4_HYDRODYN.is_file()),
    reason=f"OC4 DeepCwind r-test decks not present under {OC4_DECKS}",
)


@pytest.mark.integration
@_needs_oc4
class TestOC4DeepCwind:
    def _tower(self):
        from pybmodes.models import Tower

        return Tower.from_elastodyn_with_mooring(
            OC4_ELASTODYN, OC4_MOORDYN, OC4_HYDRODYN,
        )

    def test_ballast_matches_the_published_platform(self):
        ps = self._tower()._bmi.support
        assert ps.mass_pform == pytest.approx(OC4_PLATFORM_MASS, rel=1e-3)
        assert -ps.cm_pform == pytest.approx(OC4_PLATFORM_CM_Z, abs=0.01)

    def test_roll_and_pitch_restoring_is_positive(self):
        k = self._tower()._bmi.support.hydro_K
        assert k[3, 3] > 0.0 and k[4, 4] > 0.0

    def test_all_six_rigid_body_modes_are_returned(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res = self._tower().run(n_modes=8, check_model=False)
        assert set(res.mode_labels[:6]) == {
            "surge", "sway", "heave", "roll", "pitch", "yaw",
        }

    def test_heave_matches_the_single_dof_estimate(self):
        """Heave barely couples to the tower, so it should sit close to
        sqrt(C33 / (m_total + A33)) / 2π with the ballasted mass."""
        tower = self._tower()
        ps = tower._bmi.support
        res = tower.run(n_modes=8, check_model=False)
        f = dict(zip(res.mode_labels, res.frequencies))["heave"]
        m_tower = trapezoid(tower._sp.mass_den, tower._sp.span_loc) * (
            tower._bmi.radius + ps.draft)
        m_total = ps.mass_pform + m_tower + tower._bmi.tip_mass.mass
        k33 = ps.hydro_K[2, 2] + ps.mooring_K[2, 2]
        f_est = np.sqrt(k33 / (m_total + ps.hydro_M[2, 2])) / (2 * np.pi)
        assert f == pytest.approx(f_est, rel=0.02)
