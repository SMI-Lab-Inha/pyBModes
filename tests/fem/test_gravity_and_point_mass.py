"""Self-weight geometric softening and discrete mid-span point masses.

Two features share this module because they share the FE plumbing: a
lumped mass at an arbitrary station (issue #35) and the axial force its
weight, plus the beam's own, puts into the column (issue #134).

Analytical references
---------------------
* **Euler buckling of a cantilever under a tip axial load** —
  ``P_cr = pi^2 EI / (4 L^2)``. Timoshenko, S. P. and Gere, J. M.
  (1961), *Theory of Elastic Stability*, 2nd ed., McGraw-Hill, Ch. 2.
* **Self-weight buckling of a uniform cantilever column** —
  ``q_cr L^3 / EI = 7.8373``, with ``q`` the weight per unit length.
  Greenhill, A. G. (1881), "Determination of the greatest height
  consistent with stability that a vertical pole or mast can be made",
  *Proc. Cambridge Philos. Soc.* 4, 65-73; tabulated in Timoshenko and
  Gere (1961) Art. 2.16.
* **Cantilever carrying a single lumped mass at station ``a``** — with a
  negligible beam mass the tip-free portion beyond ``a`` is unloaded, so
  the lump sees the static cantilever stiffness at its own station,
  ``k = 3 EI / a^3`` and ``f = sqrt(k / m) / 2 pi``. Blevins, R. D.
  (1979), *Formulas for Natural Frequency and Mode Shape*, Table 8-1.

Nothing here reads any external deck; every number is either constructed
inline or taken from the published closed forms above.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.assembly import assemble
from pybmodes.fem.element import _shape_h, _shape_hu, point_mass_element_matrix
from pybmodes.fem.nondim import RM, make_params
from pybmodes.models._pipeline import _gravity_axial_force
from pybmodes.models.tower import Tower, _coerce_gravity

from .._synthetic_bmi import write_bmi, write_uniform_sec_props

L_PHYS = 100.0
EI_PHYS = 1.0e10

# Greenhill (1881): a uniform cantilever column buckles under its own
# weight at q_cr L^3 / EI = 7.8373.
GREENHILL = 7.8373


def _uniform_cantilever_gk(
    nselt: int,
    *,
    g: float,
    mass_den: float,
    tip_mass: float,
) -> np.ndarray:
    """Global stiffness of a uniform cantilever including the self-weight
    geometric term, at gravity ``g`` (m/s^2)."""
    nd = make_params(radius=L_PHYS, hub_rad=0.0, rot_rpm=0.0)

    eiy_nd = EI_PHYS / nd.ref4
    rmas_nd = mass_den / RM

    eli = 1.0 / nselt
    el = np.full(nselt, eli)
    xb = np.array([1.0 - (i + 1) * eli for i in range(nselt)])
    rmas = np.full(nselt, rmas_nd)

    axf_g, grav_w = _gravity_axial_force(
        g, nd, el, xb, rmas, tip_mass / nd.ref_mr, [],
    )

    gk, _gm, _ = assemble(
        nselt=nselt,
        el=el,
        xb=xb,
        cfe=np.zeros(nselt),
        eiy=np.full(nselt, eiy_nd),
        eiz=np.full(nselt, eiy_nd),
        gj=np.full(nselt, 1.0e3 * eiy_nd),
        eac=np.full(nselt, 100.0),
        rmas=rmas,
        skm1=np.full(nselt, 1.0e-5),
        skm2=np.full(nselt, 1.0e-5),
        eg=np.zeros(nselt),
        ea=np.zeros(nselt),
        omega2=0.0,
        sec_loc=np.array([0.0, 1.0]),
        str_tw=np.zeros(2),
        hub_conn=1,
        elm_axf_g=axf_g,
        elm_grav_w=grav_w,
    )
    return gk


def _critical_gravity(
    *, mass_den: float, tip_mass: float, nselt: int = 60,
    lo: float = 1.0e-3, hi: float = 1.0e4,
) -> float:
    """Bisect for the gravity at which the tangent stiffness loses
    positive definiteness — the numerical buckling load."""

    def stable(g: float) -> bool:
        return float(np.linalg.eigvalsh(_uniform_cantilever_gk(
            nselt, g=g, mass_den=mass_den, tip_mass=tip_mass,
        )).min()) > 0.0

    assert stable(lo), "lower bracket is already buckled"
    assert not stable(hi), "upper bracket has not buckled"
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class TestGeometricStiffnessAgainstBuckling:
    """The self-weight axial term reproduces two published buckling loads."""

    def test_tip_load_matches_euler(self):
        """A cantilever with a tip weight buckles at pi^2 EI / (4 L^2)."""
        # Negligible distributed mass so the axial force is essentially
        # the constant tip weight all the way down.
        mass_den = 1.0e-3
        tip_mass = 1.0e5
        g_cr = _critical_gravity(mass_den=mass_den, tip_mass=tip_mass)
        p_cr_fem = tip_mass * g_cr
        p_cr_ref = np.pi**2 * EI_PHYS / (4.0 * L_PHYS**2)
        assert p_cr_fem == pytest.approx(p_cr_ref, rel=2.0e-3)

    def test_self_weight_matches_greenhill(self):
        """A cantilever with no tip mass buckles at q_cr L^3 / EI = 7.8373."""
        mass_den = 1.0e3
        g_cr = _critical_gravity(mass_den=mass_den, tip_mass=0.0)
        ratio = mass_den * g_cr * L_PHYS**3 / EI_PHYS
        assert ratio == pytest.approx(GREENHILL, rel=2.0e-3)

    def test_zero_gravity_leaves_stiffness_untouched(self):
        """g = 0 must reproduce the baseline matrix bit-for-bit."""
        with_zero = _uniform_cantilever_gk(
            12, g=0.0, mass_den=1.0e3, tip_mass=1.0e5,
        )
        nd = make_params(radius=L_PHYS, hub_rad=0.0, rot_rpm=0.0)
        eiy_nd = EI_PHYS / nd.ref4
        eli = 1.0 / 12
        el = np.full(12, eli)
        xb = np.array([1.0 - (i + 1) * eli for i in range(12)])
        baseline, _gm, _ = assemble(
            nselt=12, el=el, xb=xb, cfe=np.zeros(12),
            eiy=np.full(12, eiy_nd), eiz=np.full(12, eiy_nd),
            gj=np.full(12, 1.0e3 * eiy_nd), eac=np.full(12, 100.0),
            rmas=np.full(12, 1.0e3 / RM),
            skm1=np.full(12, 1.0e-5), skm2=np.full(12, 1.0e-5),
            eg=np.zeros(12), ea=np.zeros(12), omega2=0.0,
            sec_loc=np.array([0.0, 1.0]), str_tw=np.zeros(2), hub_conn=1,
        )
        assert np.allclose(with_zero, baseline, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Tower-level behaviour
# ---------------------------------------------------------------------------

def _synthetic_tower(
    tmp_path, *, mass_den: float = 1.0e3, tip_mass: float = 0.0,
    n_elements: int = 30, flp_stff: float = EI_PHYS,
) -> Tower:
    write_uniform_sec_props(
        tmp_path / "secs.dat",
        mass_den=mass_den, flp_stff=flp_stff, edge_stff=flp_stff,
        tor_stff=1.0e12, axial_stff=1.0e14,
    )
    bmi = write_bmi(
        tmp_path / "tower.bmi", beam_type=2, radius=L_PHYS, hub_conn=1,
        tip_mass=tip_mass, n_elements=n_elements, sec_props_file="secs.dat",
    )
    return Tower(bmi)


class TestGravityOnTower:
    def test_gravity_lowers_the_first_frequency(self, tmp_path):
        base = _synthetic_tower(tmp_path).run(4, check_model=False)
        with_g = _synthetic_tower(tmp_path).run(4, check_model=False, gravity=True)
        assert with_g.frequencies[0] < base.frequencies[0]
        # A 100 m / 1e10 N.m^2 column is far from its buckling weight, so
        # the shift is a small percentage rather than a collapse.
        shift = 1.0 - with_g.frequencies[0] / base.frequencies[0]
        assert 0.0 < shift < 0.2

    def test_gravity_false_matches_no_keyword(self, tmp_path):
        a = _synthetic_tower(tmp_path).run(4, check_model=False)
        b = _synthetic_tower(tmp_path).run(4, check_model=False, gravity=False)
        assert np.allclose(a.frequencies, b.frequencies, rtol=0.0, atol=0.0)

    def test_explicit_g_matches_true(self, tmp_path):
        a = _synthetic_tower(tmp_path).run(4, check_model=False, gravity=True)
        b = _synthetic_tower(tmp_path).run(4, check_model=False, gravity=9.80665)
        assert np.allclose(a.frequencies, b.frequencies)

    def test_larger_g_softens_more(self, tmp_path):
        f = [
            _synthetic_tower(tmp_path).run(
                4, check_model=False, gravity=g,
            ).frequencies[0]
            for g in (0.0, 5.0, 9.80665, 20.0)
        ]
        assert np.all(np.diff(f) < 0.0)

    def test_rna_weight_adds_to_the_softening(self, tmp_path):
        light = _synthetic_tower(tmp_path, tip_mass=1.0e3)
        heavy = _synthetic_tower(tmp_path, tip_mass=3.0e5)
        f_light = light.run(4, check_model=False, gravity=True).frequencies[0]
        f_light0 = light.run(4, check_model=False).frequencies[0]
        f_heavy = heavy.run(4, check_model=False, gravity=True).frequencies[0]
        f_heavy0 = heavy.run(4, check_model=False).frequencies[0]
        assert (1.0 - f_heavy / f_heavy0) > (1.0 - f_light / f_light0)


class TestGravityArgumentValidation:
    @pytest.mark.parametrize("hub_conn", [1, 3])
    def test_supported_boundary_conditions(self, hub_conn):
        assert _coerce_gravity(True, hub_conn) == pytest.approx(9.80665)

    def test_floating_is_rejected(self):
        with pytest.raises(ValueError, match="buoyancy"):
            _coerce_gravity(True, 2)

    def test_cable_is_rejected(self):
        with pytest.raises(ValueError, match="pinned-free"):
            _coerce_gravity(True, 4)

    def test_off_by_default(self):
        assert _coerce_gravity(False, 1) == 0.0

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_non_physical_g_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            _coerce_gravity(bad, 1)

    def test_zero_g_is_accepted_on_any_bc(self):
        # g = 0 is a no-op, so it must not trip the floating guard.
        assert _coerce_gravity(0.0, 2) == 0.0


class TestPointMassShapeFunctions:
    """Partition of unity — the interpolation the lump rides on."""

    @pytest.mark.parametrize("xi", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_axial_shape_sums_to_one(self, xi):
        assert _shape_hu(xi).sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("xi", [0.0, 0.3, 0.7, 1.0])
    def test_hermite_displacement_components_sum_to_one(self, xi):
        h = _shape_h(xi, 0.25)
        assert h[0] + h[2] == pytest.approx(1.0)

    def test_end_values_pick_the_nodal_dofs(self):
        assert _shape_h(0.0, 0.4) == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert _shape_h(1.0, 0.4) == pytest.approx([0.0, 0.0, 1.0, 0.0])

    def test_element_matrix_is_symmetric_and_psd(self):
        em = point_mass_element_matrix(0.37, 0.2, 3.0)
        assert np.allclose(em, em.T)
        assert np.linalg.eigvalsh(em).min() >= -1.0e-12

    def test_mass_lands_on_the_translation_blocks_only(self):
        em = point_mass_element_matrix(0.5, 0.2, 1.0)
        assert np.allclose(em[12:15, :], 0.0)
        assert np.allclose(em[:, 12:15], 0.0)


class TestPointMassOnTower:
    def test_lump_matches_static_cantilever_stiffness(self, tmp_path):
        """A lump on a near-massless cantilever vibrates at sqrt(3EI/m a^3)."""
        a = 60.0
        m_lump = 5.0e5
        # A beam mass this far below the lump makes the generalised mass
        # matrix badly conditioned, which the dense LAPACK subset path
        # does not handle well; 101 elements puts the solve on the sparse
        # shift-invert path, where it is clean. Nothing about the lump
        # placement needs the fine mesh.
        tower = _synthetic_tower(tmp_path, mass_den=1.0e-2, n_elements=101)
        tower.add_point_mass(a, m_lump)
        f = tower.run(6, check_model=False).frequencies[0]
        f_ref = np.sqrt(3.0 * EI_PHYS / (m_lump * a**3)) / (2.0 * np.pi)
        assert f == pytest.approx(f_ref, rel=5.0e-3)

    def test_lump_at_the_top_matches_tip_mass(self, tmp_path):
        m_lump = 2.0e5
        with_tip = _synthetic_tower(tmp_path, tip_mass=m_lump, n_elements=40)
        with_lump = _synthetic_tower(tmp_path, n_elements=40)
        with_lump.add_point_mass(L_PHYS, m_lump)
        a = with_tip.run(4, check_model=False).frequencies
        b = with_lump.run(4, check_model=False).frequencies
        assert np.allclose(a[:2], b[:2], rtol=1.0e-6)

    def test_position_is_resolved_off_the_mesh_nodes(self, tmp_path):
        """The lump sits between nodes on both meshes, so a shape-function
        placement must give the same answer while a snap-to-node would not."""
        a = 63.7
        m_lump = 4.0e5
        coarse = _synthetic_tower(tmp_path, n_elements=17)
        fine = _synthetic_tower(tmp_path, n_elements=53)
        coarse.add_point_mass(a, m_lump)
        fine.add_point_mass(a, m_lump)
        f_c = coarse.run(4, check_model=False).frequencies[0]
        f_f = fine.run(4, check_model=False).frequencies[0]
        assert f_c == pytest.approx(f_f, rel=2.0e-3)

    def test_lump_lowers_the_frequency(self, tmp_path):
        bare = _synthetic_tower(tmp_path).run(4, check_model=False)
        loaded = _synthetic_tower(tmp_path)
        loaded.add_point_mass(70.0, 2.0e5)
        assert loaded.run(4, check_model=False).frequencies[0] < bare.frequencies[0]

    def test_several_lumps_accumulate(self, tmp_path):
        one = _synthetic_tower(tmp_path)
        one.add_point_mass(40.0, 1.0e5)
        two = _synthetic_tower(tmp_path)
        two.add_point_mass(40.0, 1.0e5).add_point_mass(80.0, 1.0e5)
        assert len(two._bmi.point_masses) == 2
        assert (two.run(4, check_model=False).frequencies[0]
                < one.run(4, check_model=False).frequencies[0])

    def test_lump_weight_softens_under_gravity(self, tmp_path):
        tower = _synthetic_tower(tmp_path)
        tower.add_point_mass(95.0, 5.0e5)
        f0 = tower.run(4, check_model=False).frequencies[0]
        fg = tower.run(4, check_model=False, gravity=True).frequencies[0]
        bare = _synthetic_tower(tmp_path)
        b0 = bare.run(4, check_model=False).frequencies[0]
        bg = bare.run(4, check_model=False, gravity=True).frequencies[0]
        assert (1.0 - fg / f0) > (1.0 - bg / b0)

    @pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
    def test_non_physical_mass_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError, match="mass"):
            _synthetic_tower(tmp_path).add_point_mass(50.0, bad)

    def test_negative_height_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="height"):
            _synthetic_tower(tmp_path).add_point_mass(-1.0, 1000.0)

    def test_height_above_the_beam_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="above the flexible beam"):
            _synthetic_tower(tmp_path).add_point_mass(L_PHYS + 5.0, 1000.0)
