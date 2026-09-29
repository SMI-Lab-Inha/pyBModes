"""The dense symmetric path solves the inverted pencil, so its lowest
modes do not depend on how many were requested.

``scipy.linalg.eigh(K, M)`` reduces the pencil through a Cholesky factor
of ``M`` and returns each eigenvalue to an *absolute* accuracy of about
``eps * lambda_max`` (the Weyl bound for a backward-stable symmetric
eigensolver; Golub and Van Loan 2013, *Matrix Computations*, 4th ed.,
ch. 8). The relative error on the lowest mode is therefore about
``eps * lambda_max / lambda_min``. The ElastoDyn adapters floor rotary
inertia at 1e-6, which pushes that ratio to ~1e15 on the NREL 5MW blade,
so the reduction returned first-flap frequencies of 0.715 / 0.710 /
0.766 / 0.728 Hz for 4 / 6 / 10 / 20 requested modes.

Solving ``M x = mu (K + s M) x`` with ``mu = 1 / (lambda + s)`` puts the
wanted modes at the top of the spectrum, where the same absolute error
is a relative one — the standard shift-invert spectral transformation
(Ericsson and Ruhe 1980, *Math. Comp.* 35(152), 1251-1268; Bathe 2014,
*Finite Element Procedures*, 2nd ed., ch. 11), and the transformation the
sparse ``eigsh(sigma=0)`` path already applies.

References for the physics: a cantilever whose beam mass is negligible
next to a tip lump is a spring-mass oscillator on the static tip
stiffness ``3 EI / L^3`` (Blevins 1979, *Formulas for Natural Frequency
and Mode Shape*, Table 8-1).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pybmodes.cli import _resolve_examples_root
from pybmodes.fem import solver
from pybmodes.fem.assembly import assemble
from pybmodes.fem.nondim import RM, ROMG, make_params, nondim_tip_mass
from pybmodes.fem.solver import eigvals_to_hz, solve_modes
from pybmodes.io.bmi import TipMassProps

NREL5MW_LAND = (
    _resolve_examples_root() / "reference_decks" / "nrel5mw_land"
    / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
)
_needs_deck = pytest.mark.skipif(
    not NREL5MW_LAND.is_file(),
    reason=f"bundled NREL 5MW deck not present at {NREL5MW_LAND}",
)

L = 100.0
EI = 1.0e10
M_TIP = 4.0e5


def _cantilever_with_tip_lump(
    nselt: int, mass_den: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Same construction as ``tests/fem/test_ill_conditioned_mass.py``."""
    nd = make_params(radius=L, hub_rad=0.0, rot_rpm=0.0)
    eiy = EI / nd.ref4
    eli = 1.0 / nselt
    el = np.full(nselt, eli)
    xb = np.array([1.0 - (i + 1) * eli for i in range(nselt)])
    tip = nondim_tip_mass(
        TipMassProps(mass=M_TIP, cm_offset=0.0, cm_axial=0.0, ixx=0.0,
                     iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0),
        nd, beam_type=2, id_form=1, hub_conn=1,
    )
    gk, gm, _ = assemble(
        nselt=nselt, el=el, xb=xb, cfe=np.zeros(nselt),
        eiy=np.full(nselt, eiy), eiz=np.full(nselt, eiy),
        gj=np.full(nselt, 1.0e3 * eiy), eac=np.full(nselt, 100.0),
        rmas=np.full(nselt, mass_den / RM),
        skm1=np.full(nselt, 1.0e-5), skm2=np.full(nselt, 1.0e-5),
        eg=np.zeros(nselt), ea=np.zeros(nselt), omega2=0.0,
        sec_loc=np.array([0.0, 1.0]), str_tw=np.zeros(2), hub_conn=1,
        tip_mass=tip,
    )
    return gk, gm


def _frequencies(model_factory, n_modes: int) -> np.ndarray:
    with warnings.catch_warnings():
        # The bundled blade carries two expected stiffness-jump warnings.
        warnings.simplefilter("ignore", UserWarning)
        return np.asarray(
            model_factory().run(n_modes=n_modes).frequencies, dtype=float,
        )


def _blade():
    from pybmodes.models import RotatingBlade

    return RotatingBlade.from_elastodyn(NREL5MW_LAND)


def _tower():
    from pybmodes.models import Tower

    return Tower.from_elastodyn(NREL5MW_LAND)


@_needs_deck
class TestNModesIndependence:
    """The regression: the lowest modes must not move with ``n_modes``."""

    @pytest.mark.parametrize("factory", [_blade, _tower], ids=["blade", "tower"])
    def test_lowest_modes_agree_for_4_and_20_requested(self, factory):
        f4 = _frequencies(factory, 4)
        f20 = _frequencies(factory, 20)
        np.testing.assert_allclose(f20[:4], f4, rtol=1.0e-6)

    def test_blade_first_flap_across_every_window(self):
        """The four windows that spread 8 % before the fix."""
        flap = [_frequencies(_blade, n)[0] for n in (4, 6, 10, 20)]
        assert max(flap) - min(flap) < 1.0e-6 * min(flap)

    @pytest.mark.parametrize("factory", [_blade, _tower], ids=["blade", "tower"])
    def test_dense_agrees_with_sparse(self, factory, monkeypatch):
        """Both paths are shift-invert at heart, so they must agree."""
        dense = _frequencies(factory, 10)
        monkeypatch.setattr(solver, "_SPARSE_NDOF_THRESHOLD", 0)
        sparse = _frequencies(factory, 10)
        np.testing.assert_allclose(dense, sparse, rtol=1.0e-6)

    def test_the_dense_path_actually_ran(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            diag = _blade().run(n_modes=10).diagnostics
        assert diag is not None
        assert diag.path == "dense_symmetric"
        assert diag.residual_fallback is False
        # The mass-reduced route left this deck at a backward error of
        # ~5e-2; the inverted one solves it to roundoff.
        assert diag.max_residual < 1.0e-6

    def test_the_mass_reduced_route_really_did_depend_on_the_window(self):
        """Pins the failure the fix removes, on the same matrices."""
        from scipy.linalg import eigh

        import pybmodes.models._pipeline as pipeline

        captured = {}
        original = pipeline.solve_modes

        def _capture(gk, gm, n_modes=None, **kwargs):
            captured["pair"] = (gk.copy(), gm.copy())
            return original(gk, gm, n_modes, **kwargs)

        mp = pytest.MonkeyPatch()
        mp.setattr(pipeline, "solve_modes", _capture)
        try:
            _frequencies(_blade, 4)
        finally:
            mp.undo()
        gk, gm = captured["pair"]
        gk = 0.5 * (gk + gk.T)
        gm = 0.5 * (gm + gm.T)
        low = [
            eigh(gk, gm, subset_by_index=(0, n - 1), eigvals_only=True)[0]
            for n in (4, 10)
        ]
        assert abs(low[0] - low[1]) / low[1] > 1.0e-2


class TestNearSingularMassSolvesDirectly:
    """The heavy-lump case the residual retry was built for no longer
    needs it: factorising K is unaffected by a near-singular M."""

    @pytest.mark.parametrize("nselt", [13, 27, 53])
    def test_analytic_frequency_without_a_retry(self, nselt):
        gk, gm = _cantilever_with_tip_lump(nselt, 1.0e-2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eigvals, _v, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        f = float(eigvals_to_hz(eigvals, ROMG)[0])
        f_ref = float(np.sqrt(3.0 * EI / (M_TIP * L**3)) / (2.0 * np.pi))
        assert f == pytest.approx(f_ref, rel=5.0e-3)
        assert diag.path == "dense_symmetric"
        assert diag.residual_fallback is False
        assert diag.max_residual < 1.0e-6


class TestShiftSelection:
    """``K`` is not always definite; the shift has to cope."""

    @staticmethod
    def _free_free(seed: int, n_zero: int) -> tuple[np.ndarray, np.ndarray]:
        n = 14
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(n, n))
        m = a @ a.T + n * np.eye(n)
        b = rng.normal(size=(n, n - n_zero))
        k = b @ b.T
        return 0.5 * (k + k.T), 0.5 * (m + m.T)

    @pytest.mark.parametrize("n_zero", [1, 3, 6])
    def test_rigid_body_modes_and_elastic_modes(self, n_zero):
        from scipy.linalg import eigh

        gk, gm = self._free_free(11, n_zero)
        ref = eigh(gk, gm, eigvals_only=True)
        vals, vecs = solver._solve_dense_symmetric(gk, gm, 8)
        scale = float(np.max(np.abs(ref)))
        assert np.all(np.abs(vals[:n_zero]) < 1.0e-9 * scale)
        np.testing.assert_allclose(vals[n_zero:], ref[n_zero:8], rtol=1.0e-9)
        residual = gk @ vecs - (gm @ vecs) * vals
        assert np.max(np.abs(residual)) < 1.0e-8 * np.max(np.abs(gk))

    def test_negative_eigenvalues_are_kept(self):
        """An indefinite K — a column loaded past its buckling weight —
        must come back with its negative modes, as ``eigh`` would."""
        from scipy.linalg import eigh

        rng = np.random.default_rng(3)
        q, _r = np.linalg.qr(rng.normal(size=(10, 10)))
        gk = q @ np.diag([-2.0, -0.5, 0.3, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0,
                          21.0]) @ q.T
        gk = 0.5 * (gk + gk.T)
        gm = np.eye(10)
        vals, _v = solver._solve_dense_symmetric(gk, gm, 5)
        np.testing.assert_allclose(
            vals, eigh(gk, gm, eigvals_only=True)[:5], rtol=1.0e-10,
        )


class TestFullSpectrum:
    def test_full_request_matches_a_subset_at_the_bottom(self):
        gk, gm = _cantilever_with_tip_lump(27, 1.0e-2)
        low, _v = solve_modes(gk, gm, n_modes=6)
        full, _w = solve_modes(gk, gm, n_modes=None)
        assert full.size == gk.shape[0]
        # Relative accuracy of mode i is about eps * lambda_i / lambda_1;
        # this pencil spans 3e7 across the six, so 1e-6 is roundoff.
        np.testing.assert_allclose(full[:6], low, rtol=1.0e-6)

    def test_full_request_keeps_the_top_of_the_spectrum(self):
        """The inverted form is weakest at the top, where the splice takes
        the mass-reduced form instead; the largest eigenvalue must match
        that form, which is accurate there."""
        from scipy.linalg import eigh

        gk, gm = _cantilever_with_tip_lump(27, 1.0e-2)
        full, _w = solve_modes(gk, gm, n_modes=None)
        top = eigh(0.5 * (gk + gk.T), 0.5 * (gm + gm.T), eigvals_only=True)
        assert full[-1] == pytest.approx(top[-1], rel=1.0e-9)
        assert np.all(np.isfinite(full))
        assert np.all(np.diff(full) >= 0.0)
