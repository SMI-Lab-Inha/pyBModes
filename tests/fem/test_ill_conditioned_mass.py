"""The symmetric eigensolvers degrade silently on a near-singular mass
matrix, and the solver has to notice.

Both ``scipy.linalg.eigh`` and ``scipy.sparse.linalg.eigsh`` reduce
``K x = lambda M x`` through a Cholesky factor of the mass matrix. When
that matrix is nearly singular — a very light beam carrying a very heavy
lump — the reduction loses accuracy, and LAPACK returns confidently
wrong low modes rather than raising. On the case pinned below the dense
symmetric path reported 0.103 Hz against a true 0.0436 Hz, a factor of
2.4, with no error and no warning.

The guard is the backward error ``||K x - lambda M x|| / ||K x||``. It
has two conditions, and the second matters more than the first: the
error must exceed the retry threshold, *and* the general path must beat
it decisively. Being above the threshold alone is not evidence of a
breakdown — the bundled NREL 5MW land deck sits at ~2e-2 because its
adapter leaves ``M`` at cond ~4e10, and there the general path is only
1.4x better while splitting a degenerate fore-aft / side-side pair the
symmetric solver resolves exactly. Swapping for that would churn a
validated frequency by 0.84 % and break the FA / SS classifier
downstream. A real breakdown improves by nine orders, not by a factor.

Analytical reference: a cantilever whose beam mass is negligible next to
a tip lump behaves as a spring-mass oscillator on the static tip
stiffness ``3 EI / L^3``, giving ``f = sqrt(3 EI / (m L^3)) / 2 pi``
(Blevins 1979, Table 8-1).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.assembly import assemble
from pybmodes.fem.nondim import RM, ROMG, make_params, nondim_tip_mass
from pybmodes.fem.solver import eigvals_to_hz, solve_modes
from pybmodes.io.bmi import TipMassProps

L = 100.0
EI = 1.0e10
M_TIP = 4.0e5

# The lump dominates the beam by ~4e5:1 here, which drives cond(M) past
# 1e12 and is what breaks the Cholesky reduction.
LIGHT = 1.0e-2
REALISTIC = 1.0e3


def _analytic() -> float:
    return float(np.sqrt(3.0 * EI / (M_TIP * L**3)) / (2.0 * np.pi))


def _cantilever_with_tip_lump(
    nselt: int, mass_den: float,
) -> tuple[np.ndarray, np.ndarray]:
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


def _first_frequency(nselt: int, mass_den: float):
    gk, gm = _cantilever_with_tip_lump(nselt, mass_den)
    with pytest.warns(RuntimeWarning, match="backward error"):
        eigvals, _vecs, diag = solve_modes(
            gk, gm, n_modes=4, return_diagnostics=True,
        )
    return float(eigvals_to_hz(eigvals, ROMG)[0]), diag


class TestIllConditionedMassIsCaught:
    """Sizes chosen to straddle the dense / sparse dispatch threshold."""

    @pytest.mark.parametrize("nselt", [13, 27, 53])
    def test_dense_path_recovers_the_analytic_frequency(self, nselt):
        f, diag = _first_frequency(nselt, LIGHT)
        assert f == pytest.approx(_analytic(), rel=5.0e-3)
        assert diag.residual_fallback is True
        assert diag.path == "dense_general"

    @pytest.mark.parametrize("nselt", [13, 27, 53])
    def test_the_symmetric_result_would_have_been_wrong(self, nselt):
        """Without the guard the answer is not merely imprecise."""
        from scipy.linalg import eigh

        gk, gm = _cantilever_with_tip_lump(nselt, LIGHT)
        raw = eigh(
            0.5 * (gk + gk.T), 0.5 * (gm + gm.T), subset_by_index=(0, 3),
        )[0]
        f_raw = float(eigvals_to_hz(raw, ROMG)[0])
        assert abs(f_raw - _analytic()) / _analytic() > 0.25

    def test_backward_error_drops_after_the_retry(self):
        _f, diag = _first_frequency(27, LIGHT)
        assert diag.max_residual < 1.0e-6


class TestHealthyProblemsAreUntouched:
    """The guard must not fire on a mass distribution any real structure
    would have — the standing rule for a numerical gate."""

    @pytest.mark.parametrize("nselt", [13, 27, 53])
    def test_realistic_mass_ratio_stays_on_the_symmetric_path(self, nselt):
        import warnings as _w

        gk, gm = _cantilever_with_tip_lump(nselt, REALISTIC)
        with _w.catch_warnings():
            _w.simplefilter("error")
            eigvals, _vecs, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.residual_fallback is False
        assert diag.symmetric is True
        f = float(eigvals_to_hz(eigvals, ROMG)[0])
        # The beam mass now matters, so this sits a little below the
        # massless-beam closed form rather than on it.
        assert 0.9 * _analytic() < f < _analytic()

    def test_healthy_residuals_keep_real_margin_under_the_threshold(self):
        """A 400:1 lump-to-beam ratio is already a demanding model and is
        the worst healthy backward error in the suite, at ~1.3e-3. The
        threshold sits well above that and an order below the
        degraded regime, so neither side is close.

        A false positive would cost time rather than accuracy in any
        case: the retry keeps whichever of the two solves has the lower
        backward error.
        """
        from pybmodes.options import DEFAULT_SOLVER_OPTIONS as opt

        worst = 0.0
        for nselt in (13, 27, 53):
            gk, gm = _cantilever_with_tip_lump(nselt, REALISTIC)
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
            worst = max(worst, diag.max_residual)
        assert worst < 0.05 * opt.residual_retry_threshold


class TestMarginalImprovementIsRefused:
    """The decisive-win condition, pinned directly.

    A synthetic pair where the symmetric solve is above the threshold but
    the general path cannot do materially better must keep the symmetric
    result — the behaviour that protects the bundled land deck's
    degenerate fore-aft / side-side pair.
    """

    def test_marginal_gain_keeps_the_symmetric_result(self, monkeypatch):
        import dataclasses

        import pybmodes.fem.solver as solvermod

        gk, gm = _cantilever_with_tip_lump(27, LIGHT)

        # Force the "general path is barely better" situation by making
        # the improvement bar unreachable, leaving the threshold tripped.
        monkeypatch.setattr(
            solvermod, "_SOLVER_OPTIONS",
            dataclasses.replace(
                solvermod._SOLVER_OPTIONS, residual_retry_improvement=0.0,
            ),
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.residual_fallback is False
        assert diag.path == "dense_symmetric"

    def test_degenerate_pair_survives_a_symmetric_solve(self):
        """The property the improvement bar exists to protect: a tower
        with EI_FA == EI_SS returns its bending modes as an exactly
        degenerate pair, which the general path would split."""
        gk, gm = _cantilever_with_tip_lump(27, REALISTIC)
        eigvals, _v = solve_modes(gk, gm, n_modes=4)
        f = eigvals_to_hz(eigvals, ROMG)
        assert f[0] == pytest.approx(f[1], rel=1.0e-9)


class TestRigidBodyModesAreNotMistakenForBreakdown:
    """A free-free model's zero-frequency modes must not corrupt the result.

    For a rigid-body mode ``K x ~ 0`` and ``lambda ~ 0``, so the relative
    residual is a ratio of two near-zero quantities and reads ~1 however
    exact the eigenpair is. Two attempts to *identify* such modes and
    exclude them both failed — an eigenvalue-relative cutoff takes a
    rigid-only subset's own noise as its scale, and a strain-relative one
    cannot tell a rigid mode from a genuinely soft one (the 0.08 Hz lump
    mode of a 1e10 N.m^2 beam carries less strain than a floating
    platform's rigid modes do, and excluding it blinded the guard to a
    case it had caught).

    So they are not identified at all. Both candidate solves are measured
    the same way and the retry needs a decisive win, so a mode the metric
    cannot speak to says the same nothing twice. The retry also preserves
    zero eigenvalues, which is what makes a false positive merely wasteful
    instead of destructive.
    """

    def _free_free_with_a_zero_mode(self):
        """A symmetric platform with no yaw restoring: one exactly zero
        eigenvalue among otherwise well-conditioned elastic modes."""
        n = 12
        rng = np.random.default_rng(7)
        a = rng.normal(size=(n, n))
        m = a @ a.T + n * np.eye(n)          # SPD, well conditioned
        b = rng.normal(size=(n, n - 1))
        k = b @ b.T                          # rank n-1: one zero mode
        return 0.5 * (k + k.T), 0.5 * (m + m.T)

    def test_zero_mode_survives_whichever_path_runs(self):
        gk, gm = self._free_free_with_a_zero_mode()
        eigvals, _v, diag = solve_modes(
            gk, gm, n_modes=6, return_diagnostics=True,
        )
        # The zero mode survives rather than being filtered away, and the
        # spectrum keeps its full width either way.
        assert abs(eigvals[0]) < 1.0e-8 * abs(eigvals).max()
        assert eigvals.size == 6
        assert diag.n_returned == 6

    def test_the_metric_really_does_read_about_one_there(self):
        """Why the naive raw-maximum reading is untrustworthy, pinned so
        the reasoning above stays anchored to a number."""
        from pybmodes.fem.solver import _modal_residuals

        gk, gm = self._free_free_with_a_zero_mode()
        eigvals, eigvecs = solve_modes(gk, gm, n_modes=6)
        raw = _modal_residuals(gk, gm, eigvals, eigvecs)
        rigid = np.argmin(np.abs(eigvals))
        assert raw[rigid] > 0.1
        # Every elastic mode is exact, so the ~1 is the metric failing,
        # not the solve.
        elastic = np.ones(raw.size, dtype=bool)
        elastic[rigid] = False
        assert raw[elastic].max() < 1.0e-8

    def test_the_retry_preserves_zero_eigenvalues(self):
        """The property that makes a false positive harmless: the
        alternative solve keeps the rigid modes rather than filtering
        them, so both candidates describe the same spectrum."""
        from pybmodes.fem.solver import _solve_dense_general

        gk, gm = self._free_free_with_a_zero_mode()
        dropped, _v = _solve_dense_general(gk, gm, 6)
        kept, _w = _solve_dense_general(gk, gm, 6, keep_rigid_body=True)
        assert np.min(np.abs(kept)) == 0.0
        assert np.min(np.abs(dropped)) > 0.0
        # The default filter loses the zero mode and shifts the rest up.
        assert kept[1] == pytest.approx(dropped[0], rel=1.0e-9)

    def _six_rigid_dofs(self):
        """A model with six genuinely free rigid-body DOFs above a set of
        elastic ones — an unmoored floating platform in the limit."""
        n = 14
        rng = np.random.default_rng(11)
        a = rng.normal(size=(n, n))
        m = a @ a.T + n * np.eye(n)
        b = rng.normal(size=(n, n - 6))
        k = b @ b.T                          # rank n-6: six zero modes
        return 0.5 * (k + k.T), 0.5 * (m + m.T)

    @pytest.mark.parametrize("n_modes", [1, 3, 6])
    def test_a_rigid_only_subset_keeps_its_modes(self, n_modes):
        """The case that broke both classification attempts.

        Every requested mode is rigid-body, so the metric reads ~1 on all
        of them and no reference scale drawn from the subset can say
        otherwise. The retry may well run; what matters is that it cannot
        take modes away, because it now preserves zero eigenvalues and
        has to win decisively to be accepted at all.
        """
        gk, gm = self._six_rigid_dofs()
        eigvals, _v, diag = solve_modes(
            gk, gm, n_modes=n_modes, return_diagnostics=True,
        )
        assert eigvals.size == n_modes
        assert np.all(np.abs(eigvals) < 1.0e-8 * float(np.linalg.norm(gk)))
        assert diag.n_returned == n_modes

    def test_a_mixed_subset_is_measured_on_its_elastic_modes(self):
        """With elastic modes present the metric is meaningful again and
        reports them as exact."""
        gk, gm = self._six_rigid_dofs()
        eigvals, _v, diag = solve_modes(
            gk, gm, n_modes=10, return_diagnostics=True,
        )
        assert eigvals.size == 10
        assert np.max(np.abs(eigvals)) > 1.0e-8 * float(np.linalg.norm(gk))
        # Six rigid modes at the bottom, four exact elastic ones above.
        assert np.sum(np.abs(eigvals) < 1.0e-12 * np.max(eigvals)) == 6
        assert max(diag.residuals[6:]) < 1.0e-8


class TestRigidModesCannotMaskAnElasticBreakdown:
    """A corrupted elastic mode must be caught even when rigid-body modes
    sit alongside it.

    Comparing the two candidates on their *maxima* fails here: a
    rigid-body mode reads ~1 in both, so it floors the alternative's
    maximum and no improvement among the elastic modes can clear a
    decisive-win bar unless the symmetric solve is worse than ~10. A
    free-free model with an elastic mode corrupted to a backward error of
    ~0.8 would then pass silently — the exact failure this guard exists
    to catch, reintroduced by the rigid modes' presence.

    Comparing per mode removes the floor.
    """

    def _rigid_plus_ill_conditioned(self, nselt: int = 27):
        """Six free rigid DOFs bolted onto the ill-conditioned cantilever,
        then rotated so the two blocks are not separable by inspection."""
        gk, gm = _cantilever_with_tip_lump(nselt, LIGHT)
        n = gk.shape[0]
        big_k = np.zeros((n + 6, n + 6))
        big_m = np.zeros((n + 6, n + 6))
        big_k[:n, :n] = gk
        big_m[:n, :n] = gm
        # Zero stiffness, unit mass on the six extra DOFs: rigid-body.
        big_m[n:, n:] = np.eye(6) * float(np.trace(gm)) / n
        q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(n + 6, n + 6)))
        k_rot = q.T @ big_k @ q
        m_rot = q.T @ big_m @ q
        return 0.5 * (k_rot + k_rot.T), 0.5 * (m_rot + m_rot.T)

    def test_the_rigid_modes_really_do_floor_the_maximum(self):
        """The mechanism, pinned: on the maxima the alternative cannot
        look decisively better even though it is exact where it counts."""
        from pybmodes.fem.solver import _modal_residuals, _solve_dense_general

        gk, gm = self._rigid_plus_ill_conditioned()
        from scipy.linalg import eigh

        w, v = eigh(gk, gm, subset_by_index=(0, 9))
        v = v / np.linalg.norm(v, axis=0)
        sym_r = _modal_residuals(gk, gm, w, v)
        aw, av = _solve_dense_general(gk, gm, 10, keep_rigid_body=True)
        av = av / np.linalg.norm(av, axis=0)
        alt_r = _modal_residuals(gk, gm, aw, av)
        # The alternative's maximum is pinned near 1 by the rigid modes...
        assert alt_r.max() > 0.1
        # ...so a maxima comparison sees no decisive win.
        assert not alt_r.max() < 0.1 * sym_r.max()
        # ...yet some mode really is corrupted and really is fixed.
        assert ((sym_r > 0.1) & (alt_r < 0.1 * sym_r)).any()

    def test_the_breakdown_is_caught_and_corrected(self):
        gk, gm = self._rigid_plus_ill_conditioned()
        with pytest.warns(RuntimeWarning, match="do not satisfy"):
            eigvals, _v, diag = solve_modes(
                gk, gm, n_modes=10, return_diagnostics=True,
            )
        assert diag.residual_fallback is True
        assert eigvals.size == 10

    def test_the_corrected_spectrum_matches_the_underlying_one(self):
        """The rotation and the extra rigid DOFs do not change the
        cantilever's own eigenvalues, so the corrected solve must still
        contain the analytic lump frequency."""
        gk, gm = self._rigid_plus_ill_conditioned()
        with pytest.warns(RuntimeWarning):
            eigvals, _v = solve_modes(gk, gm, n_modes=10)
        f = eigvals_to_hz(eigvals, ROMG)
        assert np.min(np.abs(f - _analytic())) < 5.0e-3 * _analytic()


class TestDiagnosticsContract:
    def test_residual_fallback_defaults_to_false(self):
        gk, gm = _cantilever_with_tip_lump(13, REALISTIC)
        _v, _x, diag = solve_modes(gk, gm, n_modes=4, return_diagnostics=True)
        assert diag.residual_fallback is False

    def test_plain_two_tuple_return_still_works(self):
        """The retry must not change the historical return shape."""
        gk, gm = _cantilever_with_tip_lump(13, LIGHT)
        with pytest.warns(RuntimeWarning):
            out = solve_modes(gk, gm, n_modes=4)
        assert isinstance(out, tuple)
        assert len(out) == 2
