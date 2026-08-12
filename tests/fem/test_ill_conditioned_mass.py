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

import warnings

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

    def test_degenerate_pair_comes_back_as_a_pair(self):
        """The property the improvement bar exists to protect: a tower
        with EI_FA == EI_SS returns its bending modes as one degenerate
        pair rather than two separated modes.

        The tolerance is loose on purpose. How exactly the pair resolves
        depends on the LAPACK build — this same model splits it at ~1e-16
        on one and ~3e-4 on another — so pinning it tightly tests the
        vendor's BLAS rather than pyBmodes. What matters here is that the
        two remain the same mode to engineering precision.
        """
        gk, gm = _cantilever_with_tip_lump(27, REALISTIC)
        eigvals, _v = solve_modes(gk, gm, n_modes=4)
        f = eigvals_to_hz(eigvals, ROMG)
        assert f[0] == pytest.approx(f[1], rel=1.0e-2)


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
        from pybmodes.fem.solver import (
            _general_spectrum_for_retry,
            _solve_dense_general,
        )

        gk, gm = self._free_free_with_a_zero_mode()
        dropped, _v = _solve_dense_general(gk, gm, 6)
        kept, _w, _ok = _general_spectrum_for_retry(gk, gm, 6)
        assert abs(kept[0]) < 1.0e-8 * float(np.max(np.abs(kept)))
        assert np.min(dropped) > 0.0
        # The default filter loses the zero mode and shifts the rest up.
        assert kept[1] == pytest.approx(dropped[0], rel=1.0e-6)

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

    def test_the_maxima_rule_misses_what_the_per_mode_rule_catches(self):
        """The mechanism, pinned on residual vectors directly.

        Stated as arithmetic rather than run through LAPACK on purpose:
        whether a given matrix pair happens to exhibit the masking
        depends on how badly that build's BLAS degrades, which is not
        what this is about. These are the numbers the two rules see —
        two rigid-body modes reading ~1 in both candidates, one elastic
        mode corrupted to 0.8 and fixed to 1e-9, one mode already exact.
        """
        from pybmodes.fem.solver import _compare_candidate_modes

        sym_r = np.array([1.0, 1.0, 0.8, 1.0e-12])
        alt_r = np.array([1.0, 1.0, 1.0e-9, 1.0e-12])

        # The rigid modes floor the alternative's maximum at ~1, so a
        # maxima rule sees no decisive win and leaves the corruption in.
        assert not (alt_r.max() < 0.1 * sym_r.max())

        # Per mode, the corrupted one is unmissable and the rigid ones
        # register as exactly what they are: no improvement either way.
        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 4, 4)
        assert improved.tolist() == [False, False, True, False]
        assert not regressed.any()

    def test_a_shorter_alternative_is_never_an_improvement(self):
        from pybmodes.fem.solver import _compare_candidate_modes

        sym_r = np.array([1.0, 0.8, 1.0e-12])
        alt_r = np.array([1.0e-9, 1.0e-9])
        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 2, 3)
        assert not improved.any()
        assert not regressed.any()

    def test_the_result_is_never_made_worse(self):
        """The portable guarantee when rigid modes and an ill-conditioned
        mass matrix coincide.

        Whether the rescue can *fire* here is build-dependent, and
        deliberately so. QZ may represent this problem's theoretically
        real zero modes as small complex-conjugate pairs that
        ``real_if_close`` will not coerce; where those land in the
        spectrum differs between LAPACK builds. On one they sit at the
        stiff end, far above anything a caller asks for, and the retry
        proceeds. On another they are the zero modes themselves, and
        keeping the alternative would mean backfilling the gap with
        elastic modes and silently shifting the spectrum.

        So the guard verifies its ordering and declines when it cannot,
        and the property asserted here is the one that holds either way:
        the requested modes come back, in ascending order, and nothing is
        lost. ``TestIllConditionedMassIsCaught`` pins the rescue itself on
        the case it was built for, which has no rigid modes and behaves
        identically everywhere.
        """
        gk, gm = self._rigid_plus_ill_conditioned()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigvals, _v, diag = solve_modes(
                gk, gm, n_modes=10, return_diagnostics=True,
            )
        assert eigvals.size == 10
        assert diag.n_returned == 10
        assert np.all(np.diff(eigvals) >= -1.0e-12 * max(1.0, abs(eigvals).max()))

    def test_an_unverifiable_ordering_declines_the_swap(self, monkeypatch):
        """When the alternative drops a mode from inside the window, its
        ordering cannot be trusted and the symmetric result stands."""
        import pybmodes.fem.solver as solvermod

        real_fn = solvermod._general_spectrum_for_retry

        def unsound(gk, gm, n_modes):
            vals, vecs, _ok = real_fn(gk, gm, n_modes)
            return vals, vecs, False

        monkeypatch.setattr(solvermod, "_general_spectrum_for_retry", unsound)
        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.residual_fallback is False
        assert diag.path == "dense_symmetric"


class TestNegativeEigenvaluesSurviveTheRetry:
    """An indefinite ``K`` must not have its unstable modes filtered away.

    ``run(gravity=...)`` past a column's buckling weight drives
    eigenvalues genuinely negative, and ``eigh`` returns them as-is. If
    the alternative solve filtered them, it would come back the same
    *length* — the gap backfilled from higher up — while describing a
    shifted spectrum, and the per-index comparison would be reading two
    different spectra against each other.
    """

    def _indefinite(self):
        """The ill-conditioned cantilever with a genuinely negative mode
        bolted on and the pair rotated together."""
        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        n = gk.shape[0]
        big_k = np.zeros((n + 2, n + 2))
        big_m = np.zeros((n + 2, n + 2))
        big_k[:n, :n] = gk
        big_m[:n, :n] = gm
        scale = float(np.trace(gk)) / n
        big_k[n, n] = -scale          # unstable
        big_k[n + 1, n + 1] = scale
        big_m[n:, n:] = np.eye(2) * float(np.trace(gm)) / n
        q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(n + 2, n + 2)))
        k_rot, m_rot = q.T @ big_k @ q, q.T @ big_m @ q
        return 0.5 * (k_rot + k_rot.T), 0.5 * (m_rot + m_rot.T)

    def test_the_default_filter_would_drop_the_unstable_mode(self):
        from pybmodes.fem.solver import (
            _general_spectrum_for_retry,
            _solve_dense_general,
        )

        gk, gm = self._indefinite()
        dropped, _v = _solve_dense_general(gk, gm, 6)
        kept, _w, _ok = _general_spectrum_for_retry(gk, gm, 6)
        assert kept.min() < 0.0            # the unstable mode is present
        assert dropped.min() > 0.0         # and absent from the default
        # Same length, shifted spectrum — the trap the index comparison
        # would otherwise walk into.
        assert kept.size == dropped.size
        assert kept[1] == pytest.approx(dropped[0], rel=1.0e-6)

    def test_the_symmetric_solve_reports_it_too(self):
        """Both paths must agree that the mode exists, or index matching
        is meaningless."""
        from scipy.linalg import eigh

        gk, gm = self._indefinite()
        w = eigh(gk, gm, subset_by_index=(0, 5))[0]
        assert w.min() < 0.0

    def test_solve_modes_keeps_the_unstable_mode(self):
        gk, gm = self._indefinite()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigvals, _v = solve_modes(gk, gm, n_modes=6)
        assert eigvals.size == 6
        assert eigvals.min() < 0.0


class TestAcceptedSkewDoesNotTriggerTheRetry:
    """A tolerated asymmetry must not be read as a solver failure.

    ``_is_effectively_symmetric`` accepts skew up to ``symmetry_rtol``
    times ``max|K|``, and the symmetric paths then solve the symmetrised
    matrices. That skew is only small *relative to the largest* entry: in
    a model with a wide dynamic range it can be comparable to a soft
    mode's own eigenvalue. Measuring the resulting modes against the
    unsymmetrised matrices makes an exact solve look broken, and the
    general path then "wins decisively" only because it is answering a
    different question — returning the skewed spectrum in place of the
    symmetric one the caller was promised.
    """

    def _wide_range_with_accepted_skew(self):
        from pybmodes.options import DEFAULT_SOLVER_OPTIONS as opt

        # Eigenvalues spanning 1 down to 1e-12, so max|K| is 1 and the
        # tolerated skew is ~1e-12 — the same size as the softest mode.
        # The symmetry test compares max|A - A.T|, which is twice the
        # off-diagonal skew, so stay under half the tolerance.
        d = np.array([1.0, 1.0e-6, 1.0e-12])
        gm = np.eye(3)
        s = 0.4 * opt.symmetry_rtol * max(1.0, float(np.max(np.abs(d))))
        # Couple the two softest modes: skew between the stiff ones would
        # be negligible against their own scale and prove nothing.
        gk = np.diag(d) + np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, s],
                                    [0.0, -s, 0.0]])
        return gk, gm, d

    def test_the_pair_is_accepted_as_symmetric(self):
        from pybmodes.fem.solver import _is_effectively_symmetric

        gk, gm, _d = self._wide_range_with_accepted_skew()
        assert _is_effectively_symmetric(gk)
        assert _is_effectively_symmetric(gm)

    def test_no_retry_and_the_symmetric_spectrum_is_returned(self):
        gk, gm, d = self._wide_range_with_accepted_skew()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eigvals, _v, diag = solve_modes(
                gk, gm, n_modes=3, return_diagnostics=True,
            )
        assert diag.residual_fallback is False
        assert diag.path == "dense_symmetric"
        # The symmetrised problem's spectrum, not the skewed one.
        assert np.allclose(np.sort(eigvals), np.sort(d), rtol=1.0e-6)

    def test_the_diagnostics_report_the_solved_problem_too(self):
        """Telemetry meant to be auditable must not charge a correct
        solve for the skew it was told to discard."""
        gk, gm, _d = self._wide_range_with_accepted_skew()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _v, _x, diag = solve_modes(gk, gm, n_modes=3,
                                       return_diagnostics=True)
        assert diag.residual_fallback is False
        assert diag.max_residual < 1.0e-8
        assert max(diag.residuals) < 1.0e-8

    def test_measuring_against_the_unsymmetrised_pair_would_have_tripped(self):
        """The mechanism: the same exact modes look broken when judged
        against matrices they were never solved on."""
        from pybmodes.fem.solver import _modal_residuals

        gk, gm, _d = self._wide_range_with_accepted_skew()
        gk_s = 0.5 * (gk + gk.T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigvals, eigvecs = solve_modes(gk, gm, n_modes=3)
        against_solved = _modal_residuals(gk_s, gm, eigvals, eigvecs)
        against_raw = _modal_residuals(gk, gm, eigvals, eigvecs)
        assert against_solved.max() < 1.0e-8
        assert against_raw.max() > 0.1


class TestTheRetryVerifiesItsOrderingRatherThanAssumingIt:
    """Index matching is only sound if the alternative recovered every
    mode.

    A symmetric problem has ``ngd`` real eigenvalues, so if ``eig``
    returns fewer after its real / finite filter, one was discarded — and
    a truncated request would have backfilled the gap from higher up,
    leaving equal indices pointing at different modes. The sign filter was
    one route to that and is switched off; a rounding-induced complex pair
    is another that no flag can prevent, so completeness is checked rather
    than assumed.
    """

    def test_a_sound_ordering_is_the_precondition_for_swapping(self):
        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        with pytest.warns(RuntimeWarning):
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        # This case does swap, so the precondition held.
        assert diag.residual_fallback is True

    def test_a_drop_above_the_window_leaves_the_ordering_sound(self):
        """Complex pairs at the stiff end are routine and harmless — they
        sit far above anything the caller asked for, so the window is
        still the smallest ``n_modes``.

        Built from an explicit rotation block rather than by hoping a
        real matrix produces one, so it says the same thing on every
        LAPACK build. ``[[a, -b], [b, a]]`` has eigenvalues ``a ± bi``;
        with ``a`` far above the window and ``b`` far too large for
        ``real_if_close`` to coerce, the pair is dropped from well above
        the modes being compared.
        """
        from pybmodes.fem.solver import _general_spectrum_for_retry

        gm = np.eye(4)
        gk = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0e6, -1.0e3],
            [0.0, 0.0, 1.0e3, 1.0e6],
        ])
        vals, _v, sound = _general_spectrum_for_retry(gk, gm, 2)
        assert sound is True
        assert vals.size == 2
        assert np.allclose(vals, [1.0, 2.0])

    def test_a_drop_inside_the_window_makes_the_ordering_unsound(self):
        """A discarded eigenvalue below the top of the window means the
        window is not the smallest ``n_modes`` and cannot be compared by
        index."""
        from pybmodes.fem.solver import _general_spectrum_for_retry

        # Two well-separated real modes plus a complex pair placed below
        # them, which no coercion will make real.
        gm = np.eye(4)
        gk = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0e-3],
            [0.0, 0.0, 1.0e-3, 0.0],
        ])
        vals, _v, sound = _general_spectrum_for_retry(gk, gm, 2)
        assert sound is False
        assert vals.size == 2


class TestTheRetryIsScopedToTheDensePath:
    """Only the dense symmetric path is retried, and that is about which
    matrix each routine factorises.

    ``eigh`` reduces through a Cholesky factor of the **mass** matrix,
    which is the failure this guard exists for. ``eigsh(sigma=0,
    mode='normal')`` factorises ``K`` instead and is unaffected — the
    mesh sweep that motivated the work returns correct frequencies on
    exactly the meshes large enough to take the sparse path.

    It also removes a spectrum mismatch by construction: ``which="LM"``
    selects the modes nearest zero *in magnitude* while the retry selects
    the algebraically smallest, and with negative eigenvalues present
    those are different sets that must never be compared by index.
    """

    def test_the_sparse_path_is_left_alone(self, monkeypatch):
        import pybmodes.fem.solver as solvermod

        # Force the sparse path on a small problem, then make the
        # residual look terrible. The retry must still not run.
        gk, gm = _cantilever_with_tip_lump(13, LIGHT)
        monkeypatch.setattr(solvermod, "_SPARSE_NDOF_THRESHOLD", 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.path == "sparse_shift_invert"
        assert diag.residual_fallback is False

    def test_the_sparse_path_gets_the_ill_conditioned_case_right(self):
        """Why leaving it alone is safe rather than a gap: it factorises
        K, so a near-singular M does not degrade it."""
        gk, gm = _cantilever_with_tip_lump(101, LIGHT)
        eigvals, _v, diag = solve_modes(
            gk, gm, n_modes=4, return_diagnostics=True,
        )
        assert diag.path == "sparse_shift_invert"
        f = float(eigvals_to_hz(eigvals, ROMG)[0])
        assert f == pytest.approx(_analytic(), rel=5.0e-3)


class TestTheRetryCostIsBounded:
    """A guard against a silent wrong answer must not be able to turn one
    into a silent hang.

    The retry is a dense ``eig``, whose cost grows as ``ngd^3`` with a
    much larger constant than the ``eigh`` it checks. Normally that is
    bounded by the sparse dispatch threshold, but a sparse solve that
    *fails to converge* falls back to dense at any size.
    """

    def test_a_system_above_the_ceiling_is_not_retried(self, monkeypatch):
        import dataclasses

        import pybmodes.fem.solver as solvermod

        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        ngd = gk.shape[0]
        monkeypatch.setattr(
            solvermod, "_SOLVER_OPTIONS",
            dataclasses.replace(
                solvermod._SOLVER_OPTIONS, residual_retry_max_ndof=ngd - 1,
            ),
        )
        called = []
        real = solvermod._general_spectrum_for_retry
        monkeypatch.setattr(
            solvermod, "_general_spectrum_for_retry",
            lambda *a, **k: (called.append(1), real(*a, **k))[1],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert called == []
        assert diag.residual_fallback is False

    def test_at_the_ceiling_it_still_runs(self, monkeypatch):
        import dataclasses

        import pybmodes.fem.solver as solvermod

        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        monkeypatch.setattr(
            solvermod, "_SOLVER_OPTIONS",
            dataclasses.replace(
                solvermod._SOLVER_OPTIONS,
                residual_retry_max_ndof=gk.shape[0],
            ),
        )
        with pytest.warns(RuntimeWarning):
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.residual_fallback is True


class TestARetryFailureIsNotAHardFailure:
    """A pencil defective enough to break the symmetric reduction can
    also break ``eig``. Turning that into an exception would make the
    guard destroy usable results on exactly the inputs it exists for."""

    def test_a_raising_retry_keeps_the_symmetric_result(self, monkeypatch):
        import pybmodes.fem.solver as solvermod

        def boom(gk, gm, n_modes):
            raise np.linalg.LinAlgError("did not converge")

        monkeypatch.setattr(solvermod, "_general_spectrum_for_retry", boom)
        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigvals, _v, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.residual_fallback is False
        assert diag.path == "dense_symmetric"
        assert eigvals.size == 4

    def test_a_value_error_is_handled_the_same_way(self, monkeypatch):
        import pybmodes.fem.solver as solvermod

        def boom(gk, gm, n_modes):
            raise ValueError("array must not contain infs or NaNs")

        monkeypatch.setattr(solvermod, "_general_spectrum_for_retry", boom)
        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.residual_fallback is False


class TestARetryThatTradesModesIsRefused:
    """Accepting replaces the whole spectrum, so a candidate that fixes
    one mode while ruining another is not an improvement to the result.

    The failing shape: the general solve is exact on the modes that
    prompted the retry but pushes a previously acceptable mode above the
    failure threshold. Judging on "any mode improved" would take it and
    hand back a spectrum with a new bad mode in place of an old one.
    """

    def test_a_trade_is_not_an_improvement(self):
        from pybmodes.fem.solver import _compare_candidate_modes

        # Mode 0 rescued decisively; mode 3 was fine and is now above the
        # threshold and an order worse.
        sym_r = np.array([0.52, 1.1e-3, 5.7e-6, 0.017])
        alt_r = np.array([1.1e-10, 3.0e-11, 3.5e-5, 0.26])

        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 4, 4)
        assert improved[0]
        assert regressed[3]
        # The rule the caller applies.
        assert not (improved.any() and not regressed.any())

    def test_a_mode_that_worsens_but_stays_acceptable_is_not_a_regression(self):
        """Mode 2 above goes from 5.7e-6 to 3.5e-5 — six times worse and
        entirely irrelevant, since it is nowhere near the threshold."""
        from pybmodes.fem.solver import _compare_candidate_modes

        sym_r = np.array([0.52, 5.7e-6])
        alt_r = np.array([1.1e-10, 3.5e-5])
        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 2, 2)
        assert improved[0]
        assert not regressed.any()

    def test_rigid_body_noise_is_not_a_regression(self):
        """Residuals that read ~1 in both candidates wobble either way.
        A bare ``alt > sym`` test would call that a regression and block
        every rescue that happens to sit beside a free-free mode."""
        from pybmodes.fem.solver import _compare_candidate_modes

        sym_r = np.array([1.0, 0.8])
        alt_r = np.array([1.0001, 1.0e-9])
        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 2, 2)
        assert improved[1]
        assert not regressed.any()

    def test_end_to_end_a_trading_candidate_is_declined(self, monkeypatch):
        import pybmodes.fem.solver as solvermod

        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        real = solvermod._general_spectrum_for_retry

        def trading(gk_, gm_, n_modes):
            vals, vecs, sound = real(gk_, gm_, n_modes)
            # Corrupt the last returned mode so it is decisively worse.
            vecs = vecs.copy()
            vecs[:, -1] = np.roll(vecs[:, -1], 1)
            return vals, vecs, sound

        monkeypatch.setattr(
            solvermod, "_general_spectrum_for_retry", trading,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _v, _x, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.residual_fallback is False
        assert diag.path == "dense_symmetric"


class TestTheModeCountWarningStaysHonest:
    """A shortfall is only newsworthy when modes were actually discarded.

    Two benign cases would otherwise be reported as a defective
    eigenproblem: asking for more modes than the system has, which every
    path truncates, and a residual retry, which relabels the path
    ``"dense_general"`` while preserving the whole spectrum.
    """

    def test_an_overlarge_request_after_a_retry_is_not_reported(self):
        gk, gm = _cantilever_with_tip_lump(13, LIGHT)
        ngd = gk.shape[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eigvals, _v, diag = solve_modes(
                gk, gm, n_modes=ngd + 500, return_diagnostics=True,
            )
        assert diag.residual_fallback is True
        assert eigvals.size == ngd
        assert not any(
            "recovered only" in str(w.message) for w in caught
        ), [str(w.message) for w in caught]

    def test_an_overlarge_request_without_a_retry_is_not_reported(self):
        gk, gm = _cantilever_with_tip_lump(13, REALISTIC)
        ngd = gk.shape[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eigvals, _v = solve_modes(gk, gm, n_modes=ngd + 500)
        assert eigvals.size == ngd
        assert not any("recovered only" in str(w.message) for w in caught)

    def test_a_genuine_shortfall_is_still_reported(self):
        """An asymmetric pencil whose modes really are filtered away must
        still say so."""
        # Antisymmetric K: every eigenvalue is imaginary, so the general
        # path recovers none of them.
        gk = np.array([[0.0, -1.0], [1.0, 0.0]])
        gm = np.eye(2)
        with pytest.warns(RuntimeWarning, match="recovered only"):
            solve_modes(gk, gm, n_modes=2)


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
