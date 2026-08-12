"""The symmetric eigensolvers degrade silently on a near-singular mass
matrix, and the solver has to notice.

``scipy.linalg.eigh`` reduces ``K x = lambda M x`` through a Cholesky
factor of the **mass** matrix. When that matrix is nearly singular — a
very light beam carrying a very heavy lump — the reduction loses
accuracy, and LAPACK returns confidently wrong low modes rather than
raising. On the case pinned below the dense symmetric path reported
0.103 Hz against a true 0.0436 Hz, a factor of 2.4, with no error and no
warning.

The sparse path is exempt and must stay exempt: ``eigsh(sigma=0,
mode='normal')`` factorises ``K`` instead, so a near-singular ``M`` does
not degrade it, and its ``which="LM"`` window selects a different set of
modes that must never be compared against the retry's by index.

The guard is the backward error ``||K x - lambda M x|| / ||K x||``, and
almost every test here exists because some reading of it turned out to
be wrong. Three things it does *not* establish, each learned the hard
way and each pinned below:

- **A large error is not evidence of a breakdown.** The bundled NREL 5MW
  land deck sits at ~2e-2 because its adapter leaves ``M`` at cond ~4e10;
  swapping there churns a validated frequency by 0.84 % and splits a
  degenerate fore-aft / side-side pair the symmetric solver resolves
  exactly, which the FA / SS classifier depends on.
- **A small error is not evidence of a rescue.** A rigid-body mode's
  residual divides one roundoff quantity by another, so its value is
  arbitrary: 12.4, 0.79 and 0.076 have all been measured on healthy
  models, and the last sits *below* the failure threshold. No absolute
  bar can separate that from a solved mode.
- **A better mode does not make a better spectrum.** Accepting replaces
  every mode, so a candidate that rescues one while pushing another past
  the threshold is a trade, not an improvement.

What does separate the populations is the *size* of the win. Genuine
rescues improve by 1e5 to 1e10; rigid roundoff by 11x to 16x.

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
    residual divides one roundoff quantity by another and its value is
    arbitrary. Such modes are deliberately *not* identified — three
    attempts to do so failed, which the module docstring of
    :mod:`pybmodes.fem.solver` records. They are neutralised instead, by
    judging the size of the win and by preserving zero eigenvalues so a
    false positive is wasteful rather than destructive.
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
        """The case that broke the eigenvalue-scale classifier.

        Every requested mode is rigid-body, so no reference scale drawn
        from the subset can say which of them is meaningful. The retry
        may well run; what matters is that it cannot
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


class TestTheWarningAttributesTheCauseHonestly:
    """The near-singular mass matrix motivated this guard but is not the
    only thing that trips it, and naming it unconditionally sends the
    reader to check something that may be perfectly fine."""

    def test_a_singular_mass_names_the_mass_matrix(self):
        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        with pytest.warns(RuntimeWarning, match="nearly singular here"):
            solve_modes(gk, gm, n_modes=4)

    def test_a_well_conditioned_mass_does_not_blame_it(self):
        """``M = I`` with a stiffness spectrum spanning 1e-16 to 1 trips
        the guard at ``cond(M) = 1``."""
        rng = np.random.default_rng(474)
        q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        gk = q @ np.diag([1.0e-16, 1.0e-8, 1.0]) @ q.T
        gk = 0.5 * (gk + gk.T)
        gm = np.eye(3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _v, _x, diag = solve_modes(gk, gm, n_modes=3,
                                       return_diagnostics=True)
        text = " ".join(str(w.message) for w in caught)
        if diag.residual_fallback:
            assert "well conditioned" in text
            assert "nearly singular" not in text
            assert "stiffness ratio" in text

    def test_the_message_states_what_was_measured(self):
        gk, gm = _cantilever_with_tip_lump(27, LIGHT)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_modes(gk, gm, n_modes=4)
        text = " ".join(str(w.message) for w in caught)
        assert "do not satisfy" in text
        assert "cond =" in text


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

    def test_a_small_crossing_of_the_threshold_is_still_a_regression(self):
        """A mode sliding from just under the threshold to well over it
        worsens by less than the decisive factor, so the mirrored test
        alone lets it through. Any crossing counts."""
        from pybmodes.fem.solver import _compare_candidate_modes

        sym_r = np.array([0.52, 0.09])
        alt_r = np.array([1.0e-10, 0.8])
        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 2, 2)
        assert improved[0]
        assert regressed[1]
        assert not (improved.any() and not regressed.any())

    def test_an_exact_mode_driven_to_just_under_the_bar_is_a_regression(self):
        """Gating the worsening test on the threshold itself would let a
        mode go from machine precision to 0.1 — fifteen orders — while
        staying nominally acceptable. It is gated a tenth lower."""
        from pybmodes.fem.solver import _compare_candidate_modes

        sym_r = np.array([0.52, 1.0e-16])
        alt_r = np.array([1.0e-10, 0.099])
        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 2, 2)
        assert improved[0]
        assert regressed[1]

    def test_an_acceptable_mode_cannot_be_degraded_as_collateral(self):
        """The guarantee, checked over **every** pair rather than only
        the tenfold ones (an earlier version of this test swept only
        those, which is how it certified a bound the rule did not hold).

        An earlier version of this test swept only ``alt > 10 * sym``,
        which is why it certified a bound the rule did not actually hold:
        a mode sliding from 0.02 to 0.099 is under fivefold and lands at
        the edge of tolerance, and nothing looked at it.

        The claim is narrow and about the acceptable side only. A mode
        that was acceptable can end up above the regression floor
        solely by having improved, never as collateral of someone else's
        rescue.
        """
        from pybmodes.fem.solver import _compare_candidate_modes
        from pybmodes.options import DEFAULT_SOLVER_OPTIONS as opt

        t = opt.residual_retry_threshold
        bound = opt.residual_regression_floor
        grid = np.logspace(-16, 2, 37)
        for s in grid:
            if s > t:
                continue                      # not the acceptable side
            for a in grid:
                imp, reg = _compare_candidate_modes(
                    np.array([s]), np.array([a]), 1, 1,
                )
                if reg[0] or imp[0]:
                    continue
                assert a <= bound or a <= s, (
                    f"sym={s:.2e} -> alt={a:.2e} escaped unflagged above "
                    f"the {bound:.2e} bound without improving"
                )

    def test_a_mode_already_failing_gets_no_verdict(self):
        """The deliberate hole, stated so it is not mistaken for one.

        Above the threshold neither candidate is trustworthy, and a
        rigid-body mode lives entirely there. Judging that region either
        way turns roundoff into a decision.
        """
        from pybmodes.fem.solver import _compare_candidate_modes
        from pybmodes.options import DEFAULT_SOLVER_OPTIONS as opt

        t = opt.residual_retry_threshold
        for sym, alt in [(0.2, 5.0), (12.39, 0.794), (0.794, 12.39), (1.0, 1.0)]:
            assert sym > t
            _imp, reg = _compare_candidate_modes(
                np.array([sym]), np.array([alt]), 1, 1,
            )
            assert not reg[0], f"sym={sym} alt={alt} should carry no verdict"

    def test_improved_and_regressed_are_mutually_exclusive(self):
        """A single mode cannot be both, or the caller's rule would be
        reading a contradiction."""
        from pybmodes.fem.solver import _compare_candidate_modes

        grid = np.logspace(-16, 2, 37)
        for s in grid:
            for a in grid:
                imp, reg = _compare_candidate_modes(
                    np.array([s]), np.array([a]), 1, 1,
                )
                assert not (imp[0] and reg[0]), f"sym={s:.2e} alt={a:.2e}"

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

    @pytest.mark.parametrize("alt_rigid", [12.39, 0.794, 0.0762, 0.05])
    def test_rigid_body_noise_cannot_justify_a_swap(self, alt_rigid):
        """Rigid residuals divide roundoff by roundoff, so the value is
        arbitrary — 12.39, 0.794 and 0.0762 have all been measured on
        healthy models, and the last two sit *below* the failure
        threshold. No threshold can exclude them; a resolution bar near
        machine precision can, because roundoff does not land there.
        """
        from pybmodes.fem.solver import _compare_candidate_modes

        sym_r = np.array([0.848, 3.0e-15, 2.0e-15])
        alt_r = np.array([alt_rigid, 1.0e-15, 2.0e-15])
        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 3, 3)
        assert not improved.any()
        assert not regressed.any()

    def test_the_two_populations_are_separated_by_the_ratio(self):
        """The measurement the rule is calibrated on, kept as a test so
        the constants cannot drift away from their evidence.

        The mode that justifies a swap is judged by *how much* the
        candidate improves it, because a rigid residual's value is
        arbitrary while its ratio is not. Genuine rescues improve by 1e5
        to 1e10; rigid roundoff by 11x to 16x.
        """
        from pybmodes.options import DEFAULT_SOLVER_OPTIONS as opt

        rescues = [(1.56e0, 4.10e-10), (3.01e0, 1.96e-09),
                   (7.55e1, 4.93e-09), (3.98e1, 3.17e-04)]
        noise = [(12.39, 0.794), (0.848, 0.0762)]

        worst_rescue = max(a / s for s, a in rescues)
        best_noise = min(a / s for s, a in noise)
        assert worst_rescue < opt.residual_retry_improvement < best_noise
        # And the absolute bar admits every rescue.
        assert max(a for _s, a in rescues) <= opt.residual_retry_resolved

    @pytest.mark.parametrize("seed", [14946, 7, 101, 2024, 55555])
    def test_healthy_free_free_pencils_never_trigger_a_swap(self, seed):
        """The empirical half of the argument, over several draws rather
        than one. A rank-deficient K with a well-conditioned M is a
        healthy free-free model; whatever its null-mode roundoff happens
        to read, it must never replace the spectrum."""
        rng = np.random.default_rng(seed)
        n = 3
        a = rng.normal(size=(n, n))
        gm = a @ a.T + n * np.eye(n)
        b = rng.normal(size=(n, n - 1))
        gk = b @ b.T
        gk, gm = 0.5 * (gk + gk.T), 0.5 * (gm + gm.T)
        assert np.linalg.cond(gm) < 100.0          # genuinely healthy
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eigvals, _v, diag = solve_modes(
                gk, gm, n_modes=n, return_diagnostics=True,
            )
        assert diag.residual_fallback is False
        assert eigvals.size == n

    def test_a_healthy_free_free_pencil_is_left_alone(self):
        """End to end on the shape from the report: rank-deficient K, a
        well-conditioned M, exact elastic modes."""
        rng = np.random.default_rng(19)
        a = rng.normal(size=(4, 4))
        gm = a @ a.T + 4.0 * np.eye(4)
        b = rng.normal(size=(4, 3))
        gk = b @ b.T
        gk, gm = 0.5 * (gk + gk.T), 0.5 * (gm + gm.T)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eigvals, _v, diag = solve_modes(
                gk, gm, n_modes=4, return_diagnostics=True,
            )
        assert diag.residual_fallback is False
        assert eigvals.size == 4

    def test_an_improvement_that_leaves_the_mode_failing_is_declined(self):
        """The cost of the rule, stated rather than hidden: a hundredfold
        gain that still ends above the threshold is not acted on, because
        neither result is trustworthy there."""
        from pybmodes.fem.solver import _compare_candidate_modes

        sym_r = np.array([50.0])
        alt_r = np.array([0.5])
        improved, regressed = _compare_candidate_modes(sym_r, alt_r, 1, 1)
        assert not improved.any()
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


class TestTheGuardDoesNotTaxEverySolve:
    """Residuals are computed on every solve that asks for diagnostics,
    so measuring them must not allocate a copy of the matrices.

    ``0.5 (A + A.T) v == 0.5 (A v + A.T v)``, and only the second form
    avoids a dense ngd-square pair. At ngd = 1500 that is a 54 MB peak
    against 0.3 MB, and it falls on the large sparse solves the dense
    copy would hurt most.
    """

    def _pair(self, n):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(n, n))
        gm = a @ a.T + n * np.eye(n)
        b = rng.normal(size=(n, n))
        gk = b @ b.T
        return 0.5 * (gk + gk.T), 0.5 * (gm + gm.T)

    def test_symmetrised_products_match_materialising(self):
        from pybmodes.fem.solver import _modal_residuals

        gk, gm = self._pair(40)
        rng = np.random.default_rng(5)
        v = np.linalg.qr(rng.normal(size=(40, 4)))[0]
        w = np.linspace(1.0, 2.0, 4)
        ks, ms = 0.5 * (gk + gk.T), 0.5 * (gm + gm.T)
        assert np.allclose(
            _modal_residuals(gk, gm, w, v, symmetrise=True),
            _modal_residuals(ks, ms, w, v),
            rtol=1.0e-12, atol=1.0e-15,
        )

    def test_an_asymmetric_pair_is_measured_unsymmetrised(self):
        """The flag must actually change the basis, not be inert."""
        from pybmodes.fem.solver import _modal_residuals

        gk = np.array([[1.0, 0.9], [0.0, 2.0]])
        gm = np.eye(2)
        rng = np.random.default_rng(9)
        v = np.linalg.qr(rng.normal(size=(2, 2)))[0]
        w = np.array([1.0, 2.0])
        raw = _modal_residuals(gk, gm, w, v)
        sym = _modal_residuals(gk, gm, w, v, symmetrise=True)
        assert not np.allclose(raw, sym)

    def test_the_threshold_check_reads_the_caller_s_own_matrices(
        self, monkeypatch,
    ):
        """The check runs on every eligible solve, healthy ones included,
        before the threshold has been tested. It must therefore read the
        arrays it was given rather than a symmetrised copy of them.

        Asserted by identity rather than by measuring memory: the solve
        legitimately allocates elsewhere — ``eigh`` takes matrices, so
        ``_solve_dense_symmetric`` builds the pair it needs — which would
        swamp any peak-usage threshold and make the test meaningless.
        """
        import pybmodes.fem.solver as solvermod

        seen = []
        real = solvermod._modal_residuals

        def spy(k, m, vals, vecs, *, symmetrise=False):
            seen.append((k is gk, m is gm, symmetrise))
            return real(k, m, vals, vecs, symmetrise=symmetrise)

        monkeypatch.setattr(solvermod, "_modal_residuals", spy)
        gk, gm = self._pair(60)
        _v, _x, diag = solve_modes(gk, gm, n_modes=4, return_diagnostics=True)

        assert diag.path == "dense_symmetric"
        assert diag.residual_fallback is False
        assert seen, "the threshold check did not run"
        # Every call took the caller's arrays and asked for the
        # symmetrised basis, rather than being handed a built pair.
        assert all(is_gk and is_gm and sym for is_gk, is_gm, sym in seen), seen

    def test_measuring_does_not_allocate_a_matrix_copy(self):
        import tracemalloc

        from pybmodes.fem.solver import _modal_residuals

        n = 400
        gk, gm = self._pair(n)
        rng = np.random.default_rng(7)
        v = np.linalg.qr(rng.normal(size=(n, 5)))[0]
        w = np.linspace(1.0, 2.0, 5)

        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        _modal_residuals(gk, gm, w, v, symmetrise=True)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()

        # Well under a single dense copy, which is what materialising the
        # symmetrised pair would have cost twice over.
        assert peak < 0.25 * gk.nbytes, (
            f"peak {peak / 1e6:.1f} MB against a {gk.nbytes / 1e6:.1f} MB "
            f"matrix — the symmetrised pair is being materialised"
        )


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
