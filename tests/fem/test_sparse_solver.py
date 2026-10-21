"""Tests for the sparse-solver dispatch in ``pybmodes.fem.solver``.

The solver gates the sparse shift-invert path on four conditions:

  ngd > _SPARSE_NDOF_THRESHOLD AND symmetric matrices AND a small
  subset of modes was requested (``n_modes is not None`` and
  ``n_modes < ngd // 2``) AND a positive-definite K.

The last one exists because shift-invert at zero returns the
eigenvalues smallest in magnitude, which are the lowest only when none
is negative. Without it an indefinite K (a column loaded past buckling)
lost its unstable modes on the sparse path while the dense path kept
them, so the answer changed with mesh refinement alone.

These tests build small synthetic symmetric / asymmetric problems
(no FEM pipeline needed) so they're fast and self-contained.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pybmodes.fem import solver
from pybmodes.fem.solver import (
    _SPARSE_NDOF_THRESHOLD,
    solve_modes,
)


def _spd_problem(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a symmetric positive-definite (K, M) pair of size n×n with
    a known smooth eigenvalue spectrum. Constructed as random SPD
    matrices via ``A.T @ A + I`` so eigenvalues are well-spaced and
    the problem is far from degenerate."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(n, n))
    K = A.T @ A + np.eye(n) * (n + 1.0)
    B = rng.standard_normal(size=(n, n))
    M = B.T @ B + np.eye(n) * (n + 1.0)
    # Symmetric to machine precision
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
    return K, M


def _asymmetric_problem(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a deliberately asymmetric (K, M) pair so the solver
    routes through the dense general ``eig`` path."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(n, n))
    K = A.T @ A + np.eye(n) * (n + 1.0)
    # Inject a non-symmetric coupling block big enough to clear the
    # _is_effectively_symmetric threshold (1e-12 * max|K|).
    K[0, n - 1] += 1.5 * float(np.max(np.abs(K)))
    B = rng.standard_normal(size=(n, n))
    M = B.T @ B + np.eye(n) * (n + 1.0)
    return K, M


# ---------------------------------------------------------------------------
# 1. Sparse vs dense numerical agreement
# ---------------------------------------------------------------------------

def test_sparse_matches_dense_frequencies() -> None:
    """For a symmetric SPD problem of size n > threshold, the sparse
    shift-invert path returns the same lowest-k eigenvalues as the
    dense ``eigh`` path to rtol = 1e-8."""
    n = _SPARSE_NDOF_THRESHOLD + 100
    K, M = _spd_problem(n, seed=1)
    n_modes = 8

    # Force the sparse path by calling the helper directly.
    sparse_eigvals, sparse_eigvecs = solver._solve_sparse_shift_invert(
        K, M, n_modes,
    )
    # Force the dense path the same way.
    dense_eigvals, _dense_eigvecs = solver._solve_dense_symmetric(
        K, M, n_modes,
    )

    np.testing.assert_allclose(
        np.sort(sparse_eigvals)[:n_modes],
        np.sort(dense_eigvals)[:n_modes],
        rtol=1.0e-8,
    )


# ---------------------------------------------------------------------------
# 2. Sparse path is taken above the size threshold
# ---------------------------------------------------------------------------

def test_sparse_triggered_above_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """When n_dof > threshold AND symmetric AND n_modes is a small
    subset, ``solve_modes`` logs that it took the sparse path."""
    n = _SPARSE_NDOF_THRESHOLD + 50
    K, M = _spd_problem(n, seed=2)
    n_modes = 6
    with caplog.at_level(logging.INFO, logger="pybmodes.fem.solver"):
        solve_modes(K, M, n_modes=n_modes)
    sparse_msgs = [
        rec for rec in caplog.records
        if "sparse shift-invert" in rec.getMessage()
    ]
    assert sparse_msgs, (
        "expected an INFO log line announcing the sparse path; "
        f"saw records: {[r.getMessage() for r in caplog.records]}"
    )


def test_sparse_skipped_below_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """At n_dof below the threshold the dense path runs; the sparse
    path is not invoked."""
    n = _SPARSE_NDOF_THRESHOLD - 100
    K, M = _spd_problem(n, seed=3)
    with caplog.at_level(logging.INFO, logger="pybmodes.fem.solver"):
        solve_modes(K, M, n_modes=6)
    # Either no log was emitted or it was the dense one.
    messages = [r.getMessage() for r in caplog.records]
    assert all("sparse" not in m for m in messages), (
        "small problem should not trigger the sparse path; "
        f"saw: {messages}"
    )


# ---------------------------------------------------------------------------
# 3. Asymmetric problems route through dense `eig`
# ---------------------------------------------------------------------------

def test_dense_fallback_on_asymmetric(caplog: pytest.LogCaptureFixture) -> None:
    """An asymmetric problem above the size threshold takes the dense
    general-matrix path; the sparse symmetric solver is skipped."""
    n = _SPARSE_NDOF_THRESHOLD + 50
    K, M = _asymmetric_problem(n, seed=4)
    with caplog.at_level(logging.INFO, logger="pybmodes.fem.solver"):
        eigvals, eigvecs = solve_modes(K, M, n_modes=6)
    messages = [r.getMessage() for r in caplog.records]
    # Sparse path should NOT have run.
    assert all("sparse" not in m for m in messages), (
        f"asymmetric problem should bypass the sparse path; "
        f"saw: {messages}"
    )
    # Dense general path should have run.
    assert any("dense general" in m for m in messages), (
        f"expected dense general-eig path; saw: {messages}"
    )
    # Sanity: results are well-formed.
    assert eigvals.size == 6
    assert eigvecs.shape == (n, 6)
    assert np.all(np.isfinite(eigvals))


# ---------------------------------------------------------------------------
# Indefinite K: the sparse path must not change the returned window
# ---------------------------------------------------------------------------

_N_BIG = _SPARSE_NDOF_THRESHOLD + 1


def _rotated(d: np.ndarray, seed: int) -> np.ndarray:
    """``Q diag(d) Q^T`` with a random orthogonal ``Q``: a fully coupled
    symmetric matrix with a chosen spectrum."""
    rng = np.random.default_rng(seed)
    q, _r = np.linalg.qr(rng.standard_normal((d.size, d.size)))
    k = q @ np.diag(d) @ q.T
    return 0.5 * (k + k.T)


def _lowest(gk: np.ndarray, gm: np.ndarray, n: int) -> np.ndarray:
    from scipy.linalg import eigh

    return eigh(gk, gm, eigvals_only=True, subset_by_index=(0, n - 1))


class TestIndefiniteStiffnessKeepsTheLowestWindow:
    def test_large_negative_mode_is_not_dropped(self):
        """The reported case: shift-invert at zero picked [1, 2, 3] here
        because |-100| is larger than 3."""
        gk = np.diag(np.concatenate([[-100.0], np.arange(1.0, _N_BIG)]))
        vals, _vecs, diag = solve_modes(
            gk, np.eye(_N_BIG), 3, return_diagnostics=True,
        )
        np.testing.assert_allclose(vals, [-100.0, 1.0, 2.0], rtol=1e-12)
        assert diag.path == "dense_symmetric"
        assert diag.sparse_fallback is False

    def test_coupled_stiffness_with_scaled_mass(self):
        rng = np.random.default_rng(4)
        d = np.concatenate([[-50.0, -7.0], rng.uniform(1.0, 400.0, _N_BIG - 2)])
        gk = _rotated(d, seed=5)
        gm = np.diag(rng.uniform(0.5, 20.0, _N_BIG))
        vals, vecs = solve_modes(gk, gm, 4)
        np.testing.assert_allclose(vals, _lowest(gk, gm, 4), rtol=1e-9)
        assert vals[0] < 0.0 and vals[1] < 0.0
        residual = gk @ vecs - (gm @ vecs) * vals
        assert np.max(np.abs(residual)) < 1e-8 * np.max(np.abs(gk))

    def test_zero_modes_stay_in_the_window(self):
        d = np.concatenate([np.zeros(3), np.arange(1.0, _N_BIG - 2)])
        gk = _rotated(d, seed=6)
        vals, _vecs, diag = solve_modes(
            gk, np.eye(_N_BIG), 5, return_diagnostics=True,
        )
        assert diag.path == "dense_symmetric"
        assert np.all(np.abs(vals[:3]) < 1e-9 * float(np.max(d)))
        np.testing.assert_allclose(vals[3:], [1.0, 2.0], rtol=1e-9)

    def test_positive_definite_stiffness_still_goes_sparse(self):
        gk, gm = _spd_problem(_N_BIG, seed=7)
        vals, _vecs, diag = solve_modes(gk, gm, 4, return_diagnostics=True)
        assert diag.path == "sparse_shift_invert"
        np.testing.assert_allclose(vals, _lowest(gk, gm, 4), rtol=1e-8)


class TestPositiveDefiniteCheck:
    @staticmethod
    def _check(a: np.ndarray) -> bool:
        return solver._is_positive_definite(solver._symmetric_sparse(a))

    def test_matches_a_dense_cholesky(self):
        from scipy.linalg import cho_factor

        rng = np.random.default_rng(8)
        for shift in (-5.0, -0.5, 0.5, 5.0):
            a = rng.standard_normal((60, 60))
            k = 0.5 * (a + a.T) + shift * 10.0 * np.eye(60)
            try:
                cho_factor(k)
                expected = True
            except np.linalg.LinAlgError:
                expected = False
            assert self._check(k) is expected, shift

    def test_banded_tridiagonal(self):
        n = 200
        k = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        assert self._check(k)          # second-difference matrix, SPD
        k[0, 0] = -1.0
        assert not self._check(k)

    def test_singular_and_empty_are_not_definite(self):
        assert not self._check(np.diag([1.0, 0.0, 2.0]))
        assert not self._check(np.zeros((4, 4)))

    def test_wide_coupling_is_still_exact(self):
        """A far off-diagonal coupling widens the band to the whole
        matrix; the check must still see it."""
        n = 40
        k = np.eye(n)
        k[0, n - 1] = k[n - 1, 0] = 2.0   # makes the 2x2 minor indefinite
        assert not self._check(k)
        k[0, n - 1] = k[n - 1, 0] = 0.5
        assert self._check(k)

    def test_symmetrises_tiny_asymmetry(self):
        k = np.diag([3.0, 2.0, 1.0])
        k[0, 1], k[1, 0] = 1.0, 1.0 + 1e-14
        assert self._check(k)


def test_buckled_tower_does_not_depend_on_the_mesh():
    """A real model: the bundled NREL 5MW land tower under gravity with
    1000 times its RNA mass is far past buckling. On the default mesh
    (dense path) its lowest four eigenvalues are all negative, reported
    as 0 Hz. Refined past the sparse threshold it used to return two
    positive modes among them instead."""
    import warnings

    from pybmodes.cli import _resolve_examples_root
    from pybmodes.models import Tower

    bmi = (
        _resolve_examples_root() / "sample_inputs" / "reference_turbines"
        / "01_nrel5mw_land" / "01_nrel5mw_land_tower.bmi"
    )

    def run(nodes):
        tower = Tower(bmi)
        if nodes is not None:
            tower.refine_mesh(nodes)
        tower._bmi.tip_mass.mass *= 1000.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tower.run(n_modes=4, check_model=False, gravity=True)

    coarse, fine = run(None), run(120)
    assert fine.diagnostics.path == "dense_symmetric"
    np.testing.assert_array_equal(coarse.frequencies, 0.0)
    np.testing.assert_array_equal(fine.frequencies, 0.0)
