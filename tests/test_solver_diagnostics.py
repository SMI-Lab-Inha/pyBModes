"""SolverDiagnostics: certification-grade telemetry from solve_modes.

The solver reports the path it took, whether the sparse path fell back
to dense, the mode-count guarantee, per-mode backward-error residuals,
and a mass-matrix conditioning estimate (review item 3). These tests pin
that contract on hand-built matrices (self-contained) plus the run()
attachment on a bundled sample.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.fem.solver import SolverDiagnostics, solve_modes


def _tridiag(n: int) -> np.ndarray:
    a = 2.0 * np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = -1.0
    return a


def test_default_return_is_two_tuple() -> None:
    """Without return_diagnostics the historical two-tuple is returned."""
    out = solve_modes(np.eye(3), np.eye(3), n_modes=2)
    assert isinstance(out, tuple) and len(out) == 2


def test_dense_symmetric_diagnostics() -> None:
    k = _tridiag(4)
    m = np.eye(4)
    eigvals, eigvecs, diag = solve_modes(
        k, m, n_modes=2, return_diagnostics=True,
    )
    assert isinstance(diag, SolverDiagnostics)
    assert diag.path == "dense_symmetric"
    assert diag.symmetric is True
    assert diag.n_requested == 2
    assert diag.n_returned == 2 == eigvecs.shape[1]
    assert diag.sparse_fallback is False
    assert diag.fallback_reason is None
    # A clean SPD generalised solve has machine-precision residuals.
    assert diag.max_residual < 1e-9
    assert len(diag.residuals) == 2
    assert all(r < 1e-9 for r in diag.residuals)
    # Mass matrix is the identity here, so cond == 1.
    assert diag.matrix_cond is not None
    assert diag.matrix_cond == pytest.approx(1.0, abs=1e-9)


def test_general_path_flagged_for_asymmetric_stiffness() -> None:
    """A genuinely asymmetric K routes through the dense general path."""
    k = _tridiag(4)
    k[0, 1] += 0.5            # break symmetry beyond the tolerance
    m = np.eye(4)
    _, _, diag = solve_modes(k, m, n_modes=2, return_diagnostics=True)
    assert diag.path == "dense_general"
    assert diag.symmetric is False


def test_general_path_shortfall_warns_and_records_count() -> None:
    """When the general (non-symmetric) path recovers fewer valid modes
    than requested, the shortfall is both warned and recorded. An
    asymmetric K with a negative eigenvalue yields only one positive-real
    mode, so a request for two falls short."""
    k = np.array([[1.0, 2.0], [0.0, -1.0]])   # asymmetric, eigvals 1 and -1
    m = np.eye(2)
    with pytest.warns(RuntimeWarning, match="recovered only"):
        _, _, diag = solve_modes(k, m, n_modes=2, return_diagnostics=True)
    assert diag.path == "dense_general"
    assert diag.n_requested == 2
    assert diag.n_returned < 2


def test_symmetric_truncation_does_not_warn() -> None:
    """Codex P2: asking for more modes than a symmetric system has DOFs
    is benign — the dense symmetric path truncates to min(n_modes, ngd)
    and must NOT emit the 'defective eigenproblem' warning (which would
    mislead and would fail warnings-as-errors callers). The shortfall is
    still recorded in the diagnostics."""
    import warnings as _w

    k = _tridiag(2)
    m = np.eye(2)
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)   # any RuntimeWarning fails
        _, _, diag = solve_modes(k, m, n_modes=5, return_diagnostics=True)
    assert diag.path == "dense_symmetric"
    assert diag.n_requested == 5
    assert diag.n_returned == 2                     # truncated, recorded


_LAND_TOWER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "pybmodes" / "_examples" / "sample_inputs"
    / "reference_turbines" / "01_nrel5mw_land"
    / "01_nrel5mw_land_tower.bmi"
)


@pytest.mark.skipif(not _LAND_TOWER.is_file(), reason="bundled sample absent")
def test_run_attaches_diagnostics() -> None:
    """``Tower.run`` attaches the diagnostics from its FEM solve.

    The bundled land deck comes through the ElastoDyn adapter, which uses
    a near-rigid axial stiffness and floors rotary inertia, leaving the
    mass matrix ill-conditioned (cond ~ 1e10). The diagnostics correctly
    surface that — a non-trivial residual and a large conditioning
    number — which is exactly the point of the feature, so we assert the
    fields are populated rather than that the residual is tiny (the
    well-conditioned case below pins the near-machine-precision path)."""
    from pybmodes.models import Tower

    res = Tower(_LAND_TOWER).run(n_modes=6, check_model=False)
    assert res.diagnostics is not None
    assert res.diagnostics.path == "dense_symmetric"
    assert res.diagnostics.n_requested == 6
    assert res.diagnostics.n_returned == 6
    assert res.diagnostics.sparse_fallback is False
    assert len(res.diagnostics.residuals) == 6
    assert np.isfinite(res.diagnostics.max_residual)
    # The conditioning estimate is reported for this dense solve and
    # reflects the adapter's known ill-conditioning.
    assert res.diagnostics.matrix_cond is not None
    assert res.diagnostics.matrix_cond > 1.0
