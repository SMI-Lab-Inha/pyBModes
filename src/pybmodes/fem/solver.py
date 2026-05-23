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

"""Generalized eigenvalue solver for the reduced FEM system.

Solves: ``K ψ = λ M ψ``.

Three dispatch paths in priority order:

1. **Sparse symmetric shift-invert** — selected when the assembled
   matrices are effectively symmetric, the system has more than
   :data:`_SPARSE_NDOF_THRESHOLD` DOFs, and the caller asked for a
   subset (i.e. ``n_modes is not None`` and small relative to
   ``ngd``). Routes through ``scipy.sparse.linalg.eigsh`` with
   ``sigma=0`` shift-invert; an order-of-magnitude faster than the
   dense LAPACK solve for the few-lowest-modes case on a 500+ DOF
   tower mesh.
2. **Dense symmetric** — ``scipy.linalg.eigh`` on the symmetrised
   matrices. Path used for small / mid-size symmetric problems and
   when the sparse path fails to converge (logged as a warning).
3. **Dense general** — ``scipy.linalg.eig`` for genuinely asymmetric
   systems (offshore decks where the rigid-arm transformation makes
   the platform-support block non-symmetric). Matches BModes JJ.

Note on the user-spec mode choice: ``eigsh(..., sigma=0,
mode='buckling')`` reduces to ``OP = K^-1 K = I`` for ``sigma=0``,
which is degenerate. The standard scipy idiom for "smallest
eigenvalues of ``K x = λ M x`` via shift-invert near zero" is
``mode='normal'`` (giving ``OP = K^-1 M``; ``which='LM'`` returns
the largest ``1/λ``, i.e. the smallest ``λ``). The implementation
below uses ``mode='normal'`` accordingly.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from scipy.linalg import eig, eigh

from pybmodes.options import DEFAULT_SOLVER_OPTIONS as _SOLVER_OPTIONS

_log = logging.getLogger(__name__)

# Above this reduced-system size the dense conditioning estimate
# (``np.linalg.cond``, an O(ngd^3) SVD) is skipped and reported as
# ``None`` to keep the per-solve cost negligible. Real blade / tower
# meshes sit well under this, so the estimate is populated for them; a
# 500+ DOF spliced monopile takes the sparse path anyway, where the
# estimate is not meaningful.
_COND_DENSE_MAX = 800


@dataclass(frozen=True)
class SolverDiagnostics:
    """Numerical-health record for one :func:`solve_modes` call.

    Returned alongside the eigenpairs when ``return_diagnostics=True`` and
    carried on :class:`pybmodes.models.result.ModalResult.diagnostics`. It
    makes the solve auditable for certification-grade work: which path
    ran, whether the sparse path silently fell back to dense, how many
    modes were actually recovered versus requested, the per-mode
    backward-error residuals, and a mass-matrix conditioning estimate.

    Attributes
    ----------
    path : which solver path produced the result. One of
        ``"sparse_shift_invert"``, ``"dense_symmetric"``,
        ``"dense_general"``.
    symmetric : whether the assembled matrices were treated as symmetric
        (``eigh`` / sparse) rather than routed through the general
        ``eig`` path.
    n_requested : modes asked for (``None`` means the full spectrum).
    n_returned : modes actually returned. Fewer than ``n_requested``
        means the general path filtered out complex / non-positive
        eigenvalues and could not recover enough valid modes (a warning
        is also emitted in that case).
    sparse_fallback : ``True`` when the sparse shift-invert path was
        attempted and failed, so the result came from the dense fallback.
    fallback_reason : the repr of the exception that triggered the
        fallback, or ``None`` when no fallback happened.
    max_residual : the largest per-mode relative residual
        ``||K x - λ M x|| / ||K x||`` over the returned modes (``0.0``
        when no modes were returned). A healthy modal solve sits near
        machine precision; a large value flags an ill-conditioned or
        defective eigenproblem.
    residuals : the per-mode relative residuals, one per returned mode.
    matrix_cond : 2-norm condition number of the (symmetrised) mass
        matrix, or ``None`` when not computed (sparse path, or a system
        larger than the dense-conditioning size limit).
    """

    path: Literal["sparse_shift_invert", "dense_symmetric", "dense_general"]
    symmetric: bool
    n_requested: int | None
    n_returned: int
    sparse_fallback: bool
    fallback_reason: str | None
    max_residual: float
    residuals: tuple[float, ...]
    matrix_cond: float | None

# Sparse path activates once the reduced system has more than this
# many DOFs and the caller asked for a small subset of modes. Below
# the threshold, ``eigh``'s LAPACK back-end is faster than the
# factorisation + Arnoldi cycle ``eigsh`` incurs.
#
# Kept as a module-level constant for backward compatibility; the
# value is read from :data:`pybmodes.options.DEFAULT_SOLVER_OPTIONS`
# so a single override site exists. Future PRs will thread a
# ``SolverOptions`` instance through :func:`solve_modes` directly.
_SPARSE_NDOF_THRESHOLD = _SOLVER_OPTIONS.sparse_ndof_threshold


@overload
def solve_modes(
    gk: np.ndarray, gm: np.ndarray, n_modes: int | None = ...,
    *, return_diagnostics: Literal[False] = ...,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def solve_modes(
    gk: np.ndarray, gm: np.ndarray, n_modes: int | None = ...,
    *, return_diagnostics: Literal[True],
) -> tuple[np.ndarray, np.ndarray, SolverDiagnostics]: ...


def solve_modes(
    gk: np.ndarray,
    gm: np.ndarray,
    n_modes: int | None = None,
    *,
    return_diagnostics: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, SolverDiagnostics]
):
    """Solve the generalised eigenproblem ``K ψ = λ M ψ``.

    Parameters
    ----------
    gk      : (ngd, ngd) global stiffness matrix
    gm      : (ngd, ngd) global mass matrix
    n_modes : number of lowest modes to return (``None`` = all)
    return_diagnostics : when ``True``, also return a
        :class:`SolverDiagnostics` record (path taken, sparse-to-dense
        fallback, mode-count guarantee, per-mode residuals, mass-matrix
        conditioning). Default ``False`` keeps the historical
        two-tuple return for existing callers.

    Returns
    -------
    eigvals : (n_modes,) eigenvalues λ, sorted ascending (λ = (ω_nd)²)
    eigvecs : (ngd, n_modes) eigenvectors, columns correspond to eigvals,
              each normalised to unit L2 norm.
    diagnostics : :class:`SolverDiagnostics`, only when
        ``return_diagnostics=True``.
    """
    ngd = gk.shape[0]
    sym = _is_effectively_symmetric(gk) and _is_effectively_symmetric(gm)

    path: Literal["sparse_shift_invert", "dense_symmetric", "dense_general"]
    sparse_fallback = False
    fallback_reason: str | None = None
    eigvals: np.ndarray | None = None
    eigvecs: np.ndarray | None = None

    # Sparse path — symmetric, big enough, small-subset request.
    if (
        sym
        and ngd > _SPARSE_NDOF_THRESHOLD
        and n_modes is not None
        and n_modes < ngd // 2
    ):
        try:
            eigvals, eigvecs = _solve_sparse_shift_invert(gk, gm, n_modes)
            path = "sparse_shift_invert"
            _log.info(
                "solve_modes: sparse shift-invert path "
                "(ngd=%d, n_modes=%d)",
                ngd, n_modes,
            )
        except Exception as exc:
            # eigsh can fail to converge on near-singular K, on
            # poorly-conditioned M, or when MKL throws an ARPACK
            # error. Fall back to dense in any such case so the
            # solver remains robust — but record that the path changed
            # so the caller can audit it (it is no longer silent).
            sparse_fallback = True
            fallback_reason = repr(exc)
            eigvals = eigvecs = None
            _log.warning(
                "solve_modes: sparse path failed (%r); "
                "falling back to dense eigh",
                exc,
            )

    if eigvals is None or eigvecs is None:
        if sym:
            eigvals, eigvecs = _solve_dense_symmetric(gk, gm, n_modes)
            path = "dense_symmetric"
            _log.info("solve_modes: dense symmetric eigh (ngd=%d)", ngd)
        else:
            eigvals, eigvecs = _solve_dense_general(gk, gm, n_modes)
            path = "dense_general"
            _log.info("solve_modes: dense general eig (ngd=%d)", ngd)

    _normalize_columns_l2(eigvecs)

    # Mode-count guarantee: the general path filters complex / non-
    # positive eigenvalues, so it can return fewer modes than requested.
    # Surface that rather than letting it pass silently (a downstream
    # broadcast would otherwise fail with an opaque shape error).
    n_returned = int(eigvecs.shape[1])
    if n_modes is not None and n_returned < n_modes:
        warnings.warn(
            f"solve_modes recovered only {n_returned} of the requested "
            f"{n_modes} modes via the {path} path. The eigenproblem is "
            f"likely near-degenerate or defective (a non-symmetric "
            f"PlatformSupport block can do this); the missing modes had "
            f"complex or non-positive eigenvalues and were filtered out.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not return_diagnostics:
        return eigvals, eigvecs

    diagnostics = _build_diagnostics(
        gk, gm, eigvals, eigvecs, path=path, symmetric=sym,
        n_requested=n_modes, sparse_fallback=sparse_fallback,
        fallback_reason=fallback_reason,
    )
    return eigvals, eigvecs, diagnostics


def _build_diagnostics(
    gk: np.ndarray,
    gm: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    *,
    path: Literal["sparse_shift_invert", "dense_symmetric", "dense_general"],
    symmetric: bool,
    n_requested: int | None,
    sparse_fallback: bool,
    fallback_reason: str | None,
) -> SolverDiagnostics:
    """Assemble a :class:`SolverDiagnostics` for a completed solve."""
    residuals = _modal_residuals(gk, gm, eigvals, eigvecs)
    cond = _mass_matrix_cond(gm, path)
    return SolverDiagnostics(
        path=path,
        symmetric=symmetric,
        n_requested=n_requested,
        n_returned=int(eigvecs.shape[1]),
        sparse_fallback=sparse_fallback,
        fallback_reason=fallback_reason,
        max_residual=float(residuals.max()) if residuals.size else 0.0,
        residuals=tuple(float(r) for r in residuals),
        matrix_cond=cond,
    )


def _modal_residuals(
    gk: np.ndarray, gm: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray,
) -> np.ndarray:
    """Per-mode relative backward error ``||K x - λ M x|| / ||K x||``.

    The honest health metric for a generalised modal solve. Cheap
    (matrix-times-thin-matrix), so computed for every path.
    """
    if eigvecs.size == 0:
        return np.empty(0, dtype=float)
    kx = gk @ eigvecs                                 # (ngd, k)
    mx = gm @ eigvecs
    num = np.linalg.norm(kx - mx * eigvals[np.newaxis, :], axis=0)
    den = np.linalg.norm(kx, axis=0)
    return np.asarray(num / np.where(den > 0.0, den, 1.0), dtype=float)


def _mass_matrix_cond(
    gm: np.ndarray,
    path: Literal["sparse_shift_invert", "dense_symmetric", "dense_general"],
) -> float | None:
    """2-norm conditioning of the (symmetrised) mass matrix, or ``None``.

    Skipped for the sparse path and for systems above
    :data:`_COND_DENSE_MAX`, where the O(ngd^3) SVD would dominate the
    solve cost without adding actionable information (those systems take
    the sparse path precisely because they are large).
    """
    if path == "sparse_shift_invert" or gm.shape[0] > _COND_DENSE_MAX:
        return None
    try:
        return float(np.linalg.cond(0.5 * (gm + gm.T)))
    except np.linalg.LinAlgError:
        return float("inf")


# ---------------------------------------------------------------------------
# Path implementations
# ---------------------------------------------------------------------------

def _solve_sparse_shift_invert(
    gk: np.ndarray, gm: np.ndarray, n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse symmetric generalised eigensolve via shift-invert near zero.

    Why ``mode='normal'`` and not ``mode='buckling'`` (the buckling
    mode reduces to OP = I when sigma = 0): for the generalised
    problem ``K x = λ M x`` with shift ``σ = 0``,
    ``OP = (K - σM)^-1 · M = K^-1 M`` under ``mode='normal'``. The
    eigenvalues of OP are ``1/λ``; ``which='LM'`` returns the largest,
    i.e. the smallest ``λ`` — exactly the modal-analysis ask.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import eigsh

    # Symmetrise to suppress sub-ULP scatter before factorisation.
    gk_sym = 0.5 * (gk + gk.T)
    gm_sym = 0.5 * (gm + gm.T)
    K_sp = csc_matrix(gk_sym)
    M_sp = csc_matrix(gm_sym)

    eigvals, eigvecs = eigsh(
        K_sp,
        k=n_modes,
        M=M_sp,
        sigma=0.0,
        which="LM",
        mode="normal",
    )

    # eigsh's shift-invert returns the eigenvalues unsorted; sort
    # ascending for a stable downstream contract.
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


def _solve_dense_symmetric(
    gk: np.ndarray, gm: np.ndarray, n_modes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense LAPACK eigh on the symmetrised matrices. ``n_modes=None``
    requests the full spectrum; otherwise a subset slice is taken."""
    gk_sym = 0.5 * (gk + gk.T)
    gm_sym = 0.5 * (gm + gm.T)
    if n_modes is not None:
        subset = (0, min(n_modes, gk.shape[0]) - 1)
        eigvals, eigvecs = eigh(gk_sym, gm_sym, subset_by_index=subset)
    else:
        eigvals, eigvecs = eigh(gk_sym, gm_sym)
    return np.asarray(eigvals), np.asarray(eigvecs)


def _solve_dense_general(
    gk: np.ndarray, gm: np.ndarray, n_modes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense LAPACK ``eig`` for genuinely asymmetric problems. Filters
    eigenvalues to the real, positive, finite subset (matches BModes
    JJ's general-matrix path)."""
    eigvals_all, eigvecs_all = eig(gk, gm)
    eigvals_real = np.real_if_close(eigvals_all, tol=1000)
    valid = (
        np.isreal(eigvals_real)
        & np.isfinite(eigvals_real.real)
        & (eigvals_real.real > 0.0)
    )
    eigvals = eigvals_real.real[valid]
    eigvecs = np.real_if_close(eigvecs_all[:, valid], tol=1000).real
    order = np.argsort(eigvals)
    if n_modes is not None:
        order = order[: min(n_modes, order.size)]
    return eigvals[order], eigvecs[:, order]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_effectively_symmetric(a: np.ndarray) -> bool:
    """Return True for exact / small-roundoff asymmetry; False for input
    asymmetry beyond ``rtol * max|a|``.

    Tolerance is read from :class:`pybmodes.options.SolverOptions`
    (default ``1e-12``)."""
    scale = max(1.0, float(np.max(np.abs(a))))
    return bool(np.max(np.abs(a - a.T)) <= _SOLVER_OPTIONS.symmetry_rtol * scale)


def _normalize_columns_l2(eigvecs: np.ndarray) -> None:
    """Normalise each column of ``eigvecs`` to unit L2 norm in place.

    Mode-shape consumers (extract_mode_shapes, MAC tracking, polynomial
    fits) assume L2-normalised columns. Both the dense and sparse
    paths route through this helper so the convention is uniform.
    """
    norms = np.linalg.norm(eigvecs, axis=0)
    nonzero = norms > 0.0
    eigvecs[:, nonzero] /= norms[nonzero]


def eigvals_to_hz(eigvals: np.ndarray, romg: float) -> np.ndarray:
    """Convert non-dimensional eigenvalues to Hz.

    ``freq_Hz = sqrt(λ_nd) * romg / (2π)``

    Parameters
    ----------
    eigvals : non-dimensional eigenvalues (``λ = (ω / romg)²``)
    romg    : reference angular velocity (rad/s) used in
              non-dimensionalisation (typically ``romg = 10.0`` rad/s)
    """
    return np.asarray(
        np.sqrt(np.maximum(eigvals, 0.0)) * romg / (2.0 * np.pi)
    )
