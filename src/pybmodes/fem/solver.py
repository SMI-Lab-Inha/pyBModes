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
   the platform-support block non-symmetric). Matches BModes JJ. Also
   the retry path when a symmetric solve comes back with a large
   backward error — see below.

The residual retry
------------------

``scipy.linalg.eigh`` reduces ``K x = λ M x`` through a Cholesky factor
of the **mass** matrix, and that reduction degrades once ``M`` is nearly
singular — a very light beam carrying a very heavy lump. The failure is
silent: LAPACK returns confidently wrong low modes rather than raising.
On the case in ``tests/fem/test_ill_conditioned_mass.py`` it reported
0.103 Hz against a true 0.0436 Hz.

:func:`solve_modes` therefore measures the backward error of a dense
symmetric solve and, above
:attr:`~pybmodes.options.SolverOptions.residual_retry_threshold`, solves
again through the general path and compares. Every other rule below
exists because some simpler version of that comparison was wrong.

**Only the dense path.** ``eigsh(sigma=0, mode='normal')`` factorises
``K``, not ``M``, so the sparse path does not have this failure and is
never retried. That also avoids comparing two different mode sets: its
``which="LM"`` window selects the modes nearest zero in magnitude while
the retry selects the algebraically smallest.

**Per mode, not on the maxima.** A rigid-body mode's residual divides one
roundoff quantity by another. Taking maxima lets that noise floor the
candidate's worst value and hide a genuinely corrupted elastic mode
beside it.

**Judged by the size of the win.** No absolute bar separates a rescue
from rigid noise, because the noise value is arbitrary — 0.076, 0.79 and
12.4 have all been measured on healthy models, and the first is *below*
the failure threshold. Identifying such modes was tried three times and
abandoned: not by eigenvalue scale, which a rigid-only subset makes its
own reference; not by strain, which a genuinely soft mode also has
little of; and not by which side of the threshold the value falls on.
What does separate them is the ratio. Rescues improve by 1e5 to 1e10,
roundoff by 11x to 16x, so acceptance needs
:attr:`~pybmodes.options.SolverOptions.residual_retry_improvement` *and*
a candidate that reaches
:attr:`~pybmodes.options.SolverOptions.residual_retry_resolved`.

**Non-regressive.** Accepting replaces the whole spectrum, so a candidate
that rescues one mode while pushing another past the threshold is a
trade, not an improvement.

**The same matrices throughout.** Both symmetric paths symmetrise
internally, so the measurement and the retry use the symmetrised pair.
The tolerated skew is small only relative to ``max|K|``, which in a
wide-dynamic-range model can still swamp a soft mode's own eigenvalue;
judging an exact solve against the raw matrices reads as a failure, and
``eig`` on those same matrices then "wins" by answering a different
question.

**The same spectrum throughout.** The retry keeps every real eigenvalue,
zeros and negatives included, and verifies that nothing was discarded
from inside the returned window. ``eigh`` filters nothing, so any filter
here would return a different set of the same length, backfilled from
higher up, and equal indices would stop meaning equal modes. Both
omissions are reachable: a free-free model's zero modes, and the negative
eigenvalues an indefinite ``K`` produces once ``run(gravity=...)`` loads
a column past its buckling weight.

**It can always decline.** If the alternative raises on the same
defective pencil, or the system is larger than
:attr:`~pybmodes.options.SolverOptions.residual_retry_max_ndof`, or its
ordering cannot be verified, the symmetric result stands.

What this does not promise
--------------------------

The rescue is reliable and platform-independent for the case it was built
for: a near-singular mass matrix with no rigid-body modes. Elsewhere it
is *safe* but not always *effective*. Where rigid-body modes and a
near-singular mass coincide, QZ may return the theoretically real zero
modes as complex-conjugate pairs, and where those land differs between
LAPACK builds; inside the requested window the ordering cannot be
verified and the retry declines. Declining is deliberate — a guard added
to stop a silent wrong answer must not be able to introduce one — and
``max_residual`` still reports the problem.

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
    symmetric : whether the assembled matrices were **classified** as
        symmetric, i.e. whether their asymmetry was within
        :attr:`~pybmodes.options.SolverOptions.symmetry_rtol`. This is a
        property of the input, not a record of which routine ran, so a
        residual retry leaves it ``True`` while moving ``path`` to
        ``"dense_general"``. That pairing is not a contradiction: the
        matrices were symmetric, and the general routine was used on their
        symmetrised form because the symmetric one had failed on it.
        ``residual_fallback`` is what distinguishes that case from a
        genuinely asymmetric solve.
    n_requested : modes asked for (``None`` means the full spectrum).
    n_returned : modes actually returned. Fewer than ``n_requested``
        means the general path filtered out complex / non-positive
        eigenvalues and could not recover enough valid modes (a warning
        is also emitted in that case).
    sparse_fallback : ``True`` when the sparse shift-invert path was
        attempted and failed, so the result came from the dense fallback.
    fallback_reason : the repr of the exception that triggered the
        fallback, or ``None`` when no fallback happened.
    residual_fallback : ``True`` when a symmetric path returned modes whose
        backward error exceeded
        :attr:`~pybmodes.options.SolverOptions.residual_retry_threshold`
        and the result was redone through the general dense path. The
        symmetric routines factorise the mass matrix, which a very light
        beam carrying a very heavy lump makes nearly singular; there they
        return wrong low modes rather than failing, so the residual is
        what catches it.
    max_residual : the largest per-mode relative residual
        ``||K x - λ M x|| / ||K x||`` over the returned modes (``0.0``
        when no modes were returned). A healthy modal solve sits near
        machine precision; a large value flags an ill-conditioned or
        defective eigenproblem.

        Measured against the matrices the returned modes **actually
        solve**: the symmetrised pair on the symmetric paths, which
        symmetrise internally, and the raw pair on the general one.
        Measuring a symmetric solve against the raw matrices would charge
        it for the skew it was told to discard — on a model with a wide
        dynamic range that reads as a large backward error for a solve
        that is exact, which is the opposite of what this field is for.
    residuals : the per-mode relative residuals, one per returned mode,
        on the same basis as ``max_residual``.
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
    residual_fallback: bool = False

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

    # Accuracy guarantee for the symmetric paths. Both ``eigh`` and
    # ``eigsh`` reduce ``K x = λ M x`` through a Cholesky factor of one of
    # the matrices, and that reduction degrades once the factored matrix
    # is nearly singular — a very light beam carrying a very heavy lump
    # does exactly that to ``M``. The failure is silent: LAPACK returns
    # confidently wrong low modes rather than raising. The backward error
    # catches it (healthy solves sit at ~1e-4 or below, degraded ones
    # above 1), and the general path, which factorises neither matrix,
    # stays exact there.
    # The matrices the returned modes actually solve. Both symmetric
    # paths symmetrise internally, so for them the diagnostics — and the
    # retry decision below — have to be measured against that pair, not
    # against the raw one. Reporting the raw backward error would flag a
    # correct solve as defective in telemetry meant to be auditable.
    res_k, res_m = (0.5 * (gk + gk.T), 0.5 * (gm + gm.T)) if sym else (gk, gm)

    residual_fallback = False
    # Only the *dense* symmetric path is retried, and that is a statement
    # about which matrix each routine factorises rather than a
    # convenience. ``eigh`` reduces through a Cholesky factor of the mass
    # matrix, which is the one this guard exists for. ``eigsh(sigma=0,
    # mode='normal')`` factorises ``K`` instead, so a near-singular ``M``
    # does not degrade it — the mesh sweep that motivated this work
    # returns correct frequencies on exactly the meshes large enough to
    # take the sparse path.
    #
    # Excluding it also removes a mismatch that would otherwise need
    # separate handling: ``which="LM"`` on ``OP = K^-1 M`` selects the
    # modes nearest zero *in magnitude*, while the retry selects the
    # algebraically smallest. With negative eigenvalues present — a
    # post-buckling ``run(gravity=...)`` column — those are different
    # sets, and a per-index comparison between them would be pairing
    # unrelated modes.
    #
    # The size ceiling matters only because a sparse solve that fails to
    # converge falls back to the dense path at *any* size, where an
    # unbounded ``eig`` could spend minutes on a result already in hand.
    if (
        sym
        and path == "dense_symmetric"
        and ngd <= _SOLVER_OPTIONS.residual_retry_max_ndof
    ):
        # Measure — and retry — against the matrices the symmetric paths
        # actually solved. Both symmetrise internally, and the accepted
        # skew is only guaranteed small relative to ``max|K|``: in a model
        # with a wide dynamic range it can still be large relative to a
        # soft mode's own eigenvalue. Judging an exact symmetric solve
        # against the unsymmetrised matrices would then show a residual
        # above the threshold, and ``eig`` on those same unsymmetrised
        # matrices would "win decisively" purely by answering a different
        # question — replacing a correct spectrum with the skew's.
        gk_s, gm_s = res_k, res_m
        sym_r = _modal_residuals(gk_s, gm_s, eigvals, eigvecs)
        if sym_r.size and float(sym_r.max()) > _SOLVER_OPTIONS.residual_retry_threshold:
            try:
                alt_vals, alt_vecs, ordering_sound = _general_spectrum_for_retry(
                    gk_s, gm_s, n_modes,
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                # The alternative is a best-effort second opinion, not a
                # requirement. A pencil defective enough to break the
                # symmetric reduction can also break ``eig``, and turning
                # that into a hard failure would make this guard destroy
                # usable results on exactly the inputs it was added to
                # help. Decline and keep what we have.
                _log.warning(
                    "solve_modes: residual retry failed (%r); keeping the "
                    "symmetric result", exc,
                )
                alt_vals = np.empty(0)
                alt_vecs = np.empty((eigvecs.shape[0], 0))
                ordering_sound = False
            if alt_vecs.size:
                _normalize_columns_l2(alt_vecs)
            alt_r = _modal_residuals(gk_s, gm_s, alt_vals, alt_vecs)
            improved, regressed = (
                _compare_candidate_modes(
                    sym_r, alt_r, alt_vals.size, eigvals.size,
                )
                if ordering_sound
                else (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
            )
            # Accepting replaces the whole spectrum, not just the modes
            # that prompted the retry, so a candidate that fixes one mode
            # while ruining another is not an improvement to the result.
            if improved.any() and not regressed.any():
                idx = int(np.argmax(np.where(improved, sym_r[:improved.size], 0.0)))
                warnings.warn(
                    f"the symmetric eigensolver returned "
                    f"{int(improved.sum())} mode(s) that do not satisfy "
                    f"K x = lambda M x — worst at index {idx}, backward "
                    f"error {sym_r[idx]:.2e} against {alt_r[idx]:.2e} from "
                    f"the general dense path. The returned modes come from "
                    f"the general solve, which factorises neither matrix. "
                    + _retry_cause(gm_s),
                    RuntimeWarning,
                    stacklevel=2,
                )
                eigvals, eigvecs = alt_vals, alt_vecs
                path = "dense_general"
                residual_fallback = True

    # Mode-count guarantee: the general path filters complex / non-
    # positive eigenvalues, so it can return fewer modes than requested.
    # Surface that rather than letting it pass silently (a downstream
    # broadcast would otherwise fail with an opaque shape error).
    #
    # Gate on modes actually *discarded*, not on the path label. Two
    # benign shortfalls would otherwise be reported as a defective
    # eigenproblem. Asking for more modes than the system has is one:
    # every path truncates to ``min(n_modes, ngd)``, which is a request
    # the caller can reasonably make. A residual retry is the other — it
    # relabels the path ``"dense_general"`` while preserving the whole
    # spectrum, so nothing was filtered, and a 117-DOF system asked for
    # 1000 modes would be reported as defective for returning its 117.
    n_returned = int(eigvecs.shape[1])
    n_available = ngd if n_modes is None else min(n_modes, ngd)
    if (
        path == "dense_general"
        and not residual_fallback
        and n_returned < n_available
    ):
        warnings.warn(
            f"solve_modes recovered only {n_returned} of the "
            f"{n_available} modes available via the general "
            f"(non-symmetric) eig path. The eigenproblem is likely "
            f"near-degenerate or defective (a non-symmetric "
            f"PlatformSupport block can do this); the missing modes had "
            f"complex or non-positive eigenvalues and were filtered out.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not return_diagnostics:
        return eigvals, eigvecs

    diagnostics = _build_diagnostics(
        res_k, res_m, eigvals, eigvecs, path=path, symmetric=sym,
        n_requested=n_modes, sparse_fallback=sparse_fallback,
        fallback_reason=fallback_reason,
        residual_fallback=residual_fallback,
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
    residual_fallback: bool = False,
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
        residual_fallback=residual_fallback,
    )


# Above this the mass matrix is ill-conditioned enough for the Cholesky
# reduction to be the credible culprit; below it, something else in the
# pencil is.
_MASS_COND_ATTRIBUTION = 1.0e8


def _retry_cause(gm: np.ndarray) -> str:
    """The explanatory half of the retry warning, attributed honestly.

    The near-singular mass matrix is the *motivating* case, not the only
    one: the symmetric reduction degrades on an ill-conditioned pencil
    generally, and a stiffness spectrum spanning 1e-16 to 1 with ``M = I``
    triggers this guard while ``cond(M) = 1``. Naming the mass matrix
    there would send the reader to check a mass distribution that is
    perfectly fine.

    So the cause is measured before it is asserted. The condition number
    is only computed on this branch, which is rare, and is skipped for a
    system large enough for the O(n^3) estimate to matter — where the
    text falls back to naming both possibilities.
    """
    if gm.shape[0] > _COND_DENSE_MAX:
        return (
            "This happens when the pencil is ill-conditioned — most often "
            "a nearly singular mass matrix, from a very light beam "
            "carrying a very heavy lump, but a very wide stiffness range "
            "does it too. Worth checking the section properties for an "
            "extreme mass or stiffness ratio."
        )
    try:
        cond = float(np.linalg.cond(gm))
    except np.linalg.LinAlgError:
        cond = float("inf")
    if cond > _MASS_COND_ATTRIBUTION:
        return (
            f"The symmetric reduction goes through a Cholesky factor of "
            f"the mass matrix, which is nearly singular here "
            f"(cond = {cond:.1e}) — a very light beam carrying a very "
            f"heavy lump does this. Worth checking the mass distribution "
            f"is the one you intended."
        )
    return (
        f"The mass matrix is well conditioned (cond = {cond:.1e}), so the "
        f"reduction was defeated by the pencil rather than by the mass: a "
        f"stiffness range wide enough to put a soft mode at the level of "
        f"roundoff will do it. Worth checking the section properties for "
        f"an extreme stiffness ratio."
    )


def _compare_candidate_modes(
    sym_r: np.ndarray,
    alt_r: np.ndarray,
    n_alt: int,
    n_sym: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-mode verdicts on the alternative: ``(improved, regressed)``.

    Both are needed because accepting the retry replaces the **whole**
    spectrum, not the modes that prompted it. A candidate that fixes one
    mode while ruining another is not an improvement to the result even
    though it is an improvement to that mode, so "some mode got
    decisively better" is only half the test; the other half is that no
    mode got decisively worse.

    A mode has regressed when it was acceptable, comes back worse, and
    lands above the regression floor. Any worsening counts at any ratio:
    the improvement side's margin answers a different question — rescue
    or noise — and borrowing it here left a mode free to slide from 0.02
    to 0.099 unflagged, which is the edge of tolerance.

    The landing bound is what keeps this usable rather than paralysing:
    a mode going from 5.7e-6 to 3.5e-5 is six times worse and three
    orders below anything that matters, and flagging it would block
    nearly every legitimate rescue.

    A mode already failing in the symmetric solve gets **no regression
    verdict**. Above the threshold neither candidate is trustworthy, and
    judging that region would let noise veto every rescue that happens to
    sit beside a free-free mode. ``max_residual`` still reports it.

    Rigid-body modes are kept out of the *acceptance* side by the
    resolution bar rather than by being identified, which three attempts
    established cannot be done reliably here — not by eigenvalue scale,
    not by strain, and not by which side of the failure threshold the
    residual happens to fall on. Dividing one roundoff quantity by
    another produces an arbitrary number: 12.4, 0.79 and 0.076 have all
    been measured on healthy models, and the last is *below* the failure
    threshold, so it looked exactly like a mode being resolved. The one
    thing roundoff reliably does not do is land near machine precision.

    The cost is a real case declined: a breakdown the alternative
    improves substantially without resolving is not acted on. Neither
    result is trustworthy there, so keeping the original and reporting
    the backward error is the honest outcome.

    Together the two verdicts give the guarantee the caller relies on: a
    mode that was acceptable can only end up above the regression floor
    by having *improved*, never as collateral.

    The comparison has to be **per mode**, not on the two maxima. A
    rigid-body mode's backward error is a ratio of two near-zero
    quantities and reads ~1 in *both* candidates however exact each is,
    so it sets a floor under the alternative's maximum: with one present,
    ``max(alt_r)`` stays near 1, and no improvement elsewhere can drive
    it below the required fraction of ``max(sym_r)`` unless the symmetric
    solve is worse still by that same fraction inverted. A free-free
    model with a genuinely corrupted elastic mode at a backward error of
    ~0.8 would sail through, which is exactly the breakdown this guard
    exists to catch.

    Comparing mode by mode removes the floor: the rigid modes contribute
    ~1 against ~1 and register as no improvement, while a corrupted
    elastic mode contributes ~0.8 against ~1e-9 and registers clearly.
    Both candidates are sorted ascending over the same spectrum (the
    retry preserves rigid-body modes for this reason), so equal indices
    describe the same mode.

    Returns two boolean masks over the compared modes, both empty when
    the alternative recovered fewer modes than the symmetric solve —
    losing a mode is never an improvement, whatever the residuals say.
    """
    empty = np.zeros(0, dtype=bool)
    if n_alt < n_sym:
        return empty, empty
    n = min(sym_r.size, alt_r.size)
    if n == 0:
        return empty, empty
    threshold = _SOLVER_OPTIONS.residual_retry_threshold
    factor = _SOLVER_OPTIONS.residual_retry_improvement
    sym, alt = sym_r[:n], alt_r[:n]
    # What separates a rescue from noise is the *size* of the win, not
    # which side of a line the candidate lands on. A rigid-body mode's
    # residual divides one near-zero quantity by another, so its value is
    # arbitrary — 12.4, 0.79 and 0.076 have all been measured on healthy
    # models, and the last is below the failure threshold, so no absolute
    # threshold can exclude it. Its *ratio*, though, stays around 11x to
    # 16x, while a genuine rescue improves by 1e5 to 1e10. Four orders
    # separate the two populations.
    #
    # The absolute bar is kept as a second condition for the case the
    # ratio cannot see: a wildly broken 1e6 against a candidate at 100
    # clears any ratio while both remain garbage.
    resolved = _SOLVER_OPTIONS.residual_retry_resolved
    improved = (sym > threshold) & (alt <= resolved) & (alt < factor * sym)
    # A mode that was acceptable must not come back materially worse.
    # Any worsening counts, at any ratio: the improvement side's margin
    # answers "is this a rescue or noise", which is a different question,
    # and borrowing it here left a mode free to slide from 0.02 to 0.099
    # unflagged. What bounds this instead is where the mode *lands* —
    # below the regression floor the change cannot matter, which is what
    # keeps harmless churn (5.7e-6 to 3.5e-5) from blocking every rescue.
    #
    # Modes already failing in the symmetric solve get no verdict at all.
    # Neither value is trustworthy there, and a rigid-body mode — whose
    # residual divides roundoff by roundoff and has been seen to read
    # 12.4 against 0.79 on a healthy pencil — lives entirely in that
    # region. Judging it would be judging noise, and doing so in this
    # direction would let that noise veto every legitimate rescue.
    # ``max_residual`` still reports such a mode to the caller.
    floor = _SOLVER_OPTIONS.residual_regression_floor
    regressed = (sym <= threshold) & (alt > floor) & (alt > sym)
    return improved, regressed


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


def _general_spectrum_for_retry(
    gk: np.ndarray, gm: np.ndarray, n_modes: int | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """The general solve as the retry in :func:`solve_modes` needs it.

    Separate from :func:`_solve_dense_general` so the asymmetric
    production path keeps the BModes-matching filter it is validated
    against, untouched.

    Two differences, both about making per-index comparison against a
    symmetric solve meaningful.

    **No sign filter.** ``eigh`` filters nothing, so discarding
    non-positive eigenvalues here would return a *different set* — the
    same length, since a truncated request backfills the gap from higher
    up — and equal indices would stop meaning equal modes. Both omissions
    are reachable: a free-free model's zero-frequency modes, and the
    negative eigenvalues an indefinite ``K`` produces once
    ``run(gravity=...)`` loads a column past its buckling weight.

    **A verified ordering.** Complex eigenvalues cannot be kept, and on a
    symmetric problem ``eig`` does emit a few rounding-induced conjugate
    pairs — routinely, and harmlessly, because they land at the stiff end
    of the spectrum far above any mode a caller asks for. Demanding that
    none appear is therefore too strict to be useful. What actually
    matters is narrower: whether anything discarded would have fallen
    *inside* the returned window. The third return value reports that,
    and the caller declines to swap when it is ``False``, since an
    unverifiable ordering is not a basis for replacing a result.
    """
    vals_all, vecs_all = eig(gk, gm)
    closed = np.real_if_close(vals_all, tol=1000)
    keep_mask = np.isreal(closed) & np.isfinite(closed.real)

    vals = closed.real[keep_mask]
    vecs = np.real_if_close(vecs_all[:, keep_mask], tol=1000).real
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    keep = vals.size if n_modes is None else min(n_modes, vals.size)
    dropped = vals_all[~keep_mask]
    ordering_sound = True
    if dropped.size and keep:
        # Sound exactly when every discarded eigenvalue sits above the
        # window, so the window really is the smallest ``keep`` modes.
        ordering_sound = bool(
            np.min(dropped.real) > vals[keep - 1]
        )
    return vals[:keep], vecs[:, :keep], ordering_sound


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
