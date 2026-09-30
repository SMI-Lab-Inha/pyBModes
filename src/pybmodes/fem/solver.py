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
   matrices, applied to the *inverted* pencil ``M x = μ (K + s M) x``
   so that the lowest modes are resolved to relative rather than
   absolute accuracy (see :func:`_solve_dense_inverted`). Path used for
   small / mid-size symmetric problems and when the sparse path fails
   to converge (logged as a warning).
3. **Dense general** — ``scipy.linalg.eig`` for genuinely asymmetric
   systems (offshore decks where the rigid-arm transformation makes
   the platform-support block non-symmetric). Matches BModes JJ. Also
   the retry path when a symmetric solve comes back with a large
   backward error — see below.

The residual retry
------------------

``scipy.linalg.eigh(K, M)`` reduces ``K x = λ M x`` through a Cholesky
factor of the **mass** matrix, and that reduction degrades once ``M`` is
nearly singular — a very light beam carrying a very heavy lump. The
failure is silent: LAPACK returns confidently wrong low modes rather
than raising. On the case in ``tests/fem/test_ill_conditioned_mass.py``
it reported 0.103 Hz against a true 0.0436 Hz.

The dense symmetric path no longer reduces that way first: it solves
the inverted pencil, which factorises ``K`` and gets that case right
directly (``tests/fem/test_dense_inverted_solver.py``). The mass-reduced
form remains as the fallback for a pencil no shift makes definite, and
the retry below still stands behind it.

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

    # The residual retry — see the module docstring for the rule and for
    # why each of its clauses exists. In short: dense ``eigh`` reduces
    # through a Cholesky factor of the *mass* matrix and fails silently
    # when that is nearly singular, and the backward error is what
    # catches it.
    #
    # Everything below measures against the matrices the returned modes
    # actually solve — the symmetrised pair on a symmetric path, since
    # both symmetrise internally. That is passed as a flag rather than by
    # building the pair here: materialising it costs two dense ngd-square
    # allocations, and a large sparse solve would pay for them on every
    # call without ever needing them.
    residual_fallback = False
    # Dense symmetric only — ``eigsh`` factorises ``K``, so it does not
    # have this failure, and its mode window is a different set that must
    # not be index-compared. Size-capped because a sparse solve that
    # fails to converge falls back to dense at *any* size. Both reasons
    # in full in the module docstring.
    if (
        sym
        and path == "dense_symmetric"
        and ngd <= _SOLVER_OPTIONS.residual_retry_max_ndof
    ):
        # Measured against the matrices the symmetric paths actually
        # solved. Both symmetrise internally, and the accepted skew is
        # only guaranteed small relative to ``max|K|``: in a model with a
        # wide dynamic range it can still be large relative to a soft
        # mode's own eigenvalue. Judging an exact symmetric solve against
        # the unsymmetrised matrices would then show a residual above the
        # threshold, and ``eig`` on those same unsymmetrised matrices
        # would "win decisively" purely by answering a different question
        # — replacing a correct spectrum with the skew's.
        #
        # Through the products, not the pair: this runs on every eligible
        # solve, healthy ones included, and the threshold below has not
        # been tested yet.
        sym_r = _modal_residuals(gk, gm, eigvals, eigvecs, symmetrise=True)
        if sym_r.size and float(sym_r.max()) > _SOLVER_OPTIONS.residual_retry_threshold:
            # Now the pair is worth building: ``eig`` needs matrices
            # rather than products. Bounded by the size cap above, and
            # this branch is rare.
            gk_s = 0.5 * (gk + gk.T)
            gm_s = 0.5 * (gm + gm.T)
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
        gk, gm, eigvals, eigvecs, path=path, symmetric=sym,
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
    """Assemble a :class:`SolverDiagnostics` for a completed solve.

    ``symmetric`` selects the basis the residuals are measured on: a
    symmetric path solved the symmetrised pair, so charging its modes for
    the skew it was told to discard would report a correct solve as
    defective. After a residual retry the modes came from ``eig`` on that
    same symmetrised pair, so the basis is unchanged.
    """
    residuals = _modal_residuals(
        gk, gm, eigvals, eigvecs, symmetrise=symmetric,
    )
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


# The two routes to ``sym(A) v`` peak at ``3 n k`` and ``2 n^2`` bytes of
# temporaries, so they cross over at ``k = 2 n / 3`` — measured, not
# derived: a first estimate of ``n / 3`` was wrong by a factor of two and
# would have taken the more expensive route across a third of the range.
# Columns per pass of the residual sweep. Every temporary in the sweep
# is ``ngd x`` this, so the peak is bounded by it rather than by the
# number of modes the caller asked for.
#
# Only the peak depends on it — the result does not, since the residual
# is per mode and blocking merely partitions the columns. Chosen at the
# knee of the measured time curve: sweeping the full spectrum of a
# 2000-DOF system took 1.4 s at 16 and 32 columns, where per-call BLAS
# overhead dominates, then 0.76 s at 64 and 0.42 s at 128, against a
# 0.38 s floor that 256 and above buy with two to twelve times the peak.
_RESIDUAL_BLOCK = 128


def _prefer_materialised(n_rows: int, n_cols: int) -> bool:
    """Is ``sym(A) v`` cheaper built than split into two products?

    The split form peaks at three ``n x k`` temporaries and the built one
    at an ``n x n`` copy plus the ``n x k`` result, so they cross over at
    ``3 n k = 2 n^2``, i.e. ``k = 2n/3``.

    Measured, not derived. A flop-count estimate put the crossover at
    ``n/3``, which would have taken the dearer route across a third of
    the range.
    """
    return n_cols * 3 >= n_rows * 2


def _apply(a: np.ndarray, v: np.ndarray, symmetrise: bool) -> np.ndarray:
    """``A v``, or ``sym(A) v`` by whichever route is cheaper.

    ``0.5 (A + A.T) v == 0.5 (A v + A.T v)``, and the right-hand side
    avoids a dense ngd-square allocation — but only while ``v`` is
    narrow. At full width the two products cost more than the copy they
    were avoiding, so the route is tested rather than assumed.

    Blocking the caller's sweep does not remove the need for the test.
    It bounds the block at :data:`_RESIDUAL_BLOCK` columns, which is
    narrow relative to a large ``ngd`` but not to a small one: below
    ``ngd = 192`` a full-spectrum request still hands this a block wider
    than the crossover. Measured at ``ngd = 100``, testing the width
    there runs the sweep in 116 us against 178 us for the same peak.
    """
    if not symmetrise:
        return np.asarray(a @ v)
    if v.ndim > 1 and _prefer_materialised(a.shape[0], v.shape[1]):
        return np.asarray(0.5 * (a + a.T) @ v)
    return np.asarray(0.5 * (a @ v + a.T @ v))


def _modal_residuals(
    gk: np.ndarray, gm: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray,
    *, symmetrise: bool = False,
) -> np.ndarray:
    """Per-mode relative backward error ``||K x - λ M x|| / ||K x||``.

    The honest health metric for a generalised modal solve, and cheap
    enough to compute on every path.

    ``symmetrise`` measures against ``sym(A)`` instead, which is what a
    symmetric path actually solved.

    **Swept in column blocks.** The residual is per mode, so no step
    needs every mode present at once, and holding them all would tie the
    peak to a number the caller chooses: ``n_modes=None`` is the public
    default and returns the whole spectrum, which at the retry size cap
    would put four ngd-square arrays live at once — over 100 MB — purely
    to measure. Blocking bounds every temporary at ``ngd`` by
    :data:`_RESIDUAL_BLOCK` instead, and leaves the common thin case
    (a handful of modes out of a few thousand DOFs) in a single pass,
    computed exactly as before.
    """
    if eigvecs.size == 0:
        return np.empty(0, dtype=float)
    n_modes_out = eigvecs.shape[1]
    num = np.empty(n_modes_out, dtype=float)
    den = np.empty(n_modes_out, dtype=float)
    for lo in range(0, n_modes_out, _RESIDUAL_BLOCK):
        hi = min(lo + _RESIDUAL_BLOCK, n_modes_out)
        block = eigvecs[:, lo:hi]
        kx = _apply(gk, block, symmetrise)             # (ngd, <= block)
        mx = _apply(gm, block, symmetrise)
        den[lo:hi] = np.linalg.norm(kx, axis=0)
        num[lo:hi] = np.linalg.norm(
            kx - mx * eigvals[np.newaxis, lo:hi], axis=0,
        )
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
    """Dense symmetric solve on the symmetrised matrices. ``n_modes=None``
    requests the full spectrum; otherwise the lowest ``n_modes``.

    Routed through :func:`_solve_dense_inverted`, which factorises the
    stiffness side the way the sparse shift-invert path does, so the two
    paths answer to the same precision. The mass-reduced standard form is
    kept only as the fallback for a pencil no shift can make definite
    (:func:`_solve_dense_mass_reduced`); the residual retry in
    :func:`solve_modes` still stands behind that route.
    """
    gk_sym = 0.5 * (gk + gk.T)
    gm_sym = 0.5 * (gm + gm.T)
    try:
        return _solve_dense_inverted(gk_sym, gm_sym, n_modes)
    except np.linalg.LinAlgError as exc:
        _log.info(
            "solve_modes: no definite shift found (%r); using the "
            "mass-reduced dense eigh", exc,
        )
    return _solve_dense_mass_reduced(gk_sym, gm_sym, n_modes)


def _solve_dense_mass_reduced(
    gk_sym: np.ndarray, gm_sym: np.ndarray, n_modes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """LAPACK ``eigh(K, M)``: reduction through a Cholesky factor of ``M``.

    Every eigenvalue it returns carries an absolute error of order
    ``eps * lambda_max``. That is harmless when the spectrum is compact
    and ruinous when it is not — see :func:`_solve_dense_inverted`.
    """
    if n_modes is not None:
        subset = (0, min(n_modes, gk_sym.shape[0]) - 1)
        eigvals, eigvecs = eigh(gk_sym, gm_sym, subset_by_index=subset)
    else:
        eigvals, eigvecs = eigh(gk_sym, gm_sym)
    return np.asarray(eigvals), np.asarray(eigvecs)


# How many of the lowest eigenvalues set the scale of a shift. Enough to
# see past the six rigid-body modes of a free-free beam to the first
# elastic ones, few enough that the scale stays that of the low end.
_SHIFT_SCALE_MODES = 12
# Attempts at a definite shift, growing tenfold each time, before the
# mass-reduced route is used instead.
_SHIFT_ATTEMPTS = 20


def _solve_dense_inverted(
    gk_sym: np.ndarray, gm_sym: np.ndarray, n_modes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest modes of ``K x = λ M x`` from the inverted pencil.

    The standard reduction ``eigh(K, M)`` solves ``L^-1 K L^-T y = λ y``
    with ``M = L L^T``, and a backward-stable symmetric eigensolver gets
    each ``λ`` of that matrix to an **absolute** accuracy of about
    ``eps * λ_max``. The error on the lowest mode is therefore
    ``eps * λ_max / λ_min`` relative — the spectral range of the pencil,
    not ``cond(M)`` alone. A beam with a small rotary-inertia floor (the
    ElastoDyn adapters floor it at 1e-6) has ``λ_max / λ_min`` near 1e15,
    so the lowest modes come back with an error of order one, and one
    that moves with the ``subset_by_index`` window: the NREL 5MW blade's
    first flap mode at 12.1 rpm read 0.715 / 0.710 / 0.766 / 0.728 Hz for
    4 / 6 / 10 / 20 requested modes. A very heavy lump on a very light
    beam is the same failure reached through a near-singular ``M``.

    Solving ``M x = μ (K + s M) x`` instead, with ``μ = 1 / (λ + s)``,
    puts the wanted modes at the *top* of the spectrum, where the same
    absolute accuracy ``eps * μ_max`` is relative accuracy — the dense
    counterpart of the shift-invert the sparse path already uses. The
    returned modes then no longer depend on how many were requested.

    The shift ``s`` is zero when ``K`` is positive definite. Otherwise —
    rigid-body modes, or the negative eigenvalues a column loaded past
    its buckling weight has — it is set from a first estimate of the
    lowest eigenvalues and grown until ``K + s M`` factorises. The error
    on mode ``i`` is then about ``eps (λ_i + s)^2 / (λ_1 + s)``, and with
    ``s`` at least twice the magnitude of the lowest eigenvalues that
    stays relative to the modes in the window. A ``K`` that factorises
    despite rigid-body modes is left unshifted: their eigenvalues are
    roundoff of order ``eps λ_max``, which leaves the relative error of a
    low elastic mode near ``λ_i / λ_max``.

    The inverted form is relatively accurate at the low end only; the
    mass-reduced form at the high end. A request reaching into the upper
    half of the spectrum takes each mode from whichever form is more
    accurate there, so a full-spectrum request stays accurate throughout.

    Raises :class:`numpy.linalg.LinAlgError` when no shift makes the
    pencil definite (a singular ``M`` along a direction ``K`` does not
    stiffen), for the caller to fall back on.
    """
    from scipy.linalg import cho_factor

    ngd = gk_sym.shape[0]
    k = ngd if n_modes is None else min(n_modes, ngd)
    if k <= 0:
        return np.empty(0), np.empty((ngd, 0))

    def _definite(shift: float) -> bool:
        try:
            cho_factor(gk_sym + shift * gm_sym, lower=True,
                       check_finite=False)
        except np.linalg.LinAlgError:
            return False
        return True

    shift = 0.0
    if not _definite(shift):
        # The same estimate whatever ``k`` is, so the shift, and with it
        # every returned mode, does not depend on the requested window.
        rough = eigh(
            gk_sym, gm_sym, eigvals_only=True,
            subset_by_index=(0, min(ngd, _SHIFT_SCALE_MODES) - 1),
        )
        scale = float(np.max(np.abs(rough)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.finfo(float).eps) * (
                float(np.abs(np.diag(gk_sym)).max())
                / max(float(np.abs(np.diag(gm_sym)).max()), 1.0e-300)
            )
        shift = 2.0 * scale
        for _ in range(_SHIFT_ATTEMPTS):
            if _definite(shift):
                break
            shift *= 10.0
        else:
            raise np.linalg.LinAlgError(
                "no shift makes K + s M positive definite"
            )

    eigvals, eigvecs = _inverted_window(gk_sym, gm_sym, k, shift)
    if shift > 0.0:
        eigvals, eigvecs = _resolve_swamped_modes(
            gk_sym, gm_sym, k, shift, eigvals, eigvecs,
        )

    if 2 * k > ngd:
        eigvals, eigvecs = _splice_upper_spectrum(
            gk_sym, gm_sym, k, shift, eigvals, eigvecs,
        )
    return eigvals, eigvecs


def _inverted_window(
    gk_sym: np.ndarray, gm_sym: np.ndarray, k: int, shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` largest ``μ`` of ``M x = μ (K + s M) x``, as ascending
    ``λ = 1/μ - s``. ``μ <= 0`` (a massless direction) maps to ``+inf``
    and so sorts to the top, where the splice replaces it."""
    ngd = gk_sym.shape[0]
    mu, vecs = eigh(
        gm_sym, gk_sym + shift * gm_sym,
        subset_by_index=(ngd - k, ngd - 1),
    )
    mu = np.asarray(mu)[::-1]
    vecs = np.asarray(vecs)[:, ::-1]
    with np.errstate(divide="ignore"):
        lam = np.where(mu > 0.0, 1.0 / np.where(mu > 0.0, mu, 1.0) - shift,
                       np.inf)
    return lam, np.ascontiguousarray(vecs)


# A backward-stable solve of the inverted pencil fixes each μ to about
# ``ngd * eps * μ_max``; this factor is the safety margin on that bound.
_MU_RESOLUTION_ULPS = 100.0
# A mode is swamped by the shift when the uncertainty that μ resolution
# leaves on ``λ = 1/μ - s`` exceeds this fraction of ``|λ|``.
_SWAMPED_RTOL = 1.0e-8


def _resolve_swamped_modes(
    gk_sym: np.ndarray,
    gm_sym: np.ndarray,
    k: int,
    shift: float,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover modes a large shift has pushed below the μ resolution.

    With ``μ = 1/(λ + s)`` the uncertainty on ``λ`` is about
    ``δμ (λ + s)^2``. When ``s`` is much larger than the soft end of the
    spectrum (one strongly negative eigenvalue forces a large shift),
    every soft mode maps to the same ``μ`` to working precision: their
    order is arbitrary, a window edge falling among them picks any of
    them, and ``1/μ - s`` returns noise. ``K = diag(-1e16, 1, 2, 3, 4)``
    with ``M = I`` gave ``[-1e16, 4]`` for two modes.

    The subspace such a cluster spans is still well determined, because
    it is separated from the rest of the μ spectrum. So the window is
    grown until its edge falls in a resolvable gap, and each cluster that
    holds a swamped mode is re-solved by Rayleigh-Ritz on ``(K, M)``
    restricted to that subspace. A Rayleigh quotient is only quadratically
    sensitive to the cluster's mixing with the well-separated modes, so
    this recovers the soft modes to the accuracy ``(K, M)`` itself allows.
    The lowest ``k`` of the result are returned.
    """
    ngd = gk_sym.shape[0]
    eps = float(np.finfo(float).eps)

    def swamped(lam: np.ndarray, tol_mu: float) -> np.ndarray:
        finite = np.isfinite(lam)
        out = np.zeros(lam.shape, dtype=bool)
        dlam = tol_mu * (lam[finite] + shift) ** 2
        out[finite] = dlam > _SWAMPED_RTOL * np.abs(lam[finite])
        return out

    def mu_of(lam: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.where(np.isfinite(lam), 1.0 / (lam + shift), 0.0)

    mu_max = float(mu_of(eigvals[:1])[0])
    if not np.isfinite(mu_max) or mu_max <= 0.0:
        return eigvals, eigvecs
    tol_mu = _MU_RESOLUTION_ULPS * ngd * eps * mu_max
    if not swamped(eigvals, tol_mu).any():
        return eigvals, eigvecs

    # Grow the window until the μ just beyond it is resolvably below the
    # last μ inside it, so no cluster is cut in two.
    m = k
    while m < ngd:
        lam_w, vec_w = _inverted_window(gk_sym, gm_sym, m + 1, shift)
        mu_w = mu_of(lam_w)
        if mu_w[m - 1] - mu_w[m] > tol_mu:
            lam_w, vec_w = lam_w[:m], vec_w[:, :m]
            break
        m = min(ngd, 2 * m)
    else:
        lam_w, vec_w = _inverted_window(gk_sym, gm_sym, ngd, shift)
    mu_w = mu_of(lam_w)
    flagged = swamped(lam_w, tol_mu)

    lam_out = np.array(lam_w, dtype=float)
    vec_out = np.array(vec_w, dtype=float)
    start = 0
    for stop in range(1, len(mu_w) + 1):
        if stop < len(mu_w) and mu_w[stop - 1] - mu_w[stop] <= tol_mu:
            continue
        cols = np.arange(start, stop)
        cols = cols[np.isfinite(lam_w[cols])]
        if cols.size and flagged[cols].any():
            v = vec_w[:, cols]
            kr = v.T @ gk_sym @ v
            mr = v.T @ gm_sym @ v
            vals, y = eigh(0.5 * (kr + kr.T), 0.5 * (mr + mr.T))
            lam_out[cols] = vals
            vec_out[:, cols] = v @ y
        start = stop

    order = np.argsort(lam_out, kind="stable")[:k]
    return lam_out[order], np.ascontiguousarray(vec_out[:, order])


def _splice_upper_spectrum(
    gk_sym: np.ndarray,
    gm_sym: np.ndarray,
    k: int,
    shift: float,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Take each mode from whichever form resolves it more accurately.

    The inverted form's error on mode ``i`` is about
    ``eps (λ_i + s)^2 / (λ_1 + s)``; the mass-reduced form's is about
    ``eps λ_max``. Both are monotone in ``i``, so there is one crossover,
    and below it the two forms describe the same modes in the same order.
    """
    full_vals, full_vecs = eigh(gk_sym, gm_sym)
    full_vals = np.asarray(full_vals)
    lam_max = float(np.max(np.abs(full_vals)))
    base = float(eigvals[0]) + shift
    with np.errstate(over="ignore", invalid="ignore"):
        inverted_err = (eigvals + shift) ** 2 / base
    use_mass_reduced = ~(inverted_err < lam_max)
    if not use_mass_reduced.any():
        return eigvals, eigvecs
    cut = int(np.argmax(use_mass_reduced))
    vals = np.concatenate([eigvals[:cut], full_vals[cut:k]])
    vecs = np.concatenate(
        [eigvecs[:, :cut], np.asarray(full_vecs)[:, cut:k]], axis=1,
    )
    return vals, vecs


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
