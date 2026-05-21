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

"""Map ModalResult + poly fit to named ElastoDyn input parameters."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.fitting.poly_fit import PolyFitResult, fit_mode_shape
from pybmodes.models.result import ModalResult

# Degenerate-pair resolver tunables. See ``_rotate_degenerate_pairs``.
_DEGENERATE_FREQ_RTOL = 1.0e-4
_DEGENERATE_PURITY_THRESHOLD = 0.99

# Polynomial-fit conditioning thresholds. The condition number of the
# reduced design matrix solved by ``fit_mode_shape`` depends only on the
# spanwise sampling locations, so it's the same for every fit done on
# a given tower (FA1, FA2, SS1, SS2 share one cond value). For typical
# tower meshes (10-15 stations on [0, 1]) we observe ~1e2-1e3. Anything
# above ``_FIT_COND_WARN`` flags the fit as suspect; above
# ``_FIT_COND_FAIL`` the basis is too ill-conditioned to trust the
# coefficient breakdown — although the reconstructed shape may still
# be fine, individual C2..C6 values can vary by orders of magnitude
# under tiny perturbations of the input data.
_FIT_COND_WARN = 1.0e4
_FIT_COND_FAIL = 1.0e6


@dataclass
class BladeElastoDynParams:
    """Polynomial fits for the three blade mode shapes required by ElastoDyn."""

    BldFl1Sh: PolyFitResult
    BldFl2Sh: PolyFitResult
    BldEdgSh: PolyFitResult

    def as_dict(self) -> dict[str, float]:
        """Flat {ElastoDyn_param_name: coefficient_value} for writer."""
        out: dict[str, float] = {}
        for ed_name, fit in [
            ("BldFl1Sh", self.BldFl1Sh),
            ("BldFl2Sh", self.BldFl2Sh),
            ("BldEdgSh", self.BldEdgSh),
        ]:
            for k, c in zip(range(2, 7), fit.coefficients()):
                out[f"{ed_name}({k})"] = float(c)
        return out


@dataclass
class TowerElastoDynParams:
    """Polynomial fits for the four tower mode shapes required by ElastoDyn.

    Field names match the OpenFAST tower sub-file (*_ElastoDyn_tower.dat):
      TwFAM1Sh / TwFAM2Sh - fore-aft modes 1 and 2
      TwSSM1Sh / TwSSM2Sh - side-side modes 1 and 2
    """

    TwFAM1Sh: PolyFitResult
    TwFAM2Sh: PolyFitResult
    TwSSM1Sh: PolyFitResult
    TwSSM2Sh: PolyFitResult

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ed_name, fit in [
            ("TwFAM1Sh", self.TwFAM1Sh),
            ("TwFAM2Sh", self.TwFAM2Sh),
            ("TwSSM1Sh", self.TwSSM1Sh),
            ("TwSSM2Sh", self.TwSSM2Sh),
        ]:
            for k, c in zip(range(2, 7), fit.coefficients()):
                out[f"{ed_name}({k})"] = float(c)
        return out


@dataclass(frozen=True)
class TowerFamilyMemberReport:
    """Diagnostic view of one scored FA/SS tower family candidate.

    ``fa_participation`` / ``ss_participation`` / ``torsion_participation``
    are normalised modal kinetic-energy fractions (unit-mass
    approximation: ``Σ φ_axis² / Σ φ_total²`` over all FEM nodes).
    Sum to 1 for every mode. ``torsion_rejected`` is ``True`` when
    the torsion fraction crosses ``_TORSION_REJECT_THRESHOLD`` (10 %);
    such modes are dropped from the family selection because they
    are no longer pure bending modes ElastoDyn's polynomial ansatz
    can faithfully represent.
    """

    mode_number: int
    frequency_hz: float
    family_rank: int
    is_fa: bool
    fa_rms: float
    ss_rms: float
    direction_ratio: float
    fit_rms: float
    fit_is_good: bool
    selected: bool
    fa_participation: float = 0.0
    ss_participation: float = 0.0
    torsion_participation: float = 0.0
    torsion_rejected: bool = False


@dataclass(frozen=True)
class TowerSelectionReport:
    """Structured report of tower-mode family scoring and final selection.

    ``rejected_fa_modes`` / ``rejected_ss_modes`` carry the mode
    numbers of family candidates that were dropped because their
    torsion-participation crossed
    :data:`_TORSION_REJECT_THRESHOLD`. They're empty when every
    candidate passed the gate.
    """

    fa_family: tuple[TowerFamilyMemberReport, ...]
    ss_family: tuple[TowerFamilyMemberReport, ...]
    selected_fa_modes: tuple[int, int]
    selected_ss_modes: tuple[int, int]
    rejected_fa_modes: tuple[int, ...] = ()
    rejected_ss_modes: tuple[int, ...] = ()


def _component_strength(span_loc: np.ndarray, displacement: np.ndarray) -> float:
    """Return a spanwise RMS-like displacement strength for classification."""
    x = np.asarray(span_loc, dtype=float)
    y = np.asarray(displacement, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            "span_loc and displacement must have the same shape for mode classification"
        )
    if y.size == 0:
        return 0.0
    if y.size == 1:
        return float(abs(y[0]))

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    span = float(x_sorted[-1] - x_sorted[0])
    if span <= 0.0:
        return float(np.sqrt(np.mean(y_sorted**2)))

    dx = np.diff(x_sorted)
    y2 = y_sorted**2
    area = np.sum(0.5 * dx * (y2[:-1] + y2[1:]))
    return float(np.sqrt(area / span))


def _classify_fa_dominant(
    span_loc: np.ndarray,
    fa_disp: np.ndarray,
    ss_disp: np.ndarray,
) -> bool:
    """Return True iff the FA component dominates the SS component over the span.

    Compares the spanwise RMS-like strengths of the two displacement curves.
    On a near-tie (relative gap below the default ``np.isclose`` tolerance),
    the tip displacement breaks the tie — that is the only place the tip
    appears in the rule.

    This is the single source of truth for FA-vs-SS classification across
    the blade and tower paths; both ``_is_fa_dominated`` and
    ``_tower_candidate`` route through it so a borderline mode lands on the
    same side regardless of which classifier is invoked.
    """
    fa = _component_strength(span_loc, fa_disp)
    ss = _component_strength(span_loc, ss_disp)
    if np.isclose(fa, ss):
        return abs(fa_disp[-1]) >= abs(ss_disp[-1])
    return fa > ss


def _is_fa_dominated(shape: NodeModeShape) -> bool:
    """True if spanwise flap/fore-aft motion dominates lag/side-side motion."""
    return _classify_fa_dominant(
        shape.span_loc, shape.flap_disp, shape.lag_disp
    )


def _sorted_modes(
    shapes: list[NodeModeShape],
    fa_dominated: bool,
) -> list[NodeModeShape]:
    """Return modes of the given type, sorted by ascending frequency.

    Sorting is what callers of ``compute_blade_params`` rely on to pick
    1st-flap / 2nd-flap / 1st-edge: the lowest-frequency match comes first.
    A user constructing ``ModalResult`` by hand may not feed shapes in
    frequency order, so we sort defensively rather than trust the caller.
    """
    selected = [s for s in shapes if _is_fa_dominated(s) == fa_dominated]
    selected.sort(key=lambda s: s.freq_hz)
    return selected


@dataclass
class _TowerModeCandidate:
    """One tower mode interpreted as either FA or SS bending for ElastoDyn fitting."""

    shape: NodeModeShape
    fa_disp: np.ndarray
    ss_disp: np.ndarray
    fa_rms: float
    ss_rms: float
    fit_disp: np.ndarray
    fit: PolyFitResult
    is_fa: bool


# Modal kinetic-energy participation thresholds used by the
# torsion-contamination filter inside ``_select_tower_family``.
#
# A tower bending mode is a clean candidate for ElastoDyn's polynomial
# ansatz only when one bending axis dominates and torsion stays well
# below the bending energy. The two thresholds split the unit cube
# of (T_FA, T_SS, T_tor) participations into three regions:
#
#   * FA candidate  — T_FA > _BENDING_ACCEPT  AND  T_tor < _TORSION_REJECT
#   * SS candidate  — T_SS > _BENDING_ACCEPT  AND  T_tor < _TORSION_REJECT
#   * hybrid / rejected — T_tor >= _TORSION_REJECT
#
# The 10 % torsion gate matches the convention used elsewhere in the
# codebase for "this mode is no longer pure bending" (see the hybrid-
# mode classifier in the Bir 2010 monopile case study). Tower modes
# typically sit at T_tor < 1 %; values in the 1-3 % range are "near
# threshold" and surface in the report without triggering rejection.
_TORSION_REJECT_THRESHOLD: float = 0.10
_BENDING_ACCEPT_THRESHOLD: float = 0.85


def _kinetic_participation(shape: NodeModeShape) -> tuple[float, float, float]:
    """Return ``(T_FA, T_SS, T_tor)`` modal kinetic-energy fractions.

    Computed as ``Σ φ_axis² / Σ φ_total²`` summed over all FEM nodes
    — the unit-mass-matrix approximation of ``φᵀ M φ`` per axis. For
    uniform towers this matches the exact M-weighted participation
    to roughly 1 %; for tapered or RNA-dominated towers the
    approximation is looser but still useful for the FA / SS / hybrid
    classification thresholds. Sums to 1 for every mode.
    """
    fa = float(np.sum(np.asarray(shape.flap_disp, dtype=float) ** 2))
    ss = float(np.sum(np.asarray(shape.lag_disp, dtype=float) ** 2))
    tw = float(np.sum(np.asarray(shape.twist, dtype=float) ** 2))
    total = fa + ss + tw
    if total <= 0.0:
        return (0.0, 0.0, 0.0)
    return (fa / total, ss / total, tw / total)


@dataclass(frozen=True)
class _TowerFamilySelectionConfig:
    """Selection knobs for choosing ElastoDyn FA/SS tower mode families."""

    good_fit_rms: float = 0.09


@dataclass(frozen=True)
class _TowerFamilyMemberScore:
    """Scored view of one candidate within an FA or SS tower family."""

    candidate: _TowerModeCandidate
    family_rank: int
    fit_is_good: bool
    direction_ratio: float
    fa_participation: float
    ss_participation: float
    torsion_participation: float
    torsion_rejected: bool


@dataclass(frozen=True)
class _TowerFamilySelectionResult:
    """Selected family members plus their scored candidate list.

    ``rejected_modes`` carries the mode numbers of candidates that
    failed the torsion-contamination gate; empty when every candidate
    in the family passed.
    """

    first: _TowerModeCandidate
    second: _TowerModeCandidate
    scores: tuple[_TowerFamilyMemberScore, ...]
    rejected_modes: tuple[int, ...] = ()

    def __iter__(self):
        """Support tuple-style unpacking as `(first, second)`."""
        yield self.first
        yield self.second


def _remove_root_rigid_motion(
    span_loc: np.ndarray,
    displacement: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """Remove rigid-body root translation/rotation before tower polynomial fitting.

    ElastoDyn tower polynomials enforce zero displacement and zero slope at the
    base. Offshore/support-compliant tower modes can include rigid-body motion at
    the base, so we subtract the affine root contribution first and fit the
    bending part only.
    """

    x = np.asarray(span_loc, dtype=float)
    y = np.asarray(displacement, dtype=float)
    yp = np.asarray(slope, dtype=float)
    return y - y[0] - yp[0] * x


# ---------------------------------------------------------------------------
# Degenerate-eigenpair resolution
# ---------------------------------------------------------------------------
#
# A perfectly axisymmetric tower (``EI_FA == EI_SS``, no asymmetric tip-mass
# inertia) has an exactly degenerate FA/SS bending pair. Inside that
# 2D eigenspace the eigensolver's choice of basis is arbitrary — it can
# return two clean pure-FA and pure-SS shapes, or any rotation thereof.
# Mixed eigenvectors fit polynomials just as accurately (the FA-direction
# component of a 50/50 mix is still a scaled copy of the true FA shape),
# but they make participation-based mode classification look ambiguous and
# the resulting ``ModalResult`` confusing to inspect downstream. We
# therefore detect such pairs and rotate them back to FA/SS alignment
# before running the classifier.

def _is_degenerate_pair(shape_i: NodeModeShape, shape_j: NodeModeShape) -> bool:
    """True when consecutive modes are within ``_DEGENERATE_FREQ_RTOL``."""
    f_i = max(abs(shape_i.freq_hz), 1.0e-30)
    return abs(shape_j.freq_hz - shape_i.freq_hz) / f_i < _DEGENERATE_FREQ_RTOL


def _shape_participation(shape: NodeModeShape) -> tuple[float, float]:
    """Return ``(p_FA, p_SS)`` participation fractions for a tower mode.

    Twist contributions are included in the denominator so the metric
    matches the diagnostic in ``cases/*/run.py``. Used as the post-rotation
    purity check; symmetric-degenerate pairs of a tower with no torsional
    coupling should rotate to ``p_FA ≥ 0.99`` / ``p_SS ≥ 0.99``.
    """
    fa = float(np.sum(shape.flap_disp ** 2))
    ss = float(np.sum(shape.lag_disp ** 2))
    tw = float(np.sum(shape.twist ** 2))
    total = fa + ss + tw
    if total <= 0.0:
        return (0.0, 0.0)
    return (fa / total, ss / total)


def _rotate_shape_pair(
    shape_i: NodeModeShape,
    shape_j: NodeModeShape,
    theta: float,
) -> tuple[NodeModeShape, NodeModeShape]:
    """Apply 2D rotation by ``theta`` (radians) to a shape pair.

    Builds two new ``NodeModeShape`` instances with rotated arrays:

        new_i = cos(θ) · shape_i + sin(θ) · shape_j
        new_j = -sin(θ) · shape_i + cos(θ) · shape_j

    Mode numbers, frequencies, and span locations are inherited from the
    inputs (``shape_i`` for new_i, ``shape_j`` for new_j) — the rotation
    is purely a basis change inside the degenerate eigenspace, so
    frequencies are preserved by construction.
    """
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    def rot(arr_i: np.ndarray, arr_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cos_t * arr_i + sin_t * arr_j, -sin_t * arr_i + cos_t * arr_j

    fd_a, fd_b = rot(shape_i.flap_disp, shape_j.flap_disp)
    fs_a, fs_b = rot(shape_i.flap_slope, shape_j.flap_slope)
    ld_a, ld_b = rot(shape_i.lag_disp, shape_j.lag_disp)
    ls_a, ls_b = rot(shape_i.lag_slope, shape_j.lag_slope)
    tw_a, tw_b = rot(shape_i.twist, shape_j.twist)

    return (
        NodeModeShape(
            mode_number=shape_i.mode_number,
            freq_hz=shape_i.freq_hz,
            span_loc=shape_i.span_loc.copy(),
            flap_disp=fd_a, flap_slope=fs_a,
            lag_disp=ld_a, lag_slope=ls_a,
            twist=tw_a,
        ),
        NodeModeShape(
            mode_number=shape_j.mode_number,
            freq_hz=shape_j.freq_hz,
            span_loc=shape_j.span_loc.copy(),
            flap_disp=fd_b, flap_slope=fs_b,
            lag_disp=ld_b, lag_slope=ls_b,
            twist=tw_b,
        ),
    )


def _resolve_degenerate_pair(
    shape_i: NodeModeShape,
    shape_j: NodeModeShape,
) -> tuple[NodeModeShape, NodeModeShape, float]:
    """Rotate a degenerate FA/SS pair to align with the FA and SS axes.

    Returns ``(fa_aligned, ss_aligned, theta)``. The rotation angle is
    derived analytically from the FA-DOF projections of the input
    eigenvectors:

        a² = ‖shape_i.flap_disp‖²
        b² = ‖shape_j.flap_disp‖²
        c  = ⟨shape_i.flap_disp, shape_j.flap_disp⟩
        θ  = ½ · arctan2(2·c, a² − b²)

    This θ maximises ‖fa_aligned.flap_disp‖² over all rotations of the
    2D basis, equivalently moving as much FA content as possible into
    the first returned shape (and the orthogonal SS content into the
    second).
    """
    fa_i = shape_i.flap_disp
    fa_j = shape_j.flap_disp
    a2 = float(np.dot(fa_i, fa_i))
    b2 = float(np.dot(fa_j, fa_j))
    cross = float(np.dot(fa_i, fa_j))
    theta = 0.5 * float(np.arctan2(2.0 * cross, a2 - b2))
    fa_aligned, ss_aligned = _rotate_shape_pair(shape_i, shape_j, theta)
    return fa_aligned, ss_aligned, theta


def _rotate_degenerate_pairs(
    shapes: list[NodeModeShape],
) -> list[NodeModeShape]:
    """Return a copy of ``shapes`` with degenerate FA/SS pairs rotated.

    Walks consecutive mode pairs. When a pair's relative frequency gap
    is below ``_DEGENERATE_FREQ_RTOL`` (1e-4), runs
    :func:`_resolve_degenerate_pair` and verifies the rotated pair has
    p_FA ≥ 0.99 in the first slot and p_SS ≥ 0.99 in the second. If both
    pass, the rotated shapes replace the originals in the returned list.
    Otherwise the originals are kept and a ``RuntimeWarning`` is emitted
    — the pair is genuinely coupled (e.g. anisotropic stiffness or
    asymmetric tip mass that the resolver's symmetric model can't undo)
    and downstream classification has to handle it via participation
    ratio alone.

    Three-fold and higher degeneracies are not handled here; the loop
    advances by 2 after a successful rotation so overlapping pairs aren't
    re-rotated, and 3+ identical eigenvalues are vanishingly rare in
    tower modal analysis.

    The input list is not mutated; the returned list is a fresh copy.
    """
    out = list(shapes)
    n = len(out)
    i = 0
    while i < n - 1:
        if _is_degenerate_pair(out[i], out[i + 1]):
            fa_aligned, ss_aligned, _theta = _resolve_degenerate_pair(out[i], out[i + 1])
            p_fa, _ = _shape_participation(fa_aligned)
            _, p_ss = _shape_participation(ss_aligned)
            if (p_fa >= _DEGENERATE_PURITY_THRESHOLD
                    and p_ss >= _DEGENERATE_PURITY_THRESHOLD):
                out[i], out[i + 1] = fa_aligned, ss_aligned
                i += 2
                continue
            warnings.warn(
                f"Degenerate eigenpair (modes "
                f"{out[i].mode_number}/{out[i + 1].mode_number} at "
                f"{out[i].freq_hz:.4f} Hz) did not separate cleanly into "
                f"FA / SS after rotation (p_FA={p_fa:.3f}, p_SS={p_ss:.3f}); "
                f"leaving original eigenvectors unchanged. The pair may be "
                f"genuinely coupled (anisotropic stiffness, asymmetric tip "
                f"mass, or torsion contamination) rather than a clean "
                f"symmetric degeneracy.",
                RuntimeWarning,
                stacklevel=3,
            )
        i += 1
    return out


def _tower_candidate(shape: NodeModeShape) -> _TowerModeCandidate:
    """Build the best FA/SS interpretation of a tower mode for ElastoDyn fitting."""

    fa_disp = _remove_root_rigid_motion(shape.span_loc, shape.flap_disp, shape.flap_slope)
    ss_disp = _remove_root_rigid_motion(shape.span_loc, shape.lag_disp, shape.lag_slope)

    fa_strength = _component_strength(shape.span_loc, fa_disp)
    ss_strength = _component_strength(shape.span_loc, ss_disp)

    # Route through the shared classifier so blade and tower paths agree on
    # tie-handling: spanwise strength wins, the tip breaks ties.
    is_fa = _classify_fa_dominant(shape.span_loc, fa_disp, ss_disp)
    fit_disp = fa_disp if is_fa else ss_disp
    fit = fit_mode_shape(shape.span_loc, fit_disp)

    return _TowerModeCandidate(
        shape=shape,
        fa_disp=fa_disp,
        ss_disp=ss_disp,
        fa_rms=fa_strength,
        ss_rms=ss_strength,
        fit_disp=fit_disp,
        fit=fit,
        is_fa=is_fa,
    )


def _tower_family_candidates(
    candidates: list[_TowerModeCandidate],
    is_fa: bool,
) -> list[_TowerModeCandidate]:
    """Return one directional tower family sorted by ascending frequency."""
    family = [c for c in candidates if c.is_fa == is_fa]
    family.sort(key=lambda c: c.shape.freq_hz)
    return family


def _score_tower_family(
    family: list[_TowerModeCandidate],
    is_fa: bool,
    config: _TowerFamilySelectionConfig,
) -> list[_TowerFamilyMemberScore]:
    """Annotate a directional tower family with explicit selection metrics
    including modal kinetic-energy participation and the torsion-
    contamination flag."""
    scores: list[_TowerFamilyMemberScore] = []
    for idx, candidate in enumerate(family):
        major = candidate.fa_rms if is_fa else candidate.ss_rms
        minor = candidate.ss_rms if is_fa else candidate.fa_rms
        direction_ratio = float("inf") if minor == 0.0 else major / minor
        tfa, tss, ttor = _kinetic_participation(candidate.shape)
        torsion_rejected = bool(ttor >= _TORSION_REJECT_THRESHOLD)
        scores.append(
            _TowerFamilyMemberScore(
                candidate=candidate,
                family_rank=idx + 1,
                fit_is_good=candidate.fit.rms_residual <= config.good_fit_rms,
                direction_ratio=direction_ratio,
                fa_participation=tfa,
                ss_participation=tss,
                torsion_participation=ttor,
                torsion_rejected=torsion_rejected,
            )
        )
    return scores


def _select_tower_family(
    candidates: list[_TowerModeCandidate],
    is_fa: bool,
    config: _TowerFamilySelectionConfig | None = None,
) -> _TowerFamilySelectionResult:
    """Select the 1st/2nd FA or SS tower bending modes for ElastoDyn.

    Two-gate filter:

    1. **Direction**: only candidates classified as the requested
       axis (FA or SS) are considered.
    2. **Torsion contamination**: candidates whose modal kinetic-
       energy torsion fraction crosses
       :data:`_TORSION_REJECT_THRESHOLD` (10 %) are dropped from the
       selection — they're hybrid modes ElastoDyn's polynomial
       ansatz can't faithfully represent. Their mode numbers are
       returned in :attr:`_TowerFamilySelectionResult.rejected_modes`
       so the user can see what was dropped.

    Among the candidates that survive both gates, the lowest-frequency
    candidate becomes the first family member; the next higher-frequency
    candidate whose clamped-base polynomial fit is still good
    (``rms_residual <= 0.09``) becomes the second. If no such
    higher-frequency candidate has a good fit, we fall back to the
    best-fit candidate among the surviving ones.

    Robustness: if fewer than two candidates survive the torsion gate
    we fall back to the full (un-filtered) family and emit an empty
    ``rejected_modes`` — better to return SOMETHING than to crash on a
    pathological deck where every higher mode is torsion-contaminated.
    """

    config = config or _TowerFamilySelectionConfig()
    full_family = _tower_family_candidates(candidates, is_fa=is_fa)

    if len(full_family) < 2:
        kind = "FA" if is_fa else "SS"
        raise ValueError(
            f"Need >= 2 {kind} modes; found {len(full_family)}. "
            "Increase n_modes in Tower.run()."
        )

    # Score the full family so the report can show participations for
    # every candidate (including the rejected ones).
    full_scores = _score_tower_family(full_family, is_fa=is_fa, config=config)

    # Torsion-contamination filter: drop rejected candidates from the
    # selection pool. The full_scores tuple is preserved as-is so the
    # report shows which candidates were dropped.
    rejected_modes = tuple(
        s.candidate.shape.mode_number for s in full_scores if s.torsion_rejected
    )
    clean_scores = [s for s in full_scores if not s.torsion_rejected]
    if len(clean_scores) < 2:
        # Fall back to the full family; this is a "no clean candidates"
        # situation that callers should investigate, but the alternative
        # (raising) breaks the validate / patch flow on pathological
        # decks. Report empty rejected_modes so downstream consumers
        # don't see a misleading rejection list when we had to use the
        # contaminated modes anyway.
        clean_scores = list(full_scores)
        rejected_modes = ()

    first = clean_scores[0].candidate

    for score in clean_scores[1:]:
        if score.fit_is_good:
            return _TowerFamilySelectionResult(
                first=first,
                second=score.candidate,
                scores=tuple(full_scores),
                rejected_modes=rejected_modes,
            )

    second = min(
        clean_scores[1:], key=lambda s: s.candidate.fit.rms_residual,
    ).candidate
    return _TowerFamilySelectionResult(
        first=first,
        second=second,
        scores=tuple(full_scores),
        rejected_modes=rejected_modes,
    )


def _family_report(
    selection: _TowerFamilySelectionResult,
) -> tuple[TowerFamilyMemberReport, ...]:
    """Convert one internal family-selection result to a public report."""
    selected_modes = {
        selection.first.shape.mode_number,
        selection.second.shape.mode_number,
    }
    return tuple(
        TowerFamilyMemberReport(
            mode_number=score.candidate.shape.mode_number,
            frequency_hz=score.candidate.shape.freq_hz,
            family_rank=score.family_rank,
            is_fa=score.candidate.is_fa,
            fa_rms=score.candidate.fa_rms,
            ss_rms=score.candidate.ss_rms,
            direction_ratio=score.direction_ratio,
            fit_rms=score.candidate.fit.rms_residual,
            fit_is_good=score.fit_is_good,
            selected=score.candidate.shape.mode_number in selected_modes,
            fa_participation=score.fa_participation,
            ss_participation=score.ss_participation,
            torsion_participation=score.torsion_participation,
            torsion_rejected=score.torsion_rejected,
        )
        for score in selection.scores
    )


def compute_blade_params(modal: ModalResult) -> BladeElastoDynParams:
    """Fit polynomials to the 1st/2nd flap and 1st edge modes."""

    flap_modes = _sorted_modes(modal.shapes, fa_dominated=True)
    edge_modes = _sorted_modes(modal.shapes, fa_dominated=False)

    if len(flap_modes) < 2:
        raise ValueError(
            f"Need >= 2 flap modes; found {len(flap_modes)}. "
            "Increase n_modes in RotatingBlade.run()."
        )
    if len(edge_modes) < 1:
        raise ValueError(
            f"Need >= 1 edge mode; found {len(edge_modes)}. "
            "Increase n_modes in RotatingBlade.run()."
        )

    return BladeElastoDynParams(
        BldFl1Sh=fit_mode_shape(flap_modes[0].span_loc, flap_modes[0].flap_disp),
        BldFl2Sh=fit_mode_shape(flap_modes[1].span_loc, flap_modes[1].flap_disp),
        BldEdgSh=fit_mode_shape(edge_modes[0].span_loc, edge_modes[0].lag_disp),
    )


def compute_tower_params(modal: ModalResult) -> TowerElastoDynParams:
    """Fit polynomials to the 1st/2nd FA and 1st/2nd SS tower modes."""

    params, _ = compute_tower_params_report(modal)
    return params


def compute_tower_params_report(
    modal: ModalResult,
) -> tuple[TowerElastoDynParams, TowerSelectionReport]:
    """Compute tower ElastoDyn parameters and return a selection report.

    Pre-rotates any degenerate FA/SS eigenpair (consecutive modes whose
    relative frequency gap is below ``_DEGENERATE_FREQ_RTOL``) into clean
    direction-aligned shapes before classification. The original
    ``modal.shapes`` is not mutated; the rotation only feeds the candidate
    builder used here. See :func:`_rotate_degenerate_pairs`.

    Emits a ``RuntimeWarning`` if the polynomial-fit design-matrix
    condition number exceeds ``_FIT_COND_WARN`` (1e4) — the fit may be
    unreliable because the reduced Vandermonde basis is poorly
    conditioned at the mesh stations supplied. Emits a stronger warning
    above ``_FIT_COND_FAIL`` (1e6); the reconstructed shape may still be
    accurate but individual C2..C6 coefficients are likely to swing
    wildly under small perturbations of the input data.
    """
    shapes_for_fit = _rotate_degenerate_pairs(modal.shapes)
    candidates = [_tower_candidate(shape) for shape in shapes_for_fit]
    fa_sel = _select_tower_family(candidates, is_fa=True)
    ss_sel = _select_tower_family(candidates, is_fa=False)

    params = TowerElastoDynParams(
        TwFAM1Sh=fa_sel.first.fit,
        TwFAM2Sh=fa_sel.second.fit,
        TwSSM1Sh=ss_sel.first.fit,
        TwSSM2Sh=ss_sel.second.fit,
    )

    # Conditioning check. cond(A) is the same for all four fits since they
    # share the spanwise sampling — but emit one warning per affected fit
    # so the message names the specific coefficient block.
    for name, fit in [
        ("TwFAM1Sh", params.TwFAM1Sh),
        ("TwFAM2Sh", params.TwFAM2Sh),
        ("TwSSM1Sh", params.TwSSM1Sh),
        ("TwSSM2Sh", params.TwSSM2Sh),
    ]:
        if fit.cond_number > _FIT_COND_FAIL:
            warnings.warn(
                f"{name} polynomial fit: design-matrix condition number "
                f"{fit.cond_number:.2e} exceeds the FAIL threshold "
                f"{_FIT_COND_FAIL:.0e}; coefficient values are unreliable "
                f"(reconstructed shape may still be accurate, but C2..C6 "
                f"individually can swing by orders of magnitude under "
                f"small perturbations of the input data).",
                RuntimeWarning,
                stacklevel=2,
            )
        elif fit.cond_number > _FIT_COND_WARN:
            warnings.warn(
                f"{name} polynomial fit: design-matrix condition number "
                f"{fit.cond_number:.2e} exceeds the WARN threshold "
                f"{_FIT_COND_WARN:.0e}; fit may be unreliable for fine "
                f"coefficient comparisons.",
                RuntimeWarning,
                stacklevel=2,
            )

    report = TowerSelectionReport(
        fa_family=_family_report(fa_sel),
        ss_family=_family_report(ss_sel),
        selected_fa_modes=(fa_sel.first.shape.mode_number, fa_sel.second.shape.mode_number),
        selected_ss_modes=(ss_sel.first.shape.mode_number, ss_sel.second.shape.mode_number),
        rejected_fa_modes=fa_sel.rejected_modes,
        rejected_ss_modes=ss_sel.rejected_modes,
    )
    return params, report
