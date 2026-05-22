"""Tests for the degenerate FA/SS eigenpair resolver in elastodyn.params.

The resolver detects consecutive modes whose relative frequency gap is below
``_DEGENERATE_FREQ_RTOL`` (1e-4) and rotates them inside their shared 2D
eigenspace so the first comes out FA-pure and the second SS-pure. This
removes a class of classifier ambiguity on perfectly symmetric structures
where the eigensolver's basis choice within the degenerate subspace is
arbitrary.
"""

from __future__ import annotations

import math
import pathlib
import warnings

import numpy as np
import pytest

from pybmodes.elastodyn.params import (
    _BENDING_ACCEPT_THRESHOLD,
    _DEGENERATE_FREQ_RTOL,
    _TORSION_REJECT_THRESHOLD,
    _bending_purity,
    _is_degenerate_pair,
    _kinetic_participation,
    _resolve_degenerate_pair,
    _rotate_degenerate_pairs,
    _rotate_shape_pair,
    _select_tower_family,
    _shape_participation,
    _tower_candidate,
    compute_tower_params_report,
)
from pybmodes.fem.normalize import NodeModeShape
from pybmodes.fitting.poly_fit import fit_mode_shape
from pybmodes.models import Tower

from ._synthetic_bmi import write_bmi, write_uniform_sec_props

# ---------------------------------------------------------------------------
# 1. Symmetric tower → degenerate pair → resolver returns clean FA/SS
# ---------------------------------------------------------------------------

def _build_symmetric_tower(tmp_path: pathlib.Path) -> pathlib.Path:
    """Tower with EI_FA == EI_SS exactly and no tip mass: degenerate by construction."""
    bmi_path = write_bmi(
        tmp_path / "sym.bmi",
        beam_type=2, radius=80.0, hub_rad=0.0,
        tow_support=0, sec_props_file="sym.dat",
    )
    write_uniform_sec_props(
        tmp_path / "sym.dat",
        n_secs=8, mass_den=5000.0,
        flp_stff=5.0e10, edge_stff=5.0e10,
    )
    return bmi_path


def test_degenerate_symmetric_tower(tmp_path):
    """Resolver undoes any rotation in the degenerate subspace and pins the
    polynomial coefficients regardless of the input basis."""
    bmi_path = _build_symmetric_tower(tmp_path)
    modal = Tower(bmi_path).run(n_modes=4)

    # The eigensolver pair {mode 1, mode 2} should be detected as degenerate.
    assert _is_degenerate_pair(modal.shapes[0], modal.shapes[1]), (
        f"modes 1, 2 freqs = {modal.shapes[0].freq_hz}, {modal.shapes[1].freq_hz} "
        f"— expected within {_DEGENERATE_FREQ_RTOL} relative gap"
    )

    # Fixed reference: resolve the pair as the eigensolver returned it.
    fa_ref, ss_ref, _ = _resolve_degenerate_pair(modal.shapes[0], modal.shapes[1])
    fa_fit_ref = fit_mode_shape(fa_ref.span_loc, fa_ref.flap_disp)
    ss_fit_ref = fit_mode_shape(ss_ref.span_loc, ss_ref.lag_disp)

    # Rotated shapes should be FA-pure / SS-pure (no torsion in this case).
    p_fa, _ = _shape_participation(fa_ref)
    _, p_ss = _shape_participation(ss_ref)
    assert p_fa > 0.99, f"FA-aligned p_FA = {p_fa:.4f}"
    assert p_ss > 0.99, f"SS-aligned p_SS = {p_ss:.4f}"

    # Polynomial fits should be near-exact for a clean Euler-Bernoulli mode.
    assert fa_fit_ref.rms_residual < 0.001
    assert ss_fit_ref.rms_residual < 0.001

    # Stability: rotate the same pair by an arbitrary angle to simulate a
    # different eigensolver basis choice, run the resolver, and verify the
    # polynomial coefficients land in the same place.
    for theta_artif_deg in (37.0, 91.0, -19.0):
        theta = math.radians(theta_artif_deg)
        rot_a, rot_b = _rotate_shape_pair(modal.shapes[0], modal.shapes[1], theta)
        fa, ss, _ = _resolve_degenerate_pair(rot_a, rot_b)

        # Same purity check after the artificial rotation pre-stage.
        p_fa_, _ = _shape_participation(fa)
        _, p_ss_ = _shape_participation(ss)
        assert p_fa_ > 0.99, (
            f"After {theta_artif_deg}° pre-rotation: p_FA = {p_fa_:.4f}"
        )
        assert p_ss_ > 0.99, (
            f"After {theta_artif_deg}° pre-rotation: p_SS = {p_ss_:.4f}"
        )

        # Coefficients invariant to the input basis (modulo overall sign,
        # which fit_mode_shape already absorbs by tip-normalisation).
        fa_fit = fit_mode_shape(fa.span_loc, fa.flap_disp)
        np.testing.assert_allclose(
            fa_fit.coefficients(), fa_fit_ref.coefficients(),
            atol=1e-6,
            err_msg=f"FA coefficients drifted under {theta_artif_deg}° pre-rotation",
        )


# ---------------------------------------------------------------------------
# 2. Asymmetric tower → resolver does nothing
# ---------------------------------------------------------------------------

def test_nondegenerate_pair_unchanged(tmp_path):
    """Asymmetric stiffness ⇒ frequency gap above threshold ⇒ no rotation."""
    bmi_path = write_bmi(
        tmp_path / "asym.bmi",
        beam_type=2, radius=80.0, hub_rad=0.0,
        tow_support=0, sec_props_file="asym.dat",
    )
    write_uniform_sec_props(
        tmp_path / "asym.dat",
        n_secs=8, mass_den=5000.0,
        flp_stff=5.0e10, edge_stff=5.0e11,  # 10× ratio
    )
    modal = Tower(bmi_path).run(n_modes=4)

    gap = (
        abs(modal.shapes[1].freq_hz - modal.shapes[0].freq_hz)
        / modal.shapes[0].freq_hz
    )
    assert gap > _DEGENERATE_FREQ_RTOL, (
        f"expected non-degenerate, got relative gap = {gap:.2e}"
    )

    rotated = _rotate_degenerate_pairs(modal.shapes)

    # Returned list must contain the original NodeModeShape *instances*
    # (no rotation applied, no new objects created).
    assert len(rotated) == len(modal.shapes)
    for orig, new in zip(modal.shapes, rotated):
        assert new is orig, (
            f"mode {orig.mode_number}: expected the same instance back, "
            f"got a fresh copy (rotation triggered when it shouldn't have)"
        )


# ---------------------------------------------------------------------------
# 3. Nearly-degenerate pair — gap just above threshold, classifier still works
# ---------------------------------------------------------------------------

def test_nearly_degenerate_pair(tmp_path):
    """A small but above-threshold stiffness asymmetry: resolver does NOT
    fire (correctly), but the participation classifier still picks FA / SS
    cleanly because the eigensolver returns nearly-pure modes once the
    degeneracy has been lifted."""
    bmi_path = write_bmi(
        tmp_path / "near.bmi",
        beam_type=2, radius=80.0, hub_rad=0.0,
        tow_support=0, sec_props_file="near.dat",
        # 16 elements => 17 spanwise stations. The default n_elements=4
        # gives only 5 stations; the constrained 6th-order polynomial
        # design matrix is structurally zero at x=0 and x=1 (2 rows out
        # of 5), leaving 3 useful equations for 4 unknowns. The fit is
        # then rank-deficient and cond_number explodes to ~1e16,
        # tripping the WARN/FAIL warnings legitimately. Real decks have
        # nselt >= 30, so we use a realistic mesh here.
        n_elements=16,
    )
    EI_FA = 5.0e10
    # tor_stff is set high enough that the first torsion mode lands
    # above the 2nd bending pair. With the default helper value
    # (1.0e7) the torsion frequencies interleave between bending-1
    # and bending-2; the classifier's flap-vs-lag tiebreaker then
    # operates on floating-point noise for those twist-dominant modes
    # and the FA/SS label flips between LAPACK builds. With
    # tor_stff = 1.0e12 the first six eigenpairs are clean
    # FA-1 / SS-1 / FA-2 / SS-2 / FA-3 / SS-3 across builds.
    write_uniform_sec_props(
        tmp_path / "near.dat",
        n_secs=8, mass_den=5000.0,
        flp_stff=EI_FA, edge_stff=EI_FA * 1.0005,  # 0.05 % EI diff
        tor_stff=1.0e12,
    )
    # n_modes >= 6 per the Tower.run() docstring: with n_modes <= 4
    # scipy.linalg.eigh invokes a subset eigenvalue routine that can
    # return a different basis in the near-degenerate FA/SS subspace
    # depending on the LAPACK build.
    modal = Tower(bmi_path).run(n_modes=6)

    gap = (
        abs(modal.shapes[1].freq_hz - modal.shapes[0].freq_hz)
        / modal.shapes[0].freq_hz
    )
    assert gap > _DEGENERATE_FREQ_RTOL, (
        f"frequency gap {gap:.2e} should be > threshold {_DEGENERATE_FREQ_RTOL}; "
        f"either tighten the EI ratio or relax the threshold"
    )

    # Resolver should not fire.
    rotated = _rotate_degenerate_pairs(modal.shapes)
    for orig, new in zip(modal.shapes, rotated):
        assert new is orig

    # Even without rotation the classifier should still hand back sensible
    # FA / SS polynomial fits — the asymmetry is enough for the eigensolver
    # to emit modes that are already FA-dominant and SS-dominant.
    params, report = compute_tower_params_report(modal)
    assert params.TwFAM1Sh.rms_residual < 0.01
    assert params.TwSSM1Sh.rms_residual < 0.01
    # The two selected first-family modes must be different mode numbers
    # (the classifier mustn't have collapsed them onto the same eigenmode).
    assert report.selected_fa_modes[0] != report.selected_ss_modes[0]


# ---------------------------------------------------------------------------
# 4. IEA-3.4 deck — live regression test for the original failure case
# ---------------------------------------------------------------------------

_IEA34_MAIN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "external/OpenFAST_files/IEA-3.4-130-RWT/openfast/IEA-3.4-130-RWT_ElastoDyn.dat"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not _IEA34_MAIN.is_file(),
    reason=f"IEA-3.4 ElastoDyn deck not present at {_IEA34_MAIN}",
)
def test_iea34_no_degeneracy_warning():
    """The IEA-3.4 modes 1-2 should resolve to clean FA/SS shapes with no
    warning emitted.

    The deck's tower is symmetric in stiffness; with a sufficiently large
    eigenvalue request scipy/LAPACK returns the degenerate pair at
    *identical* frequencies but in an arbitrarily-rotated basis (≈ 52/48
    FA/SS mix). We use ``n_modes=10`` here to match what the case-study
    script in ``cases/iea3mw_land/run.py`` exercises and to put the
    eigensolver on the full-solve path; smaller subset requests produce
    slightly-different numerical answers where the eigensolver lifts the
    degeneracy artificially and the modes come back already-separated.
    """
    tower = Tower.from_elastodyn(_IEA34_MAIN)
    modal = tower.run(n_modes=10)

    # Sanity: this deck is expected to produce a degenerate first pair
    # (gap exactly zero on a full solve — the structural model is symmetric
    # in stiffness and the rigid-RNA c.m. offset is small enough not to
    # lift the degeneracy at the FEM precision we operate at).
    assert _is_degenerate_pair(modal.shapes[0], modal.shapes[1]), (
        f"expected a degenerate first pair from the full solve; got "
        f"freqs {modal.shapes[0].freq_hz:.6f}, {modal.shapes[1].freq_hz:.6f}"
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        rotated = _rotate_degenerate_pairs(modal.shapes)
    runtime_warnings = [w for w in captured if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, (
        "unexpected RuntimeWarning(s) from the resolver:\n  "
        + "\n  ".join(str(w.message) for w in runtime_warnings)
    )

    # First rotated shape should be FA-pure (was ~ 0.52 pre-rotation).
    p_fa, _ = _shape_participation(rotated[0])
    assert p_fa > 0.99, (
        f"After rotation, FA-aligned mode has p_FA = {p_fa:.4f} "
        f"(expected > 0.99). Pre-rotation values were ~ 0.522 / 0.478."
    )
    # And the second SS-pure.
    _, p_ss = _shape_participation(rotated[1])
    assert p_ss > 0.99, f"SS-aligned mode has p_SS = {p_ss:.4f}"


# ---------------------------------------------------------------------------
# Torsion-contamination filter inside ``_select_tower_family``
# ---------------------------------------------------------------------------
#
# Three tests gate the user-facing contract added in this commit:
#
# 1. ``test_classifier_rejects_torsion_contaminated`` — a synthetic
#    candidate set with one torsion-contaminated FA mode (T_tor > 10 %)
#    drops that mode from the family selection; the next clean
#    candidate wins instead.
# 2. ``test_classifier_accepts_pure_bending`` — when every candidate
#    has T_tor < 10 %, no rejections happen and the lowest-fit-rms
#    mode is picked as usual.
# 3. ``test_iea34_mode_torsion_reported`` — integration: on the real
#    IEA-3.4-130-RWT deck, the family-selection report carries
#    populated torsion-participation values for every candidate;
#    modes with a small but non-zero twist content (~1-3 %) are NOT
#    rejected because they stay below the 10 % threshold.


def _node_shape(
    *,
    mode_number: int,
    freq_hz: float,
    flap_amp: float,
    lag_amp: float,
    twist_amp: float,
    n: int = 11,
) -> NodeModeShape:
    """Build a synthetic NodeModeShape with controlled FA / SS / twist
    amplitudes. Each component is a polynomial in span_loc shaped like
    a tower bending mode (zero at root, peaking at the tip)."""
    span = np.linspace(0.0, 1.0, n)
    shape_curve = span ** 2 * (3.0 - 2.0 * span)  # smooth 0→1 ramp
    return NodeModeShape(
        mode_number=mode_number,
        freq_hz=freq_hz,
        span_loc=span,
        flap_disp=flap_amp * shape_curve,
        flap_slope=np.zeros(n),
        lag_disp=lag_amp * shape_curve,
        lag_slope=np.zeros(n),
        twist=twist_amp * shape_curve,
    )


def test_classifier_rejects_torsion_contaminated() -> None:
    """A higher-frequency FA candidate with T_tor > 10 % is dropped
    from the selection; the next clean candidate becomes the 2nd FA."""
    # Mode 1: clean 1st FA bending — T_FA dominant, T_tor = 0.
    m1 = _node_shape(
        mode_number=1, freq_hz=0.33,
        flap_amp=1.0, lag_amp=0.0, twist_amp=0.0,
    )
    # Mode 2: torsion-contaminated FA — flap still dominates the
    # direction classification but twist energy is ≈ 30 % of total.
    m2 = _node_shape(
        mode_number=2, freq_hz=1.50,
        flap_amp=1.0, lag_amp=0.0, twist_amp=0.65,
    )
    # Mode 3: clean 2nd FA bending — should win the second slot.
    m3 = _node_shape(
        mode_number=3, freq_hz=2.10,
        flap_amp=1.0, lag_amp=0.0, twist_amp=0.0,
    )

    # Sanity: mode 2 actually trips the threshold.
    _, _, ttor_m2 = _kinetic_participation(m2)
    assert ttor_m2 > _TORSION_REJECT_THRESHOLD, (
        f"synthetic mode 2's torsion fraction = {ttor_m2:.3f}; "
        f"must exceed {_TORSION_REJECT_THRESHOLD} to trigger the gate"
    )

    candidates = [_tower_candidate(s) for s in (m1, m2, m3)]
    sel = _select_tower_family(candidates, is_fa=True)
    # Mode 2 is dropped, mode 3 takes the 2nd-FA slot.
    assert sel.first.shape.mode_number == 1
    assert sel.second.shape.mode_number == 3
    assert 2 in sel.rejected_modes, (
        f"expected mode 2 in rejected_modes; got {sel.rejected_modes!r}"
    )


def test_classifier_accepts_pure_bending() -> None:
    """When every FA candidate has T_tor < 10 %, no rejections happen
    and the family-selection report's rejected_modes list is empty."""
    shapes = [
        _node_shape(mode_number=k + 1, freq_hz=0.3 + k * 1.0,
                    flap_amp=1.0, lag_amp=0.0, twist_amp=0.05)
        for k in range(3)
    ]
    # Every shape should be well below the torsion threshold.
    for s in shapes:
        _, _, ttor = _kinetic_participation(s)
        assert ttor < _TORSION_REJECT_THRESHOLD, (
            f"shape mode={s.mode_number} has T_tor={ttor:.3f}; "
            "the fixture should keep torsion tiny"
        )
    candidates = [_tower_candidate(s) for s in shapes]
    sel = _select_tower_family(candidates, is_fa=True)
    assert sel.first.shape.mode_number == 1
    assert sel.second.shape.mode_number == 2
    assert sel.rejected_modes == (), (
        f"expected no rejections on clean inputs; got {sel.rejected_modes!r}"
    )


@pytest.mark.integration
def test_iea34_mode_torsion_reported() -> None:
    """On the real IEA-3.4-130-RWT land deck, every tower-family
    candidate has a populated torsion-participation value, and modes
    with a small twist content (≤ 5 %) are not rejected because they
    stay below the 10 % threshold."""
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
    deck = (
        REPO_ROOT / "external" / "OpenFAST_files" / "IEA-3.4-130-RWT"
        / "openfast" / "IEA-3.4-130-RWT_ElastoDyn.dat"
    )
    if not deck.is_file():
        pytest.skip(f"IEA-3.4 deck not present at {deck}")

    tower = Tower.from_elastodyn(deck)
    modal = tower.run(n_modes=10, check_model=False)
    _params, report = compute_tower_params_report(modal)

    # Sanity: every family member exposes the three new participation
    # fields, summing to 1.0 within float roundoff.
    members = (*report.fa_family, *report.ss_family)
    assert members, "no family members reported"
    for m in members:
        total = (m.fa_participation + m.ss_participation
                 + m.torsion_participation)
        assert math.isclose(total, 1.0, abs_tol=1.0e-9), (
            f"mode {m.mode_number}: participation sum = {total:.6f}; "
            "should be 1 within roundoff"
        )

    # Modes with twist content but below threshold must not be
    # rejected. We're forgiving on the exact mode number — the user's
    # original note pointed at "mode 8 with 2.2 % twist" but the
    # specific mode index depends on n_modes and the eigensolver
    # ordering. Assert the contract structurally: every reported
    # member with 0 < T_tor < threshold has torsion_rejected = False.
    for m in members:
        if 0.0 < m.torsion_participation < _TORSION_REJECT_THRESHOLD:
            assert not m.torsion_rejected, (
                f"mode {m.mode_number} with T_tor="
                f"{m.torsion_participation:.4f} was rejected, but it "
                f"falls below the {_TORSION_REJECT_THRESHOLD:.2f} "
                f"threshold and should be accepted"
            )

    # Sanity on the threshold itself.
    assert _BENDING_ACCEPT_THRESHOLD > _TORSION_REJECT_THRESHOLD, (
        "the bending-accept threshold should sit above the torsion-"
        "reject threshold"
    )


# ---------------------------------------------------------------------------
# Determinism: the rotation gate keys on bending separation, not on a
# twist-inclusive purity that flips across its threshold under floating-
# point / BLAS-ordering noise (the latent FA2/SS2-swap regression).
# ---------------------------------------------------------------------------


def _twisted_degenerate_pair(
    *, twist_amp: float, n: int = 21,
) -> tuple[NodeModeShape, NodeModeShape]:
    """A clean FA/SS degenerate bending pair where the FA mode carries a
    small twist content. The bending DOFs separate perfectly; only the
    twist-inclusive purity is dragged below the 0.99 gate."""
    span = np.linspace(0.0, 1.0, n)
    curve = span ** 2 * (3.0 - 2.0 * span)
    zeros = np.zeros(n)
    true_fa = NodeModeShape(
        mode_number=1, freq_hz=0.5, span_loc=span,
        flap_disp=curve, flap_slope=zeros,
        lag_disp=zeros, lag_slope=zeros,
        twist=twist_amp * curve,
    )
    true_ss = NodeModeShape(
        mode_number=2, freq_hz=0.5, span_loc=span,
        flap_disp=zeros, flap_slope=zeros,
        lag_disp=curve, lag_slope=zeros,
        twist=zeros,
    )
    return true_fa, true_ss


def test_degenerate_pair_rotates_despite_subthreshold_twist() -> None:
    """A degenerate bending pair given in a mixed basis still rotates to
    clean FA / SS when it carries a small (sub-torsion-threshold) twist.

    The twist amplitude (20 %) drags the *twist-inclusive* purity to
    ~0.96 — below the 0.99 accept gate — so the previous gate rejected the
    rotation, kept the mixed eigenvectors, and emitted a RuntimeWarning.
    The bending-only gate accepts it. Regression for the BLAS-ordering-
    dependent FA2/SS2 swap.
    """
    true_fa, true_ss = _twisted_degenerate_pair(twist_amp=0.2)
    mix_a, mix_b = _rotate_shape_pair(true_fa, true_ss, math.radians(30.0))

    # Sanity: the twist-inclusive metric on the aligned FA shape is below
    # the gate (this is what used to reject the rotation).
    fa_aligned, ss_aligned, _ = _resolve_degenerate_pair(mix_a, mix_b)
    p_fa_twist, _ = _shape_participation(fa_aligned)
    assert p_fa_twist < 0.99, (
        f"fixture should sit below the twist-inclusive gate; got {p_fa_twist:.4f}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # must NOT warn / reject
        rotated = _rotate_degenerate_pairs([mix_a, mix_b])

    assert rotated[0] is not mix_a, "expected the pair to be rotated, not kept"
    p_fa, _ = _bending_purity(rotated[0])
    _, p_ss = _bending_purity(rotated[1])
    assert p_fa > 0.99 and p_ss > 0.99, (
        f"bending separation should be clean; got p_FA={p_fa:.4f}, p_SS={p_ss:.4f}"
    )


def test_degenerate_resolution_basis_invariant_with_twist() -> None:
    """Resolving the same physical twisted pair from different input bases
    yields identical FA and SS polynomial coefficients (deterministic
    canonicalisation, independent of the eigensolver's arbitrary basis)."""
    true_fa, true_ss = _twisted_degenerate_pair(twist_amp=0.2)

    fa_coeffs: list[np.ndarray] = []
    ss_coeffs: list[np.ndarray] = []
    for ang in (0.0, 30.0, -75.0, 110.0):
        a, b = _rotate_shape_pair(true_fa, true_ss, math.radians(ang))
        rot = _rotate_degenerate_pairs([a, b])
        fa_coeffs.append(fit_mode_shape(rot[0].span_loc, rot[0].flap_disp).coefficients())
        ss_coeffs.append(fit_mode_shape(rot[1].span_loc, rot[1].lag_disp).coefficients())

    for fa, ss in zip(fa_coeffs[1:], ss_coeffs[1:]):
        np.testing.assert_allclose(fa, fa_coeffs[0], atol=1e-9)
        np.testing.assert_allclose(ss, ss_coeffs[0], atol=1e-9)
