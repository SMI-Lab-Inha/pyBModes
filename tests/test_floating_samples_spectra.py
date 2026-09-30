"""Regression: the bundled floating reference-turbine samples must
produce physically-valid, ``n_modes``-stable rigid-body spectra.

These tests read only repo-shipped sample data
(``src/pybmodes/_examples/sample_inputs/reference_turbines/``), so they
run in the default (self-contained) suite — same data-independence
rule as ``verify.py`` and ``test_reference_decks.py``.

Why this exists
---------------
Up to and including v1.1.0 the floating samples were emitted with
cantilever-proxy section properties (``axial_stff = 1e6·EI``,
~5e6× too stiff for a real tower, and a near-zero rotary-inertia
floor). For the clamped-base land / monopile samples that is harmless
(axial + torsion DOFs are locked at the base). For the FREE-base
floating samples (``hub_conn = 2``) it wrecked the conditioning of the
global matrices: on the OC3-Hywind-style asymmetric platform the soft
rigid-body modes collapsed into a single degenerate value whose
magnitude *drifted with the requested mode count* — the surge / sway /
heave / roll / pitch / yaw spectrum came out wrong while the
tower-bending pair stayed roughly right, so the build-time
"1st-FA > 0.3 Hz" check and the PlatformSupport round-trip test both
missed it. The fix emits physically-scaled section properties for the
floating path; these tests pin both invariants the old code violated:

1. The spectrum is ``n_modes``-invariant (a count-dependent eigenvalue
   is the signature of an ill-conditioned generalised eigenproblem).
2. The OC3 Hywind sample reproduces the BModes JJ reference spectrum
   (Jonkman 2010, NREL/TP-500-47535; BModes v1.03 JJ run) — the six
   rigid-body modes + first tower-bending pair — to engineering
   tolerance.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest

from pybmodes.models import Tower

_SAMPLES = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "pybmodes" / "_examples" / "sample_inputs"
    / "reference_turbines"
)

_FLOATING = [
    "07_nrel5mw_oc3hywind_spar",
    "08_nrel5mw_oc4semi",
    "09_iea15_umainesemi",
    "10_iea22_semi",
    "11_upscale25_centraltower",
]

# BModes JJ reference (Jonkman 2010 OC3 Hywind; BModes v1.03 JJ run,
# the same baseline test_certtest_oc3hywind validates the solver
# against on the canonical OC3Hywind.bmi deck). First nine modes:
# surge, sway, heave, roll, pitch, yaw, tower-FA, tower-SS, 2nd-FA.
_OC3_BMODES_JJ = np.array([
    0.0081183, 0.0081184, 0.0324283, 0.0393115, 0.0393271,
    0.1205100, 0.4815790, 0.4907540, 1.5696900,
])


def _tower_bmi(sample_id: str) -> pathlib.Path:
    return _SAMPLES / sample_id / f"{sample_id}_tower.bmi"


@pytest.mark.parametrize("sample_id", _FLOATING)
def test_floating_sample_spectrum_is_nmodes_stable(sample_id: str) -> None:
    """The first six (rigid-body) frequencies must not depend on how
    many modes were requested. A drift here means the global matrices
    are ill-conditioned — exactly the pre-fix failure mode."""
    bmi = _tower_bmi(sample_id)
    assert bmi.is_file(), f"bundled sample missing: {bmi}"

    f9 = Tower(bmi).run(n_modes=9, check_model=False).frequencies
    f15 = Tower(bmi).run(n_modes=15, check_model=False).frequencies

    drift = np.max(np.abs(f9[:6] - f15[:6]))
    assert drift < 1e-4, (
        f"{sample_id}: rigid-body spectrum drifts with n_modes "
        f"({drift:.3e} Hz between n=9 and n=15) — ill-conditioned "
        f"floating assembly.\n  n=9 : {np.array2string(f9[:6], precision=5)}"
        f"\n  n=15: {np.array2string(f15[:6], precision=5)}"
    )


@pytest.mark.parametrize("sample_id", _FLOATING)
def test_floating_sample_rigid_modes_not_collapsed(sample_id: str) -> None:
    """The six rigid-body modes must span a real range, not collapse
    into one degenerate value. surge≈sway is correct (axisymmetric
    horizontal plane); a *single* six-fold value is the bug."""
    f = Tower(_tower_bmi(sample_id)).run(n_modes=9, check_model=False).frequencies
    rigid = f[:6]
    assert np.all(rigid > 0.0) and np.all(np.isfinite(rigid))
    # Distinct-frequency count (cluster within 1%): a healthy floating
    # platform has surge≈sway and roll≈pitch but distinct heave and
    # yaw, so >= 3 distinct levels. The bug produced exactly 1.
    distinct = 1
    for i in range(1, 6):
        if rigid[i] > rigid[i - 1] * 1.01:
            distinct += 1
    assert distinct >= 3, (
        f"{sample_id}: rigid-body modes collapsed to {distinct} distinct "
        f"level(s): {np.array2string(rigid, precision=5)}"
    )


def test_oc3hywind_sample_matches_bmodes_jj() -> None:
    """The bundled OC3 Hywind sample reproduces the BModes JJ reference
    spectrum to engineering tolerance.

    Tolerance is 0.5 % — looser than the 0.01 % cert tolerance on the
    canonical OC3Hywind.bmi deck because this pyBmodes-authored sample
    recomputes the tower-top RNA mass (Hub + Nac + 3·Blade ≈ 347.3 t)
    rather than carrying OC3's literal 350 t, and synthesises the
    torsion constant from the thin-wall-tube identity. Both are
    documented modelling choices, not solver error; pre-fix this
    assertion would have failed by orders of magnitude (the rigid
    modes were a degenerate ~0.07–0.11 Hz cluster vs the correct
    0.008–0.12 Hz spread)."""
    f = Tower(_tower_bmi("07_nrel5mw_oc3hywind_spar")).run(
        n_modes=9, check_model=False
    ).frequencies

    rel_err = np.abs(f[:9] - _OC3_BMODES_JJ) / _OC3_BMODES_JJ
    worst = float(np.max(rel_err))
    assert worst < 5e-3, (
        f"OC3 Hywind sample vs BModes JJ: worst rel err {worst*100:.3f}% "
        f"(allowed 0.5%).\n  pyBmodes : {np.array2string(f[:9], precision=5)}"
        f"\n  BModes JJ: {np.array2string(_OC3_BMODES_JJ, precision=5)}"
    )


@pytest.mark.parametrize("sample_id", _FLOATING)
def test_floating_sample_is_statically_stable(sample_id: str) -> None:
    """Every bundled floater has positive roll / pitch restoring and returns
    all six rigid-body modes without the negative-stiffness warning.

    The OC4 DeepCwind sample used to fail this: its water ballast sits in
    HydroDyn filled members rather than ElastoDyn's PtfmMass, and until the
    deck constructors lumped that ballast into the platform the sample
    carried 3.85e6 kg of steel instead of 1.35e7 kg of platform, read
    C44 = C55 = -3.8e8 N·m/rad, and labelled tower-bending modes as roll
    and pitch."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = Tower(_tower_bmi(sample_id)).run(n_modes=9, check_model=False)
    assert set(res.mode_labels[:6]) == {
        "surge", "sway", "heave", "roll", "pitch", "yaw",
    }, (sample_id, res.mode_labels[:6])
