"""Floating-platform rigid-body mode naming (1.3.0).

`ModalResult.mode_labels` names the platform rigid-body modes
(surge / sway / heave / roll / pitch / yaw) for a free-free floating
tower. These tests are self-contained (default suite): the bundled
samples are repo-shipped, same data-independence rule as
`test_floating_samples_spectra`.

Coverage:
1. Bundled floating samples (OC3 Hywind spar, IEA-15 UMaineSemi) —
   the six lowest modes are exactly the six platform DOFs; the
   flexible tower modes above them are unlabelled (`None`).
2. A cantilever / land sample — `mode_labels` is `None` entirely
   (no rigid-body modes; must not be mislabelled).
3. `classify_platform_modes` unit behaviour: a synthetic eigenvector
   whose base node moves in one platform DOF is named that DOF; modes
   beyond the rigid-body count are `None`.
4. `mode_labels` (including `None` entries) round-trips through the
   NPZ and JSON serialisers.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.fem.boundary import NESH, active_dof_indices
from pybmodes.fem.platform_modes import classify_platform_modes
from pybmodes.models import Tower
from pybmodes.models.result import ModalResult

_SAMPLES = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "pybmodes" / "_examples" / "sample_inputs"
    / "reference_turbines"
)
_DOF_SET = {"surge", "sway", "heave", "roll", "pitch", "yaw"}


@pytest.mark.parametrize("sample_id", [
    "07_nrel5mw_oc3hywind_spar",
    "09_iea15_umainesemi",
])
def test_floating_sample_rigid_modes_named(sample_id: str) -> None:
    """The six lowest modes of a bundled floating sample are exactly
    the six platform DOFs (each once); modes above are unlabelled."""
    bmi = _SAMPLES / sample_id / f"{sample_id}_tower.bmi"
    res = Tower(bmi).run(n_modes=12, check_model=False)

    assert res.mode_labels is not None
    assert len(res.mode_labels) == len(res.frequencies)

    first6 = res.mode_labels[:6]
    assert all(lbl is not None for lbl in first6), first6
    # Exactly one of each platform DOF among the rigid-body modes.
    assert set(first6) == _DOF_SET, first6
    # Flexible tower modes above the rigid-body cluster are not named.
    assert all(lbl is None for lbl in res.mode_labels[6:]), res.mode_labels[6:]


def test_cantilever_sample_not_mislabelled() -> None:
    """A clamped-base (land) sample has no rigid-body modes — the
    classifier must never run, so mode_labels stays None."""
    bmi = _SAMPLES / "01_nrel5mw_land" / "01_nrel5mw_land_tower.bmi"
    res = Tower(bmi).run(n_modes=6, check_model=False)
    assert res.mode_labels is None


def _one_dof_eigvecs(nselt: int, dof_local: int, n_modes: int) -> np.ndarray:
    """Build compact eigenvectors (free-free → active == all DOFs)
    where mode 0's base node moves purely in FEM base DOF
    ``dof_local`` (0=axial … 5=phi) and other modes are inert."""
    ndt = NESH * nselt + 6
    ev = np.zeros((ndt, n_modes))
    ev[NESH * nselt + dof_local, 0] = 1.0
    return ev


@pytest.mark.parametrize("dof_local,expected", [
    (0, "heave"),   # axial   → heave
    (1, "surge"),   # v_disp  → surge
    (2, "pitch"),   # v_slope → pitch
    (3, "sway"),    # w_disp  → sway
    (4, "roll"),    # w_slope → roll
    (5, "yaw"),     # phi     → yaw
])
def test_classifier_single_dof_unit(dof_local: int, expected: str) -> None:
    """A base node moving purely in one FEM DOF is named the matching
    platform DOF (pins the FEM→platform reorder)."""
    nselt = 4
    n_modes = 8
    ev = _one_dof_eigvecs(nselt, dof_local, n_modes)
    active = active_dof_indices(nselt, hub_conn=2)
    Mp = np.eye(6)  # identity metric: pure single-DOF motion is unambiguous
    labels = classify_platform_modes(ev, active, nselt, Mp)

    assert labels[0] == expected
    # Inert modes carry no energy → None; modes past the rigid count
    # are None regardless.
    assert all(lbl is None for lbl in labels[1:])


def _base_eigvecs(
    nselt: int, mode_dofs: dict[int, dict[int, float]], n_modes: int
) -> np.ndarray:
    """Compact free-free eigenvectors whose base node carries the given
    FEM-DOF amplitudes per mode. ``mode_dofs[m]`` maps a local FEM base
    DOF (0=axial … 5=phi) to its amplitude in mode ``m``."""
    ndt = NESH * nselt + 6
    base0 = NESH * nselt
    ev = np.zeros((ndt, n_modes))
    for m, dofs in mode_dofs.items():
        for d, amp in dofs.items():
            ev[base0 + d, m] = amp
    return ev


def test_classifier_global_assignment_recovers_true_owner() -> None:
    """Issue #93: on an asymmetric platform the greedy "argmax per mode,
    drop later duplicates" rule mislabelled the rigid-body modes.

    Reproduces the exact mechanism: mode 0 is a genuine surge mode that,
    through surge↔yaw coupling, carries a small parasitic yaw rotation.
    The yaw inertia is ~25× the surge mass, so that tiny rotation's
    *mass-weighted energy* (80 %) outweighs the surge translation
    (20 %) — its argmax is yaw. Mode 2 is the genuine yaw mode (100 %).

    Greedy gave mode 0 = "yaw" (stealing the label) and then starved the
    true yaw mode 2 to None — i.e. "surge mode classified as yaw, third
    mode unclassified". Global assignment hands yaw to its true owner
    (mode 2, 100 % > mode 0's 80 %); mode 0's best remaining DOF is
    surge at 20 %, below the dominance threshold, so it is honestly left
    None rather than mislabelled.
    """
    nselt = 4
    n_modes = 8
    # FEM base DOFs: 0=axial(heave) 1=v_disp(surge) 2=v_slope(pitch)
    # 3=w_disp(sway) 4=w_slope(roll) 5=phi(yaw).
    mode_dofs = {
        0: {1: 1.0, 5: 0.4},   # surge + parasitic yaw (mass-inflated)
        1: {3: 1.0},           # sway
        2: {5: 1.0},           # yaw  (true owner)
        3: {4: 1.0},           # roll
        4: {2: 1.0},           # pitch
        5: {0: 1.0},           # heave
    }
    ev = _base_eigvecs(nselt, mode_dofs, n_modes)
    active = active_dof_indices(nselt, hub_conn=2)
    # Diagonal mass metric with a large yaw (phi) inertia: a 0.4-rad
    # parasitic yaw on mode 0 then carries 0.4·25·0.4 = 4 of energy vs
    # the surge translation's 1, i.e. 80 % yaw / 20 % surge.
    Mp = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 25.0])
    labels = classify_platform_modes(ev, active, nselt, Mp)

    assert labels[2] == "yaw"            # true owner wins (was None)
    assert labels[0] is None             # coupled mode (was mislabelled "yaw")
    assert labels.count("yaw") == 1      # never named twice
    # The cleanly single-DOF rigid modes are still each named once.
    assert labels[1] == "sway"
    assert labels[3] == "roll"
    assert labels[4] == "pitch"
    assert labels[5] == "heave"
    named = [lbl for lbl in labels if lbl is not None]
    assert len(named) == len(set(named))  # no DOF named twice anywhere


def test_classifier_resolves_degenerate_pair_basis() -> None:
    """Issue #93 (robustness): a symmetric platform's surge≈sway pair is
    degenerate, so the eigensolver may return any rotation of that 2-D
    eigenspace. Fed a 45°-mixed basis — mode 0 = (surge+sway)/√2,
    mode 1 = (surge−sway)/√2, both at the same frequency — each mode
    reads 50 % surge / 50 % sway and the dominance threshold would leave
    both ``None``. Passing the (equal) frequencies lets the classifier
    rotate the degenerate pair back onto its axes and name them.
    """
    nselt = 4
    n_modes = 8
    inv = 1.0 / np.sqrt(2.0)
    # FEM base DOFs: 1 = v_disp (surge), 3 = w_disp (sway).
    mode_dofs = {
        0: {1: inv, 3: inv},    # (surge + sway)/√2
        1: {1: inv, 3: -inv},   # (surge − sway)/√2
    }
    ev = _base_eigvecs(nselt, mode_dofs, n_modes)
    active = active_dof_indices(nselt, hub_conn=2)
    Mp = np.eye(6)

    # Without frequencies the degeneracy can't be detected: the mixed
    # basis stays 50/50 and neither mode clears the dominance threshold.
    no_freq = classify_platform_modes(ev, active, nselt, Mp)
    assert no_freq[0] is None and no_freq[1] is None

    # With the (equal) frequencies the pair is rotated onto its axes and
    # cleanly named — order within the degenerate pair is immaterial.
    freqs = np.array([0.01, 0.01, 0.05, 0.06, 0.07, 0.08, 0.5, 0.6])
    labels = classify_platform_modes(ev, active, nselt, Mp, frequencies=freqs)
    assert {labels[0], labels[1]} == {"surge", "sway"}


def test_classifier_truncated_frequencies_no_crash() -> None:
    """Codex P2: a frequencies array shorter than the rigid-body block
    (e.g. an externally-built mode subset) must not index past its end —
    the degeneracy alignment is skipped and the global assignment still
    labels the modes."""
    nselt = 4
    n_modes = 8
    ev = _one_dof_eigvecs(nselt, 1, n_modes)        # mode 0 = pure surge
    active = active_dof_indices(nselt, hub_conn=2)
    # Only two frequencies for an 8-mode / 6-rigid problem.
    labels = classify_platform_modes(
        ev, active, nselt, np.eye(6), frequencies=np.array([0.01, 0.02]),
    )
    assert labels[0] == "surge"
    assert all(lbl is None for lbl in labels[1:])


def test_mode_labels_roundtrip_npz_json(tmp_path) -> None:
    """mode_labels (with None entries) round-trips through both
    serialisers."""
    from pybmodes.fem.normalize import NodeModeShape

    span = np.linspace(0.0, 1.0, 5)
    shapes = [
        NodeModeShape(
            mode_number=i + 1, freq_hz=0.01 * (i + 1), span_loc=span,
            flap_disp=np.zeros(5), flap_slope=np.zeros(5),
            lag_disp=np.zeros(5), lag_slope=np.zeros(5), twist=np.zeros(5),
        )
        for i in range(4)
    ]
    res = ModalResult(
        frequencies=np.array([0.01, 0.02, 0.03, 0.5]),
        shapes=shapes,
        mode_labels=["surge", "sway", "yaw", None],
    )

    npz = tmp_path / "r.npz"
    res.save(npz)
    assert ModalResult.load(npz).mode_labels == ["surge", "sway", "yaw", None]

    js = tmp_path / "r.json"
    res.to_json(js)
    assert ModalResult.from_json(js).mode_labels == ["surge", "sway", "yaw", None]

    # A result without labels still round-trips with mode_labels None.
    plain = ModalResult(frequencies=np.array([1.0]), shapes=shapes[:1])
    plain.save(tmp_path / "p.npz")
    assert ModalResult.load(tmp_path / "p.npz").mode_labels is None
